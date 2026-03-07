"""Tests for src.training.train_stats — StatsCallback, RewardMetricsAccumulator,
per-step/per-eval JSONL writers."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Mock transformers before importing train_stats, since StatsCallback.__init__
# does a deferred `from transformers import TrainerCallback`.
if "transformers" not in sys.modules:
    _mock_transformers = MagicMock()
    _mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    _mock_transformers.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    sys.modules["transformers"] = _mock_transformers

from src.training.train_stats import (
    RewardMetricsAccumulator,
    StatsCallback,
)


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    TRAIN_STATS_PER_STEP_JSONL: str = ""
    TRAIN_STATS_PER_EVAL_JSONL: str = ""
    TRAIN_STATS_FLUSH_EVERY_STEPS: int = 1
    TRAIN_CURVES_UPDATE_EVERY_STEPS: int = 1
    TRAIN_CURVES_COMPILE_EVERY_STEPS: int = 100
    TRAIN_CURVES_COMPILE_ENABLE: bool = False
    TRAIN_CURVES_COMPILE_AT_END: bool = False
    TRAIN_CURVES_TEX_ENABLE: bool = False
    TRAIN_CURVES_DIR: str = ""
    TRAIN_CURVES_KEEP_LAST_N_POINTS: int = 0


def _make_config(tmp_path) -> _StubConfig:
    step_jsonl = os.path.join(str(tmp_path), "per_step.jsonl")
    eval_jsonl = os.path.join(str(tmp_path), "per_eval.jsonl")
    return _StubConfig(
        TRAIN_STATS_PER_STEP_JSONL=step_jsonl,
        TRAIN_STATS_PER_EVAL_JSONL=eval_jsonl,
    )


def _make_state(global_step=1, epoch=0.5, log_history=None):
    return SimpleNamespace(
        global_step=global_step,
        epoch=epoch,
        log_history=log_history or [],
    )


# ---------------------------------------------------------------------------
# RewardMetricsAccumulator
# ---------------------------------------------------------------------------

class TestRewardMetricsAccumulator:

    def test_append_and_flush(self):
        acc = RewardMetricsAccumulator()
        acc.append_batch([{"r_correct": 1.0, "r_valid": 1.0}])
        acc.append_batch([{"r_correct": 0.0, "r_valid": 0.0}])
        result = acc.flush()
        assert len(result) == 2
        assert result[0]["r_correct"] == 1.0
        assert result[1]["r_correct"] == 0.0

    def test_flush_clears_buffer(self):
        acc = RewardMetricsAccumulator()
        acc.append_batch([{"r_correct": 1.0}])
        acc.flush()
        assert acc.flush() == []

    def test_thread_safety(self):
        acc = RewardMetricsAccumulator()
        errors = []

        def writer(batch_id):
            try:
                for _ in range(50):
                    acc.append_batch([{"batch_id": batch_id}])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        result = acc.flush()
        assert len(result) == 500


# ---------------------------------------------------------------------------
# StatsCallback — on_log writes JSONL
# ---------------------------------------------------------------------------

class TestStatsCallbackOnLog:

    def test_step_jsonl_written(self, tmp_path):
        """StatsCallback.on_log writes per-step JSONL records."""
        cfg = _make_config(tmp_path)
        acc = RewardMetricsAccumulator()
        cb = StatsCallback(cfg, acc)

        state = _make_state(global_step=1, epoch=0.1)
        logs = {"loss": 2.5, "learning_rate": 5e-5}
        cb.on_log(None, state, None, logs=logs)
        cb.flush_final()

        with open(cfg.TRAIN_STATS_PER_STEP_JSONL) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["global_step"] == 1
        assert "train_total_loss" in record

    def test_multiple_steps(self, tmp_path):
        """Three on_log calls → three JSONL lines."""
        cfg = _make_config(tmp_path)
        acc = RewardMetricsAccumulator()
        cb = StatsCallback(cfg, acc)

        for step in (1, 2, 3):
            state = _make_state(global_step=step, epoch=step * 0.1)
            cb.on_log(None, state, None, logs={"loss": 1.0 / step})

        cb.flush_final()

        with open(cfg.TRAIN_STATS_PER_STEP_JSONL) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_eval_jsonl_written(self, tmp_path):
        """Eval-prefixed log keys go to per-eval JSONL."""
        cfg = _make_config(tmp_path)
        acc = RewardMetricsAccumulator()
        cb = StatsCallback(cfg, acc)

        state = _make_state(global_step=10)
        logs = {"eval_loss": 0.5, "eval_reward_mean": 0.8}
        cb.on_log(None, state, None, logs=logs)

        with open(cfg.TRAIN_STATS_PER_EVAL_JSONL) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["eval_reward_mean"] == 0.8


# ---------------------------------------------------------------------------
# Per-step JSONL format
# ---------------------------------------------------------------------------

class TestPerStepJSONLFormat:

    def test_required_fields(self, tmp_path):
        """Each per-step record has global_step, epoch, wallclock_sec, step_time_sec."""
        cfg = _make_config(tmp_path)
        acc = RewardMetricsAccumulator()
        cb = StatsCallback(cfg, acc)

        state = _make_state(global_step=5, epoch=1.0)
        cb.on_log(None, state, None, logs={"loss": 0.5})
        cb.flush_final()

        with open(cfg.TRAIN_STATS_PER_STEP_JSONL) as f:
            record = json.loads(f.readline())
        assert "global_step" in record
        assert "epoch" in record
        assert "wallclock_sec" in record
        assert "step_time_sec" in record

    def test_reward_breakdown_included(self, tmp_path):
        """Reward accumulator records are aggregated into step JSONL."""
        cfg = _make_config(tmp_path)
        acc = RewardMetricsAccumulator()
        acc.append_batch([
            {"r_correct": 1.0, "r_valid": 1.0, "depth": 3,
             "is_invalid": False, "is_parse_fail": False, "response_len": 100},
        ])
        cb = StatsCallback(cfg, acc)

        state = _make_state(global_step=1)
        cb.on_log(None, state, None, logs={"loss": 0.5})
        cb.flush_final()

        with open(cfg.TRAIN_STATS_PER_STEP_JSONL) as f:
            record = json.loads(f.readline())
        assert "reward_correctness_mean" in record
        assert "validity_rate" in record


# ---------------------------------------------------------------------------
# Resume-safe append
# ---------------------------------------------------------------------------

class TestResumeSafe:

    def test_append_after_reopen(self, tmp_path):
        """Write 5 steps, close, reopen, write 5 more → JSONL has 10 lines."""
        cfg = _make_config(tmp_path)
        acc = RewardMetricsAccumulator()

        cb1 = StatsCallback(cfg, acc)
        for step in range(1, 6):
            cb1.on_log(None, _make_state(global_step=step), None, logs={"loss": 0.5})
        cb1.flush_final()

        cb2 = StatsCallback(cfg, acc)
        for step in range(6, 11):
            cb2.on_log(None, _make_state(global_step=step), None, logs={"loss": 0.5})
        cb2.flush_final()

        with open(cfg.TRAIN_STATS_PER_STEP_JSONL) as f:
            lines = f.readlines()
        assert len(lines) == 10


# ---------------------------------------------------------------------------
# Flush cadence
# ---------------------------------------------------------------------------

class TestFlushCadence:

    def test_flush_every_respects_cadence(self, tmp_path):
        """flush_every=3 → file has >= 9 lines after 10 steps (before explicit close)."""
        cfg = _make_config(tmp_path)
        cfg.TRAIN_STATS_FLUSH_EVERY_STEPS = 3
        acc = RewardMetricsAccumulator()
        cb = StatsCallback(cfg, acc)

        for step in range(1, 11):
            cb.on_log(None, _make_state(global_step=step), None, logs={"loss": 0.5})

        # Before flush_final, at least 9 lines should be on disk (3 flushes of 3)
        with open(cfg.TRAIN_STATS_PER_STEP_JSONL) as f:
            lines = f.readlines()
        assert len(lines) >= 9

        cb.flush_final()
        with open(cfg.TRAIN_STATS_PER_STEP_JSONL) as f:
            all_lines = f.readlines()
        assert len(all_lines) == 10


# ---------------------------------------------------------------------------
# Overfit-PoC diagnostics field
# ---------------------------------------------------------------------------

class TestOverfitPocDiagnostics:

    def test_tiny_train_correctness_in_eval(self, tmp_path):
        """Eval record with tiny_train_correctness_rate propagates to JSONL."""
        cfg = _make_config(tmp_path)
        acc = RewardMetricsAccumulator()
        cb = StatsCallback(cfg, acc)

        state = _make_state(global_step=10)
        logs = {
            "eval_loss": 0.3,
            "eval_tiny_train_correctness_rate": 0.875,
        }
        cb.on_log(None, state, None, logs=logs)

        with open(cfg.TRAIN_STATS_PER_EVAL_JSONL) as f:
            record = json.loads(f.readline())
        assert record["eval_tiny_train_correctness_rate"] == 0.875
