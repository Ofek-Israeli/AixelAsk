"""Tests for src.training.checkpointing — MetadataCallback, symlinks, best
checkpoint, checkpoint_index.json, config compatibility."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Mock transformers before importing checkpointing
if "transformers" not in sys.modules:
    _mock_transformers = MagicMock()
    _mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    _mock_transformers.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    sys.modules["transformers"] = _mock_transformers

from src.training.checkpointing import (
    check_resume_compatibility,
    make_metadata_callback,
    _update_latest_symlink,
    _write_checkpoint_metadata,
    _checkpoints_dir,
    _load_index,
    _save_index,
)


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    TRAIN_OUTPUT_DIR: str = ""
    TRAIN_SAVE_BEST_BY: str = "eval_reward_mean"
    TRAIN_SAVE_MERGED_ADAPTER: bool = False
    TRAINING_MODE: str = "GRPO"
    INFERENCE_MODEL: str = "model-a"
    INFERENCE_MODEL_REVISION: str = "v1"
    TRAIN_LORA_R: int = 16
    TRAIN_LORA_ALPHA: int = 32
    TRAIN_LORA_TARGET_MODULES: str = "q_proj,v_proj"
    TRAIN_LORA_DROPOUT: float = 0.05
    TRAIN_USE_4BIT: bool = True
    REWARD_MODE: str = "weighted"
    REWARD_WEIGHT_CORRECTNESS: float = 1.0
    REWARD_WEIGHT_VALIDITY: float = 0.5
    REWARD_WEIGHT_DEPTH: float = 0.1
    REWARD_WEIGHT_INVALID_PENALTY: float = 0.5
    REWARD_DEPTH_NORMALIZATION: str = "DIVIDE_BY_MAX_DEPTH"
    REWARD_MAX_DEPTH: int = 10
    REWARD_INVALID_IF_PARSE_FAILS: bool = True
    REWARD_CORRECTNESS_PARTIAL_CREDIT: bool = False

    # We need __dataclass_fields__ to exist for _make_config_snapshot
    @property
    def __dataclass_fields__(self):
        """Return the dataclass fields for config snapshot."""
        import dataclasses
        return {f.name: f for f in dataclasses.fields(self)}


def _make_config(tmp_path) -> _StubConfig:
    output_dir = str(tmp_path / "output")
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    return _StubConfig(TRAIN_OUTPUT_DIR=output_dir)


def _make_state(global_step=50, epoch=1.0, log_history=None):
    return SimpleNamespace(
        global_step=global_step,
        epoch=epoch,
        log_history=log_history or [],
    )


def _make_args(output_dir=""):
    return SimpleNamespace(output_dir=output_dir)


# ---------------------------------------------------------------------------
# Latest symlink updated by MetadataCallback
# ---------------------------------------------------------------------------

class TestLatestSymlink:

    def test_latest_points_to_most_recent(self, tmp_path):
        """Simulate save at step 50, then step 100 → latest → checkpoint-100."""
        cfg = _make_config(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")

        cb = make_metadata_callback(cfg)

        # Simulate step 50 save
        ckpt50 = os.path.join(ckpt_dir, "checkpoint-50")
        os.makedirs(ckpt50, exist_ok=True)
        state50 = _make_state(global_step=50)
        args = _make_args(output_dir=cfg.TRAIN_OUTPUT_DIR)
        cb.on_save(args, state50, None)

        latest = os.path.join(ckpt_dir, "latest")
        assert os.path.islink(latest)
        assert "checkpoint-50" in os.readlink(latest)

        # Simulate step 100 save
        ckpt100 = os.path.join(ckpt_dir, "checkpoint-100")
        os.makedirs(ckpt100, exist_ok=True)
        state100 = _make_state(global_step=100)
        cb.on_save(args, state100, None)

        assert "checkpoint-100" in os.readlink(latest)


# ---------------------------------------------------------------------------
# Best checkpoint on metric improvement
# ---------------------------------------------------------------------------

class TestBestCheckpoint:

    def test_best_updated_on_improvement(self, tmp_path):
        """metric=0.5 then 0.7 → best/ has step with 0.7. Then 0.6 → still 0.7."""
        cfg = _make_config(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")

        cb = make_metadata_callback(cfg)
        args = _make_args(output_dir=cfg.TRAIN_OUTPUT_DIR)

        # Step 50: eval_reward_mean=0.5
        ckpt50 = os.path.join(ckpt_dir, "checkpoint-50")
        os.makedirs(ckpt50, exist_ok=True)
        state50 = _make_state(
            global_step=50,
            log_history=[{"eval_reward_mean": 0.5, "step": 50}],
        )
        cb.on_save(args, state50, None)

        best_dir = os.path.join(ckpt_dir, "best")
        assert os.path.isdir(best_dir)

        # Step 100: eval_reward_mean=0.7 (improvement)
        ckpt100 = os.path.join(ckpt_dir, "checkpoint-100")
        os.makedirs(ckpt100, exist_ok=True)
        state100 = _make_state(
            global_step=100,
            log_history=[
                {"eval_reward_mean": 0.5, "step": 50},
                {"eval_reward_mean": 0.7, "step": 100},
            ],
        )
        cb.on_save(args, state100, None)

        meta_path = os.path.join(best_dir, "checkpoint_metadata.json")
        assert os.path.isfile(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["metric_value"] == pytest.approx(0.7)

        # Step 150: eval_reward_mean=0.6 (no improvement)
        ckpt150 = os.path.join(ckpt_dir, "checkpoint-150")
        os.makedirs(ckpt150, exist_ok=True)
        state150 = _make_state(
            global_step=150,
            log_history=[
                {"eval_reward_mean": 0.5, "step": 50},
                {"eval_reward_mean": 0.7, "step": 100},
                {"eval_reward_mean": 0.6, "step": 150},
            ],
        )
        cb.on_save(args, state150, None)

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["metric_value"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Best disabled when validation empty
# ---------------------------------------------------------------------------

class TestBestDisabledEmpty:

    def test_no_best_without_eval(self, tmp_path):
        """No eval metric in log_history → no best/ directory."""
        cfg = _make_config(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")

        cb = make_metadata_callback(cfg)
        args = _make_args(output_dir=cfg.TRAIN_OUTPUT_DIR)

        ckpt50 = os.path.join(ckpt_dir, "checkpoint-50")
        os.makedirs(ckpt50, exist_ok=True)
        state50 = _make_state(
            global_step=50,
            log_history=[{"loss": 0.5, "step": 50}],
        )
        cb.on_save(args, state50, None)

        best_dir = os.path.join(ckpt_dir, "best")
        assert not os.path.isdir(best_dir)


# ---------------------------------------------------------------------------
# Final checkpoint on train_end
# ---------------------------------------------------------------------------

class TestFinalCheckpoint:

    def test_final_created_on_train_end(self, tmp_path):
        """on_train_end creates final/ directory."""
        cfg = _make_config(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")

        cb = make_metadata_callback(cfg)
        args = _make_args(output_dir=cfg.TRAIN_OUTPUT_DIR)

        # Save step 50 so latest symlink exists
        ckpt50 = os.path.join(ckpt_dir, "checkpoint-50")
        os.makedirs(ckpt50, exist_ok=True)
        state50 = _make_state(global_step=50)
        cb.on_save(args, state50, None)

        # on_train_end
        cb.on_train_end(args, state50, None)

        final_dir = os.path.join(ckpt_dir, "final")
        assert os.path.isdir(final_dir)


# ---------------------------------------------------------------------------
# checkpoint_index.json format
# ---------------------------------------------------------------------------

class TestCheckpointIndex:

    def test_index_updated(self, tmp_path):
        """Save steps 50 and 100 → checkpoint_index.json lists both."""
        cfg = _make_config(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")

        cb = make_metadata_callback(cfg)
        args = _make_args(output_dir=cfg.TRAIN_OUTPUT_DIR)

        for step in (50, 100):
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_path, exist_ok=True)
            state = _make_state(global_step=step)
            cb.on_save(args, state, None)

        index_path = os.path.join(ckpt_dir, "checkpoint_index.json")
        assert os.path.isfile(index_path)
        with open(index_path) as f:
            index = json.load(f)
        steps = [e["step"] for e in index["checkpoints"]]
        assert 50 in steps
        assert 100 in steps
        assert index["latest"] == "checkpoint-100/"


# ---------------------------------------------------------------------------
# Config compatibility checks
# ---------------------------------------------------------------------------

class TestConfigCompatibility:

    def test_fatal_mismatch_blocks_resume(self, tmp_path):
        """config_snapshot.json has LORA_R=16, current config has 32 → ValueError."""
        cfg = _make_config(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, "checkpoint-50")
        os.makedirs(ckpt_path, exist_ok=True)

        # Write snapshot with TRAIN_LORA_R=16
        snapshot = {"TRAIN_LORA_R": 16}
        with open(os.path.join(ckpt_path, "config_snapshot.json"), "w") as f:
            json.dump(snapshot, f)

        # Current config has TRAIN_LORA_R=32
        cfg.TRAIN_LORA_R = 32

        with pytest.raises(ValueError, match="Fatal config mismatch"):
            check_resume_compatibility(ckpt_path, cfg)

    def test_nonfatal_mismatch_warning(self, tmp_path):
        """Non-fatal key change → warning logged, no error."""
        cfg = _make_config(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, "checkpoint-50")
        os.makedirs(ckpt_path, exist_ok=True)

        # Snapshot must match all fatal keys to avoid fatal error,
        # but differ on a non-fatal key to trigger warning.
        from src.training.checkpointing import _make_config_snapshot
        snapshot = _make_config_snapshot(cfg)
        snapshot["SOME_NON_FATAL_KEY"] = "old_value"

        with open(os.path.join(ckpt_path, "config_snapshot.json"), "w") as f:
            json.dump(snapshot, f)

        # Should not raise
        check_resume_compatibility(ckpt_path, cfg)

    def test_missing_snapshot_no_error(self, tmp_path):
        """No config_snapshot.json → warning, no error."""
        cfg = _make_config(tmp_path)
        ckpt_path = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints", "checkpoint-50")
        os.makedirs(ckpt_path, exist_ok=True)

        check_resume_compatibility(ckpt_path, cfg)
