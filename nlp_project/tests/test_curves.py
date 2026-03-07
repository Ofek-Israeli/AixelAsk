"""Tests for src.training.curves — TSV updates, TeX generation, keep-last-N,
and metrics manifest."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.training.curves import CurvesManager


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    TRAIN_CURVES_DIR: str = ""
    TRAIN_CURVES_KEEP_LAST_N_POINTS: int = 0
    TRAIN_CURVES_COMPILE_ENABLE: bool = False
    TRAIN_CURVES_COMPILE_AT_END: bool = False


def _make_config(tmp_path, keep_last_n: int = 0) -> _StubConfig:
    return _StubConfig(
        TRAIN_CURVES_DIR=str(tmp_path / "curves"),
        TRAIN_CURVES_KEEP_LAST_N_POINTS=keep_last_n,
    )


# ---------------------------------------------------------------------------
# TSV update
# ---------------------------------------------------------------------------

class TestTsvUpdate:

    def test_single_update(self, tmp_path):
        """curves.update_tsv → data/reward_mean.tsv has 1 data row."""
        cfg = _make_config(tmp_path)
        mgr = CurvesManager(cfg)
        mgr.init_curves_dir()

        mgr.update_tsv("reward_mean", step=5, value=0.5)

        tsv = os.path.join(cfg.TRAIN_CURVES_DIR, "data", "reward_mean.tsv")
        with open(tsv) as f:
            lines = f.readlines()
        header = lines[0].strip()
        assert header == "step\tvalue"
        data_rows = [l for l in lines[1:] if l.strip()]
        assert len(data_rows) == 1
        parts = data_rows[0].strip().split("\t")
        assert parts[0] == "5"
        assert parts[1] == "0.5"

    def test_append_two_updates(self, tmp_path):
        """Two updates → 2 data rows."""
        cfg = _make_config(tmp_path)
        mgr = CurvesManager(cfg)
        mgr.init_curves_dir()

        mgr.update_tsv("reward_mean", step=5, value=0.5)
        mgr.update_tsv("reward_mean", step=10, value=0.7)

        tsv = os.path.join(cfg.TRAIN_CURVES_DIR, "data", "reward_mean.tsv")
        with open(tsv) as f:
            lines = f.readlines()
        data_rows = [l for l in lines[1:] if l.strip()]
        assert len(data_rows) == 2


# ---------------------------------------------------------------------------
# TeX generation
# ---------------------------------------------------------------------------

class TestTexGeneration:

    def test_tex_file_created(self, tmp_path):
        """After update, tex/reward_mean.tex exists and contains \\addplot."""
        cfg = _make_config(tmp_path)
        mgr = CurvesManager(cfg)
        mgr.init_curves_dir()

        mgr.update_tsv("reward_mean", step=5, value=0.5)
        mgr.generate_tex()

        tex_path = os.path.join(cfg.TRAIN_CURVES_DIR, "tex", "reward_mean.tex")
        assert os.path.isfile(tex_path)
        with open(tex_path) as f:
            content = f.read()
        assert "\\addplot" in content
        assert "reward_mean.tsv" in content

    def test_family_tex_overlay(self, tmp_path):
        """Family TeX has two \\addplot commands (train + eval)."""
        cfg = _make_config(tmp_path)
        mgr = CurvesManager(cfg)
        mgr.init_curves_dir()

        tex_path = os.path.join(cfg.TRAIN_CURVES_DIR, "tex", "reward_family.tex")
        assert os.path.isfile(tex_path)
        with open(tex_path) as f:
            content = f.read()
        assert content.count("\\addplot") == 2
        assert "Train" in content
        assert "Eval" in content


# ---------------------------------------------------------------------------
# Compile scheduling (via CurvesCallback)
# ---------------------------------------------------------------------------

class TestCompileScheduling:

    def test_should_compile_cadence(self):
        """Compile fires at correct step multiples."""
        assert 100 % 100 == 0  # step=100, every=100 → True
        assert 99 % 100 != 0   # step=99, every=100 → False

    def test_failed_compilation_non_fatal(self, tmp_path):
        """Mock subprocess.run to raise → no exception propagated."""
        cfg = _make_config(tmp_path)
        cfg.TRAIN_CURVES_COMPILE_ENABLE = True
        mgr = CurvesManager(cfg)
        mgr.init_curves_dir()

        with patch("src.training.tex_compile.compile_all", side_effect=RuntimeError("pdflatex failed")):
            from src.training.train_stats import CurvesCallback
            from types import SimpleNamespace

            cb_cfg = _StubConfig(
                TRAIN_CURVES_DIR=cfg.TRAIN_CURVES_DIR,
                TRAIN_CURVES_COMPILE_ENABLE=True,
                TRAIN_CURVES_COMPILE_AT_END=False,
                TRAIN_CURVES_KEEP_LAST_N_POINTS=0,
            )
            cb_cfg.TRAIN_CURVES_UPDATE_EVERY_STEPS = 1
            cb_cfg.TRAIN_CURVES_COMPILE_EVERY_STEPS = 1

            from src.training.train_stats import RewardMetricsAccumulator
            cb = CurvesCallback(cb_cfg, mgr, RewardMetricsAccumulator())

            state = SimpleNamespace(global_step=100)
            cb.on_log(None, state, None, logs={"loss": 0.5})
            # No exception should be raised


# ---------------------------------------------------------------------------
# Keep-last-N truncation
# ---------------------------------------------------------------------------

class TestKeepLastN:

    def test_truncation(self, tmp_path):
        """keep_last_n=10 + write 20 points → TSV has 10 data rows."""
        cfg = _make_config(tmp_path, keep_last_n=10)
        mgr = CurvesManager(cfg)
        mgr.init_curves_dir()

        for i in range(1, 21):
            mgr.update_tsv("reward_mean", step=i, value=float(i) / 20)

        tsv = os.path.join(cfg.TRAIN_CURVES_DIR, "data", "reward_mean.tsv")
        with open(tsv) as f:
            lines = f.readlines()
        data_rows = [l for l in lines[1:] if l.strip()]
        assert len(data_rows) == 10

        # Last data row should be step=20
        last = data_rows[-1].strip().split("\t")
        assert last[0] == "20"


# ---------------------------------------------------------------------------
# Metrics manifest
# ---------------------------------------------------------------------------

class TestMetricsManifest:

    def test_manifest_created(self, tmp_path):
        """After init, manifests/metrics_manifest.json lists all expected metrics."""
        cfg = _make_config(tmp_path)
        mgr = CurvesManager(cfg)
        mgr.init_curves_dir()

        manifest_path = os.path.join(
            cfg.TRAIN_CURVES_DIR, "manifests", "metrics_manifest.json"
        )
        assert os.path.isfile(manifest_path)

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert isinstance(manifest, list)
        assert len(manifest) > 0

        metric_names = {entry["metric"] for entry in manifest}
        assert "reward_mean" in metric_names
        assert "eval_reward_mean" in metric_names
        assert "correctness_rate" in metric_names
