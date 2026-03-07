"""Tests for src.training.checkpoint_resolver — resolve best/latest/merged/explicit
checkpoints, adapter-only detection, metadata extraction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from src.training.checkpoint_resolver import (
    ResolvedCheckpoint,
    resolve_test_checkpoint,
)


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    TRAIN_OUTPUT_DIR: str = ""
    TEST_TRAINED_CHECKPOINT_SOURCE: str = "best"
    TEST_TRAINED_CHECKPOINT_PATH: str = ""


def _setup_checkpoints(tmp_path) -> _StubConfig:
    """Build a mock output dir with checkpoints/."""
    output_dir = str(tmp_path / "output")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return _StubConfig(TRAIN_OUTPUT_DIR=output_dir)


def _create_checkpoint(ckpt_dir: str, name: str, files: dict[str, str] | None = None):
    """Create a checkpoint sub-directory with optional files."""
    path = os.path.join(ckpt_dir, name)
    os.makedirs(path, exist_ok=True)
    if files:
        for fname, content in files.items():
            with open(os.path.join(path, fname), "w") as f:
                f.write(content)
    return path


# ---------------------------------------------------------------------------
# Resolve best checkpoint
# ---------------------------------------------------------------------------

class TestResolveBest:

    def test_resolve_best(self, tmp_path):
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        _create_checkpoint(ckpt_dir, "best", {"adapter_config.json": "{}"})

        result = resolve_test_checkpoint(cfg)
        assert result.source == "best"
        assert "best" in result.path

    def test_best_missing_raises(self, tmp_path):
        """No best/ directory → FileNotFoundError."""
        cfg = _setup_checkpoints(tmp_path)
        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "best"

        with pytest.raises(FileNotFoundError, match="Best checkpoint not found"):
            resolve_test_checkpoint(cfg)


# ---------------------------------------------------------------------------
# Resolve latest (symlink)
# ---------------------------------------------------------------------------

class TestResolveLatest:

    def test_resolve_latest_via_symlink(self, tmp_path):
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        _create_checkpoint(ckpt_dir, "checkpoint-100", {"adapter_config.json": "{}"})
        os.symlink("checkpoint-100", os.path.join(ckpt_dir, "latest"))

        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "latest"
        result = resolve_test_checkpoint(cfg)
        assert result.source == "latest"
        assert "checkpoint-100" in result.path

    def test_latest_missing_raises(self, tmp_path):
        cfg = _setup_checkpoints(tmp_path)
        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "latest"

        with pytest.raises(FileNotFoundError, match="Latest checkpoint not found"):
            resolve_test_checkpoint(cfg)


# ---------------------------------------------------------------------------
# Resolve merged
# ---------------------------------------------------------------------------

class TestResolveMerged:

    def test_resolve_merged(self, tmp_path):
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        _create_checkpoint(ckpt_dir, "merged", {
            "config.json": "{}",
            "model.safetensors": "fake",
        })

        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "merged"
        result = resolve_test_checkpoint(cfg)
        assert result.source == "merged"
        assert "merged" in result.path

    def test_merged_missing_raises(self, tmp_path):
        cfg = _setup_checkpoints(tmp_path)
        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "merged"

        with pytest.raises(FileNotFoundError, match="CONFIG_TRAIN_SAVE_MERGED_ADAPTER"):
            resolve_test_checkpoint(cfg)


# ---------------------------------------------------------------------------
# Resolve explicit path
# ---------------------------------------------------------------------------

class TestResolveExplicitPath:

    def test_resolve_explicit(self, tmp_path):
        explicit_dir = str(tmp_path / "custom_ckpt")
        os.makedirs(explicit_dir, exist_ok=True)
        with open(os.path.join(explicit_dir, "adapter_config.json"), "w") as f:
            f.write("{}")

        cfg = _setup_checkpoints(tmp_path)
        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "explicit_path"
        cfg.TEST_TRAINED_CHECKPOINT_PATH = explicit_dir

        result = resolve_test_checkpoint(cfg)
        assert result.source == "explicit_path"
        assert explicit_dir in result.path

    def test_explicit_missing_raises(self, tmp_path):
        cfg = _setup_checkpoints(tmp_path)
        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "explicit_path"
        cfg.TEST_TRAINED_CHECKPOINT_PATH = "/nonexistent/path"

        with pytest.raises(FileNotFoundError, match="/nonexistent/path"):
            resolve_test_checkpoint(cfg)


# ---------------------------------------------------------------------------
# Override via --checkpoint flag
# ---------------------------------------------------------------------------

class TestOverride:

    def test_override_source(self, tmp_path):
        """Config has best, override_source=latest → uses latest."""
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        _create_checkpoint(ckpt_dir, "checkpoint-50", {"adapter_config.json": "{}"})
        os.symlink("checkpoint-50", os.path.join(ckpt_dir, "latest"))
        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "best"

        result = resolve_test_checkpoint(cfg, override_source="latest")
        assert result.source == "latest"
        assert "checkpoint-50" in result.path


# ---------------------------------------------------------------------------
# Adapter-only detection (file presence)
# ---------------------------------------------------------------------------

class TestAdapterOnlyDetection:

    def test_adapter_only(self, tmp_path):
        """adapter_config.json + adapter_model.safetensors but NO config.json → adapter_only."""
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        _create_checkpoint(ckpt_dir, "best", {
            "adapter_config.json": "{}",
            "adapter_model.safetensors": "fake",
        })

        result = resolve_test_checkpoint(cfg)
        assert result.is_adapter_only is True

    def test_full_weight(self, tmp_path):
        """config.json + model.safetensors → NOT adapter_only."""
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        _create_checkpoint(ckpt_dir, "best", {
            "config.json": "{}",
            "model.safetensors": "fake",
        })

        result = resolve_test_checkpoint(cfg)
        assert result.is_adapter_only is False


# ---------------------------------------------------------------------------
# Metadata extraction from checkpoint_metadata.json
# ---------------------------------------------------------------------------

class TestMetadataExtraction:

    def test_metadata_read(self, tmp_path):
        """checkpoint_metadata.json with step/metric → fields populated."""
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        meta = {
            "step": 375,
            "metric_name": "eval_reward_mean",
            "metric_value": 0.68,
        }
        _create_checkpoint(ckpt_dir, "best", {
            "adapter_config.json": "{}",
            "checkpoint_metadata.json": json.dumps(meta),
        })

        result = resolve_test_checkpoint(cfg)
        assert result.step == 375
        assert result.metric_name == "eval_reward_mean"
        assert result.metric_value == pytest.approx(0.68)

    def test_missing_metadata_graceful(self, tmp_path):
        """No checkpoint_metadata.json → step/metric_name/metric_value are None."""
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        _create_checkpoint(ckpt_dir, "best", {
            "adapter_config.json": "{}",
        })

        result = resolve_test_checkpoint(cfg)
        assert result.step is None
        assert result.metric_name is None
        assert result.metric_value is None


# ---------------------------------------------------------------------------
# Merged checkpoint detection
# ---------------------------------------------------------------------------

class TestMergedDetection:

    def test_merged_is_full_weight(self, tmp_path):
        """Merged checkpoint with config.json + model.safetensors → not adapter_only."""
        cfg = _setup_checkpoints(tmp_path)
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        _create_checkpoint(ckpt_dir, "merged", {
            "config.json": "{}",
            "model.safetensors": "fake",
        })

        cfg.TEST_TRAINED_CHECKPOINT_SOURCE = "merged"
        result = resolve_test_checkpoint(cfg)
        assert result.is_adapter_only is False
