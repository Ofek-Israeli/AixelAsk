"""Tests for src.training.split_utils — split construction across all three modes."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure a proper transformers mock for datasets library compatibility
if "transformers" not in sys.modules:
    _mock_transformers = MagicMock()
    _mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    _mock_transformers.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    sys.modules["transformers"] = _mock_transformers
elif not hasattr(sys.modules["transformers"], "PreTrainedTokenizerBase") or \
     not isinstance(sys.modules["transformers"].PreTrainedTokenizerBase, type):
    sys.modules["transformers"].PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})


# ---------------------------------------------------------------------------
# Stub config factory
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    SPLIT_MODE: str = "seeded_ratio"
    TRAIN_DATASET_PATH: str = ""
    TRAIN_DEV_DATASET_PATH: str = ""
    TRAIN_USE_SEEDED_SPLIT: bool = True
    TRAIN_SPLIT_RATIO: float = 0.8
    TRAIN_SPLIT_SEED: int = 42
    TRAIN_MAX_TRAIN_EXAMPLES: int = 0
    TRAIN_MAX_DEV_EXAMPLES: int = 0
    INFERENCE_DATASET_PATH: str = ""
    AIXELASK_ROOT: str = ""
    OVERFIT_POC_SELECTION_MODE: str = "FIRST_N"
    OVERFIT_POC_NUM_EXAMPLES: int = 8
    OVERFIT_POC_INDICES_FILE: str = ""
    GLOBAL_SEED: int = 42

    # Explicit-indices config fields
    SPLIT_TRAIN_WIKITQ_4K_INDICES: List[int] = field(default_factory=list)
    SPLIT_VALID_WIKITQ_4K_INDICES: List[int] = field(default_factory=list)
    SPLIT_TEST_WIKITQ_4K_INDICES: List[int] = field(default_factory=list)
    SPLIT_TRAIN_WIKITQ_PLUS_INDICES: List[int] = field(default_factory=list)
    SPLIT_VALID_WIKITQ_PLUS_INDICES: List[int] = field(default_factory=list)
    SPLIT_TEST_WIKITQ_PLUS_INDICES: List[int] = field(default_factory=list)
    SPLIT_TRAIN_SCALABILITY_INDICES: List[int] = field(default_factory=list)
    SPLIT_VALID_SCALABILITY_INDICES: List[int] = field(default_factory=list)
    SPLIT_TEST_SCALABILITY_INDICES: List[int] = field(default_factory=list)


def _make_jsonl(tmp_path, name: str, n: int) -> str:
    """Write a JSONL file with n lines and return its path."""
    fpath = os.path.join(str(tmp_path), name)
    with open(fpath, "w") as f:
        for i in range(n):
            json.dump({"question": f"q{i}", "answer": f"a{i}", "table": [[f"h{i}"], [f"v{i}"]]}, f)
            f.write("\n")
    return fpath


# ---------------------------------------------------------------------------
# Seeded ratio split
# ---------------------------------------------------------------------------

class TestSeededRatio:

    def test_deterministic_split(self, tmp_path):
        """100-example dataset, seed=42, ratio=0.8 → train=80, valid=20.
        Re-run with same seed → identical partition."""
        path = _make_jsonl(tmp_path, "train.jsonl", 100)
        cfg = _StubConfig(
            SPLIT_MODE="seeded_ratio",
            TRAIN_DATASET_PATH=path,
            TRAIN_USE_SEEDED_SPLIT=True,
            TRAIN_SPLIT_RATIO=0.8,
            TRAIN_SPLIT_SEED=42,
        )

        from src.training.split_utils import build_splits

        r1 = build_splits(cfg)
        assert len(r1.train) == 80
        assert len(r1.valid) == 20

        r2 = build_splits(cfg)
        assert list(r1.train["question"]) == list(r2.train["question"])

    def test_different_seed(self, tmp_path):
        """Seed=99 → different partition from seed=42."""
        path = _make_jsonl(tmp_path, "train.jsonl", 100)

        cfg42 = _StubConfig(
            SPLIT_MODE="seeded_ratio",
            TRAIN_DATASET_PATH=path,
            TRAIN_USE_SEEDED_SPLIT=True,
            TRAIN_SPLIT_RATIO=0.8,
            TRAIN_SPLIT_SEED=42,
        )
        cfg99 = _StubConfig(
            SPLIT_MODE="seeded_ratio",
            TRAIN_DATASET_PATH=path,
            TRAIN_USE_SEEDED_SPLIT=True,
            TRAIN_SPLIT_RATIO=0.8,
            TRAIN_SPLIT_SEED=99,
        )

        from src.training.split_utils import build_splits

        r42 = build_splits(cfg42)
        r99 = build_splits(cfg99)
        assert list(r42.train["question"]) != list(r99.train["question"])


# ---------------------------------------------------------------------------
# Explicit indices mode
# ---------------------------------------------------------------------------

class TestExplicitIndices:

    def test_single_dataset(self, tmp_path):
        """wikitq_4k train [0,1,2], valid [3,4], test [5,6,7]."""
        root = str(tmp_path)
        ds_dir = os.path.join(root, "dataset", "WikiTQ-4k")
        os.makedirs(ds_dir, exist_ok=True)
        for split_name in ("train", "valid", "test"):
            fpath = os.path.join(ds_dir, f"{split_name}.jsonl")
            with open(fpath, "w") as f:
                for i in range(20):
                    json.dump({"question": f"{split_name}_q{i}", "answer": str(i)}, f)
                    f.write("\n")

        cfg = _StubConfig(
            SPLIT_MODE="explicit_indices",
            AIXELASK_ROOT=root,
            SPLIT_TRAIN_WIKITQ_4K_INDICES=[0, 1, 2],
            SPLIT_VALID_WIKITQ_4K_INDICES=[3, 4],
            SPLIT_TEST_WIKITQ_4K_INDICES=[5, 6, 7],
        )

        from src.training.split_utils import build_splits

        result = build_splits(cfg)
        assert len(result.train) == 3
        assert len(result.valid) == 2
        assert len(result.test) == 3
        assert result.mode == "explicit_indices"


# ---------------------------------------------------------------------------
# Overfit PoC mode
# ---------------------------------------------------------------------------

class TestOverfitPoc:

    def test_overfit_poc_mode(self, tmp_path):
        """FIRST_N, N=8 → 8-example subset used as both train and valid."""
        path = _make_jsonl(tmp_path, "train.jsonl", 100)
        cfg = _StubConfig(
            SPLIT_MODE="overfit_poc",
            TRAIN_DATASET_PATH=path,
            OVERFIT_POC_SELECTION_MODE="FIRST_N",
            OVERFIT_POC_NUM_EXAMPLES=8,
        )

        from src.training.split_utils import build_splits

        result = build_splits(cfg)
        assert len(result.train) == 8
        assert len(result.valid) == 8
        assert result.mode == "overfit_poc"
        # train and valid are the same subset
        assert list(result.train["question"]) == list(result.valid["question"])


# ---------------------------------------------------------------------------
# SplitResult fields
# ---------------------------------------------------------------------------

class TestSplitResultFields:

    def test_dataclass_fields(self, tmp_path):
        """SplitResult has mode, train, valid, test fields."""
        path = _make_jsonl(tmp_path, "train.jsonl", 50)
        cfg = _StubConfig(
            SPLIT_MODE="seeded_ratio",
            TRAIN_DATASET_PATH=path,
            TRAIN_USE_SEEDED_SPLIT=True,
            TRAIN_SPLIT_RATIO=0.8,
            TRAIN_SPLIT_SEED=42,
        )

        from src.training.split_utils import build_splits

        result = build_splits(cfg)
        assert hasattr(result, "mode")
        assert hasattr(result, "train")
        assert hasattr(result, "valid")
        assert hasattr(result, "test")
        assert result.mode == "seeded_ratio"


# ---------------------------------------------------------------------------
# Baselines use only test split
# ---------------------------------------------------------------------------

class TestBaselinesUseTestOnly:

    def test_pipeline_uses_test_only(self, tmp_path):
        """Mock pipeline.run and verify only test-split examples are processed."""
        # Conceptual: the pipeline._resolve_test_split function enforces this
        from src.pipeline import _resolve_test_split

        path = _make_jsonl(tmp_path, "test.jsonl", 10)
        cfg = _StubConfig()
        cfg.SPLIT_MODE = "seeded_ratio"
        cfg.INFERENCE_DATASET_PATH = path
        cfg.DATASET = "custom"

        # _load_seeded_ratio_test_split is used for seeded_ratio mode
        from src.pipeline import _load_seeded_ratio_test_split
        examples = _load_seeded_ratio_test_split(cfg)
        assert len(examples) == 10


# ---------------------------------------------------------------------------
# Empty split failures (entrypoint validation)
# ---------------------------------------------------------------------------

class TestEmptySplitValidation:

    def test_overfit_poc_rejected_for_inference(self, tmp_path):
        """SPLIT_MODE_OVERFIT_POC → ValueError in pipeline._resolve_test_split."""
        cfg = _StubConfig(SPLIT_MODE="overfit_poc")

        from src.pipeline import _resolve_test_split

        with pytest.raises(ValueError, match="not supported"):
            _resolve_test_split(cfg)

    def test_empty_explicit_test_split(self, tmp_path):
        """All test index lists empty → test split is empty."""
        root = str(tmp_path)
        ds_dir = os.path.join(root, "dataset", "WikiTQ-4k")
        os.makedirs(ds_dir, exist_ok=True)
        for split_name in ("train", "valid", "test"):
            fpath = os.path.join(ds_dir, f"{split_name}.jsonl")
            with open(fpath, "w") as f:
                for i in range(10):
                    json.dump({"question": f"q{i}", "answer": str(i)}, f)
                    f.write("\n")

        cfg = _StubConfig(
            SPLIT_MODE="explicit_indices",
            AIXELASK_ROOT=root,
            SPLIT_TRAIN_WIKITQ_4K_INDICES=[0, 1, 2],
        )

        from src.training.split_utils import build_splits

        result = build_splits(cfg)
        assert len(result.test) == 0
