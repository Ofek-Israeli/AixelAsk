"""Tests for src.training.split_utils — YAML-based split construction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

if "transformers" not in sys.modules:
    _mock_transformers = MagicMock()
    _mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    _mock_transformers.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    sys.modules["transformers"] = _mock_transformers
elif not hasattr(sys.modules["transformers"], "PreTrainedTokenizerBase") or \
     not isinstance(sys.modules["transformers"].PreTrainedTokenizerBase, type):
    sys.modules["transformers"].PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})


@dataclass
class _StubConfig:
    SPLIT_YAML_PATH: str = "/fake/train_valid_test.yaml"
    AIXELASK_ROOT: str = "/fake/root"


def _mock_yaml_result() -> Dict[str, List[Dict[str, Any]]]:
    """Return a mock result from load_yaml_splits."""
    return {
        "train": [
            {"question": "q0", "answer": "a0",
             "_source_dataset": "wikitq_4k", "_source_file": "train.jsonl", "_source_index": 0},
            {"question": "q1", "answer": "a1",
             "_source_dataset": "wikitq_plus", "_source_file": "test.jsonl", "_source_index": 5},
        ],
        "valid": [
            {"question": "qv0", "answer": "av0",
             "_source_dataset": "scalability", "_source_file": "all", "_source_index": 100},
        ],
        "test": [
            {"question": "qt0", "answer": "at0",
             "_source_dataset": "wikitq_4k", "_source_file": "test.jsonl", "_source_index": 42},
        ],
    }


class TestBuildSplits:

    @patch("src.yaml_splits.load_yaml_splits")
    def test_returns_split_result(self, mock_load):
        mock_load.return_value = _mock_yaml_result()

        from src.training.split_utils import build_splits, SplitResult

        cfg = _StubConfig()
        result = build_splits(cfg)

        assert isinstance(result, SplitResult)
        mock_load.assert_called_once_with(cfg.SPLIT_YAML_PATH, cfg.AIXELASK_ROOT)

    @patch("src.yaml_splits.load_yaml_splits")
    def test_correct_sizes(self, mock_load):
        mock_load.return_value = _mock_yaml_result()

        from src.training.split_utils import build_splits

        result = build_splits(_StubConfig())
        assert len(result.train) == 2
        assert len(result.valid) == 1
        assert len(result.test) == 1

    @patch("src.yaml_splits.load_yaml_splits")
    def test_provenance_columns(self, mock_load):
        mock_load.return_value = _mock_yaml_result()

        from src.training.split_utils import build_splits

        result = build_splits(_StubConfig())
        assert "source_dataset" in result.train.column_names
        assert "source_file" in result.train.column_names
        assert "source_index" in result.train.column_names

    @patch("src.yaml_splits.load_yaml_splits")
    def test_empty_split(self, mock_load):
        data = _mock_yaml_result()
        data["test"] = []
        mock_load.return_value = data

        from src.training.split_utils import build_splits

        result = build_splits(_StubConfig())
        assert len(result.test) == 0
        assert "source_dataset" in result.test.column_names

    @patch("src.yaml_splits.load_yaml_splits")
    def test_provenance_values(self, mock_load):
        mock_load.return_value = _mock_yaml_result()

        from src.training.split_utils import build_splits

        result = build_splits(_StubConfig())
        assert result.train[0]["source_dataset"] == "wikitq_4k"
        assert result.valid[0]["source_index"] == 100
        assert result.test[0]["source_file"] == "test.jsonl"
