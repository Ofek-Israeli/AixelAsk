"""Tests for src.training.dataset_registry — canonical dataset paths, loading,
scalability offsets, and count_examples."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest

from src.training.dataset_registry import (
    DATASET_REGISTRY,
    SplitEntry,
    count_examples,
    load_examples,
    _ScalabilityOffsets,
)


# ---------------------------------------------------------------------------
# Helper — build a mock filesystem with canonical datasets
# ---------------------------------------------------------------------------

def _build_mock_filesystem(root: str, examples_per_file: int = 20) -> None:
    """Create the canonical dataset directory structure with JSONL files."""
    # WikiTQ-4k
    wikitq4k_dir = os.path.join(root, "dataset", "WikiTQ-4k")
    os.makedirs(wikitq4k_dir, exist_ok=True)
    for split in ("train", "valid", "test"):
        fpath = os.path.join(wikitq4k_dir, f"{split}.jsonl")
        with open(fpath, "w") as f:
            for i in range(examples_per_file):
                json.dump({"question": f"wikitq4k_{split}_q{i}", "answer": str(i)}, f)
                f.write("\n")

    # WikiTQ+
    wikitq_plus_dir = os.path.join(root, "dataset", "WikiTQ+")
    os.makedirs(wikitq_plus_dir, exist_ok=True)
    for split in ("train", "valid", "test"):
        fpath = os.path.join(wikitq_plus_dir, f"{split}.jsonl")
        with open(fpath, "w") as f:
            for i in range(examples_per_file):
                json.dump({"question": f"wikitqplus_{split}_q{i}", "answer": str(i)}, f)
                f.write("\n")

    # Scalability (6 sub-files)
    scal_dir = os.path.join(root, "dataset", "Scalability")
    os.makedirs(scal_dir, exist_ok=True)
    for subfile in ("0-1k", "1k-2k", "2k-3k", "3k-4k", "4k-5k", "5k+"):
        fpath = os.path.join(scal_dir, f"{subfile}.jsonl")
        with open(fpath, "w") as f:
            for i in range(10):
                json.dump({"question": f"scal_{subfile}_q{i}", "answer": str(i)}, f)
                f.write("\n")


# ---------------------------------------------------------------------------
# Registry keys
# ---------------------------------------------------------------------------

class TestRegistryKeys:

    def test_supported_datasets_exist(self):
        """Registry contains wikitq_4k, wikitq_plus, scalability."""
        assert "wikitq_4k" in DATASET_REGISTRY
        assert "wikitq_plus" in DATASET_REGISTRY
        assert "scalability" in DATASET_REGISTRY

    def test_tabfact_not_present(self):
        """No key containing 'tabfact'."""
        for key in DATASET_REGISTRY:
            assert "tabfact" not in key.lower()


# ---------------------------------------------------------------------------
# load_examples returns correct fields
# ---------------------------------------------------------------------------

class TestLoadExamples:

    def test_correct_fields(self, tmp_path):
        """load_examples returns SplitEntry objects with correct provenance."""
        root = str(tmp_path)
        _build_mock_filesystem(root)

        # Reset cached offsets for scalability
        _ScalabilityOffsets.__init__(_ScalabilityOffsets())

        entries = load_examples("wikitq_4k", "train", [0, 1], root)
        assert len(entries) == 2
        assert all(isinstance(e, SplitEntry) for e in entries)
        assert entries[0].source_dataset == "wikitq_4k"
        assert entries[0].source_file == "train.jsonl"
        assert entries[0].source_index == 0
        assert entries[1].source_index == 1
        assert "question" in entries[0].example

    def test_out_of_range_raises(self, tmp_path):
        """load_examples with index 99999 → ValueError."""
        root = str(tmp_path)
        _build_mock_filesystem(root)

        with pytest.raises(ValueError, match="out of range"):
            load_examples("wikitq_4k", "train", [99999], root)


# ---------------------------------------------------------------------------
# Scalability concatenation + offset mapping
# ---------------------------------------------------------------------------

class TestScalabilityConcatenation:

    def test_concatenation_order(self, tmp_path):
        """Index 0 from scalability matches line 0 of 0-1k.jsonl."""
        root = str(tmp_path)
        _build_mock_filesystem(root)

        offsets = _ScalabilityOffsets()
        entries = load_examples.__wrapped__(
            "scalability", "all", [0], root
        ) if hasattr(load_examples, "__wrapped__") else load_examples(
            "scalability", "all", [0], root
        )
        assert len(entries) == 1
        assert entries[0].source_dataset == "scalability"
        assert entries[0].source_index == 0
        assert "scal_0-1k_q0" in entries[0].example.get("question", "")

    def test_cross_file_boundary(self, tmp_path):
        """Index at boundary between 0-1k and 1k-2k resolves correctly."""
        root = str(tmp_path)
        _build_mock_filesystem(root)

        # Each sub-file has 10 lines, so index 10 is the first line of 1k-2k.jsonl
        entries = load_examples("scalability", "all", [10], root)
        assert len(entries) == 1
        assert entries[0].source_index == 10
        assert "scal_1k-2k_q0" in entries[0].example.get("question", "")


# ---------------------------------------------------------------------------
# count_examples
# ---------------------------------------------------------------------------

class TestCountExamples:

    def test_count_wikitq_4k(self, tmp_path):
        """count_examples matches actual line count."""
        root = str(tmp_path)
        _build_mock_filesystem(root, examples_per_file=20)

        count = count_examples("wikitq_4k", "train", root)
        assert count == 20

    def test_count_scalability(self, tmp_path):
        """Scalability count is sum across all sub-files."""
        root = str(tmp_path)
        _build_mock_filesystem(root)

        # Reset cached offsets
        _ScalabilityOffsets.__init__(_ScalabilityOffsets())

        count = count_examples("scalability", "all", root)
        assert count == 60  # 6 files × 10 lines


# ---------------------------------------------------------------------------
# Missing file error
# ---------------------------------------------------------------------------

class TestMissingFile:

    def test_unknown_dataset_key_raises(self):
        """Unknown dataset key raises KeyError."""
        with pytest.raises(KeyError, match="Unknown dataset key"):
            load_examples("nonexistent_dataset", "train", [0], "/fake/root")

    def test_tabfact_rejected(self):
        """TabFact key is explicitly rejected."""
        with pytest.raises(ValueError, match="TabFact"):
            load_examples("tabfact_plus", "train", [0], "/fake/root")
