"""Tests for src.yaml_splits — ID parsing and YAML split loading."""

from __future__ import annotations

import json
import os
import textwrap

import pytest
import yaml

from src.yaml_splits import parse_id, load_yaml_splits


# ---------------------------------------------------------------------------
# parse_id
# ---------------------------------------------------------------------------

class TestParseId:

    def test_w4k_train(self):
        ds, split, idx = parse_id("w4k-train-123")
        assert ds == "wikitq_4k"
        assert split == "train"
        assert idx == 123

    def test_w4k_valid(self):
        ds, split, idx = parse_id("w4k-valid-0")
        assert ds == "wikitq_4k"
        assert split == "valid"
        assert idx == 0

    def test_w4k_test(self):
        ds, split, idx = parse_id("w4k-test-42")
        assert ds == "wikitq_4k"
        assert split == "test"
        assert idx == 42

    def test_wplus_train(self):
        ds, split, idx = parse_id("w+-train-100")
        assert ds == "wikitq_plus"
        assert split == "train"
        assert idx == 100

    def test_wplus_valid(self):
        ds, split, idx = parse_id("w+-valid-8")
        assert ds == "wikitq_plus"
        assert split == "valid"
        assert idx == 8

    def test_wplus_test(self):
        ds, split, idx = parse_id("w+-test-71")
        assert ds == "wikitq_plus"
        assert split == "test"
        assert idx == 71

    def test_scalability(self):
        ds, split, idx = parse_id("scalability-3165")
        assert ds == "scalability"
        assert split == "all"
        assert idx == 3165

    def test_scalability_zero(self):
        ds, split, idx = parse_id("scalability-0")
        assert ds == "scalability"
        assert split == "all"
        assert idx == 0

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            parse_id("unknown-5")


# ---------------------------------------------------------------------------
# load_yaml_splits with mock filesystem
# ---------------------------------------------------------------------------

def _build_mock_filesystem(root: str) -> None:
    """Create minimal JSONL files for all datasets."""
    w4k_dir = os.path.join(root, "dataset", "WikiTQ-4k")
    wplus_dir = os.path.join(root, "dataset", "WikiTQ+")
    scal_dir = os.path.join(root, "dataset", "Scalability")

    for d in (w4k_dir, wplus_dir, scal_dir):
        os.makedirs(d, exist_ok=True)

    for split in ("train", "valid", "test"):
        for ds_dir, prefix in [(w4k_dir, "w4k"), (wplus_dir, "wplus")]:
            path = os.path.join(ds_dir, f"{split}.jsonl")
            with open(path, "w") as f:
                for i in range(20):
                    json.dump({"question": f"{prefix}_{split}_q{i}", "answer": str(i)}, f)
                    f.write("\n")

    for subfile in ("0-1k.jsonl", "1k-2k.jsonl", "2k-3k.jsonl",
                     "3k-4k.jsonl", "4k-5k.jsonl", "5k+.jsonl"):
        path = os.path.join(scal_dir, subfile)
        with open(path, "w") as f:
            for i in range(10):
                json.dump({"question": f"scal_{subfile}_{i}", "answer": str(i)}, f)
                f.write("\n")


def _write_yaml(path: str, train_ids, valid_ids, test_ids) -> None:
    """Write a multi-document YAML split file."""
    docs = [
        {"title": "Train", "ids": train_ids},
        {"title": "Validation", "ids": valid_ids},
        {"title": "Test", "ids": test_ids},
    ]
    with open(path, "w") as f:
        yaml.dump_all(docs, f)


class TestLoadYamlSplits:

    def test_basic_loading(self, tmp_path):
        root = str(tmp_path)
        _build_mock_filesystem(root)

        from src.training.dataset_registry import _ScalabilityOffsets
        _ScalabilityOffsets.__init__(_ScalabilityOffsets())

        yaml_path = os.path.join(root, "splits.yaml")
        _write_yaml(
            yaml_path,
            train_ids=["w4k-train-0", "w4k-train-1", "w+-test-3"],
            valid_ids=["w4k-valid-0"],
            test_ids=["scalability-0", "w+-train-5"],
        )

        result = load_yaml_splits(yaml_path, root)
        assert len(result["train"]) == 3
        assert len(result["valid"]) == 1
        assert len(result["test"]) == 2

    def test_provenance_fields(self, tmp_path):
        root = str(tmp_path)
        _build_mock_filesystem(root)

        from src.training.dataset_registry import _ScalabilityOffsets
        _ScalabilityOffsets.__init__(_ScalabilityOffsets())

        yaml_path = os.path.join(root, "splits.yaml")
        _write_yaml(
            yaml_path,
            train_ids=["w4k-train-5"],
            valid_ids=[],
            test_ids=["scalability-2"],
        )

        result = load_yaml_splits(yaml_path, root)

        train_ex = result["train"][0]
        assert train_ex["_source_dataset"] == "wikitq_4k"
        assert train_ex["_source_file"] == "train.jsonl"
        assert train_ex["_source_index"] == 5

        test_ex = result["test"][0]
        assert test_ex["_source_dataset"] == "scalability"
        assert test_ex["_source_file"] == "all"
        assert test_ex["_source_index"] == 2

    def test_missing_yaml_raises(self):
        with pytest.raises(FileNotFoundError):
            load_yaml_splits("/nonexistent/path.yaml", "/fake")

    def test_empty_section(self, tmp_path):
        root = str(tmp_path)
        _build_mock_filesystem(root)

        from src.training.dataset_registry import _ScalabilityOffsets
        _ScalabilityOffsets.__init__(_ScalabilityOffsets())

        yaml_path = os.path.join(root, "splits.yaml")
        _write_yaml(yaml_path, train_ids=["w4k-train-0"], valid_ids=[], test_ids=[])

        result = load_yaml_splits(yaml_path, root)
        assert len(result["train"]) == 1
        assert len(result["valid"]) == 0
        assert len(result["test"]) == 0
