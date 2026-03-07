"""Split construction from ``train_valid_test.yaml``.

Always loads train/valid/test splits from the project's canonical YAML
split file.  Returns a ``SplitResult`` bundling HF ``datasets.Dataset``
objects with provenance columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import datasets

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Bundles the three split datasets."""

    train: datasets.Dataset
    valid: datasets.Dataset
    test: datasets.Dataset


def build_splits(config: "Config") -> SplitResult:
    """Build train/valid/test splits from ``train_valid_test.yaml``.

    Returns a ``SplitResult`` whose ``.train`` / ``.valid`` / ``.test``
    are ``datasets.Dataset`` objects with provenance columns
    (``source_dataset``, ``source_file``, ``source_index``).
    """
    from src.yaml_splits import load_yaml_splits

    raw = load_yaml_splits(config.SPLIT_YAML_PATH, config.AIXELASK_ROOT)

    train_ds = _examples_to_dataset(raw["train"])
    valid_ds = _examples_to_dataset(raw["valid"])
    test_ds = _examples_to_dataset(raw["test"])

    logger.info(
        "YAML split: train=%d, valid=%d, test=%d",
        len(train_ds), len(valid_ds), len(test_ds),
    )
    return SplitResult(train=train_ds, valid=valid_ds, test=test_ds)


def _examples_to_dataset(
    examples: List[Dict[str, Any]],
) -> datasets.Dataset:
    """Convert a list of example dicts (with ``_source_*`` keys) to a
    ``datasets.Dataset`` with provenance columns."""
    if not examples:
        return datasets.Dataset.from_dict({
            "source_dataset": [],
            "source_file": [],
            "source_index": [],
        })

    clean = []
    source_datasets = []
    source_files = []
    source_indices = []
    for ex in examples:
        ex_copy = dict(ex)
        source_datasets.append(ex_copy.pop("_source_dataset", ""))
        source_files.append(ex_copy.pop("_source_file", ""))
        source_indices.append(ex_copy.pop("_source_index", -1))
        clean.append(ex_copy)

    ds = datasets.Dataset.from_list(clean)
    ds = ds.add_column("source_dataset", source_datasets)
    ds = ds.add_column("source_file", source_files)
    ds = ds.add_column("source_index", source_indices)
    return ds
