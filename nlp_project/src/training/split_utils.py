"""Split construction for all three split modes.

This module is the single authoritative place for constructing train / valid /
test splits from the project's dataset sources.  It dispatches on
``config.SPLIT_MODE`` (``seeded_ratio``, ``explicit_indices``, ``overfit_poc``)
and returns a ``SplitResult`` bundling HF ``datasets.Dataset`` objects with
provenance columns.

Library delegation: ``datasets.Dataset.from_json`` for JSONL loading,
``.select()`` for index-based selection, ``datasets.concatenate_datasets``
for multi-dataset union, ``.map()`` for provenance injection.

Custom logic: cross-split overlap check, canonical dataset-registry mapping,
provenance field injection, overfit-PoC delegation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import datasets

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SplitResult
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Bundles the three split datasets and the mode that produced them."""

    mode: str
    train: datasets.Dataset
    valid: datasets.Dataset
    test: datasets.Dataset


# ---------------------------------------------------------------------------
# Dataset key → (Config field name prefix, canonical split name) mapping
# ---------------------------------------------------------------------------

_DATASET_SPLIT_MAP: Dict[str, Dict[str, tuple[str, str]]] = {
    "wikitq_4k": {
        "train": ("SPLIT_TRAIN_WIKITQ_4K_INDICES", "train"),
        "valid": ("SPLIT_VALID_WIKITQ_4K_INDICES", "valid"),
        "test":  ("SPLIT_TEST_WIKITQ_4K_INDICES", "test"),
    },
    "wikitq_plus": {
        "train": ("SPLIT_TRAIN_WIKITQ_PLUS_INDICES", "train"),
        "valid": ("SPLIT_VALID_WIKITQ_PLUS_INDICES", "valid"),
        "test":  ("SPLIT_TEST_WIKITQ_PLUS_INDICES", "test"),
    },
    "scalability": {
        "train": ("SPLIT_TRAIN_SCALABILITY_INDICES", "all"),
        "valid": ("SPLIT_VALID_SCALABILITY_INDICES", "all"),
        "test":  ("SPLIT_TEST_SCALABILITY_INDICES", "all"),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_splits(config: "Config") -> SplitResult:
    """Build train/valid/test splits according to ``config.SPLIT_MODE``.

    Returns a ``SplitResult`` whose ``.train`` / ``.valid`` / ``.test``
    are ``datasets.Dataset`` objects with provenance columns
    (``source_dataset``, ``source_file``, ``source_index``).
    """
    mode = config.SPLIT_MODE

    if mode == "seeded_ratio":
        return _build_seeded_ratio(config)
    elif mode == "explicit_indices":
        return _build_explicit_indices(config)
    elif mode == "overfit_poc":
        return _build_overfit_poc(config)
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {mode!r}")


# ---------------------------------------------------------------------------
# SEEDED_RATIO mode
# ---------------------------------------------------------------------------

def _infer_source_dataset(path: str) -> str:
    """Best-effort dataset-family inference from a file path."""
    lower = path.lower()
    if "wikitq-4k" in lower or "wikitq_4k" in lower:
        return "wikitq_4k"
    if "wikitq+" in lower or "wikitq_plus" in lower:
        return "wikitq_plus"
    if "scalability" in lower:
        return "scalability"
    return "custom"


def _inject_provenance(
    ds: datasets.Dataset,
    source_dataset: str,
    source_file: str,
    *,
    index_offset: int = 0,
) -> datasets.Dataset:
    """Add provenance columns via ``.map()``."""
    def _add(example, idx):
        example["source_dataset"] = source_dataset
        example["source_file"] = source_file
        example["source_index"] = idx + index_offset
        return example

    return ds.map(_add, with_indices=True)


def _build_seeded_ratio(config: "Config") -> SplitResult:
    """SPLIT_MODE_SEEDED_RATIO: load from paths, optional seeded split."""
    train_path = config.TRAIN_DATASET_PATH
    source_dataset = _infer_source_dataset(train_path)
    source_file = os.path.basename(train_path)

    full_ds = datasets.Dataset.from_json(train_path)

    if config.TRAIN_DEV_DATASET_PATH:
        train_ds = full_ds
        dev_ds = datasets.Dataset.from_json(config.TRAIN_DEV_DATASET_PATH)
        dev_source_file = os.path.basename(config.TRAIN_DEV_DATASET_PATH)
        train_ds = _inject_provenance(train_ds, source_dataset, source_file)
        dev_ds = _inject_provenance(dev_ds, source_dataset, dev_source_file)
    elif config.TRAIN_USE_SEEDED_SPLIT:
        split = full_ds.train_test_split(
            test_size=1.0 - config.TRAIN_SPLIT_RATIO,
            seed=config.TRAIN_SPLIT_SEED,
        )
        train_ds = split["train"]
        dev_ds = split["test"]
        train_ds = _inject_provenance(train_ds, source_dataset, source_file)
        dev_ds = _inject_provenance(dev_ds, source_dataset, source_file)
    else:
        train_ds = full_ds
        dev_ds = datasets.Dataset.from_dict({
            col: [] for col in full_ds.column_names
        })
        train_ds = _inject_provenance(train_ds, source_dataset, source_file)
        dev_ds = _inject_provenance(dev_ds, source_dataset, source_file)

    if config.TRAIN_MAX_TRAIN_EXAMPLES > 0:
        train_ds = train_ds.select(range(
            min(config.TRAIN_MAX_TRAIN_EXAMPLES, len(train_ds))
        ))
    if config.TRAIN_MAX_DEV_EXAMPLES > 0:
        dev_ds = dev_ds.select(range(
            min(config.TRAIN_MAX_DEV_EXAMPLES, len(dev_ds))
        ))

    test_ds = _empty_dataset_like(train_ds)

    logger.info(
        "SEEDED_RATIO split: train=%d, valid=%d, test=%d",
        len(train_ds), len(dev_ds), len(test_ds),
    )
    return SplitResult(mode="seeded_ratio", train=train_ds, valid=dev_ds, test=test_ds)


# ---------------------------------------------------------------------------
# EXPLICIT_INDICES mode
# ---------------------------------------------------------------------------

def _build_explicit_indices(config: "Config") -> SplitResult:
    """SPLIT_MODE_EXPLICIT_INDICES: per-dataset per-split index selection."""
    from src.training.dataset_registry import DATASET_REGISTRY

    per_split: Dict[str, List[datasets.Dataset]] = {
        "train": [], "valid": [], "test": [],
    }

    provenance_triples: Dict[str, set] = {
        "train": set(), "valid": set(), "test": set(),
    }

    for dataset_key, split_info in _DATASET_SPLIT_MAP.items():
        for split_name, (field_name, canonical_split) in split_info.items():
            indices: List[int] = getattr(config, field_name)
            if not indices:
                continue

            if dataset_key == "scalability":
                source_file = "all"
                file_paths = DATASET_REGISTRY["scalability"]["all"]
                abs_paths = [
                    os.path.join(config.AIXELASK_ROOT, rp) for rp in file_paths
                ]
                combined = datasets.concatenate_datasets([
                    datasets.Dataset.from_json(p) for p in abs_paths
                ])
                selected = combined.select(indices)
            else:
                rel_path = DATASET_REGISTRY[dataset_key][canonical_split]
                abs_path = os.path.join(config.AIXELASK_ROOT, rel_path)
                source_file = os.path.basename(rel_path)
                full_ds = datasets.Dataset.from_json(abs_path)
                selected = full_ds.select(indices)

            actual_indices = list(indices)

            def _make_provenance_fn(ds_key, sf, idx_list):
                def _add(example, i):
                    example["source_dataset"] = ds_key
                    example["source_file"] = sf
                    example["source_index"] = idx_list[i]
                    return example
                return _add

            selected = selected.map(
                _make_provenance_fn(dataset_key, source_file, actual_indices),
                with_indices=True,
            )

            per_split[split_name].append(selected)

            for idx in indices:
                provenance_triples[split_name].add(
                    (dataset_key, source_file, idx)
                )

    _check_cross_split_overlap(provenance_triples)

    train_ds = _concat_or_empty(per_split["train"])
    valid_ds = _concat_or_empty(per_split["valid"])
    test_ds = _concat_or_empty(per_split["test"])

    logger.info(
        "EXPLICIT_INDICES split: train=%d, valid=%d, test=%d",
        len(train_ds), len(valid_ds), len(test_ds),
    )
    return SplitResult(
        mode="explicit_indices", train=train_ds, valid=valid_ds, test=test_ds,
    )


def _check_cross_split_overlap(
    provenance_triples: Dict[str, set],
) -> None:
    """Raise ``ValueError`` if any (dataset, file, index) triple appears in
    more than one split."""
    splits = ["train", "valid", "test"]
    overlaps: List[str] = []
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            common = provenance_triples[s1] & provenance_triples[s2]
            for triple in sorted(common):
                overlaps.append(
                    f"  ({triple[0]}, {triple[1]}, {triple[2]}) "
                    f"in both {s1} and {s2}"
                )
    if overlaps:
        raise ValueError(
            "Cross-split overlap detected:\n" + "\n".join(overlaps)
        )


# ---------------------------------------------------------------------------
# OVERFIT_POC mode
# ---------------------------------------------------------------------------

def _build_overfit_poc(config: "Config") -> SplitResult:
    """SPLIT_MODE_OVERFIT_POC: delegate to ``tiny_overfit_poc`` and use the
    selected subset as both train and valid."""
    from src.training.tiny_overfit_poc import select_overfit_subset

    train_path = config.TRAIN_DATASET_PATH
    source_dataset = _infer_source_dataset(train_path)
    source_file = os.path.basename(train_path)

    full_ds = datasets.Dataset.from_json(train_path)
    dataset_size = len(full_ds)

    indices = select_overfit_subset(config, dataset_size)

    subset = full_ds.select(indices)

    def _add_provenance(example, i):
        example["source_dataset"] = source_dataset
        example["source_file"] = source_file
        example["source_index"] = indices[i]
        return example

    subset = subset.map(_add_provenance, with_indices=True)

    test_ds = _empty_dataset_like(subset)

    logger.info(
        "OVERFIT_POC split: subset=%d examples used as both train and valid",
        len(subset),
    )
    return SplitResult(
        mode="overfit_poc", train=subset, valid=subset, test=test_ds,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_dataset_like(ds: datasets.Dataset) -> datasets.Dataset:
    """Create an empty ``Dataset`` with the same schema as *ds*."""
    return datasets.Dataset.from_dict({col: [] for col in ds.column_names})


def _concat_or_empty(
    parts: List[datasets.Dataset],
) -> datasets.Dataset:
    """Concatenate dataset parts, returning an empty dataset if list is empty."""
    if not parts:
        return datasets.Dataset.from_dict({
            "source_dataset": [],
            "source_file": [],
            "source_index": [],
        })
    return datasets.concatenate_datasets(parts)
