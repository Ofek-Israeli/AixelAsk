"""Canonical dataset-key to JSONL-path mapping and per-dataset loading.

Maps well-known dataset keys (``wikitq_4k``, ``wikitq_plus``,
``scalability``) to their JSONL paths relative to ``AIXELASK_ROOT``.
Provides ``load_examples`` for explicit-index split construction and
``count_examples`` for bounds checking.

The Scalability dataset consists of multiple sub-files that are
concatenated in lexicographic order, with cumulative offsets computed
once on first access.
"""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wikitq_4k": {
        "train": "dataset/WikiTQ-4k/train.jsonl",
        "valid": "dataset/WikiTQ-4k/valid.jsonl",
        "test":  "dataset/WikiTQ-4k/test.jsonl",
    },
    "wikitq_plus": {
        "train": "dataset/WikiTQ+/train.jsonl",
        "valid": "dataset/WikiTQ+/valid.jsonl",
        "test":  "dataset/WikiTQ+/test.jsonl",
    },
    "scalability": {
        "all": [
            "dataset/Scalability/0-1k.jsonl",
            "dataset/Scalability/1k-2k.jsonl",
            "dataset/Scalability/2k-3k.jsonl",
            "dataset/Scalability/3k-4k.jsonl",
            "dataset/Scalability/4k-5k.jsonl",
            "dataset/Scalability/5k+.jsonl",
        ],
    },
}


@dataclass
class SplitEntry:
    """A loaded example with its provenance metadata."""

    source_dataset: str
    source_file: str
    source_index: int
    example: dict


# ---------------------------------------------------------------------------
# Scalability offset table (computed lazily, once per process)
# ---------------------------------------------------------------------------

class _ScalabilityOffsets:
    """Lazy cumulative offset table for the multi-file Scalability dataset."""

    def __init__(self) -> None:
        self._offsets: Optional[List[int]] = None
        self._counts: Optional[List[int]] = None
        self._total: Optional[int] = None
        self._files: List[str] = DATASET_REGISTRY["scalability"]["all"]
        self._resolved_files: Optional[List[str]] = None

    def _ensure(self, aixelask_root: str) -> None:
        if self._offsets is not None:
            return
        offsets: list[int] = []
        counts: list[int] = []
        resolved: list[str] = []
        cumulative = 0
        for rel_path in self._files:
            full = os.path.join(aixelask_root, rel_path)
            if not os.path.isfile(full):
                raise FileNotFoundError(
                    f"Scalability sub-file missing: {full}"
                )
            resolved.append(full)
            offsets.append(cumulative)
            n = _count_lines(full)
            counts.append(n)
            cumulative += n
        self._offsets = offsets
        self._counts = counts
        self._total = cumulative
        self._resolved_files = resolved

    def total(self, aixelask_root: str) -> int:
        self._ensure(aixelask_root)
        assert self._total is not None
        return self._total

    def locate(self, global_idx: int, aixelask_root: str) -> tuple[str, int]:
        """Return ``(abs_file_path, local_line_idx)`` for *global_idx*."""
        self._ensure(aixelask_root)
        assert self._offsets is not None and self._resolved_files is not None
        assert self._counts is not None
        file_idx = bisect.bisect_right(self._offsets, global_idx) - 1
        if file_idx < 0:
            file_idx = 0
        local_idx = global_idx - self._offsets[file_idx]
        if local_idx < 0 or local_idx >= self._counts[file_idx]:
            raise IndexError(
                f"Index {global_idx} out of range for Scalability "
                f"(max {self._total - 1})"  # type: ignore[operator]
            )
        return self._resolved_files[file_idx], local_idx

    def load_lines(
        self, indices: list[int], aixelask_root: str
    ) -> list[tuple[int, dict]]:
        """Load specific global indices. Returns ``(global_idx, parsed_dict)``."""
        self._ensure(aixelask_root)
        assert self._offsets is not None and self._resolved_files is not None
        assert self._counts is not None

        by_file: dict[int, list[tuple[int, int]]] = {}
        for gi in indices:
            fpath, li = self.locate(gi, aixelask_root)
            fi = self._resolved_files.index(fpath)
            by_file.setdefault(fi, []).append((gi, li))

        results: list[tuple[int, dict]] = []
        for fi, pairs in sorted(by_file.items()):
            fpath = self._resolved_files[fi]
            needed_locals = {li: gi for gi, li in pairs}
            with open(fpath, "r") as f:
                for line_no, line in enumerate(f):
                    if line_no in needed_locals:
                        results.append(
                            (needed_locals[line_no], json.loads(line))
                        )
        results.sort(key=lambda t: t[0])
        return results


_scalability_offsets = _ScalabilityOffsets()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_lines(path: str) -> int:
    count = 0
    with open(path, "r") as f:
        for _ in f:
            count += 1
    return count


def _load_jsonl_lines(path: str, indices: list[int]) -> list[tuple[int, dict]]:
    """Load specific line indices from a JSONL file.

    Returns ``(index, parsed_dict)`` pairs sorted by index.
    """
    needed = set(indices)
    results: list[tuple[int, dict]] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f):
            if line_no in needed:
                results.append((line_no, json.loads(line)))
                if len(results) == len(needed):
                    break
    results.sort(key=lambda t: t[0])
    return results


def _validate_key(dataset_key: str) -> None:
    if "tabfact" in dataset_key.lower():
        raise ValueError(
            "TabFact+ is not supported in nlp_project; please choose "
            "WikiTQ-4k/WikiTQ+/Scalability or provide a non-TabFact "
            "custom dataset."
        )
    if dataset_key not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset key '{dataset_key}'. "
            f"Known keys: {sorted(DATASET_REGISTRY.keys())}"
        )


def _resolve_path(dataset_key: str, split: str, aixelask_root: str) -> str:
    """Resolve the absolute path for a single-file dataset/split."""
    entry = DATASET_REGISTRY[dataset_key]
    if split not in entry:
        raise KeyError(
            f"Split '{split}' not found for dataset '{dataset_key}'. "
            f"Available splits: {sorted(entry.keys())}"
        )
    rel = entry[split]
    if isinstance(rel, list):
        raise TypeError(
            f"Dataset '{dataset_key}' split '{split}' is multi-file; "
            f"use load_examples() directly."
        )
    return os.path.join(aixelask_root, rel)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_examples(
    dataset_key: str, split: str, aixelask_root: str
) -> int:
    """Return the total number of examples for *dataset_key* / *split*.

    For Scalability (split ``"all"``), returns the combined line count
    across all sub-files.
    """
    _validate_key(dataset_key)
    if dataset_key == "scalability":
        return _scalability_offsets.total(aixelask_root)

    path = _resolve_path(dataset_key, split, aixelask_root)
    return _count_lines(path)


def load_examples(
    dataset_key: str,
    split: str,
    indices: list[int],
    aixelask_root: str,
) -> list[SplitEntry]:
    """Load specific examples by index from the canonical JSONL.

    Parameters
    ----------
    dataset_key:
        One of ``"wikitq_4k"``, ``"wikitq_plus"``, ``"scalability"``.
    split:
        The split name (``"train"``, ``"valid"``, ``"test"`` for WikiTQ
        variants; ``"all"`` for Scalability).
    indices:
        Sorted, deduplicated zero-based line indices.
    aixelask_root:
        Absolute path to the AixelAsk repo root.

    Returns
    -------
    list[SplitEntry]
        Loaded examples with provenance metadata, ordered by index.
    """
    _validate_key(dataset_key)

    if dataset_key == "scalability":
        total = _scalability_offsets.total(aixelask_root)
        for idx in indices:
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"Index {idx} out of range for {dataset_key} "
                    f"(max {total - 1})"
                )
        raw = _scalability_offsets.load_lines(indices, aixelask_root)
        return [
            SplitEntry(
                source_dataset="scalability",
                source_file="all",
                source_index=gi,
                example=ex,
            )
            for gi, ex in raw
        ]

    path = _resolve_path(dataset_key, split, aixelask_root)
    total = _count_lines(path)
    for idx in indices:
        if idx < 0 or idx >= total:
            raise ValueError(
                f"Index {idx} out of range for {dataset_key} "
                f"(max {total - 1})"
            )

    raw = _load_jsonl_lines(path, indices)
    source_file = os.path.basename(path)
    return [
        SplitEntry(
            source_dataset=dataset_key,
            source_file=source_file,
            source_index=li,
            example=ex,
        )
        for li, ex in raw
    ]
