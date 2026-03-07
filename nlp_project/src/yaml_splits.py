"""Load train/valid/test splits from ``train_valid_test.yaml``.

Each ID in the YAML encodes the physical location of an example in the
upstream repo:

    w4k-train-123      → WikiTQ-4k,   train.jsonl, line 123
    w4k-valid-5        → WikiTQ-4k,   valid.jsonl, line 5
    w4k-test-42        → WikiTQ-4k,   test.jsonl,  line 42
    w+-train-100       → WikiTQ+,     train.jsonl, line 100
    w+-valid-8         → WikiTQ+,     valid.jsonl, line 8
    w+-test-71         → WikiTQ+,     test.jsonl,  line 71
    scalability-3165   → Scalability, global index 3165

The YAML file has three documents (``---`` separated), each with
``title`` ("Train"/"Validation"/"Test") and ``ids`` (list of strings).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import yaml

from src.training.dataset_registry import SplitEntry, load_examples

logger = logging.getLogger(__name__)

_ID_PREFIXES_FILE_MAP = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}


def parse_id(id_str: str) -> Tuple[str, str, int]:
    """Parse a YAML split ID into ``(dataset_key, canonical_split, index)``.

    Returns
    -------
    tuple
        ``(dataset_key, canonical_split, line_index)`` where
        ``canonical_split`` is the split argument accepted by
        ``dataset_registry.load_examples`` (e.g. ``"train"``, ``"all"``).
    """
    if id_str.startswith("scalability-"):
        idx = int(id_str.split("-", 1)[1])
        return ("scalability", "all", idx)

    if id_str.startswith("w4k-"):
        rest = id_str[4:]  # e.g. "train-123"
        split_name, _, idx_str = rest.rpartition("-")
        return ("wikitq_4k", split_name, int(idx_str))

    if id_str.startswith("w+-"):
        rest = id_str[3:]  # e.g. "test-71"
        split_name, _, idx_str = rest.rpartition("-")
        return ("wikitq_plus", split_name, int(idx_str))

    raise ValueError(f"Unrecognised YAML split ID: {id_str!r}")


def _load_section_examples(
    ids: List[str],
    aixelask_root: str,
) -> List[Dict[str, Any]]:
    """Resolve a list of YAML IDs to loaded example dicts with provenance."""
    grouped: Dict[Tuple[str, str], List[int]] = {}
    for id_str in ids:
        dataset_key, canonical_split, idx = parse_id(id_str)
        key = (dataset_key, canonical_split)
        grouped.setdefault(key, []).append(idx)

    all_entries: List[SplitEntry] = []
    for (dataset_key, canonical_split), indices in grouped.items():
        indices_sorted = sorted(set(indices))
        entries = load_examples(dataset_key, canonical_split, indices_sorted, aixelask_root)
        all_entries.extend(entries)

    return [
        {
            **entry.example,
            "_source_dataset": entry.source_dataset,
            "_source_file": entry.source_file,
            "_source_index": entry.source_index,
        }
        for entry in all_entries
    ]


def load_yaml_splits(
    yaml_path: str,
    aixelask_root: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Read ``train_valid_test.yaml`` and return loaded examples per split.

    Returns
    -------
    dict
        ``{"train": [...], "valid": [...], "test": [...]}`` where each
        value is a list of example dicts enriched with
        ``_source_dataset``, ``_source_file``, ``_source_index``.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Split YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    _TITLE_TO_KEY = {
        "Train": "train",
        "Validation": "valid",
        "Test": "test",
    }

    result: Dict[str, List[Dict[str, Any]]] = {
        "train": [], "valid": [], "test": [],
    }

    for doc in docs:
        if doc is None:
            continue
        title = doc.get("title", "")
        split_key = _TITLE_TO_KEY.get(title)
        if split_key is None:
            logger.warning("Unknown section title in YAML: %r — skipping", title)
            continue
        ids = doc.get("ids", [])
        if not ids:
            logger.warning("Section %r has no ids", title)
            continue
        logger.info("Loading %d examples for %r split …", len(ids), title)
        result[split_key] = _load_section_examples(ids, aixelask_root)

    logger.info(
        "YAML splits loaded: train=%d, valid=%d, test=%d",
        len(result["train"]), len(result["valid"]), len(result["test"]),
    )
    return result
