"""Tiny overfit-PoC subset selection.

Selects a small subset of indices from a dataset for the overfit proof-of-
concept.  Four selection modes are supported:

- ``FIRST_N``: deterministic first-N indices (no shuffling).
- ``RANDOM_SEEDED``: seeded random sample without replacement.
- ``FIXED_INDICES_FILE``: explicit integer indices read from a file.
- ``EXPLICIT_IDS``: comma-separated YAML split IDs resolved against
  provenance columns in the loaded dataset.

The selected subset is used as **both** train and validation in overfit-PoC
mode — the only intentional train/valid overlap in the project.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    import datasets
    from src.config import Config

logger = logging.getLogger(__name__)


def select_overfit_subset(
    config: "Config",
    dataset: "datasets.Dataset",
) -> List[int]:
    """Return sorted indices for the overfit-PoC subset.

    Parameters
    ----------
    config:
        Fully-resolved project ``Config`` (needs ``OVERFIT_POC_SELECTION_MODE``,
        ``OVERFIT_POC_NUM_EXAMPLES``, ``OVERFIT_POC_INDICES_FILE``,
        ``OVERFIT_POC_ID_LIST``, ``GLOBAL_SEED``).
    dataset:
        The full train ``datasets.Dataset`` with provenance columns
        (``source_dataset``, ``source_file``, ``source_index``).

    Returns
    -------
    list[int]
        Sorted, deduplicated list of selected zero-based indices.

    Raises
    ------
    ValueError
        If the mode is unrecognised, the requested count exceeds the
        dataset size, or requested IDs are not found.
    """
    mode = config.OVERFIT_POC_SELECTION_MODE
    n = config.OVERFIT_POC_NUM_EXAMPLES
    dataset_size = len(dataset)

    if mode == "FIRST_N":
        indices = _select_first_n(n, dataset_size)
    elif mode == "RANDOM_SEEDED":
        indices = _select_random_seeded(n, dataset_size, config.GLOBAL_SEED)
    elif mode == "FIXED_INDICES_FILE":
        indices = _select_from_file(config.OVERFIT_POC_INDICES_FILE, dataset_size)
    elif mode == "EXPLICIT_IDS":
        indices = _select_by_yaml_ids(config.OVERFIT_POC_ID_LIST, dataset)
    else:
        raise ValueError(
            f"Unknown OVERFIT_POC_SELECTION_MODE: {mode!r}. "
            f"Expected FIRST_N, RANDOM_SEEDED, FIXED_INDICES_FILE, "
            f"or EXPLICIT_IDS."
        )

    logger.info(
        "Overfit-PoC subset: mode=%s, selected %d indices (dataset_size=%d)",
        mode, len(indices), dataset_size,
    )
    return indices


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def _select_first_n(n: int, dataset_size: int) -> List[int]:
    """Take the first *n* indices — deterministic, no shuffling."""
    if n > dataset_size:
        raise ValueError(
            f"OVERFIT_POC_NUM_EXAMPLES ({n}) exceeds dataset size "
            f"({dataset_size})."
        )
    return list(range(n))


def _select_random_seeded(n: int, dataset_size: int, seed: int) -> List[int]:
    """Seeded random sample of *n* indices without replacement."""
    if n > dataset_size:
        raise ValueError(
            f"OVERFIT_POC_NUM_EXAMPLES ({n}) exceeds dataset size "
            f"({dataset_size})."
        )
    rng = random.Random(seed)
    sampled = rng.sample(range(dataset_size), n)
    sampled.sort()
    return sampled


def _select_from_file(path: str, dataset_size: int) -> List[int]:
    """Read explicit integer indices (one per line) from *path*."""
    if not path:
        raise ValueError(
            "OVERFIT_POC_SELECTION_MODE=FIXED_INDICES_FILE but "
            "OVERFIT_POC_INDICES_FILE is empty."
        )

    seen: set[int] = set()
    indices: List[int] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(
                    f"Non-integer value on line {line_no} of "
                    f"{path!r}: {token!r}"
                )
            if idx < 0 or idx >= dataset_size:
                raise ValueError(
                    f"Index {idx} (line {line_no} of {path!r}) is out of "
                    f"range [0, {dataset_size})."
                )
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)

    if not indices:
        raise ValueError(f"No valid indices found in {path!r}.")

    indices.sort()
    return indices


def _select_by_yaml_ids(
    id_list_str: str,
    dataset: "datasets.Dataset",
) -> List[int]:
    """Resolve comma-separated YAML IDs to dataset row indices."""
    from src.yaml_splits import parse_id

    if not id_list_str or not id_list_str.strip():
        raise ValueError(
            "OVERFIT_POC_SELECTION_MODE=EXPLICIT_IDS but "
            "OVERFIT_POC_ID_LIST is empty."
        )

    requested_ids = [tok.strip() for tok in id_list_str.split(",") if tok.strip()]
    if not requested_ids:
        raise ValueError("OVERFIT_POC_ID_LIST contains no valid IDs.")

    provenance_index: dict[tuple[str, str, int], int] = {}
    src_datasets = dataset["source_dataset"]
    src_files = dataset["source_file"]
    src_indices = dataset["source_index"]
    for row_idx in range(len(dataset)):
        canonical_split = src_files[row_idx].removesuffix(".jsonl")
        key = (src_datasets[row_idx], canonical_split, src_indices[row_idx])
        provenance_index[key] = row_idx

    indices: List[int] = []
    seen: set[int] = set()
    for yaml_id in requested_ids:
        dataset_key, split_name, src_idx = parse_id(yaml_id)
        key = (dataset_key, split_name, src_idx)
        row_idx = provenance_index.get(key)
        if row_idx is None:
            raise ValueError(
                f"YAML ID {yaml_id!r} (resolved to dataset={dataset_key!r}, "
                f"split={split_name!r}, index={src_idx}) not found in the "
                f"train split ({len(dataset)} examples)."
            )
        if row_idx not in seen:
            seen.add(row_idx)
            indices.append(row_idx)

    indices.sort()
    return indices
