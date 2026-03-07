"""Tiny overfit-PoC subset selection.

Selects a small subset of indices from a dataset for the overfit proof-of-
concept.  Three selection modes are supported:

- ``FIRST_N``: deterministic first-N indices (no shuffling).
- ``RANDOM_SEEDED``: seeded random sample without replacement.
- ``FIXED_INDICES_FILE``: explicit integer indices read from a file.

The selected subset is used as **both** train and validation in overfit-PoC
mode — the only intentional train/valid overlap in the project.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


def select_overfit_subset(config: "Config", dataset_size: int) -> List[int]:
    """Return sorted indices for the overfit-PoC subset.

    Parameters
    ----------
    config:
        Fully-resolved project ``Config`` (needs ``OVERFIT_POC_SELECTION_MODE``,
        ``OVERFIT_POC_NUM_EXAMPLES``, ``OVERFIT_POC_INDICES_FILE``, ``GLOBAL_SEED``).
    dataset_size:
        Total number of examples available in the source dataset.

    Returns
    -------
    list[int]
        Sorted, deduplicated list of selected zero-based indices.

    Raises
    ------
    ValueError
        If the mode is unrecognised, the requested count exceeds *dataset_size*,
        or the indices file contains out-of-bounds values.
    """
    mode = config.OVERFIT_POC_SELECTION_MODE
    n = config.OVERFIT_POC_NUM_EXAMPLES

    if mode == "FIRST_N":
        indices = _select_first_n(n, dataset_size)
    elif mode == "RANDOM_SEEDED":
        indices = _select_random_seeded(n, dataset_size, config.GLOBAL_SEED)
    elif mode == "FIXED_INDICES_FILE":
        indices = _select_from_file(config.OVERFIT_POC_INDICES_FILE, dataset_size)
    else:
        raise ValueError(
            f"Unknown OVERFIT_POC_SELECTION_MODE: {mode!r}. "
            f"Expected FIRST_N, RANDOM_SEEDED, or FIXED_INDICES_FILE."
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
