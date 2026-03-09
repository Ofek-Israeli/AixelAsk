"""Reward computation for GRPO training.

Computes a single weighted scalar reward for each generated DAG completion::

    r = w_correct * r_correct
      + w_valid   * r_valid
      - w_depth   * depth_term

Depth normalization modes:

- ``NONE``: raw integer depth.
- ``DIVIDE_BY_MAX_DEPTH`` (alias ``CLAMPED_LINEAR``):
  ``min(depth / max_depth, 1.0)``.

All weights and normalization parameters are read from the project ``Config``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

_NORMALIZED_MODES = frozenset({"DIVIDE_BY_MAX_DEPTH", "CLAMPED_LINEAR"})


def compute(
    r_correct: float,
    r_valid: float,
    depth: int,
    config: "Config",
) -> float:
    """Compute the weighted scalar reward for a single completion.

    Parameters
    ----------
    r_correct:
        Correctness signal in ``[0, 1]``.  ``1`` if the DAG-executed answer
        matches the gold answer (exact or fuzzy); ``0`` otherwise.
    r_valid:
        Validity signal: ``1`` if the DAG passes ``validate_dag``; ``0``
        otherwise.
    depth:
        Longest-path depth of the DAG (integer >= 0).
        ``CONFIG_REWARD_INVALID_DAG_DEPTH`` when the DAG is invalid or
        unparseable.
    config:
        Fully-resolved project ``Config`` carrying ``REWARD_WEIGHT_*`` and
        ``REWARD_DEPTH_NORMALIZATION`` / ``REWARD_MAX_DEPTH``.

    Returns
    -------
    float
        The scalar reward value.
    """
    w_correct = config.REWARD_WEIGHT_CORRECTNESS
    w_valid = config.REWARD_WEIGHT_VALIDITY
    w_depth = config.REWARD_WEIGHT_DEPTH

    depth_term = _normalize_depth(depth, config)

    reward = (
        w_correct * r_correct
        + w_valid * r_valid
        - w_depth * depth_term
    )
    return reward


def _normalize_depth(depth: int, config: "Config") -> float:
    """Apply the configured depth normalization."""
    mode = config.REWARD_DEPTH_NORMALIZATION

    if mode == "NONE":
        return float(depth)

    if mode in _NORMALIZED_MODES:
        max_depth = config.REWARD_MAX_DEPTH
        if max_depth <= 0:
            return float(depth)
        return min(depth / max_depth, 1.0)

    logger.warning(
        "Unknown REWARD_DEPTH_NORMALIZATION %r — falling back to NONE", mode,
    )
    return float(depth)
