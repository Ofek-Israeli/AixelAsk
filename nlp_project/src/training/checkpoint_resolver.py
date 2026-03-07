"""Checkpoint resolver for post-training test evaluation.

Centralises checkpoint path resolution, format detection (adapter-only vs
full-weight), and metadata extraction for ``test_main.py``.

Public API:

- ``ResolvedCheckpoint`` — dataclass with path, source, step, metric info.
- ``resolve_test_checkpoint(config, override_source)`` — resolves and
  validates the checkpoint, returning a ``ResolvedCheckpoint``.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ResolvedCheckpoint
# ---------------------------------------------------------------------------

@dataclass
class ResolvedCheckpoint:
    """Resolved checkpoint with path, provenance, and format information."""

    source: str
    path: str
    step: Optional[int]
    metric_name: Optional[str]
    metric_value: Optional[float]
    is_adapter_only: bool


# ---------------------------------------------------------------------------
# Source label mapping
# ---------------------------------------------------------------------------

_SOURCE_MAP = {
    "TEST_TRAINED_CHECKPOINT_BEST": "best",
    "TEST_TRAINED_CHECKPOINT_LATEST": "latest",
    "TEST_TRAINED_CHECKPOINT_MERGED": "merged",
    "TEST_TRAINED_CHECKPOINT_EXPLICIT_PATH": "explicit_path",
    "best": "best",
    "latest": "latest",
    "merged": "merged",
    "explicit_path": "explicit_path",
    "explicit": "explicit_path",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_test_checkpoint(
    config: "Config",
    override_source: Optional[str] = None,
) -> ResolvedCheckpoint:
    """Resolve the checkpoint to evaluate and return a ``ResolvedCheckpoint``.

    Parameters
    ----------
    config:
        Fully-resolved project ``Config``.
    override_source:
        If provided (from ``--checkpoint`` CLI flag), overrides
        ``CONFIG_TEST_TRAINED_CHECKPOINT_SOURCE``.

    Returns
    -------
    ResolvedCheckpoint

    Raises
    ------
    FileNotFoundError
        If the resolved checkpoint directory does not exist.
    """
    raw_source = override_source or config.TEST_TRAINED_CHECKPOINT_SOURCE
    source = _SOURCE_MAP.get(raw_source, raw_source)

    ckpt_dir = os.path.join(config.TRAIN_OUTPUT_DIR, "checkpoints")
    path = _resolve_path(source, ckpt_dir, config)

    if not os.path.isdir(path):
        raise FileNotFoundError(_error_message(source, path))

    is_adapter_only = _detect_adapter_only(path)

    step, metric_name, metric_value = _read_metadata(path)

    resolved = ResolvedCheckpoint(
        source=source,
        path=os.path.abspath(path),
        step=step,
        metric_name=metric_name,
        metric_value=metric_value,
        is_adapter_only=is_adapter_only,
    )

    logger.info(
        "Resolved checkpoint: source=%s, path=%s, step=%s, "
        "adapter_only=%s, metric=%s=%.6f" if metric_value is not None
        else "Resolved checkpoint: source=%s, path=%s, step=%s, "
        "adapter_only=%s",
        resolved.source,
        resolved.path,
        resolved.step,
        resolved.is_adapter_only,
        *(
            [resolved.metric_name, resolved.metric_value]
            if resolved.metric_value is not None else []
        ),
    )

    return resolved


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _resolve_path(source: str, ckpt_dir: str, config: "Config") -> str:
    """Map a source label to an absolute checkpoint path."""
    if source == "best":
        return os.path.join(ckpt_dir, "best")

    if source == "latest":
        latest_link = os.path.join(ckpt_dir, "latest")
        if os.path.islink(latest_link):
            target = os.readlink(latest_link)
            return os.path.join(ckpt_dir, target)
        if os.path.isdir(latest_link):
            return latest_link
        index = _load_index(ckpt_dir)
        if index and index.get("latest"):
            return os.path.join(ckpt_dir, index["latest"])
        return latest_link

    if source == "merged":
        return os.path.join(ckpt_dir, "merged")

    if source == "explicit_path":
        explicit = config.TEST_TRAINED_CHECKPOINT_PATH
        if not explicit:
            raise FileNotFoundError(
                "Checkpoint source is 'explicit_path' but "
                "CONFIG_TEST_TRAINED_CHECKPOINT_PATH is empty."
            )
        return explicit

    return os.path.join(ckpt_dir, source)


def _detect_adapter_only(path: str) -> bool:
    """Detect whether a checkpoint is adapter-only by file presence.

    Adapter-only: has ``adapter_config.json`` but no
    ``config.json`` + ``model*.safetensors``.
    """
    has_adapter_config = os.path.isfile(
        os.path.join(path, "adapter_config.json")
    )
    has_model_config = os.path.isfile(
        os.path.join(path, "config.json")
    )
    has_safetensors = bool(glob.glob(os.path.join(path, "model*.safetensors")))

    if has_adapter_config and not (has_model_config and has_safetensors):
        return True

    return False


def _read_metadata(
    path: str,
) -> tuple[Optional[int], Optional[str], Optional[float]]:
    """Read ``checkpoint_metadata.json`` if present."""
    meta_path = os.path.join(path, "checkpoint_metadata.json")
    if not os.path.isfile(meta_path):
        return None, None, None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return (
            meta.get("step"),
            meta.get("metric_name"),
            meta.get("metric_value"),
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read checkpoint metadata: %s", exc)
        return None, None, None


def _load_index(ckpt_dir: str) -> Optional[dict]:
    """Load ``checkpoint_index.json`` if present."""
    index_path = os.path.join(ckpt_dir, "checkpoint_index.json")
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _error_message(source: str, path: str) -> str:
    """Generate an actionable error message for missing checkpoints."""
    messages = {
        "best": (
            f"Best checkpoint not found at {path}. "
            "Has training completed with at least one evaluation step?"
        ),
        "latest": (
            f"Latest checkpoint not found at {path}. "
            "Has training saved at least one checkpoint?"
        ),
        "merged": (
            f"Merged checkpoint not found at {path}. "
            "Did you set CONFIG_TRAIN_SAVE_MERGED_ADAPTER=y during training?"
        ),
        "explicit_path": (
            f"Checkpoint path does not exist: {path}"
        ),
    }
    return messages.get(source, f"Checkpoint not found at {path}")
