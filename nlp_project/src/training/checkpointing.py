"""Checkpoint metadata management and project-specific save policies.

``MetadataCallback`` is a ``TrainerCallback`` that augments TRL's native
checkpoint saves with:

- ``checkpoint_metadata.json`` per checkpoint (step, timestamp, save reason).
- ``checkpoint_index.json`` manifest of all checkpoints.
- ``latest`` symlink to the most recent checkpoint.
- ``best/`` copy when ``CONFIG_TRAIN_SAVE_BEST_BY`` metric improves.
- ``final/`` checkpoint at normal training completion.
- Optional merged adapter export at end of training.
- Config compatibility checking for resume.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint index helpers
# ---------------------------------------------------------------------------

_COMPAT_FATAL_KEYS = frozenset({
    "INFERENCE_MODEL", "INFERENCE_MODEL_REVISION",
    "TRAIN_LORA_R", "TRAIN_LORA_ALPHA", "TRAIN_LORA_TARGET_MODULES",
    "TRAIN_USE_4BIT", "TRAINING_MODE",
})


def _checkpoints_dir(config: "Config") -> str:
    return os.path.join(config.TRAIN_OUTPUT_DIR, "checkpoints")


def _index_path(config: "Config") -> str:
    return os.path.join(_checkpoints_dir(config), "checkpoint_index.json")


def _load_index(config: "Config") -> Dict[str, Any]:
    path = _index_path(config)
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"checkpoints": [], "latest": None, "best": None, "final": None}


def _save_index(config: "Config", index: Dict[str, Any]) -> None:
    path = _index_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Config snapshot for compatibility checking
# ---------------------------------------------------------------------------

def _make_config_snapshot(config: "Config") -> Dict[str, Any]:
    """Create a JSON-serialisable dict of key config symbols."""
    snapshot: Dict[str, Any] = {}
    for attr in sorted(config.__dataclass_fields__):
        val = getattr(config, attr, None)
        if isinstance(val, (str, int, float, bool, type(None))):
            snapshot[attr] = val
        elif isinstance(val, list) and all(isinstance(v, (int, str)) for v in val):
            snapshot[attr] = val
    return snapshot


def _make_reward_config(config: "Config") -> Dict[str, Any]:
    return {
        "mode": config.REWARD_MODE,
        "w_correct": config.REWARD_WEIGHT_CORRECTNESS,
        "w_valid": config.REWARD_WEIGHT_VALIDITY,
        "w_depth": config.REWARD_WEIGHT_DEPTH,
        "depth_normalization": config.REWARD_DEPTH_NORMALIZATION,
        "max_depth": config.REWARD_MAX_DEPTH,
        "correctness_partial_credit": config.REWARD_CORRECTNESS_PARTIAL_CREDIT,
    }


# ---------------------------------------------------------------------------
# Compatibility check on resume
# ---------------------------------------------------------------------------

def check_resume_compatibility(
    checkpoint_path: str,
    config: "Config",
) -> None:
    """Check config compatibility between a checkpoint and current config.

    Raises ``ValueError`` on fatal mismatches; logs warnings on non-fatal ones.
    """
    snapshot_path = os.path.join(checkpoint_path, "config_snapshot.json")
    if not os.path.isfile(snapshot_path):
        logger.warning(
            "No config_snapshot.json in %s — skipping compatibility check",
            checkpoint_path,
        )
        return

    with open(snapshot_path, "r") as f:
        saved = json.load(f)

    current = _make_config_snapshot(config)

    fatal: List[str] = []
    warnings: List[str] = []

    for key in _COMPAT_FATAL_KEYS:
        saved_val = saved.get(key)
        current_val = current.get(key)
        if saved_val != current_val:
            fatal.append(
                f"  {key}: saved={saved_val!r}, current={current_val!r}"
            )

    for key in set(saved.keys()) | set(current.keys()):
        if key in _COMPAT_FATAL_KEYS:
            continue
        saved_val = saved.get(key)
        current_val = current.get(key)
        if saved_val != current_val:
            warnings.append(
                f"  {key}: saved={saved_val!r}, current={current_val!r}"
            )

    if warnings:
        logger.warning(
            "Non-fatal config changes since checkpoint:\n%s",
            "\n".join(warnings[:20]),
        )

    if fatal:
        raise ValueError(
            "Fatal config mismatch with checkpoint — cannot resume:\n"
            + "\n".join(fatal)
        )


# ---------------------------------------------------------------------------
# MetadataCallback
# ---------------------------------------------------------------------------

def make_metadata_callback(config: "Config"):
    """Build a ``TrainerCallback`` that manages checkpoint metadata.

    Returns a ``TrainerCallback`` instance ready for registration.
    """
    from transformers import TrainerCallback

    ckpt_dir = _checkpoints_dir(config)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_metric_name = config.TRAIN_SAVE_BEST_BY
    best_metric_value: Optional[float] = None
    higher_is_better = not any(
        kw in best_metric_name for kw in ("loss", "invalid", "depth")
    )

    class _MetadataCB(TrainerCallback):

        def on_save(self, args, state, control, **kwargs):
            nonlocal best_metric_value

            step = state.global_step
            ckpt_subdir = f"checkpoint-{step}"
            ckpt_path = os.path.join(ckpt_dir, ckpt_subdir)

            if not os.path.isdir(ckpt_path):
                ckpt_path = os.path.join(args.output_dir, ckpt_subdir)
                if not os.path.isdir(ckpt_path):
                    logger.warning(
                        "Expected checkpoint dir not found: %s",
                        ckpt_path,
                    )
                    return

            _write_checkpoint_metadata(
                ckpt_path, step, "periodic", config,
            )

            _update_latest_symlink(ckpt_dir, ckpt_subdir)

            index = _load_index(config)
            index["checkpoints"].append({
                "step": step,
                "path": f"{ckpt_subdir}/",
                "timestamp": _now_iso(),
                "save_reason": "periodic",
            })
            index["latest"] = f"{ckpt_subdir}/"
            _save_index(config, index)

            _try_update_best(
                state, config, ckpt_dir, ckpt_path, step,
                best_metric_name, best_metric_value, higher_is_better,
            )
            val = _get_metric_from_state(state, best_metric_name)
            if val is not None:
                if best_metric_value is None:
                    best_metric_value = val
                elif higher_is_better and val > best_metric_value:
                    best_metric_value = val
                elif not higher_is_better and val < best_metric_value:
                    best_metric_value = val

        def on_train_end(self, args, state, control, **kwargs):
            step = state.global_step
            final_dir = os.path.join(ckpt_dir, "final")
            os.makedirs(final_dir, exist_ok=True)

            latest_path = os.path.join(ckpt_dir, "latest")
            source = None
            if os.path.islink(latest_path):
                source = os.path.join(ckpt_dir, os.readlink(latest_path))
            elif os.path.isdir(latest_path):
                source = latest_path

            if source and os.path.isdir(source) and source != final_dir:
                if os.path.exists(final_dir):
                    shutil.rmtree(final_dir)
                shutil.copytree(source, final_dir)

            _write_checkpoint_metadata(
                final_dir, step, "final", config,
            )

            index = _load_index(config)
            index["checkpoints"].append({
                "step": step,
                "path": "final/",
                "timestamp": _now_iso(),
                "save_reason": "final",
            })
            index["final"] = "final/"
            _save_index(config, index)

            if config.TRAIN_SAVE_MERGED_ADAPTER:
                _do_merged_export(config, ckpt_dir, step)

    return _MetadataCB()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_checkpoint_metadata(
    ckpt_path: str,
    step: int,
    save_reason: str,
    config: "Config",
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
) -> None:
    """Write ``checkpoint_metadata.json``, ``config_snapshot.json``, and
    ``reward_config.json`` into *ckpt_path*."""
    meta = {
        "step": step,
        "timestamp": _now_iso(),
        "save_reason": save_reason,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }
    with open(os.path.join(ckpt_path, "checkpoint_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(ckpt_path, "config_snapshot.json"), "w") as f:
        json.dump(_make_config_snapshot(config), f, indent=2)

    with open(os.path.join(ckpt_path, "reward_config.json"), "w") as f:
        json.dump(_make_reward_config(config), f, indent=2)


def _update_latest_symlink(ckpt_dir: str, target_subdir: str) -> None:
    """Atomically update the ``latest`` symlink."""
    link_path = os.path.join(ckpt_dir, "latest")
    tmp_link = link_path + ".tmp"
    try:
        if os.path.islink(tmp_link) or os.path.exists(tmp_link):
            os.remove(tmp_link)
        os.symlink(target_subdir, tmp_link)
        os.rename(tmp_link, link_path)
    except OSError as exc:
        logger.warning("Could not update latest symlink: %s", exc)


def _try_update_best(
    state,
    config: "Config",
    ckpt_dir: str,
    source_ckpt_path: str,
    step: int,
    best_metric_name: str,
    current_best: Optional[float],
    higher_is_better: bool,
) -> None:
    """Copy checkpoint to ``best/`` if the tracked metric improved."""
    val = _get_metric_from_state(state, best_metric_name)
    if val is None:
        return

    improved = False
    if current_best is None:
        improved = True
    elif higher_is_better and val > current_best:
        improved = True
    elif not higher_is_better and val < current_best:
        improved = True

    if not improved:
        return

    best_dir = os.path.join(ckpt_dir, "best")
    try:
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(source_ckpt_path, best_dir)
        _write_checkpoint_metadata(
            best_dir, step, "best", config,
            metric_name=best_metric_name,
            metric_value=val,
        )

        index = _load_index(config)
        index["checkpoints"].append({
            "step": step,
            "path": "best/",
            "timestamp": _now_iso(),
            "save_reason": "best",
            "metric_name": best_metric_name,
            "metric_value": val,
        })
        index["best"] = "best/"
        _save_index(config, index)

        logger.info(
            "New best checkpoint at step %d: %s=%.6f",
            step, best_metric_name, val,
        )
    except Exception:
        logger.warning("Failed to copy best checkpoint", exc_info=True)


def _get_metric_from_state(state, metric_name: str) -> Optional[float]:
    """Extract a metric value from the trainer's log history."""
    if not state.log_history:
        return None
    for entry in reversed(state.log_history):
        if metric_name in entry:
            return entry[metric_name]
    return None


def _do_merged_export(config: "Config", ckpt_dir: str, step: int) -> None:
    """Merge the latest adapter into the base model and save to ``merged/``."""
    try:
        from src.training.lora_factory import merge_and_export
        from src.download_models import resolve_model_path

        latest_path = os.path.join(ckpt_dir, "latest")
        if os.path.islink(latest_path):
            adapter_path = os.path.join(ckpt_dir, os.readlink(latest_path))
        elif os.path.isdir(os.path.join(ckpt_dir, "final")):
            adapter_path = os.path.join(ckpt_dir, "final")
        else:
            logger.warning("No adapter found for merged export")
            return

        merged_dir = os.path.join(ckpt_dir, "merged")
        resolved_base = resolve_model_path(config)

        merge_and_export(adapter_path, merged_dir, config, resolved_base)

        _write_checkpoint_metadata(
            merged_dir, step, "merged", config,
        )

        index = _load_index(config)
        index["checkpoints"].append({
            "step": step,
            "path": "merged/",
            "timestamp": _now_iso(),
            "save_reason": "merged",
        })
        _save_index(config, index)

        logger.info("Merged adapter exported to %s", merged_dir)

    except Exception:
        logger.warning("Merged export failed (non-fatal)", exc_info=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
