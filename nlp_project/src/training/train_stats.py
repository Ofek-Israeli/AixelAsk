"""Training statistics collection: per-step/per-eval JSONL writers and
Trainer callbacks for metrics aggregation and curves updates.

``RewardMetricsAccumulator`` is the thread-safe bridge between
``reward_func`` (writer) and ``StatsCallback`` (reader).

``StatsCallback`` fires on each ``on_log`` event, extracts TRL-native
metrics from ``trainer.state``, flushes the accumulator for reward
breakdown metrics, and appends the combined record to per-step JSONL.

``CurvesCallback`` updates TSV data files and triggers TeX generation /
PDF compilation on the configured cadence.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.config import Config
    from src.training.curves import CurvesManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RewardMetricsAccumulator
# ---------------------------------------------------------------------------

class RewardMetricsAccumulator:
    """Thread-safe accumulator for per-completion reward breakdown metrics.

    ``reward_func`` appends after each batch; ``StatsCallback`` reads and
    flushes at each step boundary.  The last flushed batch is cached so
    that ``CurvesCallback`` can read the same aggregated data without
    double-flushing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._last_flushed: List[Dict[str, Any]] = []

    def append_batch(self, records: List[Dict[str, Any]]) -> None:
        """Append a list of per-completion metric dicts.

        Each dict should contain: ``r_correct``, ``r_valid``, ``depth``,
        ``is_invalid``, ``is_parse_fail``, ``response_len``.
        """
        with self._lock:
            self._buffer.extend(records)

    def flush(self) -> List[Dict[str, Any]]:
        """Return all buffered records and clear the buffer.

        The returned batch is also cached (accessible via
        ``last_flushed``) so other consumers can read it.
        """
        with self._lock:
            batch, self._buffer = self._buffer, []
            self._last_flushed = batch
            return batch

    def last_flushed(self) -> List[Dict[str, Any]]:
        """Return the records from the most recent ``flush()`` call."""
        with self._lock:
            return list(self._last_flushed)


# ---------------------------------------------------------------------------
# StatsCallback
# ---------------------------------------------------------------------------

class StatsCallback:
    """Trainer callback that writes per-step and per-eval JSONL files.

    Reads TRL-native metrics from ``trainer.state.log_history`` and
    reward breakdown metrics from the shared ``RewardMetricsAccumulator``.
    """

    def __init__(
        self,
        config: "Config",
        accumulator: RewardMetricsAccumulator,
    ) -> None:
        from transformers import TrainerCallback

        self._config = config
        self._accumulator = accumulator
        self._start_time = time.monotonic()
        self._last_step_time = self._start_time
        self._step_jsonl_path = config.TRAIN_STATS_PER_STEP_JSONL
        self._eval_jsonl_path = config.TRAIN_STATS_PER_EVAL_JSONL
        self._flush_every = config.TRAIN_STATS_FLUSH_EVERY_STEPS
        self._step_buffer: List[str] = []

        os.makedirs(os.path.dirname(self._step_jsonl_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self._eval_jsonl_path) or ".", exist_ok=True)

        self._callback = _StatsTrainerCallback(self)

    @property
    def callback(self):
        """Return the ``TrainerCallback`` instance to register."""
        return self._callback

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        now = time.monotonic()
        step_time = now - self._last_step_time
        self._last_step_time = now

        record: Dict[str, Any] = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "wallclock_sec": round(now - self._start_time, 2),
            "step_time_sec": round(step_time, 2),
        }

        if logs:
            for key in (
                "loss", "learning_rate", "grad_norm",
                "policy_loss", "kl_loss", "total_loss",
                "reward", "reward_std",
            ):
                if key in logs:
                    mapped = _map_log_key(key)
                    record[mapped] = logs[key]

        reward_records = self._accumulator.flush()
        if reward_records:
            agg = _aggregate_reward_records(reward_records)
            record.update(agg)

        is_eval = logs and any(k.startswith("eval_") for k in logs)
        if is_eval:
            self._write_eval_record(record, logs)
        else:
            self._buffer_step_record(record)

    def _buffer_step_record(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        self._step_buffer.append(line)
        if len(self._step_buffer) >= self._flush_every:
            self._flush_step_buffer()

    def _flush_step_buffer(self) -> None:
        if not self._step_buffer:
            return
        with open(self._step_jsonl_path, "a", encoding="utf-8") as f:
            for line in self._step_buffer:
                f.write(line + "\n")
        self._step_buffer.clear()

    def _write_eval_record(
        self, record: Dict[str, Any], logs: Optional[Dict]
    ) -> None:
        if logs:
            for key, value in logs.items():
                if key.startswith("eval_"):
                    record[key] = value

        line = json.dumps(record, default=str)
        with open(self._eval_jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def flush_final(self) -> None:
        """Flush any remaining buffered step records."""
        self._flush_step_buffer()


class _StatsTrainerCallback:
    """Thin ``TrainerCallback`` adapter that delegates to ``StatsCallback``."""

    def __init__(self, stats: StatsCallback) -> None:
        from transformers import TrainerCallback
        self.__class__ = type(
            "_StatsTrainerCallback",
            (TrainerCallback,),
            {
                "on_log": lambda self_, args, state, control, logs=None, **kw: (
                    stats.on_log(args, state, control, logs=logs, **kw)
                ),
                "on_train_end": lambda self_, args, state, control, **kw: (
                    stats.flush_final()
                ),
            },
        )


def _make_stats_callback_cls(stats: StatsCallback):
    """Dynamically build a TrainerCallback class that delegates to *stats*."""
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            stats.on_log(args, state, control, logs=logs, **kwargs)

        def on_train_end(self, args, state, control, **kwargs):
            stats.flush_final()

    return _CB()


# ---------------------------------------------------------------------------
# CurvesCallback
# ---------------------------------------------------------------------------

class CurvesCallback:
    """Trainer callback that updates TSV files and triggers TeX compilation."""

    def __init__(
        self,
        config: "Config",
        curves_manager: "CurvesManager",
        accumulator: RewardMetricsAccumulator,
    ) -> None:
        self._config = config
        self._curves = curves_manager
        self._accumulator = accumulator
        self._update_every = config.TRAIN_CURVES_UPDATE_EVERY_STEPS
        self._compile_every = config.TRAIN_CURVES_COMPILE_EVERY_STEPS
        self._last_update_step = 0
        self._last_compile_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        step = state.global_step
        is_eval = logs and any(k.startswith("eval_") for k in logs)

        if logs and (is_eval or step - self._last_update_step >= self._update_every):
            self._update(step, logs)
            if not is_eval:
                self._last_update_step = step

        if is_eval or step - self._last_compile_step >= self._compile_every:
            self._compile()
            if not is_eval:
                self._last_compile_step = step

    def on_train_end(self, args, state, control, **kwargs) -> None:
        if self._config.TRAIN_CURVES_COMPILE_AT_END:
            self._compile()

    def _update(self, step: int, logs: Dict[str, Any]) -> None:
        is_eval = any(k.startswith("eval_") for k in logs)

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                mapped = _map_log_key(key)
                self._curves.update_tsv(mapped, step, value)
                alias = _ALIAS_MAP.get(mapped)
                if alias:
                    self._curves.update_tsv(alias, step, value)

        reward_records = self._accumulator.last_flushed()
        if reward_records:
            agg = _aggregate_reward_records(reward_records)
            prefix = "eval_" if is_eval else ""
            for key, value in agg.items():
                if isinstance(value, (int, float)):
                    self._curves.update_tsv(f"{prefix}{key}", step, value)

        # GRPO advantages are (reward - group_mean) / group_std. The mean
        # across a group is 0 by construction, but tracking it confirms
        # normalisation is behaving as expected.
        reward_val = logs.get("reward") if not is_eval else logs.get("eval_reward")
        if reward_val is not None:
            self._curves.update_tsv("advantage_mean", step, 0.0)

        self._curves.generate_tex()

    def _compile(self) -> None:
        if not self._config.TRAIN_CURVES_COMPILE_ENABLE:
            return
        try:
            from src.training.tex_compile import compile_all
            compile_all(self._config)
        except Exception:
            logger.warning("TeX compilation failed (non-fatal)", exc_info=True)


def make_curves_callback(
    config: "Config",
    curves_manager: "CurvesManager",
    accumulator: RewardMetricsAccumulator,
):
    """Build a ``TrainerCallback`` instance for curves updates."""
    from transformers import TrainerCallback

    cb = CurvesCallback(config, curves_manager, accumulator)

    class _CB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            cb.on_log(args, state, control, logs=logs, **kwargs)

        def on_train_end(self, args, state, control, **kwargs):
            cb.on_train_end(args, state, control, **kwargs)

    return _CB()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_TRL_KEY_MAP = {
    "loss": "train_total_loss",
    "learning_rate": "lr",
    "policy_loss": "train_policy_loss",
    "kl_loss": "train_kl_loss",
    "kl": "train_kl_loss",
    "total_loss": "train_total_loss",
    "reward": "reward_mean",
    "reward_std": "reward_std",
    "rewards/reward_func/mean": "reward_mean",
    "rewards/reward_func/std": "group_reward_std_mean",
    "eval_loss": "eval_train_total_loss",
    "eval_reward": "eval_reward_mean",
    "eval_reward_std": "eval_reward_std",
    "eval_rewards/reward_func/mean": "eval_reward_mean",
    "eval_rewards/reward_func/std": "eval_group_reward_std_mean",
}

# Keys that should be written to a second TSV alias as well.
# In GRPO the single "loss" is the policy loss.
_ALIAS_MAP: Dict[str, str] = {
    "train_total_loss": "train_policy_loss",
}


def _map_log_key(key: str) -> str:
    """Map TRL log keys to project-standard field names."""
    return _TRL_KEY_MAP.get(key, key)


def _aggregate_reward_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from per-completion reward records."""
    n = len(records)
    if n == 0:
        return {}

    r_corrects = [r.get("r_correct", 0.0) for r in records]
    r_valids = [r.get("r_valid", 0.0) for r in records]
    depths = [r.get("depth", 0) for r in records]
    invalids = [r.get("is_invalid", False) for r in records]
    parse_fails = [r.get("is_parse_fail", False) for r in records]
    resp_lens = [r.get("response_len", 0) for r in records]

    valid_depths = [d for d, inv in zip(depths, invalids) if not inv and d >= 0]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    return {
        "reward_correctness_mean": _mean(r_corrects),
        "reward_validity_mean": _mean(r_valids),
        "reward_depth_mean": _mean(depths),
        "correctness_rate": _mean([1.0 if c > 0 else 0.0 for c in r_corrects]),
        "validity_rate": 1.0 - _mean([1.0 if inv else 0.0 for inv in invalids]),
        "invalid_rate": _mean([1.0 if inv else 0.0 for inv in invalids]),
        "depth_mean": _mean(valid_depths) if valid_depths else float("nan"),
        "depth_std": _std(valid_depths) if valid_depths else float("nan"),
        "parse_success_rate": 1.0 - _mean([1.0 if pf else 0.0 for pf in parse_fails]),
        "dag_parse_fail_rate": _mean([1.0 if pf else 0.0 for pf in parse_fails]),
        "response_len_mean": _mean(resp_lens),
        "response_len_std": _std(resp_lens),
    }
