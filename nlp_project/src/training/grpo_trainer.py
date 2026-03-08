"""GRPO trainer construction and reward function wiring.

``build_trainer`` assembles a TRL ``GRPOTrainer`` with the project's reward
function, dataset, and all training callbacks (``StatsCallback``,
``CurvesCallback``, ``MetadataCallback``).

The reward function calls ``dag_reward_parser.parse`` +
``dag_reward_parser.execute_for_reward`` to obtain a correctness signal,
then ``reward.compute`` for the weighted scalar reward.  Per-completion
breakdown metrics are written to the shared ``RewardMetricsAccumulator``
for ``StatsCallback`` to read.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    import datasets
    from src.config import Config
    from src.training.curves import CurvesManager
    from src.training.train_stats import RewardMetricsAccumulator

logger = logging.getLogger(__name__)


def build_trainer(
    config: "Config",
    model,
    tokenizer,
    train_dataset: "datasets.Dataset",
    eval_dataset: "datasets.Dataset",
    accumulator: "RewardMetricsAccumulator",
    curves_manager: Optional["CurvesManager"] = None,
    table_embedding_map: Optional[Dict[str, Any]] = None,
):
    """Build a ``GRPOTrainer`` with reward function and all callbacks.

    Parameters
    ----------
    config:
        Fully-resolved project ``Config``.
    model:
        PEFT-wrapped causal-LM model.
    tokenizer:
        HF tokenizer.
    train_dataset:
        Training ``datasets.Dataset`` with ``prompt``, ``gold_answer``,
        ``table``, and provenance columns.
    eval_dataset:
        Validation ``datasets.Dataset`` (same schema).
    accumulator:
        Shared ``RewardMetricsAccumulator`` for reward breakdown metrics.
    curves_manager:
        Optional ``CurvesManager`` for learning-curve updates.
    table_embedding_map:
        Pre-computed table embeddings keyed by SHA1 table ID.

    Returns
    -------
    GRPOTrainer
        Ready to call ``.train()``.
    """
    from trl import GRPOTrainer
    from src.training.train_config import build_grpo_config
    from src.training.train_stats import (
        StatsCallback,
        _make_stats_callback_cls,
        make_curves_callback,
    )
    from src.training.checkpointing import make_metadata_callback

    grpo_config = build_grpo_config(config)

    reward_fn = _build_reward_func(config, accumulator, table_embedding_map)

    callbacks = []

    stats_cb = StatsCallback(config, accumulator)
    callbacks.append(_make_stats_callback_cls(stats_cb))

    if curves_manager is not None and config.TRAIN_CURVES_TEX_ENABLE:
        callbacks.append(make_curves_callback(config, curves_manager, accumulator))

    callbacks.append(make_metadata_callback(config))

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info(
        "GRPOTrainer built: train=%d, eval=%d, callbacks=%d",
        len(train_dataset),
        len(eval_dataset),
        len(callbacks),
    )

    return trainer


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _build_reward_func(
    config: "Config",
    accumulator: "RewardMetricsAccumulator",
    table_embedding_map: Optional[Dict[str, Any]],
) -> Callable:
    """Build the reward function closure for ``GRPOTrainer``.

    TRL calls ``reward_func(completions, **kwargs)`` where *completions*
    is ``list[list[dict]]`` and extra dataset columns arrive via
    ``**kwargs``.
    """
    from src.training import dag_reward_parser, reward

    def reward_func(
        completions: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        gold_answers: List[str] = kwargs.get("gold_answer", [])
        tables: List[Any] = kwargs.get("table", [])
        questions: List[str] = kwargs.get(
            "question",
            kwargs.get("statement", [""] * len(completions)),
        )

        rewards: List[float] = []
        records: List[Dict[str, Any]] = []

        for i, completion in enumerate(completions):
            text = _extract_completion_text(completion)
            gold = gold_answers[i] if i < len(gold_answers) else ""
            table = tables[i] if i < len(tables) else {}
            question = questions[i] if i < len(questions) else ""

            parsed = dag_reward_parser.parse(text)

            r_correct = 0.0
            r_valid = 1.0 if parsed.valid else 0.0
            is_invalid = not parsed.valid
            is_parse_fail = parsed.error_category == dag_reward_parser.ERROR_JSON_PARSE
            depth = parsed.depth

            if parsed.valid and parsed.dag:
                try:
                    predicted = dag_reward_parser.execute_for_reward(
                        parsed.dag, table, question, config,
                        table_embedding_map=table_embedding_map,
                    )
                    r_correct = _compute_correctness(predicted, gold, config)
                except Exception:
                    logger.debug(
                        "execute_for_reward failed for completion %d", i,
                        exc_info=True,
                    )
                    r_correct = 0.0

            scalar = reward.compute(
                r_correct=r_correct,
                r_valid=r_valid,
                depth=depth,
                config=config,
            )
            rewards.append(scalar)

            records.append({
                "r_correct": r_correct,
                "r_valid": r_valid,
                "depth": depth,
                "is_invalid": is_invalid,
                "is_parse_fail": is_parse_fail,
                "response_len": len(text),
            })

        accumulator.append_batch(records)
        return rewards

    return reward_func


def _extract_completion_text(completion: List[Dict[str, str]]) -> str:
    """Extract the text content from a TRL completion structure."""
    if not completion:
        return ""
    if isinstance(completion, list):
        if isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return str(completion[0])
    return str(completion)


def _compute_correctness(
    predicted: str,
    gold: str,
    config: "Config",
) -> float:
    """Compute correctness score: exact match or fuzzy partial credit."""
    if not predicted or not gold:
        return 0.0

    pred_norm = predicted.strip().lower()
    gold_norm = gold.strip().lower()

    if config.REWARD_CORRECTNESS_PARTIAL_CREDIT:
        import difflib
        return difflib.SequenceMatcher(None, pred_norm, gold_norm).ratio()

    return 1.0 if pred_norm == gold_norm else 0.0
