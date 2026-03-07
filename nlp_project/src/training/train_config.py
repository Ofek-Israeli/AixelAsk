"""Training-specific configuration: extends base Config with split indices
and builds TRL ``GRPOConfig`` from Kconfig symbols.

``TrainConfig`` is a thin wrapper that augments the base ``Config`` with
structured per-dataset split index dicts.  ``build_grpo_config`` translates
Kconfig ``CONFIG_GRPO_*`` symbols into the ``GRPOConfig`` dataclass expected
by TRL's ``GRPOTrainer``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

_DATASET_KEYS = ("wikitq_4k", "wikitq_plus", "scalability")


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training-specific fields derived from the base ``Config``.

    Constructed by ``from_config`` — not instantiated directly.
    """

    base: "Config" = field(repr=False)

    split_mode: str = "seeded_ratio"

    split_train_indices: Dict[str, List[int]] = field(default_factory=dict)
    split_valid_indices: Dict[str, List[int]] = field(default_factory=dict)
    split_test_indices: Dict[str, List[int]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: "Config") -> "TrainConfig":
        """Build a ``TrainConfig`` from a fully-resolved base ``Config``."""
        train_indices: Dict[str, List[int]] = {
            "wikitq_4k": list(config.SPLIT_TRAIN_WIKITQ_4K_INDICES),
            "wikitq_plus": list(config.SPLIT_TRAIN_WIKITQ_PLUS_INDICES),
            "scalability": list(config.SPLIT_TRAIN_SCALABILITY_INDICES),
        }
        valid_indices: Dict[str, List[int]] = {
            "wikitq_4k": list(config.SPLIT_VALID_WIKITQ_4K_INDICES),
            "wikitq_plus": list(config.SPLIT_VALID_WIKITQ_PLUS_INDICES),
            "scalability": list(config.SPLIT_VALID_SCALABILITY_INDICES),
        }
        test_indices: Dict[str, List[int]] = {
            "wikitq_4k": list(config.SPLIT_TEST_WIKITQ_4K_INDICES),
            "wikitq_plus": list(config.SPLIT_TEST_WIKITQ_PLUS_INDICES),
            "scalability": list(config.SPLIT_TEST_SCALABILITY_INDICES),
        }

        return cls(
            base=config,
            split_mode=config.SPLIT_MODE,
            split_train_indices=train_indices,
            split_valid_indices=valid_indices,
            split_test_indices=test_indices,
        )

    def __getattr__(self, name: str):
        """Proxy attribute access to the base Config for convenience."""
        if name in ("base", "split_mode", "split_train_indices",
                     "split_valid_indices", "split_test_indices"):
            raise AttributeError(name)
        return getattr(self.base, name)


# ---------------------------------------------------------------------------
# GRPOConfig builder
# ---------------------------------------------------------------------------

def _resolve_seed(stage_seed: int, global_seed: int) -> int:
    """Return *stage_seed* if explicitly set, otherwise inherit *global_seed*.

    The sentinel value ``-1`` means "inherit from GLOBAL_SEED".  This
    resolution is already performed during ``load_config`` in ``config.py``,
    but we keep a local helper for any late-constructed configs.
    """
    return global_seed if stage_seed == -1 else stage_seed


def build_grpo_config(config: "Config"):
    """Translate Kconfig ``CONFIG_GRPO_*`` symbols into a TRL ``GRPOConfig``.

    Deferred import of ``trl`` so the module can be imported without the
    heavy ML stack installed (e.g. during tests or config-only validation).
    """
    from trl import GRPOConfig  # deferred

    max_steps = config.GRPO_MAX_STEPS
    if max_steps == 0:
        max_steps = -1

    training_seed = _resolve_seed(config.TRAINING_SEED, config.GLOBAL_SEED)
    dataloader_seed = _resolve_seed(config.DATALOADER_SEED, config.GLOBAL_SEED)

    grpo_cfg = GRPOConfig(
        output_dir=config.TRAIN_OUTPUT_DIR,
        num_train_epochs=config.GRPO_NUM_EPOCHS,
        max_steps=max_steps,
        per_device_train_batch_size=config.GRPO_BATCH_SIZE_PROMPTS,
        num_generations=config.GRPO_GROUP_SIZE,
        max_completion_length=config.GRPO_MAX_NEW_TOKENS,
        temperature=config.GRPO_TEMPERATURE,
        top_p=config.GRPO_TOP_P,
        top_k=config.GRPO_TOP_K if config.GRPO_TOP_K > 0 else None,
        max_grad_norm=1.0,
        cliprange=config.GRPO_CLIP_EPS,
        beta=config.GRPO_KL_COEF,
        learning_rate=config.GRPO_LR,
        gradient_accumulation_steps=config.GRPO_GRAD_ACCUM,
        save_steps=config.GRPO_SAVE_EVERY_STEPS,
        eval_steps=config.GRPO_EVAL_EVERY_STEPS,
        eval_strategy="steps",
        seed=training_seed,
        data_seed=dataloader_seed,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=config.TRAIN_USE_GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
        report_to="none",
    )

    logger.info(
        "GRPOConfig built: max_steps=%s, lr=%s, group_size=%s, "
        "batch_size=%s, seed=%s",
        grpo_cfg.max_steps,
        grpo_cfg.learning_rate,
        grpo_cfg.num_generations,
        grpo_cfg.per_device_train_batch_size,
        grpo_cfg.seed,
    )
    return grpo_cfg
