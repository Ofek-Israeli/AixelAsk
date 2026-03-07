"""Full training entrypoint.

Usage::

    python -m src.training.train_main --config .config [--resume]
              [--checkpoint-path /explicit/path] [--override KEY=VALUE ...]

Orchestrates the complete training lifecycle:

1.  Parse ``.config`` (config.py + train_config.py).
2.  Setup logging.
3.  Resolve seeds: set torch/random/numpy seeds, write ``resolved_seeds.json``.
4.  Write ``reward_config.json``.
5.  Build train/valid split via ``split_utils``.
5a. Resolve model path.
5b. Bootstrap upstream imports.
5c. Start SGLang server + init clients.
5d. Apply ``patch_request_gpt`` ONLY (NOT ``patch_dag``, NOT ``patch_dag_execution``).
5e. Precompute training-table embeddings, load ``table_embedding_map``.
6.  Build model + tokenizer via ``lora_factory``.
7.  Format datasets for GRPO via ``rl_dataset``.
8.  Build trainer via ``grpo_trainer`` (registers all callbacks).
9.  Initialise curves directory.
10. ``trainer.train()`` (or ``resume_from_checkpoint``).
11. Final stats flush.
12. Final curves compilation.
13. Optional adapter merge + export.
14. Stop SGLang server.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def main() -> None:
    """Full training startup sequence."""

    # ==================================================================
    # 1. Parse .config
    # ==================================================================
    from src.config import build_arg_parser, load_config

    parser = build_arg_parser(description="NLP Project — GRPO Training")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest (or specified) checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Explicit checkpoint path for resume (overrides config).",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)

    # ==================================================================
    # 2. Setup logging
    # ==================================================================
    from src.logging_setup import setup_logging

    setup_logging(config)
    logger.info("Configuration loaded from %s", args.config)

    # ==================================================================
    # 3. Resolve seeds
    # ==================================================================
    training_seed = config.TRAINING_SEED
    logger.info("Resolving seeds (global=%d, training=%d)", config.GLOBAL_SEED, training_seed)

    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    random.seed(training_seed)
    np.random.seed(training_seed)

    resolved_seeds = {
        "global_seed": config.GLOBAL_SEED,
        "training_seed": config.TRAINING_SEED,
        "dataloader_seed": config.DATALOADER_SEED,
        "generation_seed": config.GENERATION_SEED,
        "reward_seed": config.REWARD_SEED,
        "eval_seed": config.EVAL_SEED,
    }

    if config.SAVE_RESOLVED_SEEDS:
        seeds_path = os.path.join(config.TRAIN_OUTPUT_DIR, "resolved_seeds.json")
        os.makedirs(os.path.dirname(seeds_path) or ".", exist_ok=True)
        with open(seeds_path, "w") as f:
            json.dump(resolved_seeds, f, indent=2)
        logger.info("Resolved seeds written to %s", seeds_path)

    # ==================================================================
    # 4. Write reward_config.json
    # ==================================================================
    reward_config = {
        "mode": config.REWARD_MODE,
        "w_correct": config.REWARD_WEIGHT_CORRECTNESS,
        "w_valid": config.REWARD_WEIGHT_VALIDITY,
        "w_depth": config.REWARD_WEIGHT_DEPTH,
        "w_invalid": config.REWARD_WEIGHT_INVALID_PENALTY,
        "depth_normalization": config.REWARD_DEPTH_NORMALIZATION,
        "max_depth": config.REWARD_MAX_DEPTH,
        "invalid_if_parse_fails": config.REWARD_INVALID_IF_PARSE_FAILS,
        "correctness_partial_credit": config.REWARD_CORRECTNESS_PARTIAL_CREDIT,
    }
    reward_path = os.path.join(config.TRAIN_OUTPUT_DIR, "reward_config.json")
    os.makedirs(os.path.dirname(reward_path) or ".", exist_ok=True)
    with open(reward_path, "w") as f:
        json.dump(reward_config, f, indent=2)

    # ==================================================================
    # 5. Build train/valid split
    # ==================================================================
    from src.training.split_utils import build_splits

    logger.info("Building train/valid splits from YAML...")
    split_result = build_splits(config)
    logger.info(
        "Split result: train=%d, valid=%d, test=%d",
        len(split_result.train), len(split_result.valid), len(split_result.test),
    )

    # ==================================================================
    # 5a. Resolve model path
    # ==================================================================
    from src.download_models import resolve_model_path

    logger.info("Resolving model path...")
    resolved_model_path = resolve_model_path(config)
    logger.info("Resolved model path: %s", resolved_model_path)

    # ==================================================================
    # 5b. Bootstrap upstream imports
    # ==================================================================
    from src.config import bootstrap_upstream_imports

    bootstrap_upstream_imports(config)
    logger.info("Upstream imports bootstrapped.")

    # ==================================================================
    # 5c. Start SGLang server + init clients
    # ==================================================================
    from src import sglang_server
    from src.sglang_client import SglangClient
    from src.embedding_client import EmbeddingClient

    atexit.register(sglang_server.stop)

    logger.info("Starting SGLang server for reward-time DAG execution...")
    sglang_server.start(config, resolved_model_path)

    sglang_client = SglangClient(config, resolved_model_path)
    embedding_client = EmbeddingClient(config)

    # ==================================================================
    # 5d. Apply patch_request_gpt ONLY
    # ==================================================================
    from src import patch_request_gpt

    patch_request_gpt.init_patches(sglang_client, embedding_client, config)
    logger.info("Applied patch_request_gpt (training mode — no patch_dag, no patch_dag_execution).")

    # ==================================================================
    # 5e. Precompute training-table embeddings
    # ==================================================================
    logger.info("Precomputing training-table embeddings...")
    import scripts.save_embeddings as save_embeddings
    import scripts.final_reasoning_multi_thread_save_embedding as frm

    col_prompt_path = config.COL_PROMPT
    col_prompt = ""
    if col_prompt_path and os.path.isfile(col_prompt_path):
        with open(col_prompt_path, "r") as f:
            col_prompt = f.read()

    _precompute_table_embeddings(split_result.train, save_embeddings, col_prompt)

    table_embedding_map = frm.load_table_embedding_map(config.EMBEDDING_CACHE)
    logger.info("Loaded %d table embeddings from cache.", len(table_embedding_map))

    # ==================================================================
    # 6. Build model + tokenizer
    # ==================================================================
    from src.training.lora_factory import build_model_and_tokenizer

    logger.info("Building model + tokenizer via lora_factory...")
    model, tokenizer = build_model_and_tokenizer(config, resolved_model_path)

    # ==================================================================
    # 7. Format datasets for GRPO
    # ==================================================================
    from src.training.rl_dataset import format_for_grpo

    logger.info("Formatting datasets for GRPO...")
    train_dataset, eval_dataset = format_for_grpo(split_result, config)

    # ==================================================================
    # 8. Build trainer (includes all callbacks)
    # ==================================================================
    from src.training.train_stats import RewardMetricsAccumulator
    from src.training.curves import CurvesManager
    from src.training.grpo_trainer import build_trainer

    accumulator = RewardMetricsAccumulator()

    curves_manager: Optional[CurvesManager] = None
    if config.TRAIN_CURVES_TEX_ENABLE:
        curves_manager = CurvesManager(config)
        curves_manager.init_curves_dir()

    trainer = build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        accumulator=accumulator,
        curves_manager=curves_manager,
        table_embedding_map=table_embedding_map,
    )

    # ==================================================================
    # 9. Resolve resume checkpoint (if --resume)
    # ==================================================================
    resume_path: Optional[str] = None
    if args.resume:
        resume_path = _resolve_resume_path(args, config)
        if resume_path:
            from src.training.checkpointing import check_resume_compatibility
            check_resume_compatibility(resume_path, config)
            logger.info("Resuming from checkpoint: %s", resume_path)

    # ==================================================================
    # 10. Train!
    # ==================================================================
    logger.info("Starting training...")
    train_start = time.monotonic()

    if resume_path:
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    train_elapsed = time.monotonic() - train_start
    logger.info("Training completed in %.1f seconds.", train_elapsed)

    # ==================================================================
    # 11. Final stats flush
    # ==================================================================
    _write_final_summary(config, trainer, train_elapsed, split_result, resolved_seeds)

    # ==================================================================
    # 12. Final curves compilation
    # ==================================================================
    if curves_manager and config.TRAIN_CURVES_COMPILE_AT_END:
        logger.info("Final curves compilation...")
        try:
            from src.training.tex_compile import compile_all
            compile_all(config)
        except Exception:
            logger.warning("Final curves compilation failed (non-fatal)", exc_info=True)

    # ==================================================================
    # 13. Stop SGLang server
    # ==================================================================
    sglang_server.stop()
    logger.info("Training pipeline complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_resume_path(args, config) -> Optional[str]:
    """Determine the checkpoint path for resume."""
    if args.checkpoint_path:
        path = args.checkpoint_path
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path

    if config.TRAIN_LOAD_FROM_CHECKPOINT and config.TRAIN_CHECKPOINT_PATH:
        path = config.TRAIN_CHECKPOINT_PATH
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path

    ckpt_dir = os.path.join(config.TRAIN_OUTPUT_DIR, "checkpoints")
    latest_link = os.path.join(ckpt_dir, "latest")
    if os.path.islink(latest_link):
        target = os.readlink(latest_link)
        path = os.path.join(ckpt_dir, target)
        if os.path.isdir(path):
            return path

    index_path = os.path.join(ckpt_dir, "checkpoint_index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        checkpoints = index.get("checkpoints", [])
        for entry in reversed(checkpoints):
            p = os.path.join(ckpt_dir, entry["path"])
            if os.path.isdir(p):
                return p

    logger.warning("--resume specified but no checkpoint found; starting fresh.")
    return None


def _precompute_table_embeddings(train_ds, save_embeddings_mod, col_prompt: str) -> None:
    """Precompute embeddings for all tables in the training dataset."""
    try:
        examples = []
        for i in range(len(train_ds)):
            ex = train_ds[i]
            examples.append(ex)
        save_embeddings_mod.process_table_embeddings(examples, col_prompt)
    except Exception:
        logger.warning("Table embedding precomputation failed", exc_info=True)


def _write_final_summary(config, trainer, elapsed: float, split_result, seeds: dict) -> None:
    """Write the end-of-run summary JSON."""
    state = trainer.state

    final_train_metrics: dict = {}
    final_eval_metrics: dict = {}
    if state.log_history:
        for entry in reversed(state.log_history):
            if "loss" in entry and not final_train_metrics:
                final_train_metrics = dict(entry)
            if any(k.startswith("eval_") for k in entry) and not final_eval_metrics:
                final_eval_metrics = dict(entry)
            if final_train_metrics and final_eval_metrics:
                break

    best_info: dict = {"metric": None, "value": None, "step": None, "path": None}
    best_dir = os.path.join(config.TRAIN_OUTPUT_DIR, "checkpoints", "best")
    if os.path.isdir(best_dir):
        meta_path = os.path.join(best_dir, "checkpoint_metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            best_info = {
                "metric": meta.get("metric_name"),
                "value": meta.get("metric_value"),
                "step": meta.get("step"),
                "path": best_dir,
            }

    summary = {
        "run_metadata": {
            "training_mode": config.TRAINING_MODE,
            "global_seed": config.GLOBAL_SEED,
            "resolved_seeds": seeds,
            "reward_config": {
                "mode": config.REWARD_MODE,
                "w_correct": config.REWARD_WEIGHT_CORRECTNESS,
                "w_valid": config.REWARD_WEIGHT_VALIDITY,
                "w_depth": config.REWARD_WEIGHT_DEPTH,
                "w_invalid": config.REWARD_WEIGHT_INVALID_PENALTY,
                "depth_norm": config.REWARD_DEPTH_NORMALIZATION,
                "max_depth": config.REWARD_MAX_DEPTH,
            },
            "lora_config": {
                "r": config.TRAIN_LORA_R,
                "alpha": config.TRAIN_LORA_ALPHA,
                "dropout": config.TRAIN_LORA_DROPOUT,
                "target_modules": config.TRAIN_LORA_TARGET_MODULES.split(","),
            },
            "base_model": config.INFERENCE_MODEL,
            "train_examples": len(split_result.train),
            "dev_examples": len(split_result.valid),
            "total_steps": state.global_step,
            "total_epochs": state.epoch or 0,
            "wallclock_sec": round(elapsed, 2),
        },
        "final_metrics": {
            **{k: v for k, v in final_train_metrics.items() if isinstance(v, (int, float))},
            **{k: v for k, v in final_eval_metrics.items() if isinstance(v, (int, float))},
        },
        "best_checkpoint": best_info,
        "latest_checkpoint": {
            "step": state.global_step,
            "path": os.path.join(
                config.TRAIN_OUTPUT_DIR, "checkpoints", "latest"
            ),
        },
    }

    summary_path = config.TRAIN_STATS_FILE
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary written to %s", summary_path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error in training entrypoint.")
        sys.exit(1)
