"""Post-training test evaluation entrypoint.

Usage::

    python -m src.training.test_main --config .config
              [--checkpoint best|latest|merged|explicit]
              [--override KEY=VALUE ...]

Evaluates a trained checkpoint on the **test split only** using the full
inference pipeline (parallel DAG executor, retries, metadata enrichment).

Flow:

1.  Parse ``.config``.
2.  Resolve checkpoint via ``checkpoint_resolver``.
2a. Resolve base model path (for adapter merging).
2b. Bootstrap upstream imports.
3.  If adapter-only: merge to temp dir, register ``atexit`` cleanup.
4.  Set runtime overrides (``serving_model_path``, output paths).
5.  Setup logging.
6.  Register ``atexit(sglang_server.stop)``.
7.  Start SGLang with ``serving_model_path``.
8.  Init clients.
9.  Create runtime coordination objects.
10. Apply ALL monkey-patches.
11. ``pipeline.run()`` — TEST SPLIT ONLY.
12. Write ``test_eval_summary.json``.
13. Stop SGLang server.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def main() -> None:
    """Full post-training test evaluation sequence."""

    # ==================================================================
    # 1. Parse .config
    # ==================================================================
    from src.config import build_arg_parser, load_config

    parser = build_arg_parser(description="NLP Project — Post-Training Test Evaluation")
    parser.add_argument(
        "--checkpoint",
        default=None,
        choices=["best", "latest", "merged", "explicit"],
        help="Checkpoint source to evaluate (overrides config).",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)

    # ==================================================================
    # 2. Resolve checkpoint
    # ==================================================================
    from src.training.checkpoint_resolver import resolve_test_checkpoint

    resolved_checkpoint = resolve_test_checkpoint(
        config,
        override_source=args.checkpoint,
    )

    # ==================================================================
    # 2a. Resolve base model path
    # ==================================================================
    from src.download_models import resolve_model_path

    resolved_base_model_path = resolve_model_path(config)

    # ==================================================================
    # 2b. Bootstrap upstream imports
    # ==================================================================
    from src.config import bootstrap_upstream_imports

    bootstrap_upstream_imports(config)

    # ==================================================================
    # 3. If adapter-only: merge to temp dir
    # ==================================================================
    serving_model_path: str
    merge_dir: Optional[str] = None

    if resolved_checkpoint.is_adapter_only:
        from src.training.lora_factory import merge_and_export

        merge_dir = tempfile.mkdtemp(
            dir=config.EPHEMERAL_TMPDIR,
            prefix="merged_ckpt_",
        )
        atexit.register(shutil.rmtree, merge_dir, True)

        logger.info(
            "Adapter-only checkpoint — merging to temp dir: %s",
            merge_dir,
        )
        merge_and_export(
            resolved_checkpoint.path,
            merge_dir,
            config,
            resolved_base_model_path,
        )
        serving_model_path = merge_dir
    else:
        serving_model_path = resolved_checkpoint.path

    # ==================================================================
    # 4. Set runtime overrides
    # ==================================================================
    config.RESULT_FILE = config.TEST_TRAINED_RESULT_FILE
    config.DAG_STATS_FILE = config.TEST_TRAINED_DAG_STATS_FILE

    # ==================================================================
    # 5. Setup logging
    # ==================================================================
    from src.logging_setup import setup_logging

    setup_logging(config)

    logger.info(
        "Post-training test: checkpoint=%s (%s), serving=%s",
        resolved_checkpoint.source,
        resolved_checkpoint.path,
        serving_model_path,
    )

    # ==================================================================
    # 6. Register atexit(sglang_server.stop)
    # ==================================================================
    from src import sglang_server

    atexit.register(sglang_server.stop)

    # ==================================================================
    # 7. Start SGLang with serving_model_path
    # ==================================================================
    logger.info("Starting SGLang server with trained model...")
    sglang_server.start(config, serving_model_path)

    # ==================================================================
    # 8. Init clients
    # ==================================================================
    from src.sglang_client import SglangClient
    from src.embedding_client import EmbeddingClient

    sglang_client = SglangClient(config, serving_model_path)
    embedding_client = EmbeddingClient(config)

    # ==================================================================
    # 9. Create runtime coordination objects
    # ==================================================================
    from src.call_recorder import CallRecorder
    from src.item_context import DagMetadataStore, ExecTelemetryStore
    from src.dag_stats import DagStats

    call_recorder: Optional[CallRecorder] = CallRecorder(config)

    dag_metadata_store = DagMetadataStore()
    exec_telemetry_store = ExecTelemetryStore()

    dag_stats: Optional[DagStats] = None
    if config.DAG_STATS_ENABLE:
        dag_stats = DagStats(
            include_failed=config.DAG_STATS_INCLUDE_FAILED,
            write_per_item=config.DAG_STATS_WRITE_PER_ITEM,
            log_executor_stats=config.LOG_EXECUTOR_STATS,
            log_validity_errors=config.DAG_STATS_VALIDITY_ERRORS,
        )

    # ==================================================================
    # 10. Apply ALL monkey-patches
    # ==================================================================
    from src import patch_request_gpt, patch_dag, patch_dag_execution
    from src.dag_executor import DagExecutor

    patch_request_gpt.init_patches(
        sglang_client, embedding_client, config, call_recorder=call_recorder,
    )
    patch_dag.init_patches(
        config, call_recorder=call_recorder, dag_metadata_store=dag_metadata_store,
    )
    dag_executor = DagExecutor(config, embedding_client)
    patch_dag_execution.init_patches(dag_executor, exec_telemetry_store)

    logger.info("All patches applied (full inference mode).")

    # ==================================================================
    # 11. Run pipeline — TEST SPLIT ONLY
    # ==================================================================
    from src import pipeline

    checkpoint_provenance: Dict[str, Any] = {
        "checkpoint_source": resolved_checkpoint.source,
        "checkpoint_path": resolved_checkpoint.path,
        "checkpoint_step": resolved_checkpoint.step,
        "checkpoint_metric_name": resolved_checkpoint.metric_name,
        "checkpoint_metric_value": resolved_checkpoint.metric_value,
        "is_adapter_only": resolved_checkpoint.is_adapter_only,
    }

    logger.info("Running evaluation pipeline on test split...")
    result = pipeline.run(
        config,
        dag_metadata_store=dag_metadata_store,
        exec_telemetry_store=exec_telemetry_store,
        call_recorder=call_recorder,
        dag_stats=dag_stats,
        checkpoint_provenance=checkpoint_provenance,
    )

    logger.info(
        "Evaluation complete: %d/%d correct (%.4f), results → %s",
        result.correct_count,
        result.total_items,
        result.accuracy,
        result.result_file,
    )

    # ==================================================================
    # 12. Write test_eval_summary.json
    # ==================================================================
    summary: Dict[str, Any] = {
        "checkpoint_source": resolved_checkpoint.source,
        "checkpoint_path": resolved_checkpoint.path,
        "checkpoint_step": resolved_checkpoint.step,
        "checkpoint_metric_name": resolved_checkpoint.metric_name,
        "checkpoint_metric_value": resolved_checkpoint.metric_value,
        "is_adapter_only": resolved_checkpoint.is_adapter_only,
        "accuracy": result.accuracy,
        "total_items": result.total_items,
        "total_correct": result.correct_count,
        "total_errors": result.error_count,
        "result_file": result.result_file,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if dag_stats is not None and result.dag_stats_file:
        summary["dag_stats_file"] = result.dag_stats_file

    output_dir = config.TEST_TRAINED_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "test_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Test evaluation summary written to %s", summary_path)

    # ==================================================================
    # 13. Stop SGLang server
    # ==================================================================
    sglang_server.stop()
    logger.info("Post-training test pipeline complete.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error in test evaluation entrypoint.")
        sys.exit(1)
