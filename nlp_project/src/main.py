"""Inference / evaluation entrypoint.

Usage::

    python -m src.main --config .config [--download-only] [--override KEY=VALUE ...]

Orchestrates the full startup → evaluate → shutdown lifecycle:

1.  Parse ``.config`` via ``config.load_config``.
2.  Setup logging.
3.  If ``--download-only``: download models and exit.
3a. Resolve model path.
3b. Bootstrap upstream imports (``sys.path``).
4.  Register ``atexit(sglang_server.stop)``.
5.  Start SGLang server.
6.  Init ``sglang_client`` + ``embedding_client``.
7.  Create semaphores (passed to clients at init).
7a. Create runtime objects (``call_recorder``, stores, ``dag_stats``).
8.  Apply monkey-patches (order matters).
9.  ``pipeline.run(...)`` — test split only.
10. Stop SGLang server.
"""

from __future__ import annotations

import atexit
import logging
import sys
from threading import Semaphore
from typing import Optional

logger = logging.getLogger(__name__)


def main() -> None:
    """Full startup sequence executed as CLI entrypoint."""

    # ==================================================================
    # 1. Parse .config
    # ==================================================================
    from src.config import build_arg_parser, load_config

    parser = build_arg_parser(description="NLP Project — Inference / Evaluation")
    parser.add_argument(
        "--download-only",
        action="store_true",
        default=False,
        help="Download models and exit (no server started).",
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
    # 3. If --download-only: download and exit
    # ==================================================================
    from src.download_models import download, resolve_model_path

    if args.download_only:
        logger.info("--download-only: downloading models and exiting.")
        download(config)
        return

    # ==================================================================
    # 3a. Resolve model path
    # ==================================================================
    logger.info("Resolving model path …")
    resolved_model_path = resolve_model_path(config)
    logger.info("Resolved model path: %s", resolved_model_path)

    # ==================================================================
    # 3b. Bootstrap upstream imports
    # ==================================================================
    from src.config import bootstrap_upstream_imports

    bootstrap_upstream_imports(config)
    logger.info("Upstream imports bootstrapped (sys.path updated).")

    # ==================================================================
    # 4. Register atexit(sglang_server.stop)
    # ==================================================================
    from src import sglang_server

    atexit.register(sglang_server.stop)

    # ==================================================================
    # 5. Start SGLang server
    # ==================================================================
    logger.info("Starting SGLang server …")
    sglang_server.start(config, resolved_model_path)

    # ==================================================================
    # 6. Init sglang_client + embedding_client
    # ==================================================================
    from src.sglang_client import SglangClient
    from src.embedding_client import EmbeddingClient

    logger.info("Initialising SGLang client …")
    sglang_client = SglangClient(config, resolved_model_path)

    logger.info("Initialising embedding client …")
    embedding_client = EmbeddingClient(config)

    # ==================================================================
    # 7. Semaphores are created inside client constructors already
    #    (SglangClient uses CONFIG_SGLANG_CLIENT_CONCURRENCY,
    #     EmbeddingClient uses CONFIG_GLOBAL_EMBEDDING_CONCURRENCY)
    # ==================================================================

    # ==================================================================
    # 7a. Create runtime coordination objects
    # ==================================================================
    from src.call_recorder import CallRecorder
    from src.item_context import DagMetadataStore, ExecTelemetryStore
    from src.dag_stats import DagStats

    call_capture_requested = (
        config.LOG_LLM_CALLS_PER_ITEM
        or config.LOG_LLM_PROMPTS
        or config.LOG_LLM_RESPONSES
        or bool(config.LLM_CALLS_SIDEFILE)
    )
    call_recorder: Optional[CallRecorder] = (
        CallRecorder(config) if call_capture_requested else None
    )

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
    # 8. Apply monkey-patches (order matters)
    # ==================================================================
    from src import patch_request_gpt, patch_dag, patch_dag_execution
    from src.dag_executor import DagExecutor

    # 8a. Route LLM + embedding calls to local models
    patch_request_gpt.init_patches(
        sglang_client, embedding_client, config, call_recorder=call_recorder,
    )

    # 8b. Override DAG generation prompt + capture metadata
    patch_dag.init_patches(
        config, call_recorder=call_recorder, dag_metadata_store=dag_metadata_store,
    )

    # 8c. Construct DAG executor
    dag_executor = DagExecutor(config, embedding_client)

    # 8d. Parallel DAG execution + stage tagging
    patch_dag_execution.init_patches(dag_executor, exec_telemetry_store)

    # ==================================================================
    # 9. Run pipeline — TEST SPLIT ONLY
    # ==================================================================
    from src import pipeline

    logger.info("Running evaluation pipeline …")
    result = pipeline.run(
        config,
        dag_metadata_store=dag_metadata_store,
        exec_telemetry_store=exec_telemetry_store,
        call_recorder=call_recorder,
        dag_stats=dag_stats,
    )

    logger.info(
        "Evaluation complete: %d/%d correct (%.4f), results → %s",
        result.correct_count,
        result.total_items,
        result.accuracy,
        result.result_file,
    )
    if result.dag_stats_file:
        logger.info("DAG stats → %s", result.dag_stats_file)

    # ==================================================================
    # 10. Stop SGLang server
    # ==================================================================
    sglang_server.stop()


# ---------------------------------------------------------------------------
# Entrypoint: python -m src.main  or  python src/main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error in main entrypoint.")
        sys.exit(1)
