"""Dispatch layer for the configurable inference backend.

Delegates ``start`` / ``stop`` to either ``sglang_server`` or
``vllm_server`` based on ``config.INFERENCE_BACKEND``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


def start(config: Config, resolved_model_path: str) -> None:
    """Start the inference server selected by config."""
    backend = config.INFERENCE_BACKEND
    logger.info("Starting inference server (backend=%s)...", backend)

    if backend == "VLLM":
        from src import vllm_server
        vllm_server.start(config, resolved_model_path)
    else:
        from src import sglang_server
        sglang_server.start(config, resolved_model_path)


def stop(config: Config) -> None:
    """Stop the inference server selected by config."""
    backend = config.INFERENCE_BACKEND

    if backend == "VLLM":
        from src import vllm_server
        vllm_server.stop()
    else:
        from src import sglang_server
        sglang_server.stop()


if __name__ == "__main__":
    import time

    from src.config import build_arg_parser, load_config

    parser = build_arg_parser(description="Inference server lifecycle")
    parser.add_argument(
        "--action",
        choices=["start", "stop"],
        required=True,
        help="Start or stop the inference server.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)

    if args.action == "stop":
        stop(cfg)
    else:
        from src.logging_setup import setup_logging

        setup_logging(cfg)
        from src.download_models import resolve_model_path

        model_path = resolve_model_path(cfg)
        start(cfg, model_path)
        logger.info("Server is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop(cfg)
