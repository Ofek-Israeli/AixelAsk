"""Offline model download and path resolution via ``huggingface_hub``.

``download(config)`` fetches both the inference and embedding models.
``resolve_model_path(config)`` returns the local snapshot directory for the
configured inference model (no-op if already cached).

Both functions rely on the ``HF_HOME`` environment variable (set by
``config.py``) — ``cache_dir`` is never passed explicitly so that every
consumer resolves the same cache path.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


def download(config: Config) -> None:
    """Download both inference and embedding models to the HF cache.

    Uses ``$HF_HOME/hub/`` automatically — does **not** pass ``cache_dir``.
    """
    logger.info(
        "Downloading inference model: %s (revision=%s)",
        config.INFERENCE_MODEL,
        config.INFERENCE_MODEL_REVISION,
    )
    snapshot_download(
        repo_id=config.INFERENCE_MODEL,
        revision=config.INFERENCE_MODEL_REVISION,
    )

    logger.info("Downloading embedding model: %s", config.EMBEDDING_MODEL)
    snapshot_download(
        repo_id=config.EMBEDDING_MODEL,
    )

    logger.info("All models downloaded successfully.")


def resolve_model_path(config: Config) -> str:
    """Return the local snapshot directory for the configured inference model.

    Calls ``snapshot_download`` which is a no-op if already cached, then
    returns the absolute path to the local snapshot directory.  This path
    is passed to the inference server ``--model-path``, ``lora_factory``, and any other
    model consumer.
    """
    return snapshot_download(
        repo_id=config.INFERENCE_MODEL,
        revision=config.INFERENCE_MODEL_REVISION,
    )


# ---------------------------------------------------------------------------
# CLI entry point:  python -m src.download_models --config .config
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.config import build_arg_parser, load_config

    parser = build_arg_parser(description="Download HF models")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.override)

    from src.logging_setup import setup_logging

    setup_logging(cfg)
    download(cfg)
