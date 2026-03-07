"""Logging configuration: Rich console handler + standard file handler.

Provides ``setup_logging(config)`` that configures the root logger with:
  1. A Rich console handler at ``CONFIG_LOG_LEVEL`` with markup/highlighting.
  2. A file handler writing to ``CONFIG_LOG_FILE`` at DEBUG level.

Called once by each entrypoint after config parsing.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from src.config import Config


def setup_logging(config: Config) -> None:
    """Configure the root logger with Rich console and file handlers."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any pre-existing handlers to avoid duplicates on repeated calls
    root.handlers.clear()

    # --- Rich console handler at the configured level ---------------------
    console = Console(markup=True, highlight=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )
    console_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root.addHandler(console_handler)

    # --- File handler at DEBUG level --------------------------------------
    log_file = config.LOG_FILE
    if log_file:
        parent = os.path.dirname(log_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
