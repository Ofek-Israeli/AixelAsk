"""Local embedding client using ``sentence-transformers``.

Loads ``nomic-ai/nomic-embed-text-v1`` (or the configured model) at init
and exposes ``embed_one`` / ``embed_batch`` with automatic prefix handling
controlled by ``CONFIG_NOMIC_PREFIX_MODE``.

Concurrency is bounded by a ``threading.Semaphore`` (``CONFIG_GLOBAL_EMBEDDING_CONCURRENCY``)
acquired before every encode call.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List, Optional

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

_PREFIX_QUERY = "search_query: "
_PREFIX_DOCUMENT = "search_document: "

_HEURISTIC_LOG_LIMIT = 20


class EmbeddingClient:
    """Thread-safe embedding client backed by SentenceTransformer."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._prefix_mode = config.NOMIC_PREFIX_MODE.upper()
        self._semaphore = threading.Semaphore(config.GLOBAL_EMBEDDING_CONCURRENCY)
        self._heuristic_log_count = 0
        self._heuristic_lock = threading.Lock()

        logger.info(
            "Loading embedding model: %s (device=%s)",
            config.EMBEDDING_MODEL,
            config.EMBEDDING_DEVICE,
        )
        self._model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            device=config.EMBEDDING_DEVICE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_one(self, text: str) -> List[float]:
        """Embed a single text string and return a float vector.

        Applies the nomic prefix according to ``CONFIG_NOMIC_PREFIX_MODE``
        before encoding.
        """
        prefixed = self._apply_prefix(text)
        self._semaphore.acquire()
        try:
            vec = self._model.encode(prefixed, convert_to_numpy=True)
        finally:
            self._semaphore.release()
        return vec.tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Embed a batch of texts and return a list of float vectors.

        Uses the configured ``EMBEDDING_BATCH_SIZE`` if *batch_size* is
        not specified.
        """
        if batch_size is None:
            batch_size = self._config.EMBEDDING_BATCH_SIZE
        prefixed = [self._apply_prefix(t) for t in texts]
        self._semaphore.acquire()
        try:
            vecs = self._model.encode(
                prefixed,
                batch_size=batch_size,
                convert_to_numpy=True,
            )
        finally:
            self._semaphore.release()
        return [v.tolist() for v in vecs]

    # ------------------------------------------------------------------
    # Prefix logic
    # ------------------------------------------------------------------

    def _apply_prefix(self, text: str) -> str:
        """Prepend the appropriate nomic prefix based on configuration."""
        mode = self._prefix_mode

        if mode == "NONE":
            return text
        if mode == "ALWAYS_QUERY":
            return _PREFIX_QUERY + text
        if mode == "ALWAYS_DOCUMENT":
            return _PREFIX_DOCUMENT + text

        # AUTO mode: heuristic classification
        is_query = self._is_query_heuristic(text)
        prefix = _PREFIX_QUERY if is_query else _PREFIX_DOCUMENT

        with self._heuristic_lock:
            if self._heuristic_log_count < _HEURISTIC_LOG_LIMIT:
                self._heuristic_log_count += 1
                label = "query" if is_query else "document"
                logger.debug(
                    "AUTO prefix [%s]: '%s'",
                    label,
                    text[:80] + ("..." if len(text) > 80 else ""),
                )

        return prefix + text

    @staticmethod
    def _is_query_heuristic(text: str) -> bool:
        """Classify text as query (True) or document (False).

        Query if: contains '?', starts with 'question' (case-insensitive),
        or is shorter than 40 characters.
        """
        if "?" in text:
            return True
        if text.lstrip().lower().startswith("question"):
            return True
        if len(text) < 40:
            return True
        return False
