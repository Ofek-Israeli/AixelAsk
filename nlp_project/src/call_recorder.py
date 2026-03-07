"""Thread-safe LLM call accumulator.

Accumulates ``llm_calls`` entries during a pipeline run. Constructed in
``main.py`` / ``test_main.py`` and passed to ``patch_request_gpt.init_patches``.

All public methods are ``threading.Lock``-protected so they are safe to call
from concurrent ``ThreadPoolExecutor`` tasks.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TextIO

from src.item_context import ctx_item_index, ctx_item_question, ctx_node_id

logger = logging.getLogger(__name__)


class CallRecorder:
    """Thread-safe LLM call accumulator with optional sidefile output."""

    def __init__(self, config: Any) -> None:
        self._max_chars: int = getattr(config, "LOG_LLM_CALLS_MAX_CHARS", 0)
        sidefile_path: str = getattr(config, "LLM_CALLS_SIDEFILE", "")
        self._lock = threading.Lock()
        self._calls: Dict[str, dict] = {}
        self._sidefile: Optional[TextIO] = None
        if sidefile_path:
            self._sidefile = open(sidefile_path, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        stage: str,
        prompt: Optional[str],
        response_text: Optional[str],
        finish_reason: Optional[str],
        usage: Optional[Dict[str, int]],
        model: str,
        attempt: int,
        error: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Record one LLM call and return a unique *call_id* (UUID)."""
        call_id = uuid.uuid4().hex

        item_index = _safe_ctx_get(ctx_item_index, None)
        item_question = _safe_ctx_get(ctx_item_question, None)
        node_id = _safe_ctx_get(ctx_node_id, None)

        record: dict = {
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "item_index": item_index,
            "item_question": item_question,
            "node_id": node_id,
            "stage": stage,
            "attempt": attempt,
            "prompt": self._truncate(prompt),
            "response_text": self._truncate(response_text),
            "finish_reason": finish_reason,
            "usage": usage,
            "model": model,
            "error": error,
            "error_category": None,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "max_output_tokens": max_output_tokens,
        }

        with self._lock:
            self._calls[call_id] = record

        return call_id

    def update(
        self,
        call_id: str,
        error: Optional[str] = None,
        error_category: Optional[str] = None,
    ) -> None:
        """Annotate an existing call record with post-hoc results.

        Graceful no-op if *call_id* is unknown (logs a warning).
        """
        with self._lock:
            rec = self._calls.get(call_id)
            if rec is None:
                logger.warning(
                    "update() called with unknown call_id=%s", call_id
                )
                return
            if error is not None:
                rec["error"] = error
            if error_category is not None:
                rec["error_category"] = error_category

    def get_calls_for_item(self, item_index: int) -> List[dict]:
        """Return all call records tagged with *item_index*, sorted by timestamp."""
        with self._lock:
            return sorted(
                (r for r in self._calls.values() if r["item_index"] == item_index),
                key=lambda r: r["timestamp"],
            )

    def flush_for_item(self, item_index: int) -> None:
        """Write records for *item_index* to sidefile (if configured) then remove them."""
        with self._lock:
            matching_ids = [
                cid
                for cid, rec in self._calls.items()
                if rec["item_index"] == item_index
            ]
            records = sorted(
                (self._calls[cid] for cid in matching_ids),
                key=lambda r: r["timestamp"],
            )
            if self._sidefile and records:
                for rec in records:
                    self._sidefile.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._sidefile.flush()
            for cid in matching_ids:
                del self._calls[cid]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _truncate(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        if self._max_chars > 0 and len(text) > self._max_chars:
            return text[: self._max_chars] + "...<truncated>"
        return text


def _safe_ctx_get(var, default=None):
    """Read a ContextVar, returning *default* if not set."""
    try:
        return var.get()
    except LookupError:
        return default
