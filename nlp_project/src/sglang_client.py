"""SGLang client using the OpenAI-compatible ``/v1/chat/completions`` endpoint.

Provides ``SglangClient`` with two calling conventions:
- ``chat(prompt) -> str`` — drop-in replacement for ``request_gpt_chat``.
- ``chat_with_metadata(prompt) -> ChatResult`` — returns rich metadata.

Concurrency is bounded by a ``threading.Semaphore`` acquired inside both
methods, ensuring the total in-flight request count never exceeds
``CONFIG_SGLANG_CLIENT_CONCURRENCY``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from openai import OpenAI

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Lightweight result returned by ``chat_with_metadata``."""

    text: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model: str = ""
    error: Optional[str] = None


class SglangClient:
    """Thread-safe client for a locally-running SGLang server."""

    def __init__(self, config: Config, resolved_model_path: str) -> None:
        self._config = config
        self._model = resolved_model_path
        self._retries = config.LLM_RETRIES
        self._semaphore = threading.Semaphore(config.SGLANG_CLIENT_CONCURRENCY)

        base_url = f"http://{config.SGLANG_HOST}:{config.SGLANG_PORT}/v1"
        self._client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=60.0,
            max_retries=0,
        )

        # Pre-build the static portion of sampling params
        self._standard_params: Dict[str, Any] = {
            "model": self._model,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_OUTPUT_TOKENS,
            "top_p": config.LLM_TOP_P,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
            "presence_penalty": config.LLM_PRESENCE_PENALTY,
        }
        if config.LLM_SEED >= 0:
            self._standard_params["seed"] = config.LLM_SEED

        self._extra_body: Optional[Dict[str, Any]] = None
        if config.LLM_TOP_K > 0:
            self._extra_body = {"top_k": config.LLM_TOP_K}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, prompt: str) -> str:
        """Send *prompt* and return the response text.

        Drop-in replacement for ``request_gpt_chat``.  Retries with
        exponential backoff on transient failures.
        """
        result = self.chat_with_metadata(prompt)
        if result.error:
            raise RuntimeError(f"SGLang chat failed after retries: {result.error}")
        return result.text

    def chat_with_metadata(self, prompt: str) -> ChatResult:
        """Send *prompt* and return a ``ChatResult`` with full metadata.

        Acquires the concurrency semaphore, retries with exponential
        backoff, and returns error info rather than raising on failure.
        """
        last_error: Optional[str] = None

        for attempt in range(1, self._retries + 1):
            self._semaphore.acquire()
            try:
                response = self._client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **self._standard_params,
                    extra_body=self._extra_body,
                )
            except Exception as exc:
                last_error = f"[attempt {attempt}/{self._retries}] {exc}"
                logger.warning("SGLang request failed: %s", last_error)
            else:
                choice = response.choices[0] if response.choices else None
                usage_dict = None
                if response.usage:
                    usage_dict = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                return ChatResult(
                    text=choice.message.content if choice else "",
                    finish_reason=choice.finish_reason if choice else None,
                    usage=usage_dict,
                    model=response.model or self._model,
                    error=None,
                )
            finally:
                self._semaphore.release()

            # Exponential backoff: 1s, 2s, 4s, … capped at 60s
            backoff = min(2 ** (attempt - 1), 60)
            logger.debug("Retrying in %ds (attempt %d/%d)", backoff, attempt, self._retries)
            time.sleep(backoff)

        return ChatResult(error=last_error)
