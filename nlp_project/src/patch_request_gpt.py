"""Monkey-patch bridge for LLM chat and embedding functions.

Replaces ``utils.request_gpt.request_gpt_chat``,
``utils.request_gpt.request_gpt_chat_1``, and
``utils.request_gpt.request_gpt_embedding`` with project-local
implementations backed by ``SglangClient`` and ``EmbeddingClient``.

When a ``CallRecorder`` is provided, a recording wrapper is installed that
captures metadata for every LLM call.

**All upstream imports are deferred to inside ``init_patches``** so this
module can be imported before ``bootstrap_upstream_imports`` runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from src.item_context import (
    ctx_attempt,
    ctx_last_call_id,
    ctx_stage,
)

if TYPE_CHECKING:
    from src.call_recorder import CallRecorder
    from src.config import Config
    from src.embedding_client import EmbeddingClient
    from src.sglang_client import SglangClient

logger = logging.getLogger(__name__)


def init_patches(
    sglang_client: SglangClient,
    embedding_client: EmbeddingClient,
    config: Config,
    call_recorder: Optional[CallRecorder] = None,
) -> None:
    """Monkey-patch ``utils.request_gpt`` functions.

    Must be called **after** ``bootstrap_upstream_imports()``.
    """
    import utils.request_gpt as rg  # noqa: E402  — deferred import

    # Always patch embedding
    rg.request_gpt_embedding = embedding_client.embed_one

    if call_recorder is None:
        rg.request_gpt_chat = sglang_client.chat
        rg.request_gpt_chat_1 = sglang_client.chat
    else:
        def recording_chat(prompt: str) -> str:
            result = sglang_client.chat_with_metadata(prompt)

            captured_prompt = prompt if config.LOG_LLM_PROMPTS else None
            captured_response = result.text if config.LOG_LLM_RESPONSES else None

            call_id = call_recorder.record(
                stage=ctx_stage.get(),
                prompt=captured_prompt,
                response_text=captured_response,
                finish_reason=result.finish_reason,
                usage=result.usage,
                model=result.model,
                attempt=ctx_attempt.get(),
                error=result.error,
                temperature=config.LLM_TEMPERATURE,
                max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                top_p=config.LLM_TOP_P,
                top_k=config.LLM_TOP_K if config.LLM_TOP_K > 0 else None,
                frequency_penalty=config.LLM_FREQUENCY_PENALTY,
                presence_penalty=config.LLM_PRESENCE_PENALTY,
                seed=config.LLM_SEED if config.LLM_SEED >= 0 else None,
            )

            ctx_last_call_id.set(call_id)
            return result.text

        rg.request_gpt_chat = recording_chat
        rg.request_gpt_chat_1 = recording_chat

    logger.info(
        "Patched utils.request_gpt (recording=%s)",
        call_recorder is not None,
    )
