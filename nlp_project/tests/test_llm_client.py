"""Tests for src.llm_client — message formatting, retries, sampling params."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest import mock

import pytest

if "openai" not in sys.modules:
    _mock_openai = mock.MagicMock()
    sys.modules["openai"] = _mock_openai


@dataclass
class _StubConfig:
    SERVER_HOST: str = "127.0.0.1"
    SERVER_PORT: int = 30000
    LLM_RETRIES: int = 3
    LLM_MAX_OUTPUT_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.0
    LLM_TOP_P: float = 1.0
    LLM_TOP_K: int = 0
    LLM_FREQUENCY_PENALTY: float = 0.0
    LLM_PRESENCE_PENALTY: float = 0.0
    LLM_SEED: int = -1
    CLIENT_CONCURRENCY: int = 4


def _mock_response(text: str = "hello", model: str = "test-model"):
    choice = mock.MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"
    usage = mock.MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 3
    usage.total_tokens = 8
    resp = mock.MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = model
    return resp


class TestMessageFormat:

    def test_chat_builds_correct_messages(self):
        from src.llm_client import LlmClient

        cfg = _StubConfig()
        with mock.patch("src.llm_client.OpenAI") as MockOAI:
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.return_value = _mock_response()

            client = LlmClient(cfg, resolved_model_path="/models/Mistral")
            client.chat("What is 1+1?")

            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "What is 1+1?"


class TestRetryLogic:

    def test_retries_then_succeeds(self):
        from src.llm_client import LlmClient

        cfg = _StubConfig(LLM_RETRIES=3)

        with mock.patch("src.llm_client.OpenAI") as MockOAI, \
             mock.patch("src.llm_client.time.sleep"):
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.side_effect = [
                RuntimeError("fail-1"),
                RuntimeError("fail-2"),
                _mock_response("success"),
            ]

            client = LlmClient(cfg, resolved_model_path="/models/Mistral")
            result = client.chat("test")
            assert result == "success"

    def test_all_retries_exhausted(self):
        from src.llm_client import LlmClient

        cfg = _StubConfig(LLM_RETRIES=2)

        with mock.patch("src.llm_client.OpenAI") as MockOAI, \
             mock.patch("src.llm_client.time.sleep"):
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.side_effect = RuntimeError("always fail")

            client = LlmClient(cfg, resolved_model_path="/models/Mistral")
            with pytest.raises(RuntimeError, match="LLM chat failed"):
                client.chat("test")


class TestSamplingParams:

    def test_sampling_params_forwarded(self):
        from src.llm_client import LlmClient

        cfg = _StubConfig(
            LLM_TEMPERATURE=0.7,
            LLM_TOP_P=0.9,
            LLM_FREQUENCY_PENALTY=0.5,
            LLM_PRESENCE_PENALTY=0.1,
        )

        with mock.patch("src.llm_client.OpenAI") as MockOAI:
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.return_value = _mock_response()

            client = LlmClient(cfg, resolved_model_path="/models/Mistral")
            client.chat("test")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == pytest.approx(0.7)
            assert call_kwargs["top_p"] == pytest.approx(0.9)
            assert call_kwargs["frequency_penalty"] == pytest.approx(0.5)
            assert call_kwargs["presence_penalty"] == pytest.approx(0.1)

    def test_seed_included_when_positive(self):
        from src.llm_client import LlmClient

        cfg = _StubConfig(LLM_SEED=42)

        with mock.patch("src.llm_client.OpenAI") as MockOAI:
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.return_value = _mock_response()

            client = LlmClient(cfg, resolved_model_path="/models/Mistral")
            client.chat("test")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["seed"] == 42

    def test_seed_omitted_when_negative(self):
        from src.llm_client import LlmClient

        cfg = _StubConfig(LLM_SEED=-1)

        with mock.patch("src.llm_client.OpenAI") as MockOAI:
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.return_value = _mock_response()

            client = LlmClient(cfg, resolved_model_path="/models/Mistral")
            client.chat("test")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert "seed" not in call_kwargs

    def test_top_k_via_extra_body(self):
        from src.llm_client import LlmClient

        cfg = _StubConfig(LLM_TOP_K=50)

        with mock.patch("src.llm_client.OpenAI") as MockOAI:
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.return_value = _mock_response()

            client = LlmClient(cfg, resolved_model_path="/models/Mistral")
            client.chat("test")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert "top_k" not in call_kwargs
            assert call_kwargs["extra_body"]["top_k"] == 50

    def test_top_k_zero_omits_extra_body_key(self):
        from src.llm_client import LlmClient

        cfg = _StubConfig(LLM_TOP_K=0)

        with mock.patch("src.llm_client.OpenAI") as MockOAI:
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.return_value = _mock_response()

            client = LlmClient(cfg, resolved_model_path="/models/Mistral")
            client.chat("test")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            extra = call_kwargs.get("extra_body")
            assert extra is None or "top_k" not in extra


class TestModelField:

    def test_model_uses_resolved_local_path(self):
        from src.llm_client import LlmClient

        local_path = "/workspace/.cache/huggingface/hub/models--mistralai--Mistral/snapshots/abc123"
        cfg = _StubConfig()

        with mock.patch("src.llm_client.OpenAI") as MockOAI:
            mock_client = MockOAI.return_value
            mock_client.chat.completions.create.return_value = _mock_response()

            client = LlmClient(cfg, resolved_model_path=local_path)
            client.chat("test")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == local_path
