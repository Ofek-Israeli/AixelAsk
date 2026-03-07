"""Tests for src.embedding_client — mock SentenceTransformer, prefix modes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest import mock

import numpy as np
import pytest

# Ensure sentence_transformers is importable even without real installation
if "sentence_transformers" not in sys.modules:
    _mock_st = mock.MagicMock()
    sys.modules["sentence_transformers"] = _mock_st


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1"
    TRUST_REMOTE_CODE: bool = True
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    GLOBAL_EMBEDDING_CONCURRENCY: int = 4
    NOMIC_PREFIX_MODE: str = "AUTO"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_client(prefix_mode: str = "AUTO"):
    """Build an EmbeddingClient with a mocked SentenceTransformer."""
    from src.embedding_client import EmbeddingClient

    cfg = _StubConfig(NOMIC_PREFIX_MODE=prefix_mode)

    fake_model = mock.MagicMock()
    fake_model.encode = mock.MagicMock(
        side_effect=lambda texts, **kw: (
            np.random.rand(768).astype(np.float32)
            if isinstance(texts, str)
            else np.random.rand(len(texts) if isinstance(texts, list) else 1, 768).astype(np.float32)
        )
    )

    with mock.patch("src.embedding_client.SentenceTransformer", return_value=fake_model):
        client = EmbeddingClient(cfg)

    return client, fake_model


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------

class TestEmbedOne:

    def test_returns_list_of_float(self):
        client, _ = _make_client()
        vec = client.embed_one("test text")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
        assert len(vec) == 768

    def test_returns_expected_dimension(self):
        client, model = _make_client()
        model.encode.side_effect = lambda t, **kw: np.zeros(768, dtype=np.float32)
        vec = client.embed_one("hello")
        assert len(vec) == 768


class TestEmbedBatch:

    def test_returns_list_of_vectors(self):
        client, model = _make_client()
        model.encode.side_effect = lambda ts, **kw: np.zeros(
            (len(ts) if isinstance(ts, list) else 1, 768), dtype=np.float32
        )
        vecs = client.embed_batch(["a", "b"])
        assert isinstance(vecs, list)
        assert len(vecs) == 2
        assert all(len(v) == 768 for v in vecs)


# ---------------------------------------------------------------------------
# Prefix mode tests
# ---------------------------------------------------------------------------

class TestPrefixModes:

    def test_auto_query_with_question_mark(self):
        """AUTO: string with '?' → search_query prefix."""
        client, model = _make_client("AUTO")
        model.encode.side_effect = lambda t, **kw: np.zeros(768, dtype=np.float32)

        client.embed_one("What is the revenue?")
        actual_input = model.encode.call_args[0][0]
        assert actual_input.startswith("search_query: ")

    def test_auto_document_for_long_string(self):
        """AUTO: long string without '?' → search_document prefix."""
        client, model = _make_client("AUTO")
        model.encode.side_effect = lambda t, **kw: np.zeros(768, dtype=np.float32)

        long_text = "Toyota | Automotive | 256722 | 2019 | Japan | Manufacturing | Revenue growth stable year over year"
        client.embed_one(long_text)
        actual_input = model.encode.call_args[0][0]
        assert actual_input.startswith("search_document: ")

    def test_always_query(self):
        """ALWAYS_QUERY: any string → search_query prefix."""
        client, model = _make_client("ALWAYS_QUERY")
        model.encode.side_effect = lambda t, **kw: np.zeros(768, dtype=np.float32)

        client.embed_one("some random text that is long enough for doc normally")
        actual_input = model.encode.call_args[0][0]
        assert actual_input.startswith("search_query: ")

    def test_always_document(self):
        """ALWAYS_DOCUMENT: any string → search_document prefix."""
        client, model = _make_client("ALWAYS_DOCUMENT")
        model.encode.side_effect = lambda t, **kw: np.zeros(768, dtype=np.float32)

        client.embed_one("What is the revenue?")
        actual_input = model.encode.call_args[0][0]
        assert actual_input.startswith("search_document: ")

    def test_none_no_prefix(self):
        """NONE: no prefix added."""
        client, model = _make_client("NONE")
        model.encode.side_effect = lambda t, **kw: np.zeros(768, dtype=np.float32)

        original_text = "What is the revenue?"
        client.embed_one(original_text)
        actual_input = model.encode.call_args[0][0]
        assert actual_input == original_text
        assert not actual_input.startswith("search_query: ")
        assert not actual_input.startswith("search_document: ")
