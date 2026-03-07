"""Tests for src.call_recorder — record/update, thread safety, truncation, contextvars."""

from __future__ import annotations

import contextvars
import logging
import os
import threading
from dataclasses import dataclass
from unittest import mock

import pytest

from src.item_context import ctx_item_index
from src.call_recorder import CallRecorder


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    LOG_LLM_CALLS_MAX_CHARS: int = 0
    LLM_CALLS_SIDEFILE: str = ""


# ---------------------------------------------------------------------------
# Basic record / update
# ---------------------------------------------------------------------------

class TestRecordAndRetrieve:

    def test_record_returns_unique_call_ids(self):
        recorder = CallRecorder(_StubConfig())
        token = ctx_item_index.set(0)
        try:
            id1 = recorder.record(
                stage="dag_generation", prompt="p1", response_text="r1",
                finish_reason="stop", usage=None, model="m", attempt=1,
            )
            id2 = recorder.record(
                stage="retrieval", prompt="p2", response_text="r2",
                finish_reason="stop", usage=None, model="m", attempt=1,
            )
        finally:
            ctx_item_index.reset(token)

        assert id1 != id2

    def test_get_calls_for_item(self):
        recorder = CallRecorder(_StubConfig())
        token = ctx_item_index.set(0)
        try:
            recorder.record(
                stage="dag_generation", prompt="p1", response_text="r1",
                finish_reason="stop", usage=None, model="m", attempt=1,
            )
        finally:
            ctx_item_index.reset(token)

        calls = recorder.get_calls_for_item(0)
        assert len(calls) == 1
        assert calls[0]["stage"] == "dag_generation"


class TestUpdate:

    def test_update_annotates_record(self):
        recorder = CallRecorder(_StubConfig())
        token = ctx_item_index.set(0)
        try:
            call_id = recorder.record(
                stage="dag_generation", prompt="p1", response_text="r1",
                finish_reason="stop", usage=None, model="m", attempt=1,
            )
        finally:
            ctx_item_index.reset(token)

        recorder.update(call_id, error="validation failed", error_category="json_parse_error")
        calls = recorder.get_calls_for_item(0)
        assert calls[0]["error"] == "validation failed"
        assert calls[0]["error_category"] == "json_parse_error"

    def test_update_invalid_call_id_is_noop(self, caplog):
        recorder = CallRecorder(_StubConfig())
        with caplog.at_level(logging.WARNING):
            recorder.update("nonexistent_id", error="x")
        assert "unknown call_id" in caplog.text
        assert recorder.get_calls_for_item(0) == []

    def test_update_visible_in_get_calls(self):
        recorder = CallRecorder(_StubConfig())
        token = ctx_item_index.set(7)
        try:
            cid = recorder.record(
                stage="s", prompt=None, response_text=None,
                finish_reason=None, usage=None, model="m", attempt=1,
            )
        finally:
            ctx_item_index.reset(token)

        recorder.update(cid, error="x", error_category="y")
        rec = recorder.get_calls_for_item(7)[0]
        assert rec["error"] == "x"
        assert rec["error_category"] == "y"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_records(self):
        recorder = CallRecorder(_StubConfig())
        errors = []

        def _record(i):
            token = ctx_item_index.set(i)
            try:
                recorder.record(
                    stage=f"s{i}", prompt=f"p{i}", response_text=f"r{i}",
                    finish_reason="stop", usage=None, model="m", attempt=1,
                )
            except Exception as e:
                errors.append(e)
            finally:
                ctx_item_index.reset(token)

        def _run(i):
            ctx = contextvars.copy_context()
            ctx.run(_record, i)

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(10):
            calls = recorder.get_calls_for_item(i)
            assert len(calls) == 1


# ---------------------------------------------------------------------------
# flush_for_item
# ---------------------------------------------------------------------------

class TestFlushForItem:

    def test_clears_flushed_item(self):
        recorder = CallRecorder(_StubConfig())
        for item_idx in [0, 0, 0, 1, 1]:
            token = ctx_item_index.set(item_idx)
            try:
                recorder.record(
                    stage="s", prompt="p", response_text="r",
                    finish_reason="stop", usage=None, model="m", attempt=1,
                )
            finally:
                ctx_item_index.reset(token)

        assert len(recorder.get_calls_for_item(0)) == 3
        assert len(recorder.get_calls_for_item(1)) == 2

        recorder.flush_for_item(0)
        assert len(recorder.get_calls_for_item(0)) == 0
        assert len(recorder.get_calls_for_item(1)) == 2

    def test_sidefile_written_on_flush(self, tmp_path):
        sidefile = os.path.join(str(tmp_path), "calls.jsonl")
        cfg = _StubConfig(LLM_CALLS_SIDEFILE=sidefile)
        recorder = CallRecorder(cfg)

        token = ctx_item_index.set(0)
        try:
            cid = recorder.record(
                stage="s", prompt="p", response_text="r",
                finish_reason="stop", usage=None, model="m", attempt=1,
            )
            recorder.record(
                stage="s2", prompt="p2", response_text="r2",
                finish_reason="stop", usage=None, model="m", attempt=2,
            )
        finally:
            ctx_item_index.reset(token)

        recorder.update(cid, error="annotated", error_category="cat")
        recorder.flush_for_item(0)

        import json
        with open(sidefile) as f:
            lines = f.readlines()
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        assert rec0["error"] == "annotated"
        assert rec0["error_category"] == "cat"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestTruncation:

    def test_prompt_truncated(self):
        cfg = _StubConfig(LOG_LLM_CALLS_MAX_CHARS=50)
        recorder = CallRecorder(cfg)
        long_prompt = "x" * 200

        token = ctx_item_index.set(0)
        try:
            recorder.record(
                stage="s", prompt=long_prompt, response_text="ok",
                finish_reason="stop", usage=None, model="m", attempt=1,
            )
        finally:
            ctx_item_index.reset(token)

        calls = recorder.get_calls_for_item(0)
        stored = calls[0]["prompt"]
        assert len(stored) == 50 + len("...<truncated>")
        assert stored.endswith("...<truncated>")


# ---------------------------------------------------------------------------
# Contextvars tagging
# ---------------------------------------------------------------------------

class TestContextVarsTagging:

    def test_item_tagging_via_contextvars(self):
        recorder = CallRecorder(_StubConfig())
        token = ctx_item_index.set(5)
        try:
            recorder.record(
                stage="s", prompt="p", response_text="r",
                finish_reason="stop", usage=None, model="m", attempt=1,
            )
        finally:
            ctx_item_index.reset(token)

        assert len(recorder.get_calls_for_item(5)) == 1
        assert len(recorder.get_calls_for_item(0)) == 0

    def test_cross_item_isolation_under_concurrency(self):
        recorder = CallRecorder(_StubConfig())
        barrier = threading.Barrier(3, timeout=5)

        def _work(item_idx: int, n: int):
            token = ctx_item_index.set(item_idx)
            try:
                for _ in range(n):
                    recorder.record(
                        stage=f"s{item_idx}", prompt="p",
                        response_text="r", finish_reason="stop",
                        usage=None, model="m", attempt=1,
                    )
                barrier.wait()
            finally:
                ctx_item_index.reset(token)

        def _run(item_idx, n):
            ctx = contextvars.copy_context()
            ctx.run(_work, item_idx, n)

        threads = [
            threading.Thread(target=_run, args=(0, 3)),
            threading.Thread(target=_run, args=(1, 2)),
            threading.Thread(target=_run, args=(2, 4)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(recorder.get_calls_for_item(0)) == 3
        assert len(recorder.get_calls_for_item(1)) == 2
        assert len(recorder.get_calls_for_item(2)) == 4
