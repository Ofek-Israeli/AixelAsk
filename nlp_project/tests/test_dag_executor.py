"""Tests for src.dag_executor — topo scheduling, waves, context propagation."""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest import mock

import pytest

from src.item_context import ctx_node_id, ctx_stage, ctx_item_index


# ---------------------------------------------------------------------------
# Canonical DAG fixtures
# ---------------------------------------------------------------------------

def _linear_dag():
    """A → B → C."""
    return [
        {"NodeID": 1, "Action": "Retrieval", "Sub-Level-Question": "q1", "Next": [2], "Top k": 3},
        {"NodeID": 2, "Action": "Retrieval", "Sub-Level-Question": "q2", "Next": [3], "Top k": 3},
        {"NodeID": 3, "Action": "Reasoning", "Sub-Level-Question": "q3", "Next": [], "Top k": 3},
    ]


def _parallel_dag():
    """A → C, B → C."""
    return [
        {"NodeID": 1, "Action": "Retrieval", "Sub-Level-Question": "q1", "Next": [3], "Top k": 3},
        {"NodeID": 2, "Action": "Retrieval", "Sub-Level-Question": "q2", "Next": [3], "Top k": 3},
        {"NodeID": 3, "Action": "Reasoning", "Sub-Level-Question": "q3", "Next": [], "Top k": 3},
    ]


def _hybrid_dag():
    """A → B, A → C, B → D, C → D."""
    return [
        {"NodeID": 1, "Action": "Retrieval", "Sub-Level-Question": "q1", "Next": [2, 3], "Top k": 3},
        {"NodeID": 2, "Action": "Retrieval", "Sub-Level-Question": "q2", "Next": [4], "Top k": 3},
        {"NodeID": 3, "Action": "Retrieval", "Sub-Level-Question": "q3", "Next": [4], "Top k": 3},
        {"NodeID": 4, "Action": "Reasoning", "Sub-Level-Question": "q4", "Next": [], "Top k": 3},
    ]


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    DAG_NODE_MAX_INFLIGHT: int = 16
    SGLANG_CLIENT_CONCURRENCY: int = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_execute_dag(dag, config):
    """Simulate DagExecutor's topo-level scheduling, returning wave count and
    per-node wave assignments.  This avoids upstream imports.
    """
    node_dict = {n["NodeID"]: n for n in dag}
    in_degree = {nid: 0 for nid in node_dict}
    for node in dag:
        for succ in node["Next"]:
            if succ in in_degree:
                in_degree[succ] += 1

    ready = {nid for nid, deg in in_degree.items() if deg == 0}
    wave_number = 0
    node_waves: Dict[int, int] = {}
    frontier_widths: List[int] = []

    while ready:
        wave_number += 1
        frontier = list(ready)[:config.DAG_NODE_MAX_INFLIGHT]
        frontier_widths.append(len(frontier))
        for nid in frontier:
            node_waves[nid] = wave_number
        for nid in frontier:
            ready.discard(nid)
            for succ in node_dict[nid]["Next"]:
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        ready.add(succ)

    max_frontier = max(frontier_widths) if frontier_widths else 0
    return wave_number, max_frontier, node_waves


# ---------------------------------------------------------------------------
# Scheduling tests
# ---------------------------------------------------------------------------

class TestTopoScheduling:

    def test_linear_dag_waves(self):
        waves, max_f, node_waves = _mock_execute_dag(_linear_dag(), _StubConfig())
        assert waves == 3
        assert max_f == 1
        assert node_waves[1] == 1
        assert node_waves[2] == 2
        assert node_waves[3] == 3

    def test_parallel_dag_waves(self):
        waves, max_f, node_waves = _mock_execute_dag(_parallel_dag(), _StubConfig())
        assert waves == 2
        assert max_f == 2
        assert node_waves[1] == 1
        assert node_waves[2] == 1
        assert node_waves[3] == 2

    def test_hybrid_dag_waves(self):
        waves, max_f, node_waves = _mock_execute_dag(_hybrid_dag(), _StubConfig())
        assert waves == 3
        assert max_f == 2
        assert node_waves[1] == 1
        assert node_waves[2] == 2
        assert node_waves[3] == 2
        assert node_waves[4] == 3


class TestMockRetrievalFunction:

    def test_mock_retrieval_returns_dict(self):
        """A mock retrieval function returns expected keys."""
        def mock_retrieve(node, *args):
            return {"row_indices": [0, 1], "col_indices": [0]}

        result = mock_retrieve({"NodeID": 1, "Action": "Retrieval"})
        assert "row_indices" in result
        assert "col_indices" in result


# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

class TestContextPropagation:

    def test_ctx_stage_and_node_id(self):
        """Each node execution sees correct ctx_stage and ctx_node_id."""
        captured: Dict[int, dict] = {}

        def _run_node(node: dict):
            stage_token = ctx_stage.set("retrieval")
            nodeid_token = ctx_node_id.set(node["NodeID"])
            try:
                captured[node["NodeID"]] = {
                    "stage": ctx_stage.get(),
                    "node_id": ctx_node_id.get(),
                }
            finally:
                ctx_node_id.reset(nodeid_token)
                ctx_stage.reset(stage_token)

        dag = _linear_dag()
        for node in dag:
            ctx = contextvars.copy_context()
            ctx.run(_run_node, node)

        for node in dag:
            nid = node["NodeID"]
            assert captured[nid]["stage"] == "retrieval"
            assert captured[nid]["node_id"] == nid


class TestCrossItemIsolation:

    def test_item_context_isolation(self):
        """Two concurrent items see their own ctx_item_index."""
        results: Dict[int, list] = {5: [], 10: []}
        barrier = threading.Barrier(2)

        def _work(item_idx: int):
            token = ctx_item_index.set(item_idx)
            try:
                barrier.wait(timeout=5)
                observed = ctx_item_index.get()
                results[item_idx].append(observed)
            finally:
                ctx_item_index.reset(token)

        def _run_in_context(item_idx: int):
            ctx = contextvars.copy_context()
            ctx.run(_work, item_idx)

        threads = [
            threading.Thread(target=_run_in_context, args=(5,)),
            threading.Thread(target=_run_in_context, args=(10,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results[5] == [5], f"Item 5 saw {results[5]}"
        assert results[10] == [10], f"Item 10 saw {results[10]}"
