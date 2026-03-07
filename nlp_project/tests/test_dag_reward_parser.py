"""Tests for src.training.dag_reward_parser — parse, validate, depth, execute_for_reward."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from src.training.dag_reward_parser import (
    parse,
    ParseResult,
    _compute_dag_depth,
    _has_cycle,
    _extract_json_array,
    ERROR_JSON_PARSE,
    ERROR_MISSING_KEYS,
    ERROR_CYCLE_DETECTED,
    ERROR_EMPTY_DAG,
)


# ---------------------------------------------------------------------------
# Canonical DAG fixtures (as JSON strings)
# ---------------------------------------------------------------------------

def _valid_dag_json() -> str:
    dag = [
        {"NodeID": 1, "Sub-Level-Question": "q1", "Action": "Retrieval", "Next": [2], "Top k": 3},
        {"NodeID": 2, "Sub-Level-Question": "q2", "Action": "Retrieval", "Next": [3], "Top k": 3},
        {"NodeID": 3, "Sub-Level-Question": "q3", "Action": "Reasoning", "Next": [], "Top k": 3},
    ]
    return json.dumps(dag)


def _cyclic_dag_json() -> str:
    dag = [
        {"NodeID": 1, "Sub-Level-Question": "q1", "Action": "Retrieval", "Next": [2], "Top k": 3},
        {"NodeID": 2, "Sub-Level-Question": "q2", "Action": "Retrieval", "Next": [1], "Top k": 3},
    ]
    return json.dumps(dag)


def _missing_keys_dag_json() -> str:
    dag = [
        {"Sub-Level-Question": "q1", "Action": "Retrieval", "Next": [2], "Top k": 3},
    ]
    return json.dumps(dag)


def _parallel_dag_list():
    return [
        {"NodeID": 1, "Sub-Level-Question": "q1", "Action": "Retrieval", "Next": [3], "Top k": 3},
        {"NodeID": 2, "Sub-Level-Question": "q2", "Action": "Retrieval", "Next": [3], "Top k": 3},
        {"NodeID": 3, "Sub-Level-Question": "q3", "Action": "Reasoning", "Next": [], "Top k": 3},
    ]


def _hybrid_dag_list():
    return [
        {"NodeID": 1, "Sub-Level-Question": "q1", "Action": "Retrieval", "Next": [2, 3], "Top k": 3},
        {"NodeID": 2, "Sub-Level-Question": "q2", "Action": "Retrieval", "Next": [4], "Top k": 3},
        {"NodeID": 3, "Sub-Level-Question": "q3", "Action": "Retrieval", "Next": [4], "Top k": 3},
        {"NodeID": 4, "Sub-Level-Question": "q4", "Action": "Reasoning", "Next": [], "Top k": 3},
    ]


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------

class TestParseValid:

    @mock.patch("src.training.dag_reward_parser._call_validate_dag")
    def test_valid_dag_parses_correctly(self, mock_validate):
        """Standard DAG JSON → valid=True, depth computed, dag populated."""
        raw = _valid_dag_json()
        mock_validate.return_value = json.loads(raw)

        result = parse(raw)
        assert result.valid is True
        assert result.dag is not None
        assert len(result.dag) == 3
        assert result.depth == 3

    @mock.patch("src.training.dag_reward_parser._call_validate_dag")
    def test_depth_linear(self, mock_validate):
        """A→B→C → depth=3."""
        raw = _valid_dag_json()
        mock_validate.return_value = json.loads(raw)
        result = parse(raw)
        assert result.depth == 3

    @mock.patch("src.training.dag_reward_parser._call_validate_dag")
    def test_depth_parallel(self, mock_validate):
        dag_list = _parallel_dag_list()
        mock_validate.return_value = dag_list
        result = parse(json.dumps(dag_list))
        assert result.depth == 2

    @mock.patch("src.training.dag_reward_parser._call_validate_dag")
    def test_depth_hybrid(self, mock_validate):
        dag_list = _hybrid_dag_list()
        mock_validate.return_value = dag_list
        result = parse(json.dumps(dag_list))
        assert result.depth == 3


class TestParseInvalid:

    def test_cycle_detected(self):
        result = parse(_cyclic_dag_json())
        assert result.valid is False
        assert result.error_category == ERROR_CYCLE_DETECTED

    def test_malformed_json(self):
        result = parse('{"broken json')
        assert result.valid is False
        assert result.error_category == ERROR_JSON_PARSE

    def test_missing_keys(self):
        result = parse(_missing_keys_dag_json())
        assert result.valid is False
        assert result.error_category == ERROR_MISSING_KEYS

    def test_empty_output(self):
        result = parse("")
        assert result.valid is False
        assert result.error_category == ERROR_JSON_PARSE

    def test_empty_array(self):
        result = parse("[]")
        assert result.valid is False
        assert result.error_category == ERROR_EMPTY_DAG


# ---------------------------------------------------------------------------
# Depth computation helper
# ---------------------------------------------------------------------------

class TestComputeDagDepth:

    def test_linear_depth(self):
        dag = json.loads(_valid_dag_json())
        assert _compute_dag_depth(dag) == 3

    def test_parallel_depth(self):
        assert _compute_dag_depth(_parallel_dag_list()) == 2

    def test_hybrid_depth(self):
        assert _compute_dag_depth(_hybrid_dag_list()) == 3

    def test_single_node(self):
        dag = [{"NodeID": 1, "Sub-Level-Question": "q", "Action": "Reasoning", "Next": []}]
        assert _compute_dag_depth(dag) == 1

    def test_empty_dag(self):
        assert _compute_dag_depth([]) == 0


# ---------------------------------------------------------------------------
# Cycle detection helper
# ---------------------------------------------------------------------------

class TestHasCycle:

    def test_no_cycle(self):
        dag = json.loads(_valid_dag_json())
        assert _has_cycle(dag) is False

    def test_cycle(self):
        dag = json.loads(_cyclic_dag_json())
        assert _has_cycle(dag) is True


# ---------------------------------------------------------------------------
# execute_for_reward does NOT call get_dag
# ---------------------------------------------------------------------------

class TestExecuteForReward:

    @mock.patch("src.training.dag_reward_parser._execute_impl", return_value="mocked answer")
    def test_execute_for_reward_uses_provided_dag(self, mock_impl):
        """execute_for_reward calls _execute_impl with the provided DAG."""
        from src.training.dag_reward_parser import execute_for_reward

        dag = json.loads(_valid_dag_json())
        cfg = mock.MagicMock()
        answer = execute_for_reward(dag, [["h1"], ["v1"]], "question?", cfg)
        assert answer == "mocked answer"

    def test_does_not_call_get_dag(self):
        """Mock get_dag to raise — execute_for_reward should NOT invoke it."""
        from src.training.dag_reward_parser import execute_for_reward

        dag = json.loads(_valid_dag_json())
        cfg = mock.MagicMock()

        with mock.patch("src.training.dag_reward_parser._execute_impl", return_value="ok"):
            with mock.patch.dict(
                "sys.modules",
                {"scripts.generate_dag": mock.MagicMock(
                    get_dag=mock.MagicMock(side_effect=RuntimeError("should not be called"))
                )},
            ):
                answer = execute_for_reward(dag, [["h"]], "q?", cfg)

        assert answer == "ok"
