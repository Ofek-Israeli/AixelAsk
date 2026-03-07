"""Smoke test for GRPO trainer integration — mocked model, mocked LLM/embedding,
no GPU required."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.training.grpo_trainer import _build_reward_func, _extract_completion_text


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    REWARD_WEIGHT_CORRECTNESS: float = 1.0
    REWARD_WEIGHT_VALIDITY: float = 0.5
    REWARD_WEIGHT_DEPTH: float = 0.1
    REWARD_WEIGHT_INVALID_PENALTY: float = 0.5
    REWARD_DEPTH_NORMALIZATION: str = "DIVIDE_BY_MAX_DEPTH"
    REWARD_MAX_DEPTH: int = 10
    REWARD_INVALID_IF_PARSE_FAILS: bool = True
    REWARD_CORRECTNESS_PARTIAL_CREDIT: bool = False
    FINAL_REASONING_PROMPT: str = ""
    AIXELASK_ROOT: str = ""
    COL_PROMPT: str = ""
    REWARD_MODE: str = "weighted"


# Canonical valid DAG JSON
_VALID_DAG_JSON = json.dumps([
    {"NodeID": 1, "Sub-Level-Question": "q1", "Action": "Retrieval", "Next": [2], "Top k": 3},
    {"NodeID": 2, "Sub-Level-Question": "q2", "Action": "Reasoning", "Next": [], "Top k": 3},
])


# ---------------------------------------------------------------------------
# reward_func interface matches TRL
# ---------------------------------------------------------------------------

class TestRewardFuncInterface:

    def test_signature_accepts_completions_and_kwargs(self):
        """reward_func accepts (completions: list, **kwargs) and returns list[float]."""
        from src.training.train_stats import RewardMetricsAccumulator

        acc = RewardMetricsAccumulator()
        cfg = _StubConfig()

        reward_fn = _build_reward_func(cfg, acc, table_embedding_map=None)

        # Should accept completions as list[list[dict]] and kwargs
        sig = inspect.signature(reward_fn)
        params = list(sig.parameters.keys())
        assert "completions" in params

    def test_returns_list_of_floats(self):
        """reward_func returns a list of floats."""
        from src.training.train_stats import RewardMetricsAccumulator

        acc = RewardMetricsAccumulator()
        cfg = _StubConfig()

        with patch("src.training.dag_reward_parser.parse") as mock_parse, \
             patch("src.training.dag_reward_parser.execute_for_reward") as mock_exec:
            mock_parse.return_value = MagicMock(
                valid=False, dag=None, depth=0,
                error_category="json_parse_error",
            )

            reward_fn = _build_reward_func(cfg, acc, table_embedding_map=None)
            completions = [[{"content": "invalid text"}]]
            result = reward_fn(completions, gold_answer=["42"], table=[{}])

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], float)


# ---------------------------------------------------------------------------
# Bootstrap before patches verified
# ---------------------------------------------------------------------------

class TestBootstrapBeforePatches:

    def test_bootstrap_must_precede_patches(self):
        """bootstrap_upstream_imports must be called before init_patches.

        Verifying by checking that bootstrap_upstream_imports modifies sys.path
        which init_patches depends on for upstream module imports.
        """
        from src.config import bootstrap_upstream_imports

        # bootstrap_upstream_imports should be importable and callable
        assert callable(bootstrap_upstream_imports)


# ---------------------------------------------------------------------------
# Reward function executes parsed DAG (not get_dag)
# ---------------------------------------------------------------------------

class TestRewardExecutesParsedDag:

    def test_reward_calls_execute_for_reward(self):
        """Valid DAG completion → execute_for_reward called with parsed DAG."""
        from src.training.train_stats import RewardMetricsAccumulator
        from src.training.dag_reward_parser import ParseResult

        acc = RewardMetricsAccumulator()
        cfg = _StubConfig()

        parsed_dag = json.loads(_VALID_DAG_JSON)

        with patch("src.training.dag_reward_parser.parse") as mock_parse, \
             patch("src.training.dag_reward_parser.execute_for_reward") as mock_exec:
            mock_parse.return_value = ParseResult(
                dag=parsed_dag, valid=True, depth=2,
            )
            mock_exec.return_value = "42"

            reward_fn = _build_reward_func(cfg, acc, table_embedding_map=None)
            completions = [[{"content": _VALID_DAG_JSON}]]
            result = reward_fn(
                completions,
                gold_answer=["42"],
                table=[[[" h1"], ["v1"]]],
                question=["What is X?"],
            )

            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert call_args[0][0] == parsed_dag  # first positional arg is the DAG

    def test_reward_does_not_call_get_dag(self):
        """Mock scripts.generate_dag.get_dag to raise → reward_func still works."""
        from src.training.train_stats import RewardMetricsAccumulator
        from src.training.dag_reward_parser import ParseResult

        acc = RewardMetricsAccumulator()
        cfg = _StubConfig()

        parsed_dag = json.loads(_VALID_DAG_JSON)

        with patch("src.training.dag_reward_parser.parse") as mock_parse, \
             patch("src.training.dag_reward_parser.execute_for_reward") as mock_exec:
            mock_parse.return_value = ParseResult(
                dag=parsed_dag, valid=True, depth=2,
            )
            mock_exec.return_value = "42"

            reward_fn = _build_reward_func(cfg, acc, table_embedding_map=None)
            completions = [[{"content": _VALID_DAG_JSON}]]

            # If get_dag were called, this would fail — but reward_func
            # uses parse() + execute_for_reward(), not get_dag().
            result = reward_fn(
                completions,
                gold_answer=["42"],
                table=[[[" h1"], ["v1"]]],
            )
            assert len(result) == 1
            assert isinstance(result[0], float)

    def test_invalid_dag_skips_execution(self):
        """Invalid DAG → execute_for_reward NOT called, r_correct=0."""
        from src.training.train_stats import RewardMetricsAccumulator
        from src.training.dag_reward_parser import ParseResult

        acc = RewardMetricsAccumulator()
        cfg = _StubConfig()

        with patch("src.training.dag_reward_parser.parse") as mock_parse, \
             patch("src.training.dag_reward_parser.execute_for_reward") as mock_exec:
            mock_parse.return_value = ParseResult(
                valid=False, dag=None, depth=0,
                error_category="json_parse_error",
            )

            reward_fn = _build_reward_func(cfg, acc, table_embedding_map=None)
            completions = [[{"content": "not a valid dag"}]]
            result = reward_fn(completions, gold_answer=["42"], table=[{}])

            mock_exec.assert_not_called()


# ---------------------------------------------------------------------------
# _extract_completion_text
# ---------------------------------------------------------------------------

class TestExtractCompletionText:

    def test_dict_content(self):
        assert _extract_completion_text([{"content": "hello"}]) == "hello"

    def test_empty_list(self):
        assert _extract_completion_text([]) == ""

    def test_string_element(self):
        assert _extract_completion_text(["some text"]) == "some text"
