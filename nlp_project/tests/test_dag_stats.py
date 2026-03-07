"""Tests for src.dag_stats — structural metrics, aggregation, summary output."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading

import pytest

from src.dag_stats import DagStats, _compute_dag_structure, _aggregate


# ---------------------------------------------------------------------------
# Canonical DAG fixtures
# ---------------------------------------------------------------------------

def _linear_dag():
    """A → B → C (linear, depth=3)."""
    return [
        {"NodeID": 1, "Action": "Retrieval", "Sub-Level-Question": "q1", "Next": [2], "Top k": 3},
        {"NodeID": 2, "Action": "Retrieval", "Sub-Level-Question": "q2", "Next": [3], "Top k": 3},
        {"NodeID": 3, "Action": "Reasoning", "Sub-Level-Question": "q3", "Next": [], "Top k": 3},
    ]


def _parallel_dag():
    """A → C, B → C (parallel roots, depth=2)."""
    return [
        {"NodeID": 1, "Action": "Retrieval", "Sub-Level-Question": "q1", "Next": [3], "Top k": 3},
        {"NodeID": 2, "Action": "Retrieval", "Sub-Level-Question": "q2", "Next": [3], "Top k": 3},
        {"NodeID": 3, "Action": "Reasoning", "Sub-Level-Question": "q3", "Next": [], "Top k": 3},
    ]


def _hybrid_dag():
    """A → B, A → C, B → D, C → D (diamond, depth=3)."""
    return [
        {"NodeID": 1, "Action": "Retrieval", "Sub-Level-Question": "q1", "Next": [2, 3], "Top k": 3},
        {"NodeID": 2, "Action": "Retrieval", "Sub-Level-Question": "q2", "Next": [4], "Top k": 3},
        {"NodeID": 3, "Action": "Retrieval", "Sub-Level-Question": "q3", "Next": [4], "Top k": 3},
        {"NodeID": 4, "Action": "Reasoning", "Sub-Level-Question": "q4", "Next": [], "Top k": 3},
    ]


def _single_node_dag():
    return [
        {"NodeID": 1, "Action": "Reasoning", "Sub-Level-Question": "q1", "Next": [], "Top k": 3},
    ]


def _attempt_results_valid():
    return [{"valid": True, "error_category": None}]


def _attempt_results_valid_after_2_failures():
    return [
        {"valid": False, "error_category": "json_parse_error"},
        {"valid": False, "error_category": "missing_keys"},
        {"valid": True, "error_category": None},
    ]


# ---------------------------------------------------------------------------
# Structural metric tests
# ---------------------------------------------------------------------------

class TestComputeDagStructure:

    def test_linear_dag(self):
        s = _compute_dag_structure(_linear_dag())
        assert s["num_nodes"] == 3
        assert s["num_edges"] == 2
        assert s["dag_depth"] == 3
        assert s["num_roots"] == 1
        assert s["num_leaves"] == 1
        assert s["max_width"] == 1

    def test_parallel_dag(self):
        s = _compute_dag_structure(_parallel_dag())
        assert s["num_nodes"] == 3
        assert s["num_edges"] == 2
        assert s["dag_depth"] == 2
        assert s["num_roots"] == 2
        assert s["num_leaves"] == 1
        assert s["max_width"] == 2

    def test_hybrid_dag(self):
        s = _compute_dag_structure(_hybrid_dag())
        assert s["num_nodes"] == 4
        assert s["num_edges"] == 4
        assert s["dag_depth"] == 3
        assert s["num_roots"] == 1
        assert s["num_leaves"] == 1
        assert s["max_width"] == 2
        assert s["avg_out_degree"] == pytest.approx(1.0)

    def test_single_node(self):
        s = _compute_dag_structure(_single_node_dag())
        assert s["num_nodes"] == 1
        assert s["dag_depth"] == 1


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------

class TestComputeSummary:

    def test_basic_aggregation(self):
        stats = DagStats()
        stats.record_dag("q1", _linear_dag(), _attempt_results_valid(), True)
        stats.record_dag("q2", _parallel_dag(), _attempt_results_valid(), True)
        stats.record_dag("q3", _hybrid_dag(), _attempt_results_valid(), False)
        stats.record_failure("q4", [{"valid": False, "error_category": "json_parse_error"}], False)

        summary = stats.compute_summary()
        assert summary["total_items"] == 4
        assert summary["total_with_dag"] == 3
        assert summary["total_no_dag"] == 1
        assert summary["fraction_no_dag"] == pytest.approx(0.25)

    def test_correctness_aggregation(self):
        stats = DagStats()
        stats.record_dag("q1", _linear_dag(), _attempt_results_valid(), True)
        stats.record_dag("q2", _linear_dag(), _attempt_results_valid(), True)
        stats.record_dag("q3", _linear_dag(), _attempt_results_valid(), True)
        stats.record_dag("q4", _linear_dag(), _attempt_results_valid(), False)

        summary = stats.compute_summary()
        assert summary["total_correct"] == 3
        assert summary["total_incorrect"] == 1
        assert summary["accuracy"] == pytest.approx(0.75)
        assert summary["correctness"]["mean"] == pytest.approx(0.75)
        assert summary["correctness"]["var"] == pytest.approx(0.1875)
        assert summary["correctness"]["min"] == 0
        assert summary["correctness"]["max"] == 1

    def test_num_nodes_stats(self):
        stats = DagStats()
        stats.record_dag("q1", _linear_dag(), _attempt_results_valid(), True)
        stats.record_dag("q2", _hybrid_dag(), _attempt_results_valid(), True)

        summary = stats.compute_summary()
        num_nodes = summary["num_nodes"]
        assert num_nodes["mean"] == pytest.approx(3.5)
        assert num_nodes["min"] == 3
        assert num_nodes["max"] == 4

    def test_population_variance(self):
        """Variance uses population formula (/ N), not sample (/ N-1)."""
        stats = DagStats()
        for correct in [True, True, False, False]:
            stats.record_dag("q", _linear_dag(), _attempt_results_valid(), correct)

        summary = stats.compute_summary()
        assert summary["correctness"]["var"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# write_summary output tests
# ---------------------------------------------------------------------------

class TestWriteSummary:

    def test_produces_valid_json_and_csv(self, tmp_path):
        stats = DagStats()
        stats.record_dag("q1", _linear_dag(), _attempt_results_valid(), True)
        stats.record_dag("q2", _parallel_dag(), _attempt_results_valid(), False)

        json_path = os.path.join(str(tmp_path), "dag_stats.json")
        stats.write_summary(json_path)

        with open(json_path) as f:
            data = json.load(f)
        assert "total_items" in data
        assert data["total_items"] == 2

        csv_path = os.path.join(str(tmp_path), "dag_stats.csv")
        assert os.path.isfile(csv_path)
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "metric" in header
            rows = list(reader)
            assert len(rows) > 0

    def test_per_item_jsonl(self, tmp_path):
        stats = DagStats(write_per_item=True)
        stats.record_dag("q1", _linear_dag(), _attempt_results_valid(), True)
        stats.record_dag("q2", _parallel_dag(), _attempt_results_valid(), False)

        json_path = os.path.join(str(tmp_path), "dag_stats.json")
        stats.write_summary(json_path)

        per_item_path = os.path.join(str(tmp_path), "dag_stats_per_item.jsonl")
        assert os.path.isfile(per_item_path)
        with open(per_item_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert "is_correct_numeric" in rec
        assert "dag_depth" in rec
        assert "dag_validity_final" in rec


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_record_dag(self):
        stats = DagStats()
        errors = []

        def _record(i):
            try:
                stats.record_dag(
                    f"q{i}", _linear_dag(), _attempt_results_valid(), i % 2 == 0,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_record, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        summary = stats.compute_summary()
        assert summary["total_with_dag"] == 10


# ---------------------------------------------------------------------------
# Validity error aggregation
# ---------------------------------------------------------------------------

class TestValidityErrors:

    def test_error_counts_and_fractions(self):
        stats = DagStats(log_validity_errors=True)
        stats.record_dag(
            "q1", _linear_dag(),
            _attempt_results_valid_after_2_failures(),
            True,
        )
        stats.record_failure(
            "q2",
            [{"valid": False, "error_category": "json_parse_error"},
             {"valid": False, "error_category": "json_parse_error"}],
            False,
        )

        summary = stats.compute_summary()
        ve = summary["validity_errors"]
        assert ve["total_invalid_attempts"] == 4
        assert ve["counts"]["json_parse_error"] == 3
        assert ve["counts"]["missing_keys"] == 1
        assert ve["fractions"]["json_parse_error"] == pytest.approx(0.75)

    def test_all_first_try_success_zero_errors(self):
        stats = DagStats(log_validity_errors=True)
        for _ in range(3):
            stats.record_dag("q", _linear_dag(), _attempt_results_valid(), True)

        summary = stats.compute_summary()
        ve = summary["validity_errors"]
        assert ve["total_invalid_attempts"] == 0
        for count in ve["counts"].values():
            assert count == 0
