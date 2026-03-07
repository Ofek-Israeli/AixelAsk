"""Full pipeline smoke test — mock-based, no GPU or external services."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest

from src.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Install upstream mock modules that pipeline.run imports at the top of its body
# ---------------------------------------------------------------------------

def _install_mock_upstream():
    """Pre-populate sys.modules with mock upstream packages so pipeline.run()
    doesn't fail on ``import scripts.*``."""
    mock_frm = MagicMock()
    mock_frm.process_single_table.return_value = {
        "is_correct": True,
        "predicted_answer": "42",
        "type": "DAG",
    }
    mock_frm.load_table_embedding_map.return_value = {}

    mock_gen_dag = MagicMock()
    mock_sub_table = MagicMock()
    mock_gen_answer = MagicMock()
    mock_save_emb = MagicMock()

    # Build a scripts mock whose attributes point to the specific sub-mocks,
    # so `import scripts.X as X` resolves correctly.
    mock_scripts = MagicMock()
    mock_scripts.final_reasoning_multi_thread_save_embedding = mock_frm
    mock_scripts.generate_dag = mock_gen_dag
    mock_scripts.get_sub_table = mock_sub_table
    mock_scripts.generate_answer = mock_gen_answer
    mock_scripts.save_embeddings = mock_save_emb

    mocks = {
        "scripts": mock_scripts,
        "scripts.final_reasoning_multi_thread_save_embedding": mock_frm,
        "scripts.generate_dag": mock_gen_dag,
        "scripts.get_sub_table": mock_sub_table,
        "scripts.generate_answer": mock_gen_answer,
        "scripts.save_embeddings": mock_save_emb,
        "utils": MagicMock(),
        "utils.processing": MagicMock(),
        "utils.request_gpt": MagicMock(),
        "scripts.processing_format": MagicMock(),
    }
    return mocks, mock_frm


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    SPLIT_YAML_PATH: str = ""
    AIXELASK_ROOT: str = ""
    RESULT_FILE: str = ""
    DAG_STATS_FILE: str = ""
    DAG_STATS_ENABLE: bool = True
    DAG_STATS_INCLUDE_FAILED: bool = True
    DAG_STATS_WRITE_PER_ITEM: bool = False
    DAG_STATS_VALIDITY_ERRORS: bool = False
    LOG_EXECUTOR_STATS: bool = False
    MAX_WORKERS: int = 1
    ROW_PROMPT: str = ""
    COL_PROMPT: str = ""
    FINAL_REASONING_PROMPT: str = ""
    NOPLAN_REASONING_PROMPT: str = ""
    DAG_PROMPT_VARIANT: str = "default"
    FEWSHOT_VARIANT: str = "default"
    EMBEDDING_CACHE: str = ""
    LOG_LLM_CALLS_PER_ITEM: bool = False
    LOG_LLM_PROMPTS: bool = False
    LOG_LLM_RESPONSES: bool = False
    LLM_CALLS_SIDEFILE: str = ""
    resolved_dag_prompt_path: str = ""


def _write_dataset(path: str, n: int = 5) -> None:
    """Write a minimal JSONL test dataset."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            json.dump({
                "statement": f"question_{i}",
                "answer": str(i),
                "table": [["col1"], [f"val{i}"]],
            }, f)
            f.write("\n")


def _write_prompt(path: str, content: str = "prompt template") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Pipeline produces results with required fields
# ---------------------------------------------------------------------------

class TestPipelineResult:

    @patch("src.pipeline._resolve_test_split")
    def test_produces_result_with_required_fields(self, mock_resolve, tmp_path):
        """Pipeline produces PipelineResult with total_items, accuracy, etc."""
        result_file = str(tmp_path / "results.jsonl")
        dag_stats_file = str(tmp_path / "dag_stats.json")

        examples = [
            {"statement": "q1", "answer": "42", "table": [["h"], ["v"]],
             "_source_dataset": "wikitq_4k", "_source_file": "test.jsonl",
             "_source_index": 0},
        ]
        mock_resolve.return_value = examples

        cfg = _StubConfig(
            RESULT_FILE=result_file,
            DAG_STATS_FILE=dag_stats_file,
            DAG_STATS_ENABLE=False,
        )

        for attr in ("ROW_PROMPT", "COL_PROMPT", "FINAL_REASONING_PROMPT",
                      "NOPLAN_REASONING_PROMPT", "resolved_dag_prompt_path"):
            p = str(tmp_path / f"{attr}.txt")
            _write_prompt(p)
            setattr(cfg, attr, p)

        cfg.EMBEDDING_CACHE = str(tmp_path / "emb_cache")
        os.makedirs(cfg.EMBEDDING_CACHE, exist_ok=True)

        from src.item_context import DagMetadataStore, ExecTelemetryStore

        mock_mods, mock_frm = _install_mock_upstream()

        with patch.dict("sys.modules", mock_mods):
            from src.pipeline import run

            result = run(
                cfg,
                dag_metadata_store=DagMetadataStore(),
                exec_telemetry_store=ExecTelemetryStore(),
            )

        assert isinstance(result, PipelineResult)
        assert result.total_items == 1
        assert hasattr(result, "accuracy")
        assert hasattr(result, "result_file")
        assert hasattr(result, "correct_count")


# ---------------------------------------------------------------------------
# Source provenance fields present
# ---------------------------------------------------------------------------

class TestSourceProvenance:

    @patch("src.pipeline._resolve_test_split")
    def test_provenance_in_result(self, mock_resolve, tmp_path):
        """Result records include source_dataset, source_file, source_index."""
        result_file = str(tmp_path / "results.jsonl")

        examples = [
            {"statement": "q1", "answer": "42", "table": [["h"], ["v"]],
             "_source_dataset": "wikitq_4k", "_source_file": "test.jsonl",
             "_source_index": 7},
        ]
        mock_resolve.return_value = examples

        cfg = _StubConfig(
            RESULT_FILE=result_file,
            DAG_STATS_ENABLE=False,
        )

        for attr in ("ROW_PROMPT", "COL_PROMPT", "FINAL_REASONING_PROMPT",
                      "NOPLAN_REASONING_PROMPT", "resolved_dag_prompt_path"):
            p = str(tmp_path / f"{attr}.txt")
            _write_prompt(p)
            setattr(cfg, attr, p)

        cfg.EMBEDDING_CACHE = str(tmp_path / "emb_cache")
        os.makedirs(cfg.EMBEDDING_CACHE, exist_ok=True)

        from src.item_context import DagMetadataStore, ExecTelemetryStore

        mock_mods, mock_frm = _install_mock_upstream()

        with patch.dict("sys.modules", mock_mods):
            from src.pipeline import run

            run(
                cfg,
                dag_metadata_store=DagMetadataStore(),
                exec_telemetry_store=ExecTelemetryStore(),
            )

        assert os.path.isfile(result_file)
        with open(result_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["source_dataset"] == "wikitq_4k"
        assert record["source_file"] == "test.jsonl"
        assert record["source_index"] == 7


# ---------------------------------------------------------------------------
# DAG stats file produced
# ---------------------------------------------------------------------------

class TestDagStatsFileProduced:

    @patch("src.pipeline._resolve_test_split")
    def test_dag_stats_written(self, mock_resolve, tmp_path):
        """When DAG stats enabled, dag_stats.json is produced."""
        result_file = str(tmp_path / "results.jsonl")
        dag_stats_file = str(tmp_path / "dag_stats.json")

        examples = [
            {"statement": "q1", "answer": "42", "table": [["h"], ["v"]],
             "_source_dataset": "test", "_source_file": "f", "_source_index": 0},
        ]
        mock_resolve.return_value = examples

        cfg = _StubConfig(
            RESULT_FILE=result_file,
            DAG_STATS_FILE=dag_stats_file,
            DAG_STATS_ENABLE=True,
        )

        for attr in ("ROW_PROMPT", "COL_PROMPT", "FINAL_REASONING_PROMPT",
                      "NOPLAN_REASONING_PROMPT", "resolved_dag_prompt_path"):
            p = str(tmp_path / f"{attr}.txt")
            _write_prompt(p)
            setattr(cfg, attr, p)

        cfg.EMBEDDING_CACHE = str(tmp_path / "emb_cache")
        os.makedirs(cfg.EMBEDDING_CACHE, exist_ok=True)

        from src.item_context import DagMetadataStore, ExecTelemetryStore
        from src.dag_stats import DagStats

        dag_stats = DagStats(
            include_failed=True,
            write_per_item=False,
            log_executor_stats=False,
            log_validity_errors=False,
        )

        mock_mods, mock_frm = _install_mock_upstream()
        mock_frm.process_single_table.return_value = {
            "is_correct": False,
            "type": "Error generation",
        }

        with patch.dict("sys.modules", mock_mods):
            from src.pipeline import run

            result = run(
                cfg,
                dag_metadata_store=DagMetadataStore(),
                exec_telemetry_store=ExecTelemetryStore(),
                dag_stats=dag_stats,
            )

        assert result.dag_stats_file is not None
        assert os.path.isfile(result.dag_stats_file)


# ---------------------------------------------------------------------------
# Per-item telemetry fields in result records
# ---------------------------------------------------------------------------

class TestPerItemTelemetry:

    @patch("src.pipeline._resolve_test_split")
    def test_telemetry_fields_in_result_record(self, mock_resolve, tmp_path):
        """Result records include item_latency_sec, total_prompt_tokens,
        total_completion_tokens, num_llm_calls, exec_max_frontier_width."""
        result_file = str(tmp_path / "results.jsonl")

        examples = [
            {"statement": "q1", "answer": "42", "table": [["h"], ["v"]],
             "_source_dataset": "wikitq_4k", "_source_file": "test.jsonl",
             "_source_index": 0},
        ]
        mock_resolve.return_value = examples

        cfg = _StubConfig(
            RESULT_FILE=result_file,
            DAG_STATS_ENABLE=False,
        )

        for attr in ("ROW_PROMPT", "COL_PROMPT", "FINAL_REASONING_PROMPT",
                      "NOPLAN_REASONING_PROMPT", "resolved_dag_prompt_path"):
            p = str(tmp_path / f"{attr}.txt")
            _write_prompt(p)
            setattr(cfg, attr, p)

        cfg.EMBEDDING_CACHE = str(tmp_path / "emb_cache")
        os.makedirs(cfg.EMBEDDING_CACHE, exist_ok=True)

        from src.item_context import DagMetadataStore, ExecTelemetryStore

        mock_mods, mock_frm = _install_mock_upstream()

        with patch.dict("sys.modules", mock_mods):
            from src.pipeline import run

            result = run(
                cfg,
                dag_metadata_store=DagMetadataStore(),
                exec_telemetry_store=ExecTelemetryStore(),
            )

        with open(result_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        record = json.loads(lines[0])

        assert "item_latency_sec" in record
        assert isinstance(record["item_latency_sec"], (int, float))
        assert record["item_latency_sec"] >= 0

        assert "total_prompt_tokens" in record
        assert "total_completion_tokens" in record
        assert "total_tokens" in record
        assert "num_llm_calls" in record
        assert record["total_tokens"] == record["total_prompt_tokens"] + record["total_completion_tokens"]

        assert "exec_max_frontier_width" in record
        assert "exec_avg_frontier_width" in record


# ---------------------------------------------------------------------------
# Telemetry summary file produced
# ---------------------------------------------------------------------------

class TestTelemetrySummary:

    @patch("src.pipeline._resolve_test_split")
    def test_telemetry_summary_written(self, mock_resolve, tmp_path):
        """Pipeline writes a telemetry_summary.json next to the result file."""
        result_file = str(tmp_path / "results.jsonl")

        examples = [
            {"statement": f"q{i}", "answer": str(i), "table": [["h"], [f"v{i}"]],
             "_source_dataset": "test", "_source_file": "f", "_source_index": i}
            for i in range(3)
        ]
        mock_resolve.return_value = examples

        cfg = _StubConfig(
            RESULT_FILE=result_file,
            DAG_STATS_ENABLE=False,
        )

        for attr in ("ROW_PROMPT", "COL_PROMPT", "FINAL_REASONING_PROMPT",
                      "NOPLAN_REASONING_PROMPT", "resolved_dag_prompt_path"):
            p = str(tmp_path / f"{attr}.txt")
            _write_prompt(p)
            setattr(cfg, attr, p)

        cfg.EMBEDDING_CACHE = str(tmp_path / "emb_cache")
        os.makedirs(cfg.EMBEDDING_CACHE, exist_ok=True)

        from src.item_context import DagMetadataStore, ExecTelemetryStore

        mock_mods, mock_frm = _install_mock_upstream()

        with patch.dict("sys.modules", mock_mods):
            from src.pipeline import run

            result = run(
                cfg,
                dag_metadata_store=DagMetadataStore(),
                exec_telemetry_store=ExecTelemetryStore(),
            )

        assert result.telemetry_summary_file is not None
        assert os.path.isfile(result.telemetry_summary_file)

        with open(result.telemetry_summary_file) as f:
            summary = json.load(f)

        assert summary["total_items"] == 3
        assert "total_llm_calls" in summary
        assert "total_prompt_tokens" in summary
        assert "total_completion_tokens" in summary
        assert "total_tokens" in summary
        assert "mean_item_latency_sec" in summary
        assert "p50_item_latency_sec" in summary
        assert "p90_item_latency_sec" in summary
        assert "max_item_latency_sec" in summary
        assert summary["mean_item_latency_sec"] >= 0


# ---------------------------------------------------------------------------
# Bootstrap called before patches
# ---------------------------------------------------------------------------

class TestBootstrapBeforePatches:

    def test_main_calls_bootstrap_before_patches(self):
        """In src.main.main(), bootstrap_upstream_imports precedes init_patches.

        We verify the ordering by inspecting source line numbers.
        """
        import inspect
        from src import main as main_mod

        source = inspect.getsource(main_mod.main)
        bootstrap_pos = source.find("bootstrap_upstream_imports")
        patch_pos = source.find("init_patches")

        assert bootstrap_pos > 0, "bootstrap_upstream_imports not found in main()"
        assert patch_pos > 0, "init_patches not found in main()"
        assert bootstrap_pos < patch_pos, (
            "bootstrap_upstream_imports must be called before init_patches"
        )
