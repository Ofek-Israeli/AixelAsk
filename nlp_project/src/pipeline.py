"""Orchestration pipeline — resolves test split, computes embeddings, and
runs reasoning across all test items with cross-item parallelism.

``pipeline.run()`` is the **sole authority** for result-file writing,
DAG-stats computation, summary printing, and accuracy reporting.  Neither
``main.py`` nor ``test_main.py`` duplicate any of these responsibilities.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.item_context import (
    DagMetadataStore,
    ExecTelemetryStore,
    ctx_item_index,
    ctx_item_question,
)

if TYPE_CHECKING:
    from src.call_recorder import CallRecorder
    from src.config import Config
    from src.dag_stats import DagStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PipelineResult:
    total_items: int
    correct_count: int
    error_count: int
    accuracy: float
    result_file: str
    dag_stats_file: Optional[str]
    telemetry_summary_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Result-file writer (thread-safe)
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def _append_result(path: str, record: dict) -> None:
    """Append a single JSON-lines record to *path* under a lock."""
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(
    config: Config,
    *,
    dag_metadata_store: DagMetadataStore,
    exec_telemetry_store: ExecTelemetryStore,
    call_recorder: Optional[CallRecorder] = None,
    dag_stats: Optional[DagStats] = None,
    checkpoint_provenance: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """Execute the full evaluation pipeline on the configured test split.

    Returns a ``PipelineResult`` with aggregate metrics.
    """

    # ==================================================================
    # Step 0a — Deferred upstream imports + stale-alias rebinding
    # ==================================================================
    import scripts.final_reasoning_multi_thread_save_embedding as frm  # noqa: E402
    import scripts.generate_dag  # noqa: E402
    import scripts.get_sub_table  # noqa: E402
    import scripts.generate_answer  # noqa: E402
    import scripts.save_embeddings as save_embeddings  # noqa: E402

    frm.get_dag = scripts.generate_dag.get_dag
    frm.retrieve_final_subtable_DAG_save_embedding = (
        scripts.get_sub_table.retrieve_final_subtable_DAG_save_embedding
    )
    frm.generate_final_answer_DAG = scripts.generate_answer.generate_final_answer_DAG
    frm.generate_noplan_answer = scripts.generate_answer.generate_noplan_answer

    # ==================================================================
    # Step 0 — Resolve test split
    # ==================================================================
    test_examples = _resolve_test_split(config)

    if not test_examples:
        logger.warning("Test split is empty — nothing to evaluate.")
        return PipelineResult(
            total_items=0,
            correct_count=0,
            error_count=0,
            accuracy=0.0,
            result_file=config.RESULT_FILE,
            dag_stats_file=config.DAG_STATS_FILE if dag_stats is not None else None,
        )

    logger.info("Test split resolved: %d items", len(test_examples))

    # ==================================================================
    # Step 1 — Compute embeddings
    # ==================================================================
    logger.info("Step 1: computing table embeddings …")
    stats_output_path = os.path.join(
        os.path.dirname(config.EMBEDDING_CACHE), "embedding_run_summary.json"
    )
    _compute_embeddings(
        test_examples, config.EMBEDDING_CACHE, config.COL_PROMPT,
        save_embeddings, stats_output_path=stats_output_path,
    )
    if stats_output_path:
        logger.info("Embedding run summary: %s", stats_output_path)

    # ==================================================================
    # Step 1a — Load embedding cache
    # ==================================================================
    logger.info("Step 1a: loading embedding cache from %s …", config.EMBEDDING_CACHE)
    table_embedding_map: Dict[str, Any] = frm.load_table_embedding_map(
        config.EMBEDDING_CACHE
    )

    # ==================================================================
    # Load prompt templates for Step 2
    # ==================================================================
    def _read_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    row_prompt = _read_prompt(config.ROW_PROMPT)
    col_prompt = _read_prompt(config.COL_PROMPT)
    plan_prompt = _read_prompt(config.resolved_dag_prompt_path)
    final_reasoning_prompt = _read_prompt(config.FINAL_REASONING_PROMPT)
    noplan_reasoning_prompt = _read_prompt(config.NOPLAN_REASONING_PROMPT)

    question_type_map: Dict[str, str] = defaultdict(lambda: "hybrid")

    result_file_path = config.RESULT_FILE
    os.makedirs(os.path.dirname(result_file_path) or ".", exist_ok=True)

    # Truncate result file so we start clean
    with open(result_file_path, "w"):
        pass

    total_items = len(test_examples)
    correct_count = 0
    error_count = 0
    _counter_lock = threading.Lock()

    # Run-level telemetry accumulators (guarded by _counter_lock)
    _run_prompt_tokens = 0
    _run_completion_tokens = 0
    _run_llm_calls = 0
    _item_latencies: List[float] = []

    # ==================================================================
    # Step 2 — Reasoning (ThreadPoolExecutor, cross-item parallelism)
    # ==================================================================
    logger.info("Step 2: reasoning over %d items (max_workers=%d) …",
                total_items, config.MAX_WORKERS)

    def process_item(item_index: int, example: dict) -> None:
        """Process a single test item inside a copied context."""
        nonlocal correct_count, error_count
        nonlocal _run_prompt_tokens, _run_completion_tokens, _run_llm_calls

        token_idx = ctx_item_index.set(item_index)
        token_q = ctx_item_question.set(example.get("statement", example.get("question", "")))
        item_t0 = time.perf_counter()
        try:
            result_record: Optional[dict] = None
            try:
                raw_line = json.dumps(example, ensure_ascii=False)
                result_record = frm.process_single_table(
                    item_index,
                    raw_line,
                    row_prompt,
                    col_prompt,
                    plan_prompt,
                    final_reasoning_prompt,
                    noplan_reasoning_prompt,
                    question_type_map,
                    table_embedding_map,
                )
            except Exception:
                logger.exception("Item %d: process_single_table failed", item_index)
                result_record = {
                    "type": "Error generation",
                    "is_correct": False,
                    "error": True,
                }

            # -- Per-item post-processing ----------------------------------
            try:
                meta = dag_metadata_store.pop(item_index, default=None)
                telem = exec_telemetry_store.pop(item_index, default=None)

                if result_record is None:
                    result_record = {}

                # Derive enrichment from DAG metadata
                if meta is not None:
                    result_record["dag_regen_attempts"] = len(meta.attempt_results)
                    result_record["used_dag"] = meta.used_dag
                    if meta.dag is not None:
                        result_record["dag_depth"] = _dag_depth(meta.dag)
                    else:
                        result_record["dag_depth"] = None
                    result_record["dag_validity_final"] = 1 if meta.used_dag else 0
                    result_record["invalid_dag_attempts"] = sum(
                        1 for ar in meta.attempt_results if not ar.get("valid", False)
                    )
                    result_record["validity_error_types_seen"] = sorted({
                        ar.get("error_category", "other")
                        for ar in meta.attempt_results
                        if not ar.get("valid", False)
                    })
                else:
                    result_record["dag_regen_attempts"] = 0
                    result_record["used_dag"] = False
                    result_record["dag_depth"] = None
                    result_record["dag_validity_final"] = 0
                    result_record["invalid_dag_attempts"] = 0
                    result_record["validity_error_types_seen"] = []

                # Executor telemetry (frontier width) in per-item result
                if telem is not None:
                    result_record["exec_max_frontier_width"] = telem.exec_max_frontier_width
                    result_record["exec_avg_frontier_width"] = telem.exec_avg_frontier_width
                else:
                    result_record["exec_max_frontier_width"] = None
                    result_record["exec_avg_frontier_width"] = None

                # Correctness
                is_correct = result_record.get("is_correct", False)
                is_correct_numeric = 1 if is_correct else 0
                result_record["is_correct_numeric"] = is_correct_numeric

                # Config metadata
                result_record["dag_prompt_variant"] = config.DAG_PROMPT_VARIANT
                result_record["fewshot_variant"] = config.FEWSHOT_VARIANT

                # LLM calls + per-item token aggregation
                item_prompt_tokens = 0
                item_completion_tokens = 0
                item_num_calls = 0
                if call_recorder is not None:
                    calls = call_recorder.get_calls_for_item(item_index)
                    result_record["llm_calls"] = calls
                    for c in calls:
                        usage = c.get("usage") or {}
                        item_prompt_tokens += usage.get("prompt_tokens", 0)
                        item_completion_tokens += usage.get("completion_tokens", 0)
                    item_num_calls = len(calls)

                result_record["total_prompt_tokens"] = item_prompt_tokens
                result_record["total_completion_tokens"] = item_completion_tokens
                result_record["total_tokens"] = item_prompt_tokens + item_completion_tokens
                result_record["num_llm_calls"] = item_num_calls

                # Per-item latency
                item_latency = time.perf_counter() - item_t0
                result_record["item_latency_sec"] = round(item_latency, 4)

                # Checkpoint provenance
                if checkpoint_provenance is not None:
                    for k, v in checkpoint_provenance.items():
                        result_record[k] = v

                # Source provenance
                result_record["source_dataset"] = example.get("_source_dataset", "")
                result_record["source_file"] = example.get("_source_file", "")
                result_record["source_index"] = example.get("_source_index", item_index)

                # Write to result file
                _append_result(result_file_path, result_record)

                # Update counters + run-level accumulators
                with _counter_lock:
                    if is_correct:
                        correct_count += 1  # noqa: F841 — nonlocal
                    if result_record.get("error") or result_record.get("type") == "Error generation":
                        error_count += 1  # noqa: F841
                    _run_prompt_tokens += item_prompt_tokens
                    _run_completion_tokens += item_completion_tokens
                    _run_llm_calls += item_num_calls
                    _item_latencies.append(item_latency)

                # Record DAG stats
                if dag_stats is not None:
                    question_text = example.get("statement", example.get("question", ""))
                    if meta is not None and meta.used_dag and meta.dag is not None:
                        dag_stats.record_dag(
                            question=question_text,
                            dag=meta.dag,
                            attempt_results=meta.attempt_results,
                            is_correct=is_correct,
                            exec_telemetry=telem,
                        )
                    else:
                        attempt_results = meta.attempt_results if meta is not None else []
                        dag_stats.record_failure(
                            question=question_text,
                            attempt_results=attempt_results,
                            is_correct=is_correct,
                        )

            finally:
                if call_recorder is not None:
                    call_recorder.flush_for_item(item_index)

        finally:
            ctx_item_index.reset(token_idx)
            ctx_item_question.reset(token_q)

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {}
        for i, example in enumerate(test_examples):
            ctx = copy_context()
            future = pool.submit(ctx.run, process_item, i, example)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            exc = future.exception()
            if exc is not None:
                logger.error("Item %d raised an unhandled exception: %s", idx, exc)
                with _counter_lock:
                    error_count += 1

    # ==================================================================
    # Step 3 — DAG Statistics + Summary
    # ==================================================================
    dag_stats_file: Optional[str] = None
    if dag_stats is not None:
        logger.info("Step 3: computing DAG stats summary …")
        dag_stats.compute_summary()
        dag_stats_path = config.DAG_STATS_FILE
        dag_stats.write_summary(dag_stats_path)
        dag_stats.print_summary()
        dag_stats_file = dag_stats_path

    # ==================================================================
    # Step 4 — Telemetry summary
    # ==================================================================
    telemetry_summary_file = _write_telemetry_summary(
        result_file_path, total_items,
        _item_latencies, _run_prompt_tokens, _run_completion_tokens, _run_llm_calls,
    )

    accuracy = correct_count / total_items if total_items > 0 else 0.0
    logger.info(
        "Pipeline complete: %d/%d correct (%.4f accuracy), %d errors, "
        "%d total LLM calls, %d total tokens",
        correct_count, total_items, accuracy, error_count,
        _run_llm_calls, _run_prompt_tokens + _run_completion_tokens,
    )

    return PipelineResult(
        total_items=total_items,
        correct_count=correct_count,
        error_count=error_count,
        accuracy=accuracy,
        result_file=os.path.abspath(result_file_path),
        dag_stats_file=os.path.abspath(dag_stats_file) if dag_stats_file else None,
        telemetry_summary_file=telemetry_summary_file,
    )


# ---------------------------------------------------------------------------
# Telemetry summary
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: List[float], p: float) -> float:
    """Compute the *p*-th percentile (0–100) from a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _write_telemetry_summary(
    result_file_path: str,
    total_items: int,
    item_latencies: List[float],
    run_prompt_tokens: int,
    run_completion_tokens: int,
    run_llm_calls: int,
) -> Optional[str]:
    """Compute and write a ``telemetry_summary.json`` next to the result file."""
    result_dir = os.path.dirname(result_file_path) or "."
    summary_path = os.path.join(result_dir, "telemetry_summary.json")

    sorted_lat = sorted(item_latencies)
    n = len(sorted_lat)

    summary = {
        "total_items": total_items,
        "total_llm_calls": run_llm_calls,
        "total_prompt_tokens": run_prompt_tokens,
        "total_completion_tokens": run_completion_tokens,
        "total_tokens": run_prompt_tokens + run_completion_tokens,
        "mean_prompt_tokens_per_item": round(run_prompt_tokens / n, 2) if n else 0,
        "mean_completion_tokens_per_item": round(run_completion_tokens / n, 2) if n else 0,
        "mean_item_latency_sec": round(sum(sorted_lat) / n, 4) if n else 0,
        "p50_item_latency_sec": round(_percentile(sorted_lat, 50), 4),
        "p90_item_latency_sec": round(_percentile(sorted_lat, 90), 4),
        "p95_item_latency_sec": round(_percentile(sorted_lat, 95), 4),
        "max_item_latency_sec": round(sorted_lat[-1], 4) if n else 0,
        "min_item_latency_sec": round(sorted_lat[0], 4) if n else 0,
    }

    os.makedirs(result_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Telemetry summary written to %s", summary_path)
    return os.path.abspath(summary_path)


# ---------------------------------------------------------------------------
# Embedding pre-computation
# ---------------------------------------------------------------------------

def _compute_embeddings(
    examples: List[dict],
    cache_path: str,
    col_prompt_path: str,
    save_embeddings_mod,
    stats_output_path: str | None = None,
) -> None:
    """Write *examples* to a temp JSONL file and delegate to the upstream
    ``process_table_embeddings(input_path, output_path, col_prompt_path, ...)``.
    If stats_output_path is set, a summary of col template stats is written there.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    ) as tmp:
        for ex in examples:
            tmp.write(json.dumps(ex, ensure_ascii=False) + "\n")
        tmp_path = tmp.name
    try:
        save_embeddings_mod.process_table_embeddings(
            tmp_path, cache_path, col_prompt_path,
            stats_output_path=stats_output_path,
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test-split resolution
# ---------------------------------------------------------------------------

def _resolve_test_split(config: Config) -> List[dict]:
    """Load the test split from ``train_valid_test.yaml``.

    Each returned dict has the same schema as the original JSONL line plus
    ``_source_dataset``, ``_source_file``, and ``_source_index`` keys.
    """
    from src.yaml_splits import load_yaml_splits

    splits = load_yaml_splits(config.SPLIT_YAML_PATH, config.AIXELASK_ROOT)
    return splits["test"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _dag_depth(dag: List[dict]) -> Optional[int]:
    """Compute the longest-path depth of a DAG node list."""
    if not dag:
        return None
    from collections import defaultdict as _dd

    predecessors: Dict[int, List[int]] = _dd(list)
    node_ids: set = set()
    for node in dag:
        nid = node.get("NodeID", -1)
        node_ids.add(nid)
        for s in node.get("Next", []):
            predecessors[s].append(nid)

    depth_cache: Dict[int, int] = {}

    def _depth(nid: int, visited: set) -> int:
        if nid in depth_cache:
            return depth_cache[nid]
        if nid in visited:
            return 1
        visited.add(nid)
        preds = predecessors.get(nid, [])
        if not preds:
            depth_cache[nid] = 1
        else:
            depth_cache[nid] = 1 + max(_depth(p, visited) for p in preds)
        return depth_cache[nid]

    for nid in node_ids:
        _depth(nid, set())

    return max(depth_cache.values()) if depth_cache else 1
