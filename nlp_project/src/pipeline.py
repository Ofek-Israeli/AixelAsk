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
import os
import threading
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
    save_embeddings.process_table_embeddings(test_examples, config.COL_PROMPT)

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

    # ==================================================================
    # Step 2 — Reasoning (ThreadPoolExecutor, cross-item parallelism)
    # ==================================================================
    logger.info("Step 2: reasoning over %d items (max_workers=%d) …",
                total_items, config.MAX_WORKERS)

    def process_item(item_index: int, example: dict) -> None:
        """Process a single test item inside a copied context."""
        nonlocal correct_count, error_count

        token_idx = ctx_item_index.set(item_index)
        token_q = ctx_item_question.set(example.get("statement", example.get("question", "")))
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

                # Correctness
                is_correct = result_record.get("is_correct", False)
                is_correct_numeric = 1 if is_correct else 0
                result_record["is_correct_numeric"] = is_correct_numeric

                # Config metadata
                result_record["dag_prompt_variant"] = config.DAG_PROMPT_VARIANT
                result_record["fewshot_variant"] = config.FEWSHOT_VARIANT

                # LLM calls
                if call_recorder is not None:
                    result_record["llm_calls"] = call_recorder.get_calls_for_item(item_index)

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

                # Update counters
                with _counter_lock:
                    if is_correct:
                        correct_count += 1  # noqa: F841 — nonlocal
                    if result_record.get("error") or result_record.get("type") == "Error generation":
                        error_count += 1  # noqa: F841

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

    accuracy = correct_count / total_items if total_items > 0 else 0.0
    logger.info(
        "Pipeline complete: %d/%d correct (%.4f accuracy), %d errors",
        correct_count, total_items, accuracy, error_count,
    )

    return PipelineResult(
        total_items=total_items,
        correct_count=correct_count,
        error_count=error_count,
        accuracy=accuracy,
        result_file=os.path.abspath(result_file_path),
        dag_stats_file=os.path.abspath(dag_stats_file) if dag_stats_file else None,
    )


# ---------------------------------------------------------------------------
# Test-split resolution
# ---------------------------------------------------------------------------

def _resolve_test_split(config: Config) -> List[dict]:
    """Load and materialize the test split into a list of enriched dicts.

    Each returned dict has the same schema as the original JSONL line plus
    ``_source_dataset``, ``_source_file``, and ``_source_index`` keys.
    """

    if config.SPLIT_MODE == "overfit_poc":
        raise ValueError(
            "SPLIT_MODE_OVERFIT_POC is not supported by the inference pipeline. "
            "Overfit-PoC evaluation happens exclusively during training."
        )

    if config.SPLIT_MODE == "explicit_indices":
        return _load_explicit_test_split(config)

    # seeded_ratio — load from CONFIG_INFERENCE_DATASET_PATH
    return _load_seeded_ratio_test_split(config)


def _load_explicit_test_split(config: Config) -> List[dict]:
    """Load test examples via explicit index lists from the dataset registry."""
    from src.training.dataset_registry import load_examples, SplitEntry

    entries: List[SplitEntry] = []

    _INDEX_GROUPS = [
        ("wikitq_4k", "test", config.SPLIT_TEST_WIKITQ_4K_INDICES),
        ("wikitq_plus", "test", config.SPLIT_TEST_WIKITQ_PLUS_INDICES),
        ("scalability", "all", config.SPLIT_TEST_SCALABILITY_INDICES),
    ]

    for dataset_key, split_name, indices in _INDEX_GROUPS:
        if indices:
            loaded = load_examples(dataset_key, split_name, indices, config.AIXELASK_ROOT)
            entries.extend(loaded)

    return [
        {
            **entry.example,
            "_source_dataset": entry.source_dataset,
            "_source_file": entry.source_file,
            "_source_index": entry.source_index,
        }
        for entry in entries
    ]


def _load_seeded_ratio_test_split(config: Config) -> List[dict]:
    """Load all examples from ``CONFIG_INFERENCE_DATASET_PATH``."""
    dataset_path = config.INFERENCE_DATASET_PATH
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Inference dataset not found: {dataset_path}"
        )

    dataset_key = _derive_dataset_key(config)
    source_file = os.path.basename(dataset_path)

    examples: List[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ex["_source_dataset"] = dataset_key
            ex["_source_file"] = source_file
            ex["_source_index"] = line_idx
            examples.append(ex)

    return examples


def _derive_dataset_key(config: Config) -> str:
    """Map ``CONFIG_DATASET`` to a short key for provenance metadata."""
    mapping = {
        "DATASET_WIKITQ_4K": "wikitq_4k",
        "DATASET_WIKITQ_PLUS": "wikitq_plus",
        "DATASET_SCALABILITY": "scalability",
    }
    return mapping.get(config.DATASET, "custom")


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
