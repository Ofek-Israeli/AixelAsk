"""Thread-safe DAG metric aggregator and end-of-run summary writer.

Collects per-item DAG structural metrics during pipeline execution and
produces aggregate statistics (mean / variance / min / p50 / p90 / p95 / max)
at the end of the run.  Output formats: pretty-printed JSON, CSV sidecar,
optional per-item JSONL, and a rich console table.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.item_context import DagExecTelemetry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-item record
# ---------------------------------------------------------------------------

@dataclass
class _ItemRecord:
    question: str = ""
    is_correct_numeric: int = 0
    dag_validity_final: int = 0
    dag_attempts: int = 0
    invalid_dag_attempts: int = 0

    # Structural — None when item has no valid DAG
    num_nodes: Optional[int] = None
    num_edges: Optional[int] = None
    num_retrieval_nodes: Optional[int] = None
    num_reasoning_nodes: Optional[int] = None
    dag_depth: Optional[int] = None
    num_roots: Optional[int] = None
    num_leaves: Optional[int] = None
    max_width: Optional[int] = None
    avg_out_degree: Optional[float] = None

    # Executor telemetry (optional)
    exec_waves: Optional[int] = None
    exec_max_frontier_width: Optional[int] = None
    exec_avg_frontier_width: Optional[float] = None
    exec_max_concurrent_nodes: Optional[int] = None
    exec_retrieval_batches: Optional[int] = None

    # Per-attempt validity outcomes (1=valid, 0=invalid) for pooled stats
    attempt_validity: list = field(default_factory=list)
    # Per-attempt error categories for validity error breakdown
    attempt_error_categories: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_dag_structure(dag: list[dict]) -> dict[str, Any]:
    """Extract structural metrics from a parsed DAG node list.

    Tolerates malformed nodes gracefully — logs warnings and fills in what it
    can.
    """
    n = len(dag)
    if n == 0:
        return {}

    node_ids: set[int] = set()
    successors: dict[int, list[int]] = {}
    predecessors: dict[int, list[int]] = defaultdict(list)
    retrieval = 0
    reasoning = 0
    total_edges = 0

    for node in dag:
        try:
            nid = int(node.get("NodeID", -1))
        except (TypeError, ValueError):
            nid = -1
        node_ids.add(nid)
        nexts = node.get("Next", [])
        if not isinstance(nexts, list):
            nexts = []
        successors[nid] = nexts
        total_edges += len(nexts)
        for s in nexts:
            predecessors[s].append(nid)

        action = str(node.get("Action", "")).lower()
        if action == "retrieval":
            retrieval += 1
        elif action == "reasoning":
            reasoning += 1

    # Roots / leaves
    roots = [nid for nid in node_ids if nid not in predecessors]
    leaves = [nid for nid in node_ids if not successors.get(nid)]

    # Depth via topological DP (longest path in nodes)
    depth_map: dict[int, int] = {}

    def _depth(nid: int, visited: set) -> int:
        if nid in depth_map:
            return depth_map[nid]
        if nid in visited:
            return 1  # cycle guard
        visited.add(nid)
        preds = predecessors.get(nid, [])
        if not preds:
            depth_map[nid] = 1
        else:
            depth_map[nid] = 1 + max(_depth(p, visited) for p in preds)
        return depth_map[nid]

    for nid in node_ids:
        _depth(nid, set())
    dag_depth = max(depth_map.values()) if depth_map else 1

    # Max width by BFS depth level
    level_map: dict[int, int] = {}
    queue = list(roots)
    for r in roots:
        level_map[r] = 0
    visited_bfs: set[int] = set(roots)
    while queue:
        cur = queue.pop(0)
        for s in successors.get(cur, []):
            if s not in visited_bfs:
                visited_bfs.add(s)
                level_map[s] = level_map[cur] + 1
                queue.append(s)

    if level_map:
        width_counts: dict[int, int] = defaultdict(int)
        for lv in level_map.values():
            width_counts[lv] += 1
        max_width = max(width_counts.values())
    else:
        max_width = n

    avg_out_degree = total_edges / n if n else 0.0

    return {
        "num_nodes": n,
        "num_edges": total_edges,
        "num_retrieval_nodes": retrieval,
        "num_reasoning_nodes": reasoning,
        "dag_depth": dag_depth,
        "num_roots": len(roots),
        "num_leaves": len(leaves),
        "max_width": max_width,
        "avg_out_degree": avg_out_degree,
    }


def _percentile(sorted_vals: list, p: float) -> float:
    """Compute the *p*-th percentile (0–100) using linear interpolation."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f]) + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _aggregate(values: list) -> dict[str, float]:
    """Compute mean / var / min / p50 / p90 / p95 / max for a list of numbers."""
    if not values:
        return {"mean": 0.0, "var": 0.0, "min": 0.0, "p50": 0.0,
                "p90": 0.0, "p95": 0.0, "max": 0.0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    sv = sorted(values)
    return {
        "mean": mean,
        "var": var,
        "min": float(sv[0]),
        "p50": _percentile(sv, 50),
        "p90": _percentile(sv, 90),
        "p95": _percentile(sv, 95),
        "max": float(sv[-1]),
    }


def _extract_attempt_info(attempt_results: list[dict]) -> tuple[list[int], list[str]]:
    """Return per-attempt validity (1/0) and error category lists."""
    validity: list[int] = []
    categories: list[str] = []
    for ar in attempt_results:
        is_valid = 1 if ar.get("valid", False) else 0
        validity.append(is_valid)
        if not is_valid:
            categories.append(ar.get("error_category", "other"))
    return validity, categories


# ---------------------------------------------------------------------------
# DagStats
# ---------------------------------------------------------------------------

class DagStats:
    """Thread-safe DAG metric aggregator.

    Parameters
    ----------
    include_failed : bool
        Include items that fell back to no-plan in the aggregate summary.
    write_per_item : bool
        If *True*, per-item records are available for JSONL sidecar writing.
    log_executor_stats : bool
        Include executor telemetry fields in the summary.
    log_validity_errors : bool
        Include validity error breakdown in the summary.
    """

    def __init__(
        self,
        *,
        include_failed: bool = True,
        write_per_item: bool = False,
        log_executor_stats: bool = True,
        log_validity_errors: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._records: list[_ItemRecord] = []
        self._include_failed = include_failed
        self._write_per_item = write_per_item
        self._log_executor_stats = log_executor_stats
        self._log_validity_errors = log_validity_errors
        self._last_summary: Optional[dict] = None
        self._per_item_path: Optional[str] = None

    # -- properties --------------------------------------------------------

    @property
    def last_summary(self) -> Optional[dict]:
        return self._last_summary

    # -- recording ---------------------------------------------------------

    def record_dag(
        self,
        question: str,
        dag: list[dict],
        attempt_results: list[dict],
        is_correct: bool,
        exec_telemetry: Optional[DagExecTelemetry] = None,
    ) -> None:
        """Record metrics for an item that produced a valid DAG."""
        struct = _compute_dag_structure(dag)
        attempt_validity, attempt_cats = _extract_attempt_info(attempt_results)

        rec = _ItemRecord(
            question=question,
            is_correct_numeric=1 if is_correct else 0,
            dag_validity_final=1,
            dag_attempts=len(attempt_results),
            invalid_dag_attempts=len(attempt_results) - 1,
            attempt_validity=attempt_validity,
            attempt_error_categories=attempt_cats,
            **{k: v for k, v in struct.items()},
        )

        if exec_telemetry is not None:
            rec.exec_waves = exec_telemetry.exec_waves
            rec.exec_max_frontier_width = exec_telemetry.exec_max_frontier_width
            rec.exec_avg_frontier_width = exec_telemetry.exec_avg_frontier_width
            rec.exec_max_concurrent_nodes = exec_telemetry.exec_max_concurrent_nodes
            rec.exec_retrieval_batches = exec_telemetry.exec_retrieval_batches

        with self._lock:
            self._records.append(rec)

    def record_failure(
        self,
        question: str,
        attempt_results: list[dict],
        is_correct: bool,
    ) -> None:
        """Record metrics for an item that fell back to no-plan reasoning."""
        attempt_validity, attempt_cats = _extract_attempt_info(attempt_results)

        rec = _ItemRecord(
            question=question,
            is_correct_numeric=1 if is_correct else 0,
            dag_validity_final=0,
            dag_attempts=len(attempt_results),
            invalid_dag_attempts=len(attempt_results),
            attempt_validity=attempt_validity,
            attempt_error_categories=attempt_cats,
        )
        with self._lock:
            self._records.append(rec)

    # -- aggregation -------------------------------------------------------

    def compute_summary(self) -> dict:
        """Aggregate all recorded per-item metrics into a summary dict."""
        with self._lock:
            records = list(self._records)

        total_items = len(records)
        if total_items == 0:
            summary: dict[str, Any] = {
                "total_items": 0,
                "total_with_dag": 0,
                "total_no_dag": 0,
                "fraction_no_dag": 0.0,
                "total_correct": 0,
                "total_incorrect": 0,
                "accuracy": 0.0,
                "total_dag_attempts": 0,
                "total_invalid_dag_attempts": 0,
                "dag_attempts_distribution": {},
            }
            self._last_summary = summary
            return summary

        with_dag = [r for r in records if r.dag_validity_final == 1]
        no_dag = [r for r in records if r.dag_validity_final == 0]
        total_correct = sum(r.is_correct_numeric for r in records)

        # Scalar counters
        summary = {
            "total_items": total_items,
            "total_with_dag": len(with_dag),
            "total_no_dag": len(no_dag),
            "fraction_no_dag": len(no_dag) / total_items,
            "total_correct": total_correct,
            "total_incorrect": total_items - total_correct,
            "accuracy": total_correct / total_items,
            "total_dag_attempts": sum(r.dag_attempts for r in records),
            "total_invalid_dag_attempts": sum(r.invalid_dag_attempts for r in records),
        }

        # Attempts distribution
        dist: dict[int, int] = defaultdict(int)
        for r in records:
            dist[r.dag_attempts] += 1
        summary["dag_attempts_distribution"] = dict(sorted(dist.items()))

        # Metric aggregation: "all items" population
        all_pop = records
        dag_pop = with_dag

        def _agg(key: str, extractor, population: list) -> None:
            vals = [extractor(r) for r in population if extractor(r) is not None]
            summary[key] = _aggregate(vals)

        _agg("correctness", lambda r: r.is_correct_numeric, all_pop)
        _agg("dag_validity_final", lambda r: r.dag_validity_final, all_pop)
        _agg("dag_attempts", lambda r: r.dag_attempts, all_pop)
        _agg("invalid_dag_attempts", lambda r: r.invalid_dag_attempts, all_pop)

        # dag_validity_attempt: pooled across all individual attempts
        pooled_attempts: list[int] = []
        for r in all_pop:
            pooled_attempts.extend(r.attempt_validity)
        summary["dag_validity_attempt"] = _aggregate(pooled_attempts)

        # Structural metrics from items with valid DAGs
        structural_metrics = [
            "dag_depth", "num_nodes", "num_edges", "num_retrieval_nodes",
            "num_reasoning_nodes", "num_roots", "num_leaves", "max_width",
            "avg_out_degree",
        ]
        for metric in structural_metrics:
            _agg(metric, lambda r, m=metric: getattr(r, m), dag_pop)

        # Executor telemetry
        if self._log_executor_stats:
            exec_metrics = [
                "exec_waves", "exec_max_frontier_width",
                "exec_avg_frontier_width", "exec_max_concurrent_nodes",
                "exec_retrieval_batches",
            ]
            for metric in exec_metrics:
                _agg(metric, lambda r, m=metric: getattr(r, m), dag_pop)

        # Validity error breakdown
        if self._log_validity_errors:
            all_categories: list[str] = []
            for r in all_pop:
                all_categories.extend(r.attempt_error_categories)
            total_invalid = len(all_categories)

            known_cats = [
                "json_parse_error", "missing_keys", "bad_field_type",
                "cycle_detected", "terminal_not_reasoning",
                "invalid_next_ref", "duplicate_node_id", "other",
            ]
            cat_counts: dict[str, int] = {c: 0 for c in known_cats}
            for cat in all_categories:
                if cat in cat_counts:
                    cat_counts[cat] += 1
                else:
                    cat_counts["other"] += 1

            cat_fractions: dict[str, float] = {}
            for c in known_cats:
                cat_fractions[c] = cat_counts[c] / total_invalid if total_invalid > 0 else 0.0

            summary["validity_errors"] = {
                "total_invalid_attempts": total_invalid,
                "counts": cat_counts,
                "fractions": cat_fractions,
            }

        self._last_summary = summary
        return summary

    # -- output ------------------------------------------------------------

    def write_summary(self, path: str) -> None:
        """Write aggregate summary to JSON and a CSV sidecar."""
        if self._last_summary is None:
            self.compute_summary()
        assert self._last_summary is not None

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w") as f:
            json.dump(self._last_summary, f, indent=2, default=str)

        # CSV sidecar
        base, _ = os.path.splitext(path)
        csv_path = base + ".csv"
        agg_keys = [
            "correctness", "dag_depth", "dag_validity_final",
            "dag_validity_attempt", "num_nodes", "num_edges",
            "num_retrieval_nodes", "num_reasoning_nodes",
            "num_roots", "num_leaves", "max_width", "avg_out_degree",
            "dag_attempts", "invalid_dag_attempts",
        ]
        if self._log_executor_stats:
            agg_keys.extend([
                "exec_waves", "exec_max_frontier_width",
                "exec_avg_frontier_width", "exec_max_concurrent_nodes",
                "exec_retrieval_batches",
            ])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "var", "min", "p50", "p90", "p95", "max"])
            for key in agg_keys:
                agg = self._last_summary.get(key, {})
                if isinstance(agg, dict) and "mean" in agg:
                    writer.writerow([
                        key,
                        agg["mean"], agg["var"], agg["min"],
                        agg["p50"], agg["p90"], agg["p95"], agg["max"],
                    ])

        # Per-item JSONL sidecar
        if self._write_per_item:
            per_item_path = base + "_per_item.jsonl"
            with self._lock:
                records = list(self._records)
            with open(per_item_path, "w") as f:
                for r in records:
                    row: dict[str, Any] = {
                        "question": r.question,
                        "is_correct_numeric": r.is_correct_numeric,
                        "dag_validity_final": r.dag_validity_final,
                        "dag_depth": r.dag_depth,
                        "dag_attempts": r.dag_attempts,
                        "invalid_dag_attempts": r.invalid_dag_attempts,
                        "validity_error_types_seen": sorted(
                            set(r.attempt_error_categories)
                        ),
                    }
                    f.write(json.dumps(row, default=str) + "\n")

        logger.info("DAG stats summary written to %s", path)

    def print_summary(self) -> None:
        """Print a human-readable summary table to stdout using rich."""
        if self._last_summary is None:
            self.compute_summary()
        assert self._last_summary is not None

        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            self._print_summary_plain()
            return

        console = Console()
        table = Table(title="DAG Statistics Summary", show_lines=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("P95", justify="right")

        # Scalar rows
        s = self._last_summary
        table.add_row("Total items", str(s["total_items"]), "", "")
        table.add_row("Accuracy", f"{s['accuracy']:.4f}", "", "")
        table.add_row(
            "Fraction no-DAG", f"{s['fraction_no_dag']:.4f}", "", ""
        )

        agg_display = [
            ("correctness", "Correctness"),
            ("dag_depth", "DAG depth"),
            ("dag_validity_final", "DAG validity (final)"),
            ("dag_validity_attempt", "DAG validity (attempt)"),
            ("num_nodes", "Num nodes"),
            ("num_edges", "Num edges"),
            ("num_retrieval_nodes", "Retrieval nodes"),
            ("num_reasoning_nodes", "Reasoning nodes"),
            ("num_roots", "Num roots"),
            ("num_leaves", "Num leaves"),
            ("max_width", "Max width"),
            ("avg_out_degree", "Avg out-degree"),
            ("dag_attempts", "DAG attempts"),
            ("invalid_dag_attempts", "Invalid attempts"),
        ]
        if self._log_executor_stats:
            agg_display.extend([
                ("exec_waves", "Exec waves"),
                ("exec_max_frontier_width", "Exec max frontier"),
                ("exec_avg_frontier_width", "Exec avg frontier"),
                ("exec_max_concurrent_nodes", "Exec max concurrent"),
                ("exec_retrieval_batches", "Exec retrieval batches"),
            ])

        for key, label in agg_display:
            agg = s.get(key, {})
            if isinstance(agg, dict) and "mean" in agg:
                table.add_row(
                    label,
                    f"{agg['mean']:.4f}",
                    f"{agg['p50']:.4f}",
                    f"{agg['p95']:.4f}",
                )

        console.print(table)

    def _print_summary_plain(self) -> None:
        """Fallback plain-text summary when rich is not installed."""
        s = self._last_summary
        assert s is not None
        lines = [
            "=== DAG Statistics Summary ===",
            f"Total items:     {s['total_items']}",
            f"Accuracy:        {s['accuracy']:.4f}",
            f"Fraction no-DAG: {s['fraction_no_dag']:.4f}",
        ]
        agg_keys = [
            "correctness", "dag_depth", "dag_validity_final",
            "dag_validity_attempt", "num_nodes", "num_edges",
            "dag_attempts", "invalid_dag_attempts",
        ]
        header = f"{'Metric':<30} {'Mean':>10} {'Median':>10} {'P95':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for key in agg_keys:
            agg = s.get(key, {})
            if isinstance(agg, dict) and "mean" in agg:
                lines.append(
                    f"{key:<30} {agg['mean']:>10.4f} {agg['p50']:>10.4f} {agg['p95']:>10.4f}"
                )
        print("\n".join(lines))
