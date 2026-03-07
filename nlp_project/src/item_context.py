"""Per-item context variables and thread-safe metadata stores.

Provides ContextVar-based per-item/stage/node context for use under
concurrent.futures thread pools, plus two Lock-protected dict stores
for passing DAG metadata and executor telemetry from worker threads
back to the pipeline orchestrator.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Context variables — propagated via contextvars.copy_context().run(...)
# ---------------------------------------------------------------------------

ctx_item_index: ContextVar[int] = ContextVar("ctx_item_index")
ctx_item_question: ContextVar[str] = ContextVar("ctx_item_question")

ctx_stage: ContextVar[str] = ContextVar("ctx_stage", default="other")
# Values: "dag_generation", "retrieval", "final_reasoning",
#          "noplan_reasoning", "schema_linking", "other"

ctx_node_id: ContextVar[Optional[int]] = ContextVar("ctx_node_id", default=None)

ctx_attempt: ContextVar[int] = ContextVar("ctx_attempt", default=1)

ctx_last_call_id: ContextVar[Optional[str]] = ContextVar(
    "ctx_last_call_id", default=None
)


# ---------------------------------------------------------------------------
# DagMetadataStore — per-item DAG metadata written by patch_dag, read by
# pipeline.py after process_single_table returns.
# ---------------------------------------------------------------------------

@dataclass
class DagItemMeta:
    """Holds per-item DAG generation results."""

    attempt_results: list[dict] = field(default_factory=list)
    dag: Optional[list[dict]] = None
    used_dag: bool = False


class DagMetadataStore:
    """Thread-safe ``dict[int, DagItemMeta]`` keyed by item_index."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[int, DagItemMeta] = {}

    def store(
        self,
        item_index: int,
        attempt_results: list[dict],
        dag: Optional[list[dict]],
    ) -> None:
        """Store metadata for *item_index* (called from worker thread)."""
        meta = DagItemMeta(
            attempt_results=attempt_results,
            dag=dag,
            used_dag=dag is not None,
        )
        with self._lock:
            self._data[item_index] = meta

    def pop(
        self, item_index: int, default: Any = None
    ) -> Optional[DagItemMeta]:
        """Remove and return metadata for *item_index*, or *default*."""
        with self._lock:
            return self._data.pop(item_index, default)


# ---------------------------------------------------------------------------
# ExecTelemetryStore — per-item executor telemetry written by
# patch_dag_execution, read by pipeline.py after each item completes.
# ---------------------------------------------------------------------------

@dataclass
class DagExecTelemetry:
    """Per-item executor metrics produced by DagExecutor.execute_dag()."""

    exec_waves: int = 0
    exec_max_frontier_width: int = 0
    exec_avg_frontier_width: float = 0.0
    exec_max_concurrent_nodes: int = 0
    exec_retrieval_batches: int = 0


class ExecTelemetryStore:
    """Thread-safe ``dict[int, DagExecTelemetry]`` keyed by item_index."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[int, DagExecTelemetry] = {}

    def store(self, item_index: int, telemetry: DagExecTelemetry) -> None:
        """Store telemetry for *item_index* (called from worker thread)."""
        with self._lock:
            self._data[item_index] = telemetry

    def pop(
        self, item_index: int, default: Any = None
    ) -> Optional[DagExecTelemetry]:
        """Remove and return telemetry for *item_index*, or *default*."""
        with self._lock:
            return self._data.pop(item_index, default)
