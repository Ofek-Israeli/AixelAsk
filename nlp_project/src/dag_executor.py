"""Topo-level parallel DAG executor.

Replaces the upstream sequential ``for stage in dag_plan:`` loop with a
frontier-based topological scheduler.  Does **not** reimplement any retrieval
algorithm — only controls scheduling order and concurrency.

**All upstream imports are deferred to inside methods** so this module can
be imported before ``bootstrap_upstream_imports`` runs.
"""

from __future__ import annotations

import contextvars
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.item_context import DagExecTelemetry, ctx_node_id, ctx_stage

if TYPE_CHECKING:
    from src.config import Config
    from src.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


class DagExecutor:
    """Frontier-based parallel DAG executor (Option A — per-node embed_one)."""

    def __init__(self, config: Config, embedding_client: EmbeddingClient) -> None:
        self.config = config
        self.embedding_client = embedding_client

    def execute_dag(
        self,
        dag: List[dict],
        indexed_table: List[list],
        table_embeddings: Dict[str, Any],
        question: str,
    ) -> Tuple[Tuple[list, list, list], DagExecTelemetry]:
        """Execute all DAG nodes and return ``(subtable_tuple, telemetry)``."""
        from scripts.get_sub_table import (  # deferred
            retrieve_top_relevant_rows_cols,
            retrieve_rows_by_string_match,
        )
        from utils.request_gpt import request_gpt_embedding  # already patched

        # -- Build adjacency & in-degree ---------------------------------
        node_dict: Dict[int, dict] = {n["NodeID"]: n for n in dag}
        in_degree: Dict[int, int] = {nid: 0 for nid in node_dict}
        for node in dag:
            for succ_id in node["Next"]:
                if succ_id in in_degree:
                    in_degree[succ_id] += 1

        ready = {nid for nid, deg in in_degree.items() if deg == 0}

        header = indexed_table[0][1:]
        row_embeddings = table_embeddings["row_embeddings"]
        col_embeddings = table_embeddings["col_embeddings"]

        completed_states: Dict[int, dict] = {}
        max_inflight = self.config.DAG_NODE_MAX_INFLIGHT

        wave_number = 0
        frontier_widths: list[int] = []
        max_concurrent = 0

        # -- Per-node callable -------------------------------------------
        def _execute_node(node: dict) -> dict:
            """Execute a single DAG node inside a copied context."""
            top_rows, top_cols = retrieve_top_relevant_rows_cols(
                node,
                row_embeddings,
                col_embeddings,
                request_gpt_embedding,
                header,
                node["Top k"],
            )
            return {
                "row_indices": list(top_rows),
                "col_indices": list(top_cols),
            }

        def _run_node_in_context(node: dict) -> dict:
            """Set context vars and run a single node."""
            stage_token = ctx_stage.set("retrieval")
            nodeid_token = ctx_node_id.set(node["NodeID"])
            try:
                return _execute_node(node)
            finally:
                ctx_node_id.reset(nodeid_token)
                ctx_stage.reset(stage_token)

        # -- Frontier scheduling loop ------------------------------------
        while ready:
            wave_number += 1
            frontier_list = list(ready)[:max_inflight]
            frontier_widths.append(len(frontier_list))

            actual_concurrent = len(frontier_list)
            if actual_concurrent > max_concurrent:
                max_concurrent = actual_concurrent

            with ThreadPoolExecutor(max_workers=min(max_inflight, len(frontier_list))) as pool:
                futures = {}
                for nid in frontier_list:
                    node = node_dict[nid]
                    ctx = contextvars.copy_context()
                    future = pool.submit(ctx.run, _run_node_in_context, node)
                    futures[future] = nid

                for future in as_completed(futures):
                    nid = futures[future]
                    completed_states[nid] = future.result()

            # Release successors
            for nid in frontier_list:
                ready.discard(nid)
                for succ_id in node_dict[nid]["Next"]:
                    if succ_id in in_degree:
                        in_degree[succ_id] -= 1
                        if in_degree[succ_id] == 0:
                            ready.add(succ_id)

        # -- Assemble final subtable (mirrors upstream logic) ------------
        embedding_row_indices: list = []
        embedding_col_indices: list = []

        for nid in sorted(completed_states):
            state = completed_states[nid]
            embedding_row_indices.extend(state["row_indices"])
            embedding_col_indices.extend(state["col_indices"])

        embedding_row_indices = sorted(embedding_row_indices, key=int)

        match_row_indices = retrieve_rows_by_string_match(indexed_table, question)

        combined_rows = list(match_row_indices) + list(embedding_row_indices)
        combined_rows = list(dict.fromkeys(combined_rows))

        final_row_indices: list = []
        for row_index in combined_rows:
            if row_index > 0:
                final_row_indices.append(row_index - 1)
            final_row_indices.append(row_index)
            if row_index < len(indexed_table) - 2:
                final_row_indices.append(row_index + 1)

        final_col_indices: list = list(embedding_col_indices)
        final_row_indices = [-1] + final_row_indices
        final_col_indices = [-1, 0] + final_col_indices

        final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]
        final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]

        final_subtable = []
        for i in final_row_indices:
            subtable_row = [indexed_table[i + 1][j + 1] for j in final_col_indices]
            final_subtable.append(subtable_row)

        # -- Telemetry ---------------------------------------------------
        avg_width = (
            sum(frontier_widths) / len(frontier_widths) if frontier_widths else 0.0
        )
        telemetry = DagExecTelemetry(
            exec_waves=wave_number,
            exec_max_frontier_width=max(frontier_widths) if frontier_widths else 0,
            exec_avg_frontier_width=avg_width,
            exec_max_concurrent_nodes=max_concurrent,
            exec_retrieval_batches=wave_number,
        )

        return (final_subtable, final_row_indices, final_col_indices), telemetry
