"""Monkey-patch for parallel DAG execution and stage tagging.

Patches three upstream functions:
1. ``scripts.get_sub_table.retrieve_final_subtable_DAG_save_embedding``
   → parallel executor.
2. ``scripts.generate_answer.generate_final_answer_DAG``
   → stage-tagged wrapper (``ctx_stage="final_reasoning"``).
3. ``scripts.generate_answer.generate_noplan_answer``
   → stage-tagged wrapper (``ctx_stage="noplan_reasoning"``).

**All upstream imports are deferred to inside ``init_patches``.**
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.item_context import ctx_item_index, ctx_stage

if TYPE_CHECKING:
    from src.dag_executor import DagExecutor
    from src.item_context import ExecTelemetryStore

logger = logging.getLogger(__name__)


def init_patches(
    dag_executor_instance: DagExecutor,
    exec_telemetry_store: ExecTelemetryStore,
) -> None:
    """Apply all three patches.

    Must be called **after** ``bootstrap_upstream_imports()`` and after the
    ``DagExecutor`` has been constructed.
    """
    import scripts.get_sub_table as gst  # deferred
    import scripts.generate_answer as ga  # deferred

    # ------------------------------------------------------------------
    # Patch 1 — parallel retrieval
    # ------------------------------------------------------------------
    original_retrieve = gst.retrieve_final_subtable_DAG_save_embedding

    def parallel_retrieve(dag_plan, indexed_table, table_embeddings, question):
        result, telemetry = dag_executor_instance.execute_dag(
            dag_plan, indexed_table, table_embeddings, question,
        )
        try:
            item_idx = ctx_item_index.get()
        except LookupError:
            item_idx = -1
        exec_telemetry_store.store(item_idx, telemetry)
        return result

    gst.retrieve_final_subtable_DAG_save_embedding = parallel_retrieve

    # ------------------------------------------------------------------
    # Patch 2 — final-reasoning stage tag
    # ------------------------------------------------------------------
    original_final = ga.generate_final_answer_DAG

    def tagged_final(*args, **kwargs):
        token = ctx_stage.set("final_reasoning")
        try:
            return original_final(*args, **kwargs)
        finally:
            ctx_stage.reset(token)

    ga.generate_final_answer_DAG = tagged_final

    # ------------------------------------------------------------------
    # Patch 3 — no-plan-reasoning stage tag
    # ------------------------------------------------------------------
    original_noplan = ga.generate_noplan_answer

    def tagged_noplan(*args, **kwargs):
        token = ctx_stage.set("noplan_reasoning")
        try:
            return original_noplan(*args, **kwargs)
        finally:
            ctx_stage.reset(token)

    ga.generate_noplan_answer = tagged_noplan

    logger.info(
        "Patched get_sub_table.retrieve_final_subtable_DAG_save_embedding "
        "(parallel), generate_answer.generate_final_answer_DAG "
        "(stage=final_reasoning), generate_answer.generate_noplan_answer "
        "(stage=noplan_reasoning)",
    )
