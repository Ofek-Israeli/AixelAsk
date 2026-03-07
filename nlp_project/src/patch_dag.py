"""Monkey-patch bridge for DAG generation with prompt override.

Replaces ``scripts.generate_dag.get_dag`` with a new implementation that:
- Uses a vendored Jinja2 prompt template (no question-type routing).
- Always concatenates all few-shot examples per ``CONFIG_FEWSHOT_VARIANT``.
- Captures per-attempt metadata into a ``DagMetadataStore``.
- Manages ``ctx_attempt`` and ``ctx_last_call_id`` with token discipline.

**All upstream imports are deferred to inside ``init_patches``.**
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import jinja2

from src.item_context import (
    ctx_attempt,
    ctx_item_index,
    ctx_last_call_id,
    ctx_stage,
)

if TYPE_CHECKING:
    from src.call_recorder import CallRecorder
    from src.config import Config
    from src.item_context import DagMetadataStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validity error categories — ordered substring matching
# ---------------------------------------------------------------------------

_ERROR_CATEGORIES = [
    (("json", "decode", "parse", "expecting value"), "json_parse_error"),
    (("missing", "required key"), "missing_keys"),
    (("type", "must be", "invalid entry"), "bad_field_type"),
    (("cycle", "not a dag", "acyclic"), "cycle_detected"),
    (("terminal", "reasoning node", "must end"), "terminal_not_reasoning"),
    (("unknown node", "points to", "next"), "invalid_next_ref"),
    (("duplicate",), "duplicate_node_id"),
]


def _classify_error(message: str) -> str:
    """Classify a validation error message into a stable category."""
    lower = message.lower()
    for substrings, category in _ERROR_CATEGORIES:
        if any(s in lower for s in substrings):
            return category
    return "other"


# ---------------------------------------------------------------------------
# init_patches
# ---------------------------------------------------------------------------

def init_patches(
    config: Config,
    call_recorder: Optional[CallRecorder] = None,
    dag_metadata_store: Optional[DagMetadataStore] = None,
) -> None:
    """Replace ``scripts.generate_dag.get_dag`` with the project implementation.

    Must be called **after** ``bootstrap_upstream_imports()`` and **after**
    ``patch_request_gpt.init_patches()``.
    """
    import scripts.generate_dag as gd  # deferred
    from utils.request_gpt import request_gpt_chat  # deferred — already patched
    from utils.processing import sample_table_rows, list_to_markdown  # deferred

    original_validate_dag = gd.validate_dag

    # Resolve prompt template
    template_path = config.resolved_dag_prompt_path
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    # Resolve and concatenate few-shot examples
    fewshot_parts: list[str] = []
    for fpath in config.fewshot_files:
        with open(fpath, "r", encoding="utf-8") as f:
            fewshot_parts.append(f.read().strip())
    fewshot_text = "\n\n".join(fewshot_parts)

    num_sample_rows = config.NUM_SAMPLE_ROWS
    max_retries = config.DAG_MAX_RETRIES

    # ---------------------------------------------------------------
    # Replacement get_dag
    # ---------------------------------------------------------------
    def replacement_get_dag(*args, **kwargs):
        """Drop-in replacement for ``generate_dag.get_dag``.

        Accepts any positional/keyword combination for compatibility with
        the upstream call signature ``(table, question, question_type,
        dag_prompt_template)``.  Extracts *table* and *question*; ignores
        the rest.
        """
        # Extract table
        if args:
            table = args[0]
        else:
            table = kwargs.get("table", kwargs.get("table_text"))

        # Extract question
        if len(args) > 1:
            question = args[1]
        else:
            question = kwargs.get("question", kwargs.get("statement"))

        # Sample rows and render prompt
        num_rows = min(num_sample_rows, len(table) - 1)
        header, sampled_rows = sample_table_rows(table, num_rows)
        markdown_table = list_to_markdown(header, sampled_rows)

        rendered_prompt = jinja2.Template(template_text).render(
            fewshot=fewshot_text,
            question=question,
            table=markdown_table,
        )

        # Set stage context
        stage_token = ctx_stage.set("dag_generation")
        attempt_results: list[dict] = []
        final_dag = None

        try:
            for attempt_num in range(1, max_retries + 1):
                attempt_token = ctx_attempt.set(attempt_num)
                callid_token = ctx_last_call_id.set(None)
                try:
                    response_text = request_gpt_chat(rendered_prompt)
                    call_id = ctx_last_call_id.get()

                    dag = original_validate_dag(response_text)

                    if dag is not None:
                        attempt_results.append({
                            "valid": True,
                            "error_category": None,
                        })
                        final_dag = dag
                        break
                    else:
                        error_msg = "DAG validation returned None"
                        category = "other"
                        attempt_results.append({
                            "valid": False,
                            "error_category": category,
                        })
                        if call_recorder is not None and call_id is not None:
                            call_recorder.update(
                                call_id,
                                error=f"DAG validation failed: {error_msg}",
                                error_category=category,
                            )
                        logger.debug(
                            "Attempt %d/%d: DAG invalid (%s)",
                            attempt_num, max_retries, category,
                        )

                except ValueError as exc:
                    error_msg = str(exc)
                    category = _classify_error(error_msg)
                    call_id = ctx_last_call_id.get()
                    attempt_results.append({
                        "valid": False,
                        "error_category": category,
                    })
                    if call_recorder is not None and call_id is not None:
                        call_recorder.update(
                            call_id,
                            error=f"DAG validation failed: {error_msg}",
                            error_category=category,
                        )
                    logger.debug(
                        "Attempt %d/%d: DAG validation error (%s): %s",
                        attempt_num, max_retries, category, error_msg,
                    )

                except Exception as exc:
                    error_msg = str(exc)
                    category = "other"
                    call_id = ctx_last_call_id.get()
                    attempt_results.append({
                        "valid": False,
                        "error_category": category,
                    })
                    if call_recorder is not None and call_id is not None:
                        call_recorder.update(
                            call_id,
                            error=f"DAG generation error: {error_msg}",
                            error_category=category,
                        )
                    logger.warning(
                        "Attempt %d/%d: unexpected error: %s",
                        attempt_num, max_retries, error_msg,
                    )

                finally:
                    ctx_last_call_id.reset(callid_token)
                    ctx_attempt.reset(attempt_token)

        finally:
            ctx_stage.reset(stage_token)

        # Store metadata
        if dag_metadata_store is not None:
            item_idx = _safe_ctx_get(ctx_item_index, -1)
            dag_metadata_store.store(item_idx, attempt_results, final_dag)

        if final_dag is None:
            logger.info(
                "All %d DAG generation attempts exhausted — returning None",
                max_retries,
            )

        return final_dag

    # ---------------------------------------------------------------
    # Apply the patch
    # ---------------------------------------------------------------
    gd.get_dag = replacement_get_dag

    logger.info(
        "Patched scripts.generate_dag.get_dag (prompt=%s, fewshot=%d files)",
        os.path.basename(template_path),
        len(config.fewshot_files),
    )


def _safe_ctx_get(var, default=None):
    """Read a ContextVar, returning *default* if not set."""
    try:
        return var.get()
    except LookupError:
        return default
