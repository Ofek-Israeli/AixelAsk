"""DAG reward parser: parse model completions and execute DAGs for reward.

``parse`` extracts a DAG from a model completion string, validates it via the
upstream ``validate_dag``, and computes depth.  ``execute_for_reward`` runs the
parsed DAG synchronously against a table to obtain a predicted answer for
correctness scoring.

All upstream imports are **deferred** so this module can be imported without
the full ML/upstream stack (e.g. during config-only validation or tests).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parse result
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Result of parsing a model completion as a DAG."""

    dag: Optional[List[Dict[str, Any]]] = None
    valid: bool = False
    depth: int = 0
    error_category: Optional[str] = None


# Error category constants
ERROR_JSON_PARSE = "json_parse_error"
ERROR_MISSING_KEYS = "missing_keys"
ERROR_INVALID_ACTION = "invalid_action"
ERROR_CYCLE_DETECTED = "cycle_detected"
ERROR_TERMINAL_NOT_REASONING = "terminal_not_reasoning"
ERROR_EMPTY_DAG = "empty_dag"
ERROR_VALIDATION_FAILED = "validation_failed"


# ---------------------------------------------------------------------------
# parse()
# ---------------------------------------------------------------------------

def parse(completion_text: str) -> ParseResult:
    """Parse a model completion as a DAG JSON array and validate it.

    Uses upstream ``validate_dag`` (from ``scripts.generate_dag``) for
    structural validation.  Computes longest-path depth via BFS.

    Parameters
    ----------
    completion_text:
        Raw text produced by the model (expected to contain a JSON array).

    Returns
    -------
    ParseResult
        With ``valid=True`` and populated ``dag``/``depth`` on success, or
        ``valid=False`` and an ``error_category`` on failure.
    """
    text = completion_text.strip()

    # --- Step 1: extract JSON array from the completion ---
    dag_list = _extract_json_array(text)
    if dag_list is None:
        return ParseResult(error_category=ERROR_JSON_PARSE)

    if not dag_list:
        return ParseResult(dag=dag_list, error_category=ERROR_EMPTY_DAG)

    # --- Step 2: quick structural checks before calling upstream validator ---
    required_keys = {"NodeID", "Sub-Level-Question", "Action", "Next"}
    for node in dag_list:
        if not isinstance(node, dict):
            return ParseResult(dag=dag_list, error_category=ERROR_MISSING_KEYS)
        if not required_keys.issubset(node.keys()):
            return ParseResult(dag=dag_list, error_category=ERROR_MISSING_KEYS)
        if node["Action"] not in ("Retrieval", "Reasoning"):
            return ParseResult(dag=dag_list, error_category=ERROR_INVALID_ACTION)

    # Terminal nodes must be Reasoning
    node_ids_with_successors = set()
    for node in dag_list:
        for nxt in node.get("Next", []):
            node_ids_with_successors.add(nxt)
    for node in dag_list:
        if not node["Next"] and node["Action"] != "Reasoning":
            return ParseResult(
                dag=dag_list, error_category=ERROR_TERMINAL_NOT_REASONING,
            )

    # Cycle detection
    if _has_cycle(dag_list):
        return ParseResult(dag=dag_list, error_category=ERROR_CYCLE_DETECTED)

    # --- Step 3: upstream validation (authoritative) ---
    validated_dag = _call_validate_dag(completion_text)
    if validated_dag is None:
        return ParseResult(dag=dag_list, error_category=ERROR_VALIDATION_FAILED)

    depth = _compute_dag_depth(validated_dag)

    return ParseResult(dag=validated_dag, valid=True, depth=depth)


# ---------------------------------------------------------------------------
# execute_for_reward()
# ---------------------------------------------------------------------------

def execute_for_reward(
    dag: List[Dict[str, Any]],
    table: Any,
    question: str,
    config: "Config",
    table_embedding_map: Optional[Dict[str, Any]] = None,
) -> str:
    """Execute a parsed DAG against a table to obtain a predicted answer.

    Calls upstream helper functions **directly on their defining modules**:

    - ``scripts.get_sub_table.retrieve_final_subtable_DAG_save_embedding``
    - ``scripts.generate_answer.generate_final_answer_DAG``

    Does **NOT** call ``process_single_table`` or ``get_dag``.

    Parameters
    ----------
    dag:
        Validated DAG (list of node dicts).
    table:
        Table data (list-of-lists with header as first row).
    question:
        The natural-language question.
    config:
        Project ``Config`` (needs ``FINAL_REASONING_PROMPT``,
        ``AIXELASK_ROOT``).
    table_embedding_map:
        Pre-computed table embeddings keyed by SHA1 table ID.  If the
        current table is not in the map (or the map is ``None``),
        embeddings are computed on-the-fly.

    Returns
    -------
    str
        The predicted answer string.  Empty string on failure.
    """
    try:
        return _execute_impl(dag, table, question, config, table_embedding_map)
    except Exception:
        logger.exception("execute_for_reward failed")
        return ""


def _execute_impl(
    dag: List[Dict[str, Any]],
    table: Any,
    question: str,
    config: "Config",
    table_embedding_map: Optional[Dict[str, Any]],
) -> str:
    # Deferred upstream imports
    from utils.processing import clean_table, index_table
    from scripts.get_sub_table import retrieve_final_subtable_DAG_save_embedding
    from scripts.generate_answer import generate_final_answer_DAG

    # --- Prepare table ---
    cleaned = clean_table(table)
    indexed = index_table(cleaned)

    # --- Obtain embeddings ---
    table_embeddings = _get_table_embeddings(
        table, table_embedding_map, config,
    )

    # --- Execute DAG: retrieval ---
    final_subtable, _row_idx, _col_idx = (
        retrieve_final_subtable_DAG_save_embedding(
            dag, indexed, table_embeddings, question,
        )
    )

    # Build subtable with header for answer generation
    header = indexed[0][1:]  # strip "row index" column
    col_indices = _col_idx
    subtable_header = [header[j] for j in col_indices]
    final_subtable_with_header = [subtable_header] + final_subtable

    # --- Load final-reasoning prompt template ---
    prompt_path = config.FINAL_REASONING_PROMPT
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # --- Generate answer ---
    answer = generate_final_answer_DAG(
        question, dag, final_subtable_with_header, prompt_template,
    )

    return answer if answer else ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    """Try to extract a JSON array of objects from *text*."""
    cleaned = text.strip().strip("`").strip()

    # Try regex extraction (same pattern as upstream validate_dag)
    match = re.search(r'(\[\s*\{.*\}\s*\])', cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return None


def _has_cycle(dag_list: List[Dict[str, Any]]) -> bool:
    """Detect cycles via DFS."""
    node_map = {node["NodeID"]: node for node in dag_list}
    visited: set = set()
    on_stack: set = set()

    def dfs(node_id):
        if node_id in on_stack:
            return True
        if node_id in visited:
            return False
        visited.add(node_id)
        on_stack.add(node_id)
        node = node_map.get(node_id)
        if node:
            for nxt in node.get("Next", []):
                if nxt not in node_map or dfs(nxt):
                    return True
        on_stack.discard(node_id)
        return False

    return any(dfs(n["NodeID"]) for n in dag_list)


def _compute_dag_depth(dag: List[Dict[str, Any]]) -> int:
    """Compute the longest path length (in nodes) via BFS.

    The depth of a single-node DAG is 1.
    """
    if not dag:
        return 0

    node_map = {node["NodeID"]: node for node in dag}

    # Find roots (nodes not referenced as a successor by any other node)
    all_ids = set(node_map.keys())
    child_ids: set = set()
    for node in dag:
        for nxt in node.get("Next", []):
            child_ids.add(nxt)
    roots = all_ids - child_ids
    if not roots:
        roots = all_ids  # fallback if graph structure is unexpected

    max_depth = 0
    queue: deque = deque()
    for r in roots:
        queue.append((r, 1))

    visited_depth: Dict[str, int] = {}
    while queue:
        nid, d = queue.popleft()
        if nid in visited_depth and visited_depth[nid] >= d:
            continue
        visited_depth[nid] = d
        max_depth = max(max_depth, d)
        node = node_map.get(nid)
        if node:
            for nxt in node.get("Next", []):
                if nxt in node_map:
                    queue.append((nxt, d + 1))

    return max_depth


def _call_validate_dag(text: str) -> Optional[List[Dict[str, Any]]]:
    """Call the upstream ``validate_dag`` with deferred import."""
    try:
        from scripts.generate_dag import validate_dag
        return validate_dag(text)
    except ImportError:
        logger.warning(
            "Could not import scripts.generate_dag.validate_dag — "
            "falling back to local validation only",
        )
        return _extract_json_array(text)


def _get_table_id(table: Any) -> str:
    """Compute SHA1 table ID (mirrors upstream ``get_table_id_from_text``)."""
    table_str = json.dumps(table, sort_keys=True)
    return hashlib.sha1(table_str.encode("utf-8")).hexdigest()


def _get_table_embeddings(
    table: Any,
    table_embedding_map: Optional[Dict[str, Any]],
    config: "Config",
) -> Dict[str, Any]:
    """Look up precomputed embeddings or compute on-the-fly."""
    table_id = _get_table_id(table)

    if table_embedding_map and table_id in table_embedding_map:
        return table_embedding_map[table_id]

    logger.debug("Table %s not in cache — computing embeddings on-the-fly", table_id)
    return _compute_embeddings_on_the_fly(table, config)


def _compute_embeddings_on_the_fly(
    table: Any,
    config: "Config",
) -> Dict[str, Any]:
    """Compute row and column embeddings for *table* on-the-fly.

    Uses upstream ``get_embeddings`` from ``scripts.save_embeddings`` and
    description generators from ``scripts.processing_format``.
    """
    from utils.processing import clean_table, index_table
    from scripts.save_embeddings import get_embeddings
    from scripts.processing_format import get_row_flattened, get_col_description
    from utils.request_gpt import request_gpt_embedding

    cleaned = clean_table(table)
    indexed = index_table(cleaned)

    header = indexed[0][1:]  # strip "row index"
    rows = indexed[1:]

    row_descriptions = [get_row_flattened(header, row[1:]) for row in rows]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)

    col_prompt_path = config.COL_PROMPT if hasattr(config, "COL_PROMPT") else ""
    col_prompt = ""
    if col_prompt_path:
        try:
            with open(col_prompt_path, "r") as f:
                col_prompt = f.read()
        except FileNotFoundError:
            pass

    col_descriptions = [get_col_description(col_name, col_prompt) for col_name in header]
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    return {
        "row_embeddings": row_embeddings,
        "col_embeddings": col_embeddings,
    }
