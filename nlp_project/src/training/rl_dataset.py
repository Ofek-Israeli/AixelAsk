"""RL dataset formatting for TRL's ``GRPOTrainer``.

``format_for_grpo`` transforms ``SplitResult`` datasets into the format
expected by TRL: a ``prompt`` column containing the rendered DAG-generation
prompt, with ``gold_answer``, ``table``, and provenance columns preserved
for access by ``reward_func`` via TRL's ``**kwargs`` mechanism.

The prompt is rendered using the **same Jinja2 DAG-generation template**
used by the inference pipeline, ensuring format consistency between training
and evaluation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import datasets

if TYPE_CHECKING:
    from src.config import Config
    from src.training.split_utils import SplitResult

logger = logging.getLogger(__name__)


def format_for_grpo(
    split_result: "SplitResult",
    config: "Config",
    tokenizer=None,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Format split datasets for ``GRPOTrainer``.

    Parameters
    ----------
    split_result:
        Output of ``split_utils.build_splits``.
    config:
        Fully-resolved project ``Config``.
    tokenizer:
        HF tokenizer — used to apply the chat template so the model
        sees the same ``[INST]...[/INST]`` framing as during inference.

    Returns
    -------
    tuple[datasets.Dataset, datasets.Dataset]
        ``(train_dataset, eval_dataset)`` with columns:
        ``prompt``, ``gold_answer``, ``table``, ``source_dataset``,
        ``source_file``, ``source_index``, plus any original columns.
    """
    prompt_template = _load_prompt_template(config)
    fewshot_text = _load_fewshot_examples(config)

    from utils.processing import sample_table_rows, list_to_markdown

    num_sample_rows = config.NUM_SAMPLE_ROWS

    def _format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        table = example.get("table", example.get("table_text", []))
        question = example.get("statement", example.get("question", ""))
        gold = example.get("answer", example.get("gold_answer", ""))
        if isinstance(gold, list):
            gold = ", ".join(str(g) for g in gold)

        parsed_table = _ensure_list(table)
        num_rows = min(num_sample_rows, max(len(parsed_table) - 1, 0))
        header, sampled_rows = sample_table_rows(parsed_table, num_rows)
        markdown_table = list_to_markdown(header, sampled_rows)

        raw_prompt = _render_prompt(
            prompt_template, question, markdown_table, fewshot_text,
        )

        if tokenizer is not None:
            messages = [{"role": "user", "content": raw_prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = raw_prompt

        if isinstance(table, str):
            try:
                table = json.loads(table)
            except (json.JSONDecodeError, TypeError):
                pass

        example["prompt"] = prompt
        example["gold_answer"] = str(gold)
        if "table" not in example:
            example["table"] = table

        return example

    train_ds = split_result.train.map(_format_example)
    eval_ds = split_result.valid.map(_format_example)

    train_cols = set(train_ds.column_names)
    required = {"prompt", "gold_answer", "table", "source_dataset", "source_file", "source_index"}
    missing = required - train_cols
    if missing:
        logger.warning("Missing expected columns in train dataset: %s", missing)

    logger.info(
        "Formatted for GRPO: train=%d, eval=%d",
        len(train_ds), len(eval_ds),
    )
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def _load_prompt_template(config: "Config") -> str:
    """Load the DAG-generation prompt template."""
    path = config.resolved_dag_prompt_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DAG prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_fewshot_examples(config: "Config") -> str:
    """Load and concatenate few-shot example files."""
    parts: List[str] = []
    for fpath in config.fewshot_files:
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    parts.append(content)
    return "\n\n".join(parts)


def _ensure_list(table: Any) -> list:
    """Ensure *table* is a list-of-lists (parse JSON string if needed)."""
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(table, list):
        return table
    return []


def _render_prompt(
    template: str,
    question: str,
    table_sample: str,
    fewshot_text: str,
) -> str:
    """Render the DAG-generation prompt with question and table context.

    Uses the same Jinja2 variable names as the inference pipeline
    (``patch_dag.py``): ``fewshot``, ``question``, ``table``.
    """
    import jinja2
    return jinja2.Template(template).render(
        fewshot=fewshot_text,
        question=question,
        table=table_sample,
    )
