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
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import datasets

if TYPE_CHECKING:
    from src.config import Config
    from src.training.split_utils import SplitResult

logger = logging.getLogger(__name__)


def format_for_grpo(
    split_result: "SplitResult",
    config: "Config",
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Format split datasets for ``GRPOTrainer``.

    Parameters
    ----------
    split_result:
        Output of ``split_utils.build_splits``.
    config:
        Fully-resolved project ``Config``.

    Returns
    -------
    tuple[datasets.Dataset, datasets.Dataset]
        ``(train_dataset, eval_dataset)`` with columns:
        ``prompt``, ``gold_answer``, ``table``, ``source_dataset``,
        ``source_file``, ``source_index``, plus any original columns.
    """
    prompt_template = _load_prompt_template(config)
    fewshot_text = _load_fewshot_examples(config)

    def _format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        table = example.get("table", example.get("table_text", []))
        question = example.get("statement", example.get("question", ""))
        gold = example.get("answer", example.get("gold_answer", ""))

        sampled_rows = _sample_table_rows(table, config.NUM_SAMPLE_ROWS)

        prompt = _render_prompt(
            prompt_template, question, sampled_rows, fewshot_text,
        )

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


def _sample_table_rows(table: Any, num_rows: int) -> str:
    """Format a sample of table rows for the prompt context."""
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except (json.JSONDecodeError, TypeError):
            return str(table)[:500]

    if not isinstance(table, list) or not table:
        return str(table)[:500]

    header = table[0] if table else []
    data_rows = table[1:] if len(table) > 1 else []

    if len(data_rows) > num_rows:
        sampled = data_rows[:num_rows]
    else:
        sampled = data_rows

    lines: List[str] = []
    if header:
        lines.append(" | ".join(str(h) for h in header))
        lines.append("-" * len(lines[0]))
    for row in sampled:
        lines.append(" | ".join(str(cell) for cell in row))

    return "\n".join(lines)


def _render_prompt(
    template: str,
    question: str,
    table_sample: str,
    fewshot_text: str,
) -> str:
    """Render the DAG-generation prompt with question and table context.

    Handles both Jinja2 templates and simpler ``{placeholder}`` templates.
    """
    try:
        from jinja2 import Template
        tmpl = Template(template)
        rendered = tmpl.render(
            question=question,
            table=table_sample,
            fewshot_examples=fewshot_text,
            few_shot_examples=fewshot_text,
        )
        return rendered
    except Exception:
        prompt = template
        prompt = prompt.replace("{{question}}", question)
        prompt = prompt.replace("{{table}}", table_sample)
        prompt = prompt.replace("{{fewshot_examples}}", fewshot_text)
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{table}", table_sample)
        prompt = prompt.replace("{fewshot_examples}", fewshot_text)
        return prompt
