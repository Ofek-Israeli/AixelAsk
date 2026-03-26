"""Evaluate all (or selected) checkpoints on the validation set.

Usage::

    python -m src.training.eval_checkpoints --config .config
    python -m src.training.eval_checkpoints --config .config --checkpoints 100,500,1000,1725
    python -m src.training.eval_checkpoints --config .config --max-examples 100

For each checkpoint, generates DAG completions with the LoRA-adapted model,
parses them, executes valid DAGs for correctness, and computes the weighted
reward.  Results are saved as JSON; TSV data and a single TeX file are
maintained under output_dir/curves/ and compiled after each checkpoint to
one PDF containing all four plots (validity, correctness, depth, reward vs.
checkpoint step).

Requires two GPUs: GPU 0 runs a vLLM server (base model) for reward-time
answer generation; GPU 1 runs the LoRA-adapted model for DAG generation.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Checkpoint discovery
# ------------------------------------------------------------------

def discover_checkpoints(
    train_output_dir: str,
    filter_steps: Optional[List[int]] = None,
) -> List[Tuple[int, str]]:
    """Find ``checkpoint-N/`` directories and return sorted ``(step, path)``."""
    pattern = re.compile(r"^checkpoint-(\d+)$")
    results = []
    for name in os.listdir(train_output_dir):
        m = pattern.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(train_output_dir, name)
        if not os.path.isdir(path):
            continue
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            continue
        if filter_steps is not None and step not in filter_steps:
            continue
        results.append((step, path))
    results.sort(key=lambda x: x[0])
    return results


# ------------------------------------------------------------------
# Validation set loading
# ------------------------------------------------------------------

def load_validation_set(config, tokenizer, max_examples: Optional[int] = None):
    """Load and format the validation split for DAG generation.

    Returns a ``datasets.Dataset`` with ``prompt``, ``gold_answer``,
    ``table``, and ``question`` columns.
    """
    from src.training.split_utils import build_splits
    from src.training.rl_dataset import format_for_grpo

    split_result = build_splits(config)
    logger.info(
        "Splits loaded: train=%d, valid=%d, test=%d",
        len(split_result.train), len(split_result.valid), len(split_result.test),
    )

    _, eval_ds = format_for_grpo(split_result, config, tokenizer=tokenizer)

    if max_examples is not None and max_examples < len(eval_ds):
        eval_ds = eval_ds.select(range(max_examples))
        logger.info("Validation subset: %d examples", len(eval_ds))

    return eval_ds, split_result


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------

def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int = 1,
) -> List[str]:
    """Generate one completion per prompt using HF transformers."""
    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    completions: List[str] = []
    model.eval()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(getattr(tokenizer, "model_max_length", 4096) or 4096, 4096),
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
            )

        for i, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][i].shape[0]
            generated_ids = output[prompt_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completions.append(text)

        if (start // batch_size) % 20 == 0:
            done = min(start + batch_size, len(prompts))
            logger.info("  Generated %d/%d completions", done, len(prompts))

    return completions


# ------------------------------------------------------------------
# Per-checkpoint evaluation
# ------------------------------------------------------------------

def evaluate_checkpoint(
    step: int,
    ckpt_path: Optional[str],
    base_model,
    tokenizer,
    val_dataset,
    config,
    table_embedding_map: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single checkpoint on the validation set.

    Returns a dict with aggregated metrics.
    """
    from peft import PeftModel
    from src.training import dag_reward_parser, reward

    logger.info("=== Evaluating step %d (%s) ===", step, ckpt_path or "base model")
    t0 = time.monotonic()

    if ckpt_path is not None:
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model.eval()
        logger.info("  Adapter loaded from %s", ckpt_path)
    else:
        model = base_model
        model.eval()
        logger.info("  Using base model (no adapter)")

    prompts = val_dataset["prompt"]
    gold_answers = val_dataset["gold_answer"]
    tables = val_dataset["table"]
    questions = []
    for i in range(len(val_dataset)):
        ex = val_dataset[i]
        q = ex.get("statement", ex.get("question", ""))
        questions.append(q)

    completions = generate_completions(
        model,
        tokenizer,
        prompts,
        max_new_tokens=config.GRPO_MAX_NEW_TOKENS,
        temperature=config.GRPO_TEMPERATURE,
        top_p=config.GRPO_TOP_P,
    )

    # --- Unload adapter to free memory ---
    if ckpt_path is not None:
        base_model_ref = model.unload()
        del model
        torch.cuda.empty_cache()
        logger.info("  Adapter unloaded, memory freed")

    # --- Parse and evaluate ---
    records = []
    for i, text in enumerate(completions):
        gold = gold_answers[i] if i < len(gold_answers) else ""
        table = tables[i] if i < len(tables) else {}
        question = questions[i] if i < len(questions) else ""

        parsed = dag_reward_parser.parse(text)

        r_correct = 0.0
        r_valid = 1.0 if parsed.valid else 0.0
        depth = parsed.depth if parsed.valid else config.REWARD_INVALID_DAG_DEPTH

        if parsed.valid and parsed.dag:
            try:
                predicted = dag_reward_parser.execute_for_reward(
                    parsed.dag, table, question, config,
                    table_embedding_map=table_embedding_map,
                )
                r_correct = _compute_correctness(predicted, gold, config)
            except Exception:
                logger.debug("execute_for_reward failed for item %d", i, exc_info=True)
                r_correct = 0.0

        scalar = reward.compute(
            r_correct=r_correct,
            r_valid=r_valid,
            depth=depth,
            config=config,
        )

        records.append({
            "r_correct": r_correct,
            "r_valid": r_valid,
            "depth": depth,
            "reward": scalar,
        })

        if (i + 1) % 100 == 0:
            logger.info("  Evaluated %d/%d examples", i + 1, len(completions))

    n = len(records)
    validity_rate = sum(r["r_valid"] for r in records) / n if n else 0.0
    correctness_rate = sum(r["r_correct"] for r in records) / n if n else 0.0
    depth_mean = sum(r["depth"] for r in records) / n if n else 0.0
    reward_mean = sum(r["reward"] for r in records) / n if n else 0.0

    elapsed = time.monotonic() - t0
    result = {
        "step": step,
        "checkpoint_path": ckpt_path,
        "num_examples": n,
        "validity_rate": round(validity_rate, 4),
        "correctness_rate": round(correctness_rate, 4),
        "depth_mean": round(depth_mean, 4),
        "reward_mean": round(reward_mean, 4),
        "eval_time_sec": round(elapsed, 1),
    }
    logger.info(
        "  checkpoint-%d: validity=%.4f  correctness=%.4f  depth=%.2f  reward=%.4f  (%.0fs)",
        step, validity_rate, correctness_rate, depth_mean, reward_mean, elapsed,
    )
    return result


def _compute_correctness(predicted: str, gold: str, config) -> float:
    if not predicted or not gold:
        return 0.0
    pred_norm = predicted.strip().lower()
    gold_norm = gold.strip().lower()
    if config.REWARD_CORRECTNESS_PARTIAL_CREDIT:
        import difflib
        return difflib.SequenceMatcher(None, pred_norm, gold_norm).ratio()
    return 1.0 if pred_norm == gold_norm else 0.0


# ------------------------------------------------------------------
# Eval curves (TSV + TeX → one PDF with 4 plots)
# ------------------------------------------------------------------

EVAL_CURVES_METRICS = [
    ("validity_rate", "Validity Rate"),
    ("correctness_rate", "Correctness Rate"),
    ("depth_mean", "Depth Mean"),
    ("reward_mean", "Reward Mean"),
]

_EVAL_FOUR_PLOTS_TEX = r"""\documentclass[tikz,border=5pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}

\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 2, horizontal sep=1.2cm, vertical sep=1.2cm},
  width=7cm,
  height=5cm,
  xlabel={Step},
  grid=major,
  every axis plot/.style={line width=0.8pt},
]
\nextgroupplot[ylabel={ {{- validity_ylabel -}} }, title={Validity}, ymin=0, ymax=1.05]
\addplot[blue, mark=*, mark size=1pt] table[x=step, y=value, col sep=tab] { {{- validity_tsv -}} };
\nextgroupplot[ylabel={ {{- correctness_ylabel -}} }, title={Correctness}, ymin=0, ymax=1.05]
\addplot[blue, mark=*, mark size=1pt] table[x=step, y=value, col sep=tab] { {{- correctness_tsv -}} };
\nextgroupplot[ylabel={ {{- depth_ylabel -}} }, title={Depth}]
\addplot[blue, mark=*, mark size=1pt] table[x=step, y=value, col sep=tab] { {{- depth_tsv -}} };
\nextgroupplot[ylabel={ {{- reward_ylabel -}} }, title={Reward}]
\addplot[blue, mark=*, mark size=1pt] table[x=step, y=value, col sep=tab] { {{- reward_tsv -}} };
\end{groupplot}
\end{tikzpicture}
\end{document}
"""


def _curves_dir(output_dir: str, subdir: str) -> str:
    """Return output_dir/curves/<subdir> (e.g. data, tex, pdf)."""
    return os.path.join(output_dir, "curves", subdir)


def init_eval_curves_dirs(output_dir: str) -> None:
    """Create curves/data, curves/tex, curves/pdf and write TSV headers."""
    for sub in ("data", "tex", "pdf"):
        os.makedirs(_curves_dir(output_dir, sub), exist_ok=True)
    data_dir = _curves_dir(output_dir, "data")
    for key, _ in EVAL_CURVES_METRICS:
        tsv_path = os.path.join(data_dir, f"{key}.tsv")
        if not os.path.exists(tsv_path):
            with open(tsv_path, "w") as f:
                f.write("step\tvalue\n")
    logger.debug("Eval curves dirs initialised under %s", os.path.join(output_dir, "curves"))


def append_eval_curves_data(output_dir: str, step: int, result: Dict[str, Any]) -> None:
    """Append one row to each of the four metric TSV files (skip if step exists)."""
    data_dir = _curves_dir(output_dir, "data")
    for key, _ in EVAL_CURVES_METRICS:
        val = result.get(key)
        if val is None:
            continue
        tsv_path = os.path.join(data_dir, f"{key}.tsv")
        if not os.path.exists(tsv_path):
            with open(tsv_path, "w") as f:
                f.write("step\tvalue\n")
        existing_steps: set = set()
        with open(tsv_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts and parts[0].isdigit():
                    existing_steps.add(int(parts[0]))
        if step not in existing_steps:
            with open(tsv_path, "a") as f:
                f.write(f"{step}\t{val}\n")


def generate_eval_curves_tex(output_dir: str) -> None:
    """Write a single .tex file that plots all four metrics (2x2) using absolute TSV paths."""
    data_dir = _curves_dir(output_dir, "data")
    tex_dir = _curves_dir(output_dir, "tex")
    os.makedirs(tex_dir, exist_ok=True)

    paths = {}
    for key, ylabel in EVAL_CURVES_METRICS:
        tsv_path = os.path.abspath(os.path.join(data_dir, f"{key}.tsv"))
        paths[key] = tsv_path
        paths[f"{key}_ylabel"] = ylabel

    content = _EVAL_FOUR_PLOTS_TEX.replace(" {{- validity_ylabel -}} ", paths["validity_rate_ylabel"])
    content = content.replace(" {{- validity_tsv -}} ", paths["validity_rate"])
    content = content.replace(" {{- correctness_ylabel -}} ", paths["correctness_rate_ylabel"])
    content = content.replace(" {{- correctness_tsv -}} ", paths["correctness_rate"])
    content = content.replace(" {{- depth_ylabel -}} ", paths["depth_mean_ylabel"])
    content = content.replace(" {{- depth_tsv -}} ", paths["depth_mean"])
    content = content.replace(" {{- reward_ylabel -}} ", paths["reward_mean_ylabel"])
    content = content.replace(" {{- reward_tsv -}} ", paths["reward_mean"])

    tex_path = os.path.join(tex_dir, "eval_checkpoints.tex")
    with open(tex_path, "w") as f:
        f.write(content)
    logger.debug("Eval curves TeX written: %s", tex_path)


def compile_eval_curves_pdf(output_dir: str, config: "Config") -> None:
    """Compile curves/tex/eval_checkpoints.tex to curves/pdf/eval_checkpoints.pdf."""
    tex_dir = _curves_dir(output_dir, "tex")
    pdf_dir = _curves_dir(output_dir, "pdf")
    tex_path = os.path.join(tex_dir, "eval_checkpoints.tex")
    if not os.path.isfile(tex_path):
        return
    os.makedirs(pdf_dir, exist_ok=True)

    timeout_sec = config.TRAIN_CURVES_PDFLATEX_TIMEOUT_SEC
    engine = config.TRAIN_CURVES_LATEX_ENGINE
    tmpdir = tempfile.mkdtemp(prefix="eval_tex_", dir=getattr(config, "EPHEMERAL_TMPDIR", None) or tempfile.gettempdir())
    pdf_name = "eval_checkpoints.pdf"
    try:
        shutil.copy2(tex_path, os.path.join(tmpdir, "eval_checkpoints.tex"))
        if getattr(config, "TRAIN_CURVES_LATEXMK", True) and shutil.which("latexmk"):
            cmd = [
                "latexmk", "-pdf",
                f"-{engine}" if engine != "pdflatex" else "-pdflatex",
                f"-output-directory={tmpdir}",
                "eval_checkpoints.tex",
            ]
        else:
            exe = shutil.which(engine) or engine
            cmd = [exe, "-interaction=nonstopmode", f"-output-directory={tmpdir}", "eval_checkpoints.tex"]
        out = subprocess.run(cmd, cwd=tmpdir, timeout=timeout_sec, capture_output=True, text=True)
        tmp_pdf = os.path.join(tmpdir, pdf_name)
        if out.returncode == 0 and os.path.isfile(tmp_pdf):
            shutil.copy2(tmp_pdf, os.path.join(pdf_dir, pdf_name))
            logger.info("Eval curves PDF updated: %s", os.path.join(pdf_dir, pdf_name))
        else:
            logger.warning("Eval curves TeX compile failed (rc=%s): %s", out.returncode, (out.stderr or out.stdout or "")[-400:])
    except subprocess.TimeoutExpired:
        logger.warning("Eval curves TeX compile timed out after %s s", timeout_sec)
    except Exception as e:
        logger.warning("Eval curves TeX compile error: %s", e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ------------------------------------------------------------------
# Plotting (legacy matplotlib; superseded by TeX PDF above)
# ------------------------------------------------------------------

def plot_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Create four PDF plots from the per-checkpoint results (legacy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    steps = [r["step"] for r in results]

    metrics = [
        ("validity_rate", "DAG Validity Rate", "validity.pdf"),
        ("correctness_rate", "DAG Correctness Rate", "correctness.pdf"),
        ("depth_mean", "DAG Mean Depth", "depth.pdf"),
        ("reward_mean", "Mean Reward", "reward.pdf"),
    ]

    for key, title, filename in metrics:
        values = [r[key] for r in results]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, values, marker="o", linewidth=1.5, markersize=4)
        ax.set_xlabel("Checkpoint Step")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. Checkpoint Step")
        ax.grid(True, alpha=0.3)

        if key in ("validity_rate", "correctness_rate"):
            ax.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Plot saved: %s", path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    from src.config import build_arg_parser, load_config, bootstrap_upstream_imports
    from src.download_models import resolve_model_path

    parser = build_arg_parser(description="NLP Project — Checkpoint Evaluation")
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Comma-separated checkpoint steps to evaluate (e.g. 100,500,1000). Default: all.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max validation examples per checkpoint (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results and plots (default: <TRAIN_OUTPUT_DIR>/eval_checkpoints/).",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)

    from src.logging_setup import setup_logging
    setup_logging(config)

    # --- Discover checkpoints ---
    filter_steps = None
    if args.checkpoints:
        filter_steps = [int(s.strip()) for s in args.checkpoints.split(",")]

    checkpoints = discover_checkpoints(config.TRAIN_OUTPUT_DIR, filter_steps)

    if filter_steps is not None and 0 in filter_steps:
        checkpoints = [(0, None)] + checkpoints

    if not checkpoints:
        logger.error("No checkpoints found in %s", config.TRAIN_OUTPUT_DIR)
        sys.exit(1)
    logger.info("Found %d checkpoints: %s", len(checkpoints), [s for s, _ in checkpoints])

    output_dir = args.output_dir or os.path.join(config.TRAIN_OUTPUT_DIR, "eval_checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    init_eval_curves_dirs(output_dir)

    # --- Resolve model path ---
    resolved_model_path = resolve_model_path(config)
    logger.info("Base model: %s", resolved_model_path)

    # --- Bootstrap upstream imports ---
    bootstrap_upstream_imports(config)

    # --- Start vLLM (base model on GPU 0) for reward execution ---
    if not config.CUDA_VISIBLE_DEVICES:
        config.CUDA_VISIBLE_DEVICES = "0"

    from src import inference_server
    from src.llm_client import LlmClient
    from src.embedding_client import EmbeddingClient

    atexit.register(inference_server.stop, config)
    logger.info("Starting vLLM server (base model, GPU 0)...")
    inference_server.start(config, resolved_model_path)

    llm_client = LlmClient(config, resolved_model_path)
    embedding_client = EmbeddingClient(config)

    # Override retry count to avoid long stalls
    llm_client._retries = min(config.LLM_RETRIES, 3)

    from src import patch_request_gpt
    patch_request_gpt.init_patches(llm_client, embedding_client, config)

    # --- Precompute table embeddings ---
    logger.info("Precomputing validation-table embeddings...")
    import scripts.save_embeddings as save_embeddings
    import scripts.final_reasoning_multi_thread_save_embedding as frm

    table_embedding_map = frm.load_table_embedding_map(config.EMBEDDING_CACHE)
    logger.info("Loaded %d table embeddings from cache.", len(table_embedding_map))

    # --- Load base model on GPU 1 ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading base model on GPU 1...")
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        device_map={"": "cuda:1"},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )

    # --- Load validation set ---
    logger.info("Loading validation set...")
    val_dataset, split_result = load_validation_set(config, tokenizer, args.max_examples)
    logger.info("Validation set: %d examples", len(val_dataset))

    # Precompute embeddings for validation tables
    _precompute_embeddings_for_dataset(val_dataset, save_embeddings, config)
    table_embedding_map = frm.load_table_embedding_map(config.EMBEDDING_CACHE)
    logger.info("Table embedding map: %d entries", len(table_embedding_map))

    # --- Load existing results for incremental merge ---
    results_path = os.path.join(output_dir, "results.json")
    existing_results: Dict[int, Dict[str, Any]] = {}
    if os.path.isfile(results_path):
        try:
            with open(results_path, "r") as f:
                for entry in json.load(f):
                    existing_results[entry["step"]] = entry
            logger.info("Loaded %d existing results from %s", len(existing_results), results_path)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Could not parse existing results.json; starting fresh")

    # --- Evaluate each checkpoint ---
    total_t0 = time.monotonic()

    for step, ckpt_path in checkpoints:
        result = evaluate_checkpoint(
            step, ckpt_path, base_model, tokenizer,
            val_dataset, config, table_embedding_map,
        )
        existing_results[step] = result

        merged = sorted(existing_results.values(), key=lambda r: r["step"])
        with open(results_path, "w") as f:
            json.dump(merged, f, indent=2)

        append_eval_curves_data(output_dir, step, result)
        generate_eval_curves_tex(output_dir)
        compile_eval_curves_pdf(output_dir, config)

    total_elapsed = time.monotonic() - total_t0
    logger.info(
        "All checkpoints evaluated in %.1f seconds (%.1f min)",
        total_elapsed, total_elapsed / 60,
    )

    merged = sorted(existing_results.values(), key=lambda r: r["step"])
    with open(results_path, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info("Results written to %s (%d entries)", results_path, len(merged))

    # --- Cleanup ---
    inference_server.stop(config)
    logger.info("Checkpoint evaluation complete.")


def _precompute_embeddings_for_dataset(dataset, save_embeddings_mod, config) -> None:
    """Precompute table embeddings for all examples in the dataset."""
    import json as _json
    import tempfile

    stats_output_path = os.path.join(
        os.path.dirname(config.EMBEDDING_CACHE), "embedding_run_summary.json"
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
        ) as tmp:
            for i in range(len(dataset)):
                ex = dataset[i]
                tmp.write(_json.dumps(dict(ex), ensure_ascii=False) + "\n")
            tmp_path = tmp.name
        try:
            save_embeddings_mod.process_table_embeddings(
                tmp_path, config.EMBEDDING_CACHE, config.COL_PROMPT,
                stats_output_path=stats_output_path,
            )
        finally:
            os.unlink(tmp_path)
    except Exception:
        logger.warning("Table embedding precomputation failed", exc_info=True)


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error in checkpoint evaluation.")
        sys.exit(1)
