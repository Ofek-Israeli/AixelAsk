"""Parse Kconfig-generated ``.config`` into a Python dataclass.

Reads lines like ``CONFIG_SGLANG_PORT=30000``, strips quotes from strings,
casts ints, handles ``y``/``n`` as bools.  All other modules import from here.

CLI override semantics (``--override KEY=VALUE``):
1. Parse ``.config`` into raw key-value pairs.
2. Apply all overrides (add or overwrite).
3. Derive training output sub-paths from ``CONFIG_TRAIN_OUTPUT_DIR``.
4. Resolve persistent / prompt / dataset paths to absolute.
5. Validate (TabFact rejection, split-mode compat, overlap, fewshot existence).
6. Set ``HF_HOME`` / ``TRANSFORMERS_CACHE`` / ``TMPDIR`` environment variables.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Derived directory constants (independent of Kconfig symbols)
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIXELASK_ROOT = os.path.dirname(PROJECT_DIR)
UPSTREAM_SOURCE_ROOT = os.path.join(AIXELASK_ROOT, "AixelAsk")
UPSTREAM_SCRIPTS_DIR = os.path.join(UPSTREAM_SOURCE_ROOT, "scripts")

# ---------------------------------------------------------------------------
# Fewshot variant → file list mapping
# ---------------------------------------------------------------------------

_FEWSHOT_FILES = {
    "FEWSHOT_STANDARD_ALL3": [
        "prompt/fewshot/fewshot_parallel.txt",
        "prompt/fewshot/fewshot_sequential.txt",
        "prompt/fewshot/fewshot_hybrid.txt",
    ],
    "FEWSHOT_PARALLEL_HYBRID_EXTENDED": [
        "prompt/fewshot/fewshot_parallel.txt",
        "prompt/fewshot/fewshot_hybrid.txt",
        "prompt/fewshot/fewshot_parallel_extended.txt",
        "prompt/fewshot/fewshot_hybrid_extended.txt",
    ],
}

# DAG prompt variant → relative prompt path
_DAG_PROMPT_PATHS = {
    "DAG_PROMPT_STANDARD": "prompt/get_dag.md",
    "DAG_PROMPT_MIN_DEPTH_BASELINE": "prompt/get_dag_min_depth.md",
}

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Holds all parsed and resolved configuration values."""

    # -- Derived directory constants (not from Kconfig) --------------------
    PROJECT_DIR: str = ""
    AIXELASK_ROOT: str = ""
    UPSTREAM_SOURCE_ROOT: str = ""
    UPSTREAM_SCRIPTS_DIR: str = ""

    # -- Kconfig.model -----------------------------------------------------
    INFERENCE_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.3"
    INFERENCE_MODEL_REVISION: str = "main"
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1"
    MODEL_CACHE_DIR: str = "/workspace/.cache/huggingface"
    TRUST_REMOTE_CODE: bool = True
    NOMIC_PREFIX_MODE: str = "AUTO"

    # -- Kconfig.server ----------------------------------------------------
    SGLANG_HOST: str = "0.0.0.0"
    SGLANG_PORT: int = 30000
    SGLANG_TP_SIZE: int = 1
    SGLANG_MEM_FRACTION: str = "0.85"
    SGLANG_DTYPE: str = "float16"
    SGLANG_CONTEXT_LENGTH: int = 8192
    SGLANG_EXTRA_ARGS: str = ""
    SGLANG_HEALTH_TIMEOUT: int = 300
    SGLANG_HEALTH_INTERVAL: int = 5
    SGLANG_HEALTH_ENDPOINT: str = "/health"

    # -- Kconfig.dataset ---------------------------------------------------
    DATASET: str = "DATASET_WIKITQ_4K"
    INFERENCE_DATASET_PATH: str = "dataset/WikiTQ-4k/test.jsonl"

    # -- Kconfig.split -----------------------------------------------------
    SPLIT_MODE: str = "seeded_ratio"

    SPLIT_TRAIN_WIKITQ_4K_INDICES: list = field(default_factory=list)
    SPLIT_TRAIN_WIKITQ_PLUS_INDICES: list = field(default_factory=list)
    SPLIT_TRAIN_SCALABILITY_INDICES: list = field(default_factory=list)
    SPLIT_VALID_WIKITQ_4K_INDICES: list = field(default_factory=list)
    SPLIT_VALID_WIKITQ_PLUS_INDICES: list = field(default_factory=list)
    SPLIT_VALID_SCALABILITY_INDICES: list = field(default_factory=list)
    SPLIT_TEST_WIKITQ_4K_INDICES: list = field(default_factory=list)
    SPLIT_TEST_WIKITQ_PLUS_INDICES: list = field(default_factory=list)
    SPLIT_TEST_SCALABILITY_INDICES: list = field(default_factory=list)

    # -- Kconfig.retrieval -------------------------------------------------
    USE_DAG: bool = True
    DAG_MAX_RETRIES: int = 10
    DAG_PROMPT_VARIANT: str = "DAG_PROMPT_STANDARD"
    PLAN_PROMPT: str = ""
    FEWSHOT_VARIANT: str = "FEWSHOT_STANDARD_ALL3"
    FINAL_REASONING_PROMPT: str = "prompt/final_reasoning_DAG.md"
    NOPLAN_REASONING_PROMPT: str = "prompt/noplan_reasoning.md"
    ROW_PROMPT: str = "prompt/get_row_template.md"
    COL_PROMPT: str = "prompt/get_col_template.md"
    SCHEMA_LINKING_PROMPT: str = "prompt/prompt_schema_linking.md"
    NUM_SAMPLE_ROWS: int = 5

    # -- Kconfig.output ----------------------------------------------------
    PERSISTENT_ROOT: str = "/workspace/AixelAsk/nlp_project"
    EPHEMERAL_TMPDIR: str = "/tmp"
    RESULT_FILE: str = "output/results.jsonl"
    EMBEDDING_CACHE: str = "cache/table_embeddings.jsonl"
    LOG_FILE: str = "output/run.log"
    LOG_LEVEL: str = "INFO"
    LOG_LLM_PROMPTS: bool = False
    LOG_LLM_RESPONSES: bool = False
    LOG_LLM_CALLS_MAX_CHARS: int = 200000
    LOG_LLM_CALLS_PER_ITEM: bool = False
    LLM_CALLS_SIDEFILE: str = ""

    # -- Kconfig.runtime ---------------------------------------------------
    MAX_WORKERS: int = 5
    LLM_MAX_OUTPUT_TOKENS: int = 2048
    LLM_RETRIES: int = 30
    LLM_TEMPERATURE: float = 0.0
    LLM_TOP_P: float = 1.0
    LLM_TOP_K: int = 0
    LLM_FREQUENCY_PENALTY: float = 0.0
    LLM_PRESENCE_PENALTY: float = 0.0
    LLM_SEED: int = -1
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_RETRIES: int = 5
    SKIP_EXISTING: bool = True
    CUDA_VISIBLE_DEVICES: str = ""
    DAG_NODE_MAX_INFLIGHT: int = 16
    RETRIEVAL_PARALLELISM: int = 8
    RETRIEVAL_EMBED_BATCH_SIZE: int = 32
    REASONING_PARALLELISM: int = 8
    SGLANG_CLIENT_CONCURRENCY: int = 32
    GLOBAL_EMBEDDING_CONCURRENCY: int = 4

    # -- Kconfig.dagstats --------------------------------------------------
    DAG_STATS_ENABLE: bool = True
    DAG_STATS_FILE: str = "output/dag_stats.json"
    DAG_STATS_INCLUDE_FAILED: bool = True
    DAG_STATS_WRITE_PER_ITEM: bool = False
    LOG_EXECUTOR_STATS: bool = True
    DAG_STATS_VALIDITY_ERRORS: bool = True

    # -- Kconfig.training --------------------------------------------------
    ENABLE_TRAINING: bool = False
    TRAINING_MODE: str = "TRAINING_MODE_DISABLED"

    TRAIN_DATASET_PATH: str = "dataset/WikiTQ-4k/train.jsonl"
    TRAIN_DEV_DATASET_PATH: str = ""
    TRAIN_USE_SEEDED_SPLIT: bool = True
    TRAIN_SPLIT_RATIO: float = 0.9
    TRAIN_SPLIT_SEED: int = -1
    TRAIN_MAX_TRAIN_EXAMPLES: int = 0
    TRAIN_MAX_DEV_EXAMPLES: int = 0
    OVERFIT_POC_NUM_EXAMPLES: int = 16
    OVERFIT_POC_SELECTION_MODE: str = "FIRST_N"
    OVERFIT_POC_INDICES_FILE: str = ""

    GLOBAL_SEED: int = 42
    TRAINING_SEED: int = -1
    DATALOADER_SEED: int = -1
    GENERATION_SEED: int = -1
    REWARD_SEED: int = -1
    EVAL_SEED: int = -1
    SAVE_RESOLVED_SEEDS: bool = True

    GRPO_ENABLE: bool = True
    GRPO_NUM_EPOCHS: int = 1
    GRPO_MAX_STEPS: int = 0
    GRPO_BATCH_SIZE_PROMPTS: int = 4
    GRPO_GROUP_SIZE: int = 8
    GRPO_MAX_NEW_TOKENS: int = 1024
    GRPO_TEMPERATURE: float = 0.7
    GRPO_TOP_P: float = 0.95
    GRPO_TOP_K: int = 0
    GRPO_CLIP_EPS: float = 0.2
    GRPO_KL_COEF: float = 0.05
    GRPO_LR: float = 5e-5
    GRPO_GRAD_ACCUM: int = 2
    GRPO_SAVE_EVERY_STEPS: int = 50
    GRPO_EVAL_EVERY_STEPS: int = 25

    REWARD_MODE: str = "REWARD_WEIGHTED_SUM"
    REWARD_WEIGHT_CORRECTNESS: float = 1.0
    REWARD_WEIGHT_VALIDITY: float = 0.5
    REWARD_WEIGHT_DEPTH: float = 0.1
    REWARD_WEIGHT_INVALID_PENALTY: float = 0.5
    REWARD_DEPTH_NORMALIZATION: str = "DIVIDE_BY_MAX_DEPTH"
    REWARD_MAX_DEPTH: int = 10
    REWARD_INVALID_IF_PARSE_FAILS: bool = True
    REWARD_CORRECTNESS_PARTIAL_CREDIT: bool = False

    TRAIN_USE_4BIT: bool = True
    TRAIN_BNB_4BIT_QUANT_TYPE: str = "nf4"
    TRAIN_BNB_4BIT_COMPUTE_DTYPE: str = "bfloat16"
    TRAIN_USE_GRADIENT_CHECKPOINTING: bool = True
    TRAIN_LORA_R: int = 16
    TRAIN_LORA_ALPHA: int = 32
    TRAIN_LORA_DROPOUT: float = 0.05
    TRAIN_LORA_TARGET_MODULES: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    TRAIN_LORA_BIAS: str = "none"
    TRAIN_LOAD_FROM_CHECKPOINT: bool = False
    TRAIN_CHECKPOINT_PATH: str = ""

    TRAIN_OUTPUT_DIR: str = "output/train"
    TRAIN_SAVE_LATEST: bool = True
    TRAIN_SAVE_BEST_BY: str = "eval_reward_mean"
    TRAIN_SAVE_MERGED_ADAPTER: bool = False
    TRAIN_SAVE_ADAPTER_ONLY: bool = True

    # -- Kconfig.training_stats --------------------------------------------
    TRAIN_STATS_ENABLE: bool = True
    TRAIN_STATS_FILE: str = ""
    TRAIN_STATS_PER_STEP_JSONL: str = ""
    TRAIN_STATS_PER_EVAL_JSONL: str = ""
    TRAIN_CURVES_DIR: str = ""
    TRAIN_CURVES_TEX_ENABLE: bool = True
    TRAIN_CURVES_COMPILE_ENABLE: bool = True
    TRAIN_CURVES_UPDATE_EVERY_STEPS: int = 25
    TRAIN_CURVES_COMPILE_EVERY_STEPS: int = 100
    TRAIN_CURVES_COMPILE_AT_END: bool = True
    TRAIN_CURVES_LATEXMK: bool = True
    TRAIN_CURVES_LATEX_ENGINE: str = "pdflatex"
    TRAIN_CURVES_PDFLATEX_TIMEOUT_SEC: int = 60
    TRAIN_CURVES_KEEP_LAST_N_POINTS: int = 0
    TRAIN_STATS_FLUSH_EVERY_STEPS: int = 10

    # -- Kconfig.test_trained ----------------------------------------------
    TEST_TRAINED_CHECKPOINT_SOURCE: str = "TEST_TRAINED_CHECKPOINT_BEST"
    TEST_TRAINED_CHECKPOINT_PATH: str = ""
    TEST_TRAINED_OUTPUT_DIR: str = ""
    TEST_TRAINED_RESULT_FILE: str = ""
    TEST_TRAINED_DAG_STATS_FILE: str = ""

    # -- Resolved fewshot file list ----------------------------------------
    fewshot_files: list = field(default_factory=list)
    resolved_dag_prompt_path: str = ""


# ---------------------------------------------------------------------------
# .config file parser
# ---------------------------------------------------------------------------

_SPLIT_MODE_MAP = {
    "SPLIT_MODE_SEEDED_RATIO": "seeded_ratio",
    "SPLIT_MODE_EXPLICIT_INDICES": "explicit_indices",
    "SPLIT_MODE_OVERFIT_POC": "overfit_poc",
}

# Fields parsed as float from Kconfig string type
_FLOAT_FIELDS = {
    "CONFIG_LLM_TEMPERATURE",
    "CONFIG_LLM_TOP_P",
    "CONFIG_LLM_FREQUENCY_PENALTY",
    "CONFIG_LLM_PRESENCE_PENALTY",
    "CONFIG_TRAIN_SPLIT_RATIO",
    "CONFIG_GRPO_TEMPERATURE",
    "CONFIG_GRPO_TOP_P",
    "CONFIG_GRPO_CLIP_EPS",
    "CONFIG_GRPO_KL_COEF",
    "CONFIG_GRPO_LR",
    "CONFIG_REWARD_WEIGHT_CORRECTNESS",
    "CONFIG_REWARD_WEIGHT_VALIDITY",
    "CONFIG_REWARD_WEIGHT_DEPTH",
    "CONFIG_REWARD_WEIGHT_INVALID_PENALTY",
    "CONFIG_TRAIN_LORA_DROPOUT",
}

# Fields parsed as int
_INT_FIELDS = {
    "CONFIG_SGLANG_PORT", "CONFIG_SGLANG_TP_SIZE", "CONFIG_SGLANG_CONTEXT_LENGTH",
    "CONFIG_SGLANG_HEALTH_TIMEOUT", "CONFIG_SGLANG_HEALTH_INTERVAL",
    "CONFIG_DAG_MAX_RETRIES", "CONFIG_NUM_SAMPLE_ROWS",
    "CONFIG_MAX_WORKERS", "CONFIG_LLM_MAX_OUTPUT_TOKENS", "CONFIG_LLM_RETRIES",
    "CONFIG_LLM_TOP_K", "CONFIG_LLM_SEED",
    "CONFIG_EMBEDDING_BATCH_SIZE", "CONFIG_EMBEDDING_RETRIES",
    "CONFIG_DAG_NODE_MAX_INFLIGHT", "CONFIG_RETRIEVAL_PARALLELISM",
    "CONFIG_RETRIEVAL_EMBED_BATCH_SIZE", "CONFIG_REASONING_PARALLELISM",
    "CONFIG_SGLANG_CLIENT_CONCURRENCY", "CONFIG_GLOBAL_EMBEDDING_CONCURRENCY",
    "CONFIG_LOG_LLM_CALLS_MAX_CHARS",
    "CONFIG_GLOBAL_SEED", "CONFIG_TRAINING_SEED", "CONFIG_DATALOADER_SEED",
    "CONFIG_GENERATION_SEED", "CONFIG_REWARD_SEED", "CONFIG_EVAL_SEED",
    "CONFIG_GRPO_NUM_EPOCHS", "CONFIG_GRPO_MAX_STEPS",
    "CONFIG_GRPO_BATCH_SIZE_PROMPTS", "CONFIG_GRPO_GROUP_SIZE",
    "CONFIG_GRPO_MAX_NEW_TOKENS", "CONFIG_GRPO_TOP_K",
    "CONFIG_GRPO_GRAD_ACCUM", "CONFIG_GRPO_SAVE_EVERY_STEPS",
    "CONFIG_GRPO_EVAL_EVERY_STEPS",
    "CONFIG_REWARD_MAX_DEPTH",
    "CONFIG_TRAIN_LORA_R", "CONFIG_TRAIN_LORA_ALPHA",
    "CONFIG_OVERFIT_POC_NUM_EXAMPLES",
    "CONFIG_TRAIN_SPLIT_SEED",
    "CONFIG_TRAIN_MAX_TRAIN_EXAMPLES", "CONFIG_TRAIN_MAX_DEV_EXAMPLES",
    "CONFIG_TRAIN_CURVES_UPDATE_EVERY_STEPS",
    "CONFIG_TRAIN_CURVES_COMPILE_EVERY_STEPS",
    "CONFIG_TRAIN_CURVES_PDFLATEX_TIMEOUT_SEC",
    "CONFIG_TRAIN_CURVES_KEEP_LAST_N_POINTS",
    "CONFIG_TRAIN_STATS_FLUSH_EVERY_STEPS",
}

# Fields parsed as bool (y/n)
_BOOL_FIELDS = {
    "CONFIG_TRUST_REMOTE_CODE", "CONFIG_USE_DAG",
    "CONFIG_LOG_LLM_PROMPTS", "CONFIG_LOG_LLM_RESPONSES",
    "CONFIG_LOG_LLM_CALLS_PER_ITEM",
    "CONFIG_SKIP_EXISTING",
    "CONFIG_DAG_STATS_ENABLE", "CONFIG_DAG_STATS_INCLUDE_FAILED",
    "CONFIG_DAG_STATS_WRITE_PER_ITEM", "CONFIG_LOG_EXECUTOR_STATS",
    "CONFIG_DAG_STATS_VALIDITY_ERRORS",
    "CONFIG_ENABLE_TRAINING", "CONFIG_GRPO_ENABLE",
    "CONFIG_TRAIN_USE_SEEDED_SPLIT",
    "CONFIG_SAVE_RESOLVED_SEEDS",
    "CONFIG_REWARD_INVALID_IF_PARSE_FAILS",
    "CONFIG_REWARD_CORRECTNESS_PARTIAL_CREDIT",
    "CONFIG_TRAIN_USE_4BIT", "CONFIG_TRAIN_USE_GRADIENT_CHECKPOINTING",
    "CONFIG_TRAIN_LOAD_FROM_CHECKPOINT",
    "CONFIG_TRAIN_SAVE_LATEST", "CONFIG_TRAIN_SAVE_MERGED_ADAPTER",
    "CONFIG_TRAIN_SAVE_ADAPTER_ONLY",
    "CONFIG_TRAIN_STATS_ENABLE",
    "CONFIG_TRAIN_CURVES_TEX_ENABLE", "CONFIG_TRAIN_CURVES_COMPILE_ENABLE",
    "CONFIG_TRAIN_CURVES_COMPILE_AT_END", "CONFIG_TRAIN_CURVES_LATEXMK",
}

# Map from CONFIG_* key to Config dataclass field name
_KEY_TO_FIELD: Dict[str, str] = {}


def _build_key_to_field() -> None:
    """Build the CONFIG_* → field mapping lazily."""
    if _KEY_TO_FIELD:
        return
    for f_name in Config.__dataclass_fields__:
        kconfig_key = "CONFIG_" + f_name
        _KEY_TO_FIELD[kconfig_key] = f_name


def _parse_raw(config_path: str) -> dict[str, str]:
    """Parse a ``.config`` file into raw string key-value pairs."""
    raw: dict[str, str] = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            raw[key] = value
    return raw


def _parse_index_list(raw_value: str, symbol_name: str) -> list[int]:
    """Parse a comma-separated index string into a sorted, deduplicated int list."""
    if not raw_value.strip():
        return []
    tokens = raw_value.split(",")
    seen: set[int] = set()
    result: list[int] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            raise ValueError(
                f"Non-integer token '{token}' in {symbol_name}"
            )
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    result.sort()
    return result


# Index-list Kconfig symbols and their Config field names
_INDEX_SYMBOLS = [
    ("CONFIG_SPLIT_TRAIN_WIKITQ_4K_INDICES", "SPLIT_TRAIN_WIKITQ_4K_INDICES"),
    ("CONFIG_SPLIT_TRAIN_WIKITQ_PLUS_INDICES", "SPLIT_TRAIN_WIKITQ_PLUS_INDICES"),
    ("CONFIG_SPLIT_TRAIN_SCALABILITY_INDICES", "SPLIT_TRAIN_SCALABILITY_INDICES"),
    ("CONFIG_SPLIT_VALID_WIKITQ_4K_INDICES", "SPLIT_VALID_WIKITQ_4K_INDICES"),
    ("CONFIG_SPLIT_VALID_WIKITQ_PLUS_INDICES", "SPLIT_VALID_WIKITQ_PLUS_INDICES"),
    ("CONFIG_SPLIT_VALID_SCALABILITY_INDICES", "SPLIT_VALID_SCALABILITY_INDICES"),
    ("CONFIG_SPLIT_TEST_WIKITQ_4K_INDICES", "SPLIT_TEST_WIKITQ_4K_INDICES"),
    ("CONFIG_SPLIT_TEST_WIKITQ_PLUS_INDICES", "SPLIT_TEST_WIKITQ_PLUS_INDICES"),
    ("CONFIG_SPLIT_TEST_SCALABILITY_INDICES", "SPLIT_TEST_SCALABILITY_INDICES"),
]

# Persistent output/cache paths resolved against CONFIG_PERSISTENT_ROOT
_PERSISTENT_PATH_FIELDS = [
    "MODEL_CACHE_DIR",
    "RESULT_FILE",
    "EMBEDDING_CACHE",
    "LOG_FILE",
    "DAG_STATS_FILE",
    "LLM_CALLS_SIDEFILE",
    "TRAIN_OUTPUT_DIR",
    "TRAIN_STATS_FILE",
    "TRAIN_STATS_PER_STEP_JSONL",
    "TRAIN_STATS_PER_EVAL_JSONL",
    "TRAIN_CURVES_DIR",
    "TRAIN_CHECKPOINT_PATH",
    "OVERFIT_POC_INDICES_FILE",
    "TEST_TRAINED_OUTPUT_DIR",
    "TEST_TRAINED_RESULT_FILE",
    "TEST_TRAINED_DAG_STATS_FILE",
]

# Prompt paths resolved against PROJECT_DIR
_PROMPT_PATH_FIELDS = [
    "FINAL_REASONING_PROMPT",
    "NOPLAN_REASONING_PROMPT",
    "ROW_PROMPT",
    "COL_PROMPT",
    "SCHEMA_LINKING_PROMPT",
    "PLAN_PROMPT",
]

# Dataset paths resolved against AIXELASK_ROOT
_DATASET_PATH_FIELDS = [
    "INFERENCE_DATASET_PATH",
    "TRAIN_DATASET_PATH",
    "TRAIN_DEV_DATASET_PATH",
]


# ---------------------------------------------------------------------------
# Core load function
# ---------------------------------------------------------------------------

def load_config(
    config_path: str,
    overrides: Optional[list[str]] = None,
) -> Config:
    """Parse a ``.config`` file and return a fully resolved ``Config``.

    Parameters
    ----------
    config_path:
        Path to the ``.config`` file.
    overrides:
        List of ``"KEY=VALUE"`` strings from ``--override`` CLI args.
    """
    _build_key_to_field()

    # 1. Parse raw key-value pairs
    raw = _parse_raw(config_path)

    # 2. Apply CLI overrides
    if overrides:
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Invalid override (no '='): {ov}")
            key, _, value = ov.partition("=")
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            raw[key] = value

    # Build Config object by mapping raw values to typed fields
    cfg = Config()
    cfg.PROJECT_DIR = PROJECT_DIR
    cfg.AIXELASK_ROOT = AIXELASK_ROOT
    cfg.UPSTREAM_SOURCE_ROOT = UPSTREAM_SOURCE_ROOT
    cfg.UPSTREAM_SCRIPTS_DIR = UPSTREAM_SCRIPTS_DIR

    for kconfig_key, raw_value in raw.items():
        field_name = _KEY_TO_FIELD.get(kconfig_key)
        if field_name is None:
            continue

        if kconfig_key in _BOOL_FIELDS:
            setattr(cfg, field_name, raw_value.lower() in ("y", "yes", "true", "1"))
        elif kconfig_key in _INT_FIELDS:
            setattr(cfg, field_name, int(raw_value))
        elif kconfig_key in _FLOAT_FIELDS:
            setattr(cfg, field_name, float(raw_value))
        else:
            setattr(cfg, field_name, raw_value)

    # Parse split mode
    raw_split = raw.get("CONFIG_SPLIT_MODE", "SPLIT_MODE_SEEDED_RATIO")
    cfg.SPLIT_MODE = _SPLIT_MODE_MAP.get(raw_split, raw_split)

    # LOG_LLM_RESPONSES forced on when LOG_LLM_PROMPTS is on
    if cfg.LOG_LLM_PROMPTS:
        cfg.LOG_LLM_RESPONSES = True

    # Seed inheritance
    for seed_field in (
        "TRAINING_SEED", "DATALOADER_SEED", "GENERATION_SEED",
        "REWARD_SEED", "EVAL_SEED",
    ):
        if getattr(cfg, seed_field) == -1:
            setattr(cfg, seed_field, cfg.GLOBAL_SEED)
    if cfg.TRAIN_SPLIT_SEED == -1:
        cfg.TRAIN_SPLIT_SEED = cfg.GLOBAL_SEED

    # Parse index lists (always, for syntax checking even in seeded_ratio mode)
    for kconfig_key, field_name in _INDEX_SYMBOLS:
        raw_value = raw.get(kconfig_key, "")
        setattr(cfg, field_name, _parse_index_list(raw_value, kconfig_key))

    # 3. Derive training output sub-paths from TRAIN_OUTPUT_DIR
    if not cfg.TRAIN_STATS_FILE:
        cfg.TRAIN_STATS_FILE = os.path.join(cfg.TRAIN_OUTPUT_DIR, "train_stats_summary.json")
    if not cfg.TRAIN_STATS_PER_STEP_JSONL:
        cfg.TRAIN_STATS_PER_STEP_JSONL = os.path.join(cfg.TRAIN_OUTPUT_DIR, "train_steps.jsonl")
    if not cfg.TRAIN_STATS_PER_EVAL_JSONL:
        cfg.TRAIN_STATS_PER_EVAL_JSONL = os.path.join(cfg.TRAIN_OUTPUT_DIR, "train_evals.jsonl")
    if not cfg.TRAIN_CURVES_DIR:
        cfg.TRAIN_CURVES_DIR = os.path.join(cfg.TRAIN_OUTPUT_DIR, "curves")

    # Derive test-trained output paths
    source_label = _checkpoint_source_label(cfg.TEST_TRAINED_CHECKPOINT_SOURCE)
    if not cfg.TEST_TRAINED_OUTPUT_DIR:
        cfg.TEST_TRAINED_OUTPUT_DIR = os.path.join(
            cfg.TRAIN_OUTPUT_DIR, f"test_{source_label}"
        )
    if not cfg.TEST_TRAINED_RESULT_FILE:
        cfg.TEST_TRAINED_RESULT_FILE = os.path.join(
            cfg.TEST_TRAINED_OUTPUT_DIR, "results.jsonl"
        )
    if not cfg.TEST_TRAINED_DAG_STATS_FILE:
        cfg.TEST_TRAINED_DAG_STATS_FILE = os.path.join(
            cfg.TEST_TRAINED_OUTPUT_DIR, "dag_stats.json"
        )
    if not cfg.TEST_TRAINED_CHECKPOINT_PATH:
        ckpt_dir = os.path.join(cfg.TRAIN_OUTPUT_DIR, "checkpoints")
        cfg.TEST_TRAINED_CHECKPOINT_PATH = os.path.join(ckpt_dir, source_label)

    # 4. Path resolution — persistent paths against PERSISTENT_ROOT
    for field_name in _PERSISTENT_PATH_FIELDS:
        val = getattr(cfg, field_name, "")
        if val and not os.path.isabs(val):
            setattr(cfg, field_name, os.path.join(cfg.PERSISTENT_ROOT, val))

    # Prompt paths against PROJECT_DIR
    for field_name in _PROMPT_PATH_FIELDS:
        val = getattr(cfg, field_name, "")
        if val and not os.path.isabs(val):
            setattr(cfg, field_name, os.path.join(cfg.PROJECT_DIR, val))

    # Dataset paths against AIXELASK_ROOT
    for field_name in _DATASET_PATH_FIELDS:
        val = getattr(cfg, field_name, "")
        if val and not os.path.isabs(val):
            setattr(cfg, field_name, os.path.join(cfg.AIXELASK_ROOT, val))

    # Resolve DAG prompt path
    if cfg.PLAN_PROMPT:
        cfg.resolved_dag_prompt_path = cfg.PLAN_PROMPT
    else:
        rel = _DAG_PROMPT_PATHS.get(cfg.DAG_PROMPT_VARIANT, "prompt/get_dag.md")
        cfg.resolved_dag_prompt_path = os.path.join(cfg.PROJECT_DIR, rel)

    # Resolve fewshot files
    fewshot_rels = _FEWSHOT_FILES.get(
        cfg.FEWSHOT_VARIANT, _FEWSHOT_FILES["FEWSHOT_STANDARD_ALL3"]
    )
    cfg.fewshot_files = [
        os.path.join(cfg.PROJECT_DIR, rel) for rel in fewshot_rels
    ]

    # 5. Validation
    _validate(cfg)

    # 6. Set environment variables
    os.environ["HF_HOME"] = cfg.MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = cfg.MODEL_CACHE_DIR
    os.environ["TMPDIR"] = cfg.EPHEMERAL_TMPDIR

    # Create parent directories for output paths
    for field_name in _PERSISTENT_PATH_FIELDS:
        val = getattr(cfg, field_name, "")
        if val:
            parent = os.path.dirname(val)
            if parent:
                os.makedirs(parent, exist_ok=True)

    return cfg


def _checkpoint_source_label(source: str) -> str:
    """Map checkpoint source Kconfig choice to a short label."""
    mapping = {
        "TEST_TRAINED_CHECKPOINT_BEST": "best",
        "TEST_TRAINED_CHECKPOINT_LATEST": "latest",
        "TEST_TRAINED_CHECKPOINT_MERGED": "merged",
        "TEST_TRAINED_CHECKPOINT_EXPLICIT_PATH": "explicit",
    }
    return mapping.get(source, "best")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(cfg: Config) -> None:
    """Run all startup validations. Raises ValueError on failure."""

    # TabFact guardrail
    for path_field in _DATASET_PATH_FIELDS:
        val = getattr(cfg, path_field, "")
        if val and "tabfact" in val.lower():
            raise ValueError(
                "TabFact+ is not supported in nlp_project; please choose "
                "WikiTQ-4k/WikiTQ+/Scalability or provide a non-TabFact "
                "custom dataset."
            )

    # Scalability requires explicit_indices mode
    if cfg.DATASET in ("DATASET_SCALABILITY",) and cfg.SPLIT_MODE == "seeded_ratio":
        raise ValueError(
            "Scalability dataset requires SPLIT_MODE_EXPLICIT_INDICES "
            "(multi-file dataset)."
        )

    # Split mode compatibility with training mode
    if cfg.TRAINING_MODE == "TRAINING_MODE_OVERFIT_POC":
        if cfg.SPLIT_MODE != "overfit_poc":
            logger.warning(
                "TRAINING_MODE_OVERFIT_POC forces SPLIT_MODE to overfit_poc "
                "(was %s)", cfg.SPLIT_MODE,
            )
            cfg.SPLIT_MODE = "overfit_poc"
    elif cfg.TRAINING_MODE == "TRAINING_MODE_GRPO":
        if cfg.SPLIT_MODE == "overfit_poc":
            raise ValueError(
                "SPLIT_MODE_OVERFIT_POC is valid only with "
                "TRAINING_MODE_OVERFIT_POC, not TRAINING_MODE_GRPO."
            )

    # Fewshot file existence check
    for fpath in cfg.fewshot_files:
        if not os.path.isfile(fpath):
            raise ValueError(f"Few-shot file not found: {fpath}")

    # Cross-split overlap check for explicit_indices mode
    if cfg.SPLIT_MODE == "explicit_indices":
        _validate_explicit_indices(cfg)


def _validate_explicit_indices(cfg: Config) -> None:
    """Validate explicit-index splits: bounds and cross-split disjointness."""

    dataset_split_map = {
        "wikitq_4k": {
            "train": ("SPLIT_TRAIN_WIKITQ_4K_INDICES", "train"),
            "valid": ("SPLIT_VALID_WIKITQ_4K_INDICES", "valid"),
            "test":  ("SPLIT_TEST_WIKITQ_4K_INDICES", "test"),
        },
        "wikitq_plus": {
            "train": ("SPLIT_TRAIN_WIKITQ_PLUS_INDICES", "train"),
            "valid": ("SPLIT_VALID_WIKITQ_PLUS_INDICES", "valid"),
            "test":  ("SPLIT_TEST_WIKITQ_PLUS_INDICES", "test"),
        },
        "scalability": {
            "train": ("SPLIT_TRAIN_SCALABILITY_INDICES", "all"),
            "valid": ("SPLIT_VALID_SCALABILITY_INDICES", "all"),
            "test":  ("SPLIT_TEST_SCALABILITY_INDICES", "all"),
        },
    }

    from src.training.dataset_registry import count_examples

    # Collect all (source_dataset, source_file, source_index) triples per split
    split_triples: dict[str, set[tuple[str, str, int]]] = {
        "train": set(), "valid": set(), "test": set(),
    }

    for dataset_key, split_info in dataset_split_map.items():
        for split_name, (field_name, canonical_split) in split_info.items():
            indices = getattr(cfg, field_name)
            if not indices:
                continue

            # Bounds check
            total = count_examples(dataset_key, canonical_split, cfg.AIXELASK_ROOT)
            for idx in indices:
                if idx < 0 or idx >= total:
                    raise ValueError(
                        f"Index {idx} out of range for {dataset_key} "
                        f"(max {total - 1})"
                    )

            # Determine source_file for overlap tracking
            if dataset_key == "scalability":
                source_file = "all"
            else:
                from src.training.dataset_registry import DATASET_REGISTRY
                rel_path = DATASET_REGISTRY[dataset_key][canonical_split]
                source_file = os.path.basename(rel_path)

            for idx in indices:
                split_triples[split_name].add(
                    (dataset_key, source_file, idx)
                )

    # Cross-split disjointness
    splits = ["train", "valid", "test"]
    overlaps: list[str] = []
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            common = split_triples[s1] & split_triples[s2]
            for triple in sorted(common):
                overlaps.append(
                    f"  ({triple[0]}, {triple[1]}, {triple[2]}) in "
                    f"both {s1} and {s2}"
                )
    if overlaps:
        raise ValueError(
            "Cross-split overlap detected:\n" + "\n".join(overlaps)
        )


# ---------------------------------------------------------------------------
# bootstrap_upstream_imports
# ---------------------------------------------------------------------------

def bootstrap_upstream_imports(config: Config) -> None:
    """Add upstream AixelAsk directories to ``sys.path``.

    Must be called before any ``patch_*.init_patches()`` call or any
    direct upstream import. Idempotent: safe to call multiple times.
    """
    for p in (config.UPSTREAM_SOURCE_ROOT, config.UPSTREAM_SCRIPTS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# CLI argument parsing helper
# ---------------------------------------------------------------------------

def build_arg_parser(description: str = "NLP Project") -> argparse.ArgumentParser:
    """Build a standard CLI argument parser with ``--config`` and ``--override``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=".config",
        help="Path to the .config file (default: .config)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config key: --override KEY=VALUE (repeatable)",
    )
    return parser
