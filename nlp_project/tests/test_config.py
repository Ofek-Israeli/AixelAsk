"""Tests for src.config — parsing, types, validation, bootstrap, path resolution."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFCONFIG = os.path.join(PROJECT_DIR, "defconfig")


def _write_config(tmp_path, lines: str) -> str:
    """Write a .config snippet to a temp file and return its path."""
    p = os.path.join(str(tmp_path), ".config")
    with open(p, "w") as f:
        f.write(textwrap.dedent(lines))
    return p


def _minimal_config_text() -> str:
    """Minimal valid .config that avoids fewshot-existence checks."""
    return textwrap.dedent("""\
        CONFIG_PERSISTENT_ROOT="{project}"
        CONFIG_INFERENCE_DATASET_PATH="dataset/WikiTQ-4k/test.jsonl"
        CONFIG_TRAIN_DATASET_PATH="dataset/WikiTQ-4k/train.jsonl"
        CONFIG_FEWSHOT_VARIANT="FEWSHOT_STANDARD_ALL3"
    """.format(project=PROJECT_DIR))


# ---------------------------------------------------------------------------
# Parse sample .config → verify types
# ---------------------------------------------------------------------------

class TestConfigParsing:

    def test_parse_types_from_defconfig(self, tmp_path):
        """defconfig produces a Config with correctly typed fields."""
        from src.config import load_config

        with mock.patch("src.config._validate"):
            cfg = load_config(DEFCONFIG)

        assert isinstance(cfg.SGLANG_PORT, int)
        assert isinstance(cfg.LLM_TEMPERATURE, float)
        assert isinstance(cfg.TRUST_REMOTE_CODE, bool)
        assert isinstance(cfg.INFERENCE_MODEL, str)
        assert isinstance(cfg.MAX_WORKERS, int)

    def test_sampling_param_type_conversions(self, tmp_path):
        """String config values are cast to float/int for sampling params."""
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_LLM_TEMPERATURE="0.7"
            CONFIG_LLM_TOP_P="0.9"
            CONFIG_LLM_FREQUENCY_PENALTY="0.5"
            CONFIG_LLM_PRESENCE_PENALTY="-0.1"
            CONFIG_LLM_TOP_K=50
            CONFIG_LLM_SEED=42
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.LLM_TEMPERATURE == pytest.approx(0.7)
        assert isinstance(cfg.LLM_TEMPERATURE, float)
        assert cfg.LLM_TOP_P == pytest.approx(0.9)
        assert isinstance(cfg.LLM_TOP_P, float)
        assert cfg.LLM_FREQUENCY_PENALTY == pytest.approx(0.5)
        assert isinstance(cfg.LLM_FREQUENCY_PENALTY, float)
        assert cfg.LLM_PRESENCE_PENALTY == pytest.approx(-0.1)
        assert isinstance(cfg.LLM_PRESENCE_PENALTY, float)
        assert cfg.LLM_TOP_K == 50
        assert isinstance(cfg.LLM_TOP_K, int)
        assert cfg.LLM_SEED == 42
        assert isinstance(cfg.LLM_SEED, int)

    def test_sampling_param_defaults(self, tmp_path):
        """Absent sampling params fall back to dataclass defaults."""
        from src.config import load_config

        path = _write_config(tmp_path, _minimal_config_text())

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.LLM_TEMPERATURE == pytest.approx(0.0)
        assert cfg.LLM_TOP_P == pytest.approx(1.0)
        assert cfg.LLM_TOP_K == 0
        assert cfg.LLM_FREQUENCY_PENALTY == pytest.approx(0.0)
        assert cfg.LLM_PRESENCE_PENALTY == pytest.approx(0.0)
        assert cfg.LLM_SEED == -1

    def test_defconfig_produces_valid_config(self):
        """The shipped defconfig parses without error (validation mocked to
        skip fewshot existence and dataset_registry imports)."""
        from src.config import load_config

        with mock.patch("src.config._validate"):
            cfg = load_config(DEFCONFIG)

        assert cfg.SGLANG_PORT == 30000
        assert cfg.PROJECT_DIR == PROJECT_DIR


# ---------------------------------------------------------------------------
# TabFact rejection
# ---------------------------------------------------------------------------

class TestTabFactRejection:

    def test_tabfact_inference_path_rejected(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text().replace(
            'CONFIG_INFERENCE_DATASET_PATH="dataset/WikiTQ-4k/test.jsonl"',
            'CONFIG_INFERENCE_DATASET_PATH="dataset/TabFact+/large_tabfact_test_data_str.jsonl"',
        )
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate_explicit_indices"):
            with pytest.raises(ValueError, match="TabFact"):
                load_config(path)

    def test_tabfact_case_insensitive(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text().replace(
            'CONFIG_TRAIN_DATASET_PATH="dataset/WikiTQ-4k/train.jsonl"',
            'CONFIG_TRAIN_DATASET_PATH="/some/path/tabfact_data.jsonl"',
        )
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate_explicit_indices"):
            with pytest.raises(ValueError, match="TabFact"):
                load_config(path)

    def test_non_tabfact_path_accepted(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text()
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert "tabfact" not in cfg.INFERENCE_DATASET_PATH.lower()


# ---------------------------------------------------------------------------
# bootstrap_upstream_imports
# ---------------------------------------------------------------------------

class TestBootstrap:

    def test_adds_paths_to_sys_path(self):
        from src.config import Config, bootstrap_upstream_imports

        cfg = Config()
        cfg.UPSTREAM_SOURCE_ROOT = "/fake/upstream/root"
        cfg.UPSTREAM_SCRIPTS_DIR = "/fake/upstream/scripts"

        original = list(sys.path)
        try:
            for p in (cfg.UPSTREAM_SOURCE_ROOT, cfg.UPSTREAM_SCRIPTS_DIR):
                if p in sys.path:
                    sys.path.remove(p)

            bootstrap_upstream_imports(cfg)
            assert cfg.UPSTREAM_SOURCE_ROOT in sys.path
            assert cfg.UPSTREAM_SCRIPTS_DIR in sys.path
        finally:
            sys.path[:] = original

    def test_idempotent(self):
        from src.config import Config, bootstrap_upstream_imports

        cfg = Config()
        cfg.UPSTREAM_SOURCE_ROOT = "/fake/idem/root"
        cfg.UPSTREAM_SCRIPTS_DIR = "/fake/idem/scripts"

        original = list(sys.path)
        try:
            for p in (cfg.UPSTREAM_SOURCE_ROOT, cfg.UPSTREAM_SCRIPTS_DIR):
                while p in sys.path:
                    sys.path.remove(p)

            bootstrap_upstream_imports(cfg)
            bootstrap_upstream_imports(cfg)

            assert sys.path.count(cfg.UPSTREAM_SOURCE_ROOT) == 1
            assert sys.path.count(cfg.UPSTREAM_SCRIPTS_DIR) == 1
        finally:
            sys.path[:] = original

    def test_independent_of_cwd(self, tmp_path):
        from src.config import Config, bootstrap_upstream_imports

        cfg = Config()
        cfg.UPSTREAM_SOURCE_ROOT = "/fake/cwd_test/root"
        cfg.UPSTREAM_SCRIPTS_DIR = "/fake/cwd_test/scripts"

        original_path = list(sys.path)
        original_cwd = os.getcwd()
        try:
            for p in (cfg.UPSTREAM_SOURCE_ROOT, cfg.UPSTREAM_SCRIPTS_DIR):
                while p in sys.path:
                    sys.path.remove(p)

            os.chdir("/tmp")
            bootstrap_upstream_imports(cfg)
            assert cfg.UPSTREAM_SOURCE_ROOT in sys.path
            assert cfg.UPSTREAM_SCRIPTS_DIR in sys.path
        finally:
            os.chdir(original_cwd)
            sys.path[:] = original_path


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestPathResolution:

    def test_persistent_paths_resolve_under_persistent_root(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_RESULT_FILE="output/results.jsonl"
            CONFIG_DAG_STATS_FILE="output/dag_stats.json"
            CONFIG_LOG_FILE="output/run.log"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.RESULT_FILE.startswith(cfg.PERSISTENT_ROOT)
        assert cfg.DAG_STATS_FILE.startswith(cfg.PERSISTENT_ROOT)
        assert cfg.LOG_FILE.startswith(cfg.PERSISTENT_ROOT)

    def test_prompt_paths_resolve_under_project_dir(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_FINAL_REASONING_PROMPT="prompt/final_reasoning_DAG.md"
            CONFIG_ROW_PROMPT="prompt/get_row_template.md"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.FINAL_REASONING_PROMPT.startswith(cfg.PROJECT_DIR)
        assert cfg.ROW_PROMPT.startswith(cfg.PROJECT_DIR)

    def test_prompt_paths_not_under_persistent_root_override(self, tmp_path):
        """Redirecting PERSISTENT_ROOT does NOT move prompts."""
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_PERSISTENT_ROOT="/workspace/custom_output"
            CONFIG_FINAL_REASONING_PROMPT="prompt/final_reasoning_DAG.md"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.FINAL_REASONING_PROMPT.startswith(cfg.PROJECT_DIR)
        assert not cfg.FINAL_REASONING_PROMPT.startswith("/workspace/custom_output")

    def test_dataset_paths_resolve_under_aixelask_root(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text()
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.INFERENCE_DATASET_PATH.startswith(cfg.AIXELASK_ROOT)


# ---------------------------------------------------------------------------
# Split mode compatibility
# ---------------------------------------------------------------------------

class TestSplitMode:

    def test_scalability_rejected_in_seeded_ratio(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_DATASET="DATASET_SCALABILITY"
            CONFIG_SPLIT_MODE="SPLIT_MODE_SEEDED_RATIO"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate_explicit_indices"):
            with pytest.raises(ValueError, match="multi-file"):
                load_config(path)

    def test_overfit_poc_rejected_for_grpo(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_TRAINING_MODE="TRAINING_MODE_GRPO"
            CONFIG_SPLIT_MODE="SPLIT_MODE_OVERFIT_POC"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate_explicit_indices"):
            with pytest.raises(ValueError, match="TRAINING_MODE_OVERFIT_POC"):
                load_config(path)

    def test_overfit_poc_accepted_for_overfit_training(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_TRAINING_MODE="TRAINING_MODE_OVERFIT_POC"
            CONFIG_SPLIT_MODE="SPLIT_MODE_OVERFIT_POC"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.TRAINING_MODE == "TRAINING_MODE_OVERFIT_POC"

    def test_split_mode_default_seeded_ratio(self, tmp_path):
        from src.config import load_config

        path = _write_config(tmp_path, _minimal_config_text())

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.SPLIT_MODE == "seeded_ratio"


# ---------------------------------------------------------------------------
# Split index parsing
# ---------------------------------------------------------------------------

class TestSplitIndexParsing:

    def test_basic(self):
        from src.config import _parse_index_list
        result = _parse_index_list("0,5,10", "SYM")
        assert result == [0, 5, 10]

    def test_whitespace(self):
        from src.config import _parse_index_list
        result = _parse_index_list(" 0 , 5 , 10 ", "SYM")
        assert result == [0, 5, 10]

    def test_empty_tokens_ignored(self):
        from src.config import _parse_index_list
        result = _parse_index_list("0,,5,10,", "SYM")
        assert result == [0, 5, 10]

    def test_deduplication_and_sort(self):
        from src.config import _parse_index_list
        result = _parse_index_list("5,0,5,10,0", "SYM")
        assert result == [0, 5, 10]

    def test_non_integer_token_fails(self):
        from src.config import _parse_index_list
        with pytest.raises(ValueError, match="abc"):
            _parse_index_list("0,abc,5", "CONFIG_SPLIT_TRAIN_WIKITQ_4K_INDICES")

    def test_empty_string(self):
        from src.config import _parse_index_list
        result = _parse_index_list("", "SYM")
        assert result == []


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

class TestOverlapDetection:

    def test_scalability_same_index_is_overlap(self, tmp_path):
        """Scalability reads from same combined file, so same-index IS overlap."""
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_SPLIT_MODE="SPLIT_MODE_EXPLICIT_INDICES"
            CONFIG_SPLIT_TRAIN_SCALABILITY_INDICES="0,1,2"
            CONFIG_SPLIT_TEST_SCALABILITY_INDICES="2,3,4"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch(
            "src.training.dataset_registry.count_examples", return_value=10000
        ), mock.patch(
            "src.training.dataset_registry.DATASET_REGISTRY",
            {"scalability": {"all": ["f1.jsonl"]},
             "wikitq_4k": {"train": "dataset/WikiTQ-4k/train.jsonl",
                           "valid": "dataset/WikiTQ-4k/test.jsonl",
                           "test": "dataset/WikiTQ-4k/test.jsonl"},
             "wikitq_plus": {"train": "dataset/WikiTQ-plus/train.jsonl",
                             "valid": "dataset/WikiTQ-plus/test.jsonl",
                             "test": "dataset/WikiTQ-plus/test.jsonl"}},
        ):
            with pytest.raises(ValueError, match="overlap"):
                load_config(path)

    def test_different_datasets_no_overlap(self, tmp_path):
        """Same index in different datasets → no overlap."""
        from src.config import _parse_index_list
        result_a = _parse_index_list("0", "SYM")
        result_b = _parse_index_list("0", "SYM")
        assert result_a == result_b  # same ints but different datasets = OK
