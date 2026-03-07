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

        assert isinstance(cfg.SERVER_PORT, int)
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
        skip fewshot existence)."""
        from src.config import load_config

        with mock.patch("src.config._validate"):
            cfg = load_config(DEFCONFIG)

        assert cfg.SERVER_PORT == 30000
        assert cfg.PROJECT_DIR == PROJECT_DIR


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

    def test_split_yaml_path_resolved_against_project_dir(self, tmp_path):
        """SPLIT_YAML_PATH resolves against PROJECT_DIR."""
        from src.config import load_config

        path = _write_config(tmp_path, _minimal_config_text())

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.SPLIT_YAML_PATH.startswith(cfg.PROJECT_DIR)
        assert cfg.SPLIT_YAML_PATH.endswith("train_valid_test.yaml")
