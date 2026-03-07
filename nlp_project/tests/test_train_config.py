"""Tests for src.training.train_config — seed inheritance, mode selection, defconfigs."""

from __future__ import annotations

import os
import tempfile
import textwrap
from unittest import mock

import pytest


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFCONFIG_TRAIN_GRPO = os.path.join(PROJECT_DIR, "defconfig.train_grpo")
DEFCONFIG_TRAIN_OVERFIT_POC = os.path.join(PROJECT_DIR, "defconfig.train_overfit_poc")


def _write_config(tmp_path, lines: str) -> str:
    p = os.path.join(str(tmp_path), ".config")
    with open(p, "w") as f:
        f.write(textwrap.dedent(lines))
    return p


def _minimal_config_text() -> str:
    return textwrap.dedent("""\
        CONFIG_PERSISTENT_ROOT="{project}"
        CONFIG_FEWSHOT_VARIANT="FEWSHOT_STANDARD_ALL3"
    """.format(project=PROJECT_DIR))


# ---------------------------------------------------------------------------
# Seed inheritance
# ---------------------------------------------------------------------------

class TestSeedInheritance:

    def test_global_seed_only(self, tmp_path):
        """All stage seeds at -1 → inherit GLOBAL_SEED=42."""
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_GLOBAL_SEED=42
            CONFIG_TRAINING_SEED=-1
            CONFIG_DATALOADER_SEED=-1
            CONFIG_GENERATION_SEED=-1
            CONFIG_REWARD_SEED=-1
            CONFIG_EVAL_SEED=-1
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.TRAINING_SEED == 42
        assert cfg.DATALOADER_SEED == 42
        assert cfg.GENERATION_SEED == 42
        assert cfg.REWARD_SEED == 42
        assert cfg.EVAL_SEED == 42

    def test_seed_override(self, tmp_path):
        """GENERATION_SEED=99 overrides GLOBAL_SEED=42."""
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_GLOBAL_SEED=42
            CONFIG_GENERATION_SEED=99
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.GENERATION_SEED == 99
        assert cfg.TRAINING_SEED == 42


# ---------------------------------------------------------------------------
# Training mode selection
# ---------------------------------------------------------------------------

class TestTrainingMode:

    @pytest.mark.parametrize("mode_val,expected_mode", [
        ("TRAINING_MODE_DISABLED", "TRAINING_MODE_DISABLED"),
        ("TRAINING_MODE_GRPO", "TRAINING_MODE_GRPO"),
        ("TRAINING_MODE_OVERFIT_POC", "TRAINING_MODE_OVERFIT_POC"),
    ])
    def test_mode_selection(self, tmp_path, mode_val, expected_mode):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent(f"""\
            CONFIG_TRAINING_MODE="{mode_val}"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.TRAINING_MODE == expected_mode


# ---------------------------------------------------------------------------
# Float parsing
# ---------------------------------------------------------------------------

class TestFloatParsing:

    def test_grpo_temperature_float(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_GRPO_TEMPERATURE="0.7"
            CONFIG_GRPO_LR="5e-5"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.GRPO_TEMPERATURE == pytest.approx(0.7)
        assert isinstance(cfg.GRPO_TEMPERATURE, float)
        assert cfg.GRPO_LR == pytest.approx(5e-5)
        assert isinstance(cfg.GRPO_LR, float)

    def test_reward_weight_float(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_REWARD_WEIGHT_CORRECTNESS="1.0"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.REWARD_WEIGHT_CORRECTNESS == pytest.approx(1.0)
        assert isinstance(cfg.REWARD_WEIGHT_CORRECTNESS, float)


# ---------------------------------------------------------------------------
# LoRA target modules parsing
# ---------------------------------------------------------------------------

class TestLoRAModules:

    def test_lora_target_modules_parsed(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_TRAIN_LORA_TARGET_MODULES="q_proj,k_proj,v_proj"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        modules = cfg.TRAIN_LORA_TARGET_MODULES.split(",")
        assert modules == ["q_proj", "k_proj", "v_proj"]
        assert len(modules) == 3


# ---------------------------------------------------------------------------
# defconfig validation
# ---------------------------------------------------------------------------

class TestDefconfigs:

    def test_defconfig_train_grpo_valid(self):
        from src.config import load_config

        with mock.patch("src.config._validate"):
            cfg = load_config(DEFCONFIG_TRAIN_GRPO)

        assert cfg.SGLANG_PORT == 30000

    def test_defconfig_train_overfit_poc_valid(self):
        from src.config import load_config

        with mock.patch("src.config._validate"):
            cfg = load_config(DEFCONFIG_TRAIN_OVERFIT_POC)

        assert cfg.SGLANG_PORT == 30000


# ---------------------------------------------------------------------------
# TrainConfig from_config
# ---------------------------------------------------------------------------

class TestTrainConfigFromConfig:

    def test_proxy_attribute_access(self, tmp_path):
        """TrainConfig proxies attribute access to base Config."""
        from src.config import load_config
        from src.training.train_config import TrainConfig

        path = _write_config(tmp_path, _minimal_config_text())

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        tc = TrainConfig.from_config(cfg)
        assert tc.SGLANG_PORT == cfg.SGLANG_PORT
        assert tc.GRPO_TEMPERATURE == cfg.GRPO_TEMPERATURE

    def test_test_trained_checkpoint_source(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_TEST_TRAINED_CHECKPOINT_SOURCE="TEST_TRAINED_CHECKPOINT_LATEST"
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.TEST_TRAINED_CHECKPOINT_SOURCE == "TEST_TRAINED_CHECKPOINT_LATEST"

    def test_test_trained_output_dir_derivation(self, tmp_path):
        from src.config import load_config

        text = _minimal_config_text() + textwrap.dedent("""\
            CONFIG_TRAIN_OUTPUT_DIR="output/train_grpo"
            CONFIG_TEST_TRAINED_CHECKPOINT_SOURCE="TEST_TRAINED_CHECKPOINT_BEST"
            CONFIG_TEST_TRAINED_OUTPUT_DIR=""
        """)
        path = _write_config(tmp_path, text)

        with mock.patch("src.config._validate"):
            cfg = load_config(path)

        assert cfg.TEST_TRAINED_OUTPUT_DIR.endswith("test_best") or \
               cfg.TEST_TRAINED_OUTPUT_DIR.endswith("test_best/")
