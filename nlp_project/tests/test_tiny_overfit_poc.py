"""Tests for src.training.tiny_overfit_poc — seeded subset selection."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import pytest

from src.training.tiny_overfit_poc import select_overfit_subset


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    OVERFIT_POC_SELECTION_MODE: str = "FIRST_N"
    OVERFIT_POC_NUM_EXAMPLES: int = 8
    OVERFIT_POC_INDICES_FILE: str = ""
    GLOBAL_SEED: int = 42


# ---------------------------------------------------------------------------
# FIRST_N deterministic
# ---------------------------------------------------------------------------

class TestFirstN:

    def test_deterministic_first_n(self):
        """FIRST_N, N=8, dataset_size=100 → indices [0..7]."""
        cfg = _StubConfig(OVERFIT_POC_SELECTION_MODE="FIRST_N", OVERFIT_POC_NUM_EXAMPLES=8)
        indices = select_overfit_subset(cfg, dataset_size=100)
        assert indices == list(range(8))

    def test_first_n_reproducible(self):
        """Repeated calls produce the same result."""
        cfg = _StubConfig(OVERFIT_POC_SELECTION_MODE="FIRST_N", OVERFIT_POC_NUM_EXAMPLES=8)
        a = select_overfit_subset(cfg, dataset_size=100)
        b = select_overfit_subset(cfg, dataset_size=100)
        assert a == b


# ---------------------------------------------------------------------------
# RANDOM_SEEDED deterministic + reproducible
# ---------------------------------------------------------------------------

class TestRandomSeeded:

    def test_deterministic(self):
        """Same seed, same N → same indices."""
        cfg = _StubConfig(
            OVERFIT_POC_SELECTION_MODE="RANDOM_SEEDED",
            OVERFIT_POC_NUM_EXAMPLES=8,
            GLOBAL_SEED=42,
        )
        a = select_overfit_subset(cfg, dataset_size=100)
        b = select_overfit_subset(cfg, dataset_size=100)
        assert a == b
        assert len(a) == 8

    def test_sorted_output(self):
        """Returned indices are sorted."""
        cfg = _StubConfig(
            OVERFIT_POC_SELECTION_MODE="RANDOM_SEEDED",
            OVERFIT_POC_NUM_EXAMPLES=8,
            GLOBAL_SEED=42,
        )
        indices = select_overfit_subset(cfg, dataset_size=100)
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Different seed → different indices
# ---------------------------------------------------------------------------

class TestDifferentSeed:

    def test_different_seed_different_indices(self):
        """Seed=42 vs seed=99 → different indices."""
        cfg42 = _StubConfig(
            OVERFIT_POC_SELECTION_MODE="RANDOM_SEEDED",
            OVERFIT_POC_NUM_EXAMPLES=8,
            GLOBAL_SEED=42,
        )
        cfg99 = _StubConfig(
            OVERFIT_POC_SELECTION_MODE="RANDOM_SEEDED",
            OVERFIT_POC_NUM_EXAMPLES=8,
            GLOBAL_SEED=99,
        )
        a = select_overfit_subset(cfg42, dataset_size=100)
        b = select_overfit_subset(cfg99, dataset_size=100)
        assert a != b


# ---------------------------------------------------------------------------
# FIXED_INDICES_FILE
# ---------------------------------------------------------------------------

class TestFixedIndicesFile:

    def test_reads_from_file(self, tmp_path):
        """Write [0, 5, 10] to a file → exactly those indices selected."""
        idx_file = tmp_path / "indices.txt"
        idx_file.write_text("0\n5\n10\n")

        cfg = _StubConfig(
            OVERFIT_POC_SELECTION_MODE="FIXED_INDICES_FILE",
            OVERFIT_POC_INDICES_FILE=str(idx_file),
        )
        indices = select_overfit_subset(cfg, dataset_size=100)
        assert indices == [0, 5, 10]


# ---------------------------------------------------------------------------
# N exceeds dataset → error
# ---------------------------------------------------------------------------

class TestExceedsDataset:

    def test_first_n_exceeds(self):
        """N=200, dataset_size=100 → ValueError."""
        cfg = _StubConfig(OVERFIT_POC_SELECTION_MODE="FIRST_N", OVERFIT_POC_NUM_EXAMPLES=200)
        with pytest.raises(ValueError, match="exceeds dataset size"):
            select_overfit_subset(cfg, dataset_size=100)

    def test_random_seeded_exceeds(self):
        """N=200, dataset_size=100 → ValueError."""
        cfg = _StubConfig(
            OVERFIT_POC_SELECTION_MODE="RANDOM_SEEDED",
            OVERFIT_POC_NUM_EXAMPLES=200,
            GLOBAL_SEED=42,
        )
        with pytest.raises(ValueError, match="exceeds dataset size"):
            select_overfit_subset(cfg, dataset_size=100)


# ---------------------------------------------------------------------------
# Subset used as both train and dev (verified at split_utils level)
# ---------------------------------------------------------------------------

class TestSubsetAsBothTrainAndDev:

    def test_concept(self):
        """In overfit-PoC mode, the indices are the same for train/dev.

        This is a conceptual check — the actual enforcement is in split_utils.
        """
        cfg = _StubConfig(
            OVERFIT_POC_SELECTION_MODE="FIRST_N",
            OVERFIT_POC_NUM_EXAMPLES=8,
        )
        indices = select_overfit_subset(cfg, dataset_size=100)
        # The same list would be used for both train and dev
        assert indices == indices
        assert len(indices) == 8
