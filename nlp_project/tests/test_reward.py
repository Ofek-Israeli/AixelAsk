"""Tests for src.training.reward — weighted reward computation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.training.reward import compute


# ---------------------------------------------------------------------------
# Stub config
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    REWARD_WEIGHT_CORRECTNESS: float = 1.0
    REWARD_WEIGHT_VALIDITY: float = 0.5
    REWARD_WEIGHT_DEPTH: float = 0.1
    REWARD_DEPTH_NORMALIZATION: str = "DIVIDE_BY_MAX_DEPTH"
    REWARD_MAX_DEPTH: int = 10


# ---------------------------------------------------------------------------
# Basic weighted reward
# ---------------------------------------------------------------------------

class TestBasicReward:

    def test_basic_weighted_reward(self):
        """r = 1.0*1 + 0.5*1 - 0.1*(3/10) = 1.47."""
        cfg = _StubConfig()
        r = compute(r_correct=1.0, r_valid=1.0, depth=3, config=cfg)
        assert r == pytest.approx(1.47)

    def test_invalid_dag_penalty(self):
        """r_valid=0 → no validity bonus, depth=0 → no depth penalty."""
        cfg = _StubConfig()
        r = compute(r_correct=0.0, r_valid=0.0, depth=0, config=cfg)
        assert r == pytest.approx(0.0)

    def test_parse_failure_max_penalty(self):
        """Unparseable → r_correct=0, r_valid=0, depth=0."""
        cfg = _StubConfig()
        r = compute(r_correct=0.0, r_valid=0.0, depth=0, config=cfg)
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Depth normalization
# ---------------------------------------------------------------------------

class TestDepthNormalization:

    def test_none_mode(self):
        """NONE → raw depth used."""
        cfg = _StubConfig(REWARD_DEPTH_NORMALIZATION="NONE")
        r = compute(r_correct=1.0, r_valid=1.0, depth=5, config=cfg)
        # 1.0*1 + 0.5*1 - 0.1*5 = 1.5 - 0.5 = 1.0
        assert r == pytest.approx(1.0)

    def test_divide_by_max_depth(self):
        """DIVIDE_BY_MAX_DEPTH: depth=5, max=10 → depth_term=0.5."""
        cfg = _StubConfig(REWARD_DEPTH_NORMALIZATION="DIVIDE_BY_MAX_DEPTH", REWARD_MAX_DEPTH=10)
        r = compute(r_correct=1.0, r_valid=1.0, depth=5, config=cfg)
        # 1.0 + 0.5 - 0.1*0.5 = 1.45
        assert r == pytest.approx(1.45)

    def test_divide_by_max_depth_clamped(self):
        """depth > max_depth is clamped to 1.0."""
        cfg = _StubConfig(REWARD_DEPTH_NORMALIZATION="DIVIDE_BY_MAX_DEPTH", REWARD_MAX_DEPTH=10)
        r = compute(r_correct=1.0, r_valid=1.0, depth=15, config=cfg)
        # 1.0 + 0.5 - 0.1*1.0 = 1.4
        assert r == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# Partial credit (via difflib)
# ---------------------------------------------------------------------------

class TestPartialCredit:

    def test_exact_match(self):
        """Full credit: predicted matches gold."""
        cfg = _StubConfig()
        r = compute(r_correct=1.0, r_valid=1.0, depth=1, config=cfg)
        assert r > 0

    def test_no_match(self):
        """No credit: r_correct=0."""
        cfg = _StubConfig()
        r = compute(r_correct=0.0, r_valid=1.0, depth=1, config=cfg)
        # 0 + 0.5 - 0.1*(1/10) = 0.49
        assert r == pytest.approx(0.49)

    def test_partial_credit_in_range(self):
        """0 < r_correct < 1 gives partial reward."""
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, "forty two", "forty-two").ratio()
        assert 0.0 < ratio < 1.0

        cfg = _StubConfig()
        r = compute(r_correct=ratio, r_valid=1.0, depth=1, config=cfg)
        r_full = compute(r_correct=1.0, r_valid=1.0, depth=1, config=cfg)
        r_zero = compute(r_correct=0.0, r_valid=1.0, depth=1, config=cfg)
        assert r_zero < r < r_full


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_weights(self):
        """All weights zero → reward is zero regardless of inputs."""
        cfg = _StubConfig(
            REWARD_WEIGHT_CORRECTNESS=0.0,
            REWARD_WEIGHT_VALIDITY=0.0,
            REWARD_WEIGHT_DEPTH=0.0,
        )
        r = compute(r_correct=1.0, r_valid=1.0, depth=10, config=cfg)
        assert r == pytest.approx(0.0)

    def test_determinism(self):
        """Same inputs → same reward (no stochastic component)."""
        cfg = _StubConfig()
        r1 = compute(r_correct=0.8, r_valid=1.0, depth=4, config=cfg)
        r2 = compute(r_correct=0.8, r_valid=1.0, depth=4, config=cfg)
        assert r1 == r2
