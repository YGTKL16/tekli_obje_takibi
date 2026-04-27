"""Tests for ChaosDetector and ChaosConfig."""

import math
import pytest
from tracker.chaos import ChaosConfig, ChaosDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detector(window=10, chaos_thr=0.08, chaos_max=0.20, factor=0.15, min_samples=5):
    cfg = ChaosConfig(
        enabled=True,
        window=window,
        chaos_thr=chaos_thr,
        chaos_max=chaos_max,
        min_chaos_conf_factor=factor,
        min_samples=min_samples,
    )
    return ChaosDetector(cfg)


# ---------------------------------------------------------------------------
# Test 1: Stable confidence → no penalty
# ---------------------------------------------------------------------------

def test_no_trigger_stable_conf():
    det = _detector()
    # Fill window with constant high confidence
    confs = [0.8] * 12
    results = [det.step(c) for c in confs]
    # Rolling std ≈ 0 → below threshold → no deflation
    for orig, eff in zip(confs[-5:], results[-5:]):
        assert abs(eff - orig) < 1e-6, f"Expected no deflation, got {eff} vs {orig}"


# ---------------------------------------------------------------------------
# Test 2: Oscillating confidence → trigger fires, output deflated
# ---------------------------------------------------------------------------

def test_trigger_chaotic_conf():
    det = _detector(min_samples=4)
    # Alternate 0.2 / 0.9 → std ≈ 0.35, well above chaos_max=0.20
    confs = [0.2, 0.9] * 8
    results = [det.step(c) for c in confs]
    # After buffer fills, last result must be < raw confidence
    raw = confs[-1]
    eff = results[-1]
    assert eff < raw, f"Expected deflation but eff={eff} >= raw={raw}"
    # At t=1 (full chaos), eff = raw * factor = raw * 0.15
    # Allow some tolerance (std may cap at chaos_max exactly)
    assert eff <= raw * 0.15 + 1e-4


# ---------------------------------------------------------------------------
# Test 3: min_samples gate — no trigger before buffer fills
# ---------------------------------------------------------------------------

def test_min_samples_gate():
    det = _detector(min_samples=8)
    chaotic = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]  # 6 samples, need 8
    for i, c in enumerate(chaotic):
        eff = det.step(c)
        assert eff == c, f"Sample {i}: deflation before min_samples reached (eff={eff})"


# ---------------------------------------------------------------------------
# Test 4: Chaos step never increases confidence
# ---------------------------------------------------------------------------

def test_conf_never_increases():
    det = _detector()
    # Mix of low and moderate confidence values
    confs = [0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3]
    for c in confs:
        eff = det.step(c)
        assert eff <= c + 1e-9, f"eff={eff} > raw={c}: confidence was increased!"


# ---------------------------------------------------------------------------
# Test 5: reset() clears state — no trigger until buffer refills
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    det = _detector(min_samples=4)
    # Fill with oscillating values to warm up
    for _ in range(10):
        det.step(0.1)
        det.step(0.9)

    det.reset()

    # After reset, chaotic input should NOT trigger until min_samples reached
    chaotic = [0.1, 0.9, 0.1]  # 3 samples, need 4
    for i, c in enumerate(chaotic):
        eff = det.step(c)
        assert eff == c, f"Post-reset sample {i}: deflation before min_samples (eff={eff})"


# ---------------------------------------------------------------------------
# Test 6: ChaosConfig.from_dict — partial dict uses defaults
# ---------------------------------------------------------------------------

def test_from_dict_partial():
    cfg = ChaosConfig.from_dict({"window": 15, "chaos_thr": 0.05})
    assert cfg.window == 15
    assert cfg.chaos_thr == 0.05
    # Remaining fields default
    assert cfg.chaos_max == 0.20
    assert cfg.min_chaos_conf_factor == 0.15
    assert cfg.min_samples == 5
    assert cfg.enabled is True


def test_from_dict_none():
    cfg = ChaosConfig.from_dict(None)
    assert cfg.window == 10


def test_from_dict_empty():
    cfg = ChaosConfig.from_dict({})
    assert cfg.window == 10
