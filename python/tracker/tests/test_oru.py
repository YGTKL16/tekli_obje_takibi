"""Unit tests for the ORU backfill controller."""

from __future__ import annotations

import numpy as np
import pytest

import tracker_cpp as tc
from tracker.oru import (
    ImmSnapshot,
    OruConfig,
    OruController,
    gen_virtual_trajectory,
    velocity_cv,
)


# ── helpers ──────────────────────────────────────────────────────────


def _seed_imm(initial_z=(100.0, 200.0, 30.0, 40.0), nudge_steps: int = 2):
    """Return an IMM filter that has been advanced a few frames so its state
    is non-trivial (vx/vy != 0, P diverged from init, mu drifted)."""
    imm = tc.IMMFilter()
    z = np.asarray(initial_z, dtype=np.float32)
    imm.init(z)
    for k in range(1, nudge_steps + 1):
        imm.update((z + np.array([k, k * 0.5, 0, 0], dtype=np.float32)).astype(np.float32))
    return imm


def _capture(imm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(imm.get_state(), dtype=np.float32).copy(),
        np.array(imm.get_covariance(), dtype=np.float32).copy(),
        np.array(imm.get_model_probabilities(), dtype=np.float32).copy(),
    )


# ── tests ────────────────────────────────────────────────────────────


def test_restore_from_idempotent():
    """get -> restore -> get must be byte-identical."""
    imm = _seed_imm()
    s_a, P_a, mu_a = _capture(imm)

    # Corrupt the filter
    for _ in range(3):
        imm.predict()
    imm.update(np.array([300.0, 400.0, 30.0, 40.0], dtype=np.float32))

    # Restore
    imm.restore_from(s_a, P_a, mu_a)
    s_b, P_b, mu_b = _capture(imm)

    np.testing.assert_allclose(s_a, s_b, atol=1e-6)
    np.testing.assert_allclose(P_a, P_b, atol=1e-6)
    np.testing.assert_allclose(mu_a, mu_b, atol=1e-6)


def test_virtual_z_linear():
    """Linear interpolation between anchors over n=4 missing frames."""
    z1 = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    z2 = np.array([20.0, 40.0, 10.0, 10.0], dtype=np.float32)
    virt = gen_virtual_trajectory(z1, z2, n=4)

    # n=4 -> 3 virtual points at alphas 1/4, 2/4, 3/4
    assert virt.shape == (3, 4)
    np.testing.assert_allclose(virt[0], [5.0, 10.0, 10.0, 10.0], atol=1e-5)
    np.testing.assert_allclose(virt[1], [10.0, 20.0, 10.0, 10.0], atol=1e-5)
    np.testing.assert_allclose(virt[2], [15.0, 30.0, 10.0, 10.0], atol=1e-5)


def test_virtual_z_short_returns_empty():
    z1 = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    z2 = np.array([5.0, 5.0, 10.0, 10.0], dtype=np.float32)
    assert gen_virtual_trajectory(z1, z2, n=1).shape == (0, 4)
    assert gen_virtual_trajectory(z1, z2, n=0).shape == (0, 4)


def test_velocity_cv_basic():
    assert velocity_cv([]) == 0.0
    assert velocity_cv([5.0]) == 0.0
    assert velocity_cv([3.0, 3.0, 3.0]) == pytest.approx(0.0, abs=1e-6)
    cv = velocity_cv([1.0, 2.0, 3.0])
    # std/mean = 0.8165/2.0 = 0.4082
    assert cv == pytest.approx(0.4082, abs=1e-3)


def test_oru_round_trip_synthetic_target():
    """Constant-vel target moves 30 frames; we coast 20 and re-acquire.
    After ORU the posterior x,y must match ground truth within a few px."""
    cfg = OruConfig(
        enabled=True,
        n_min=5,
        n_max=30,
        conf_reentry_min=0.3,
        velocity_cv_max=10.0,        # disable CV gate (single-step samples)
        skip_if_singer_dom=False,
        min_velocity_samples=999,    # disable velocity gate for this test
    )
    ctrl = OruController(cfg)

    # Ground-truth: target at x=100+t, y=200, w=30, h=40 (constant vel 1 px/frame)
    def gt(t):
        return np.array([100.0 + t, 200.0, 30.0, 40.0], dtype=np.float32)

    imm = tc.IMMFilter()
    imm.init(gt(0))

    # Frames 1..10: TRACKING with real updates. Push snapshot each accept.
    for t in range(1, 11):
        imm.update(gt(t))
        s, P, mu = _capture(imm)
        ctrl.push_snapshot(s, P, mu, gt(t), frame_idx=t)

    # Frame 11: coast starts
    ctrl.on_coast_start(frame_idx=11)
    # Frames 11..30: only predict (no update). 20 missed frames.
    for t in range(11, 31):
        imm.predict()
        v = float(np.linalg.norm(imm.get_state()[4:6]))
        ctrl.record_coast_velocity(v)

    # Frame 31: re-acquire at gt(31)
    z_t2 = gt(31)
    fired = ctrl.maybe_run(imm, z_t2, conf_t2=0.9, frame_idx=31)
    assert fired, "ORU gate should pass for this synthetic case"
    # caller's normal real-z update at t2
    imm.update(z_t2, 0.9)

    posterior_xy = np.array(imm.get_state()[:2])
    expected_xy = gt(31)[:2]
    err = np.linalg.norm(posterior_xy - expected_xy)
    assert err < 5.0, f"ORU posterior off by {err:.2f}px (expected <5)"


def test_gate_skip_too_short():
    cfg = OruConfig(n_min=10)
    ctrl = OruController(cfg)
    imm = _seed_imm()
    s, P, mu = _capture(imm)
    z = np.array([100.0, 200.0, 30.0, 40.0], dtype=np.float32)
    ctrl.push_snapshot(s, P, mu, z, frame_idx=5)
    ctrl.on_coast_start(frame_idx=6)
    # Only 3 frames of coast -> n=3 < n_min=10
    fired = ctrl.maybe_run(imm, z, conf_t2=0.9, frame_idx=8)
    assert not fired
    assert ctrl.gate_skips["too_short"] == 1


def test_gate_skip_singer_dom():
    cfg = OruConfig(skip_if_singer_dom=True, singer_dom_thr=0.6, n_min=2)
    ctrl = OruController(cfg)
    imm = _seed_imm()
    s, P, mu = _capture(imm)
    # Forge mu with high Singer share
    mu_hot = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    z = np.array([100.0, 200.0, 30.0, 40.0], dtype=np.float32)
    ctrl.push_snapshot(s, P, mu_hot, z, frame_idx=10)
    ctrl.on_coast_start(frame_idx=11)
    fired = ctrl.maybe_run(imm, z, conf_t2=0.9, frame_idx=20)
    assert not fired
    assert ctrl.gate_skips["singer_dom"] == 1


def test_gate_skip_low_conf():
    cfg = OruConfig(conf_reentry_min=0.5, n_min=2)
    ctrl = OruController(cfg)
    imm = _seed_imm()
    s, P, mu = _capture(imm)
    z = np.array([100.0, 200.0, 30.0, 40.0], dtype=np.float32)
    ctrl.push_snapshot(s, P, mu, z, frame_idx=10)
    ctrl.on_coast_start(frame_idx=11)
    fired = ctrl.maybe_run(imm, z, conf_t2=0.2, frame_idx=20)
    assert not fired
    assert ctrl.gate_skips["low_reentry_conf"] == 1


def test_gate_skip_velocity_cv():
    cfg = OruConfig(velocity_cv_max=0.1, min_velocity_samples=2, n_min=2)
    ctrl = OruController(cfg)
    imm = _seed_imm()
    s, P, mu = _capture(imm)
    z = np.array([100.0, 200.0, 30.0, 40.0], dtype=np.float32)
    ctrl.push_snapshot(s, P, mu, z, frame_idx=10)
    ctrl.on_coast_start(frame_idx=11)
    # Highly variable velocity history
    for v in [1.0, 5.0, 0.2, 8.0, 0.5]:
        ctrl.record_coast_velocity(v)
    fired = ctrl.maybe_run(imm, z, conf_t2=0.9, frame_idx=20)
    assert not fired
    assert ctrl.gate_skips["velocity_cv"] == 1


def test_disabled_controller_is_noop():
    cfg = OruConfig(enabled=False)
    ctrl = OruController(cfg)
    imm = _seed_imm()
    s, P, mu = _capture(imm)
    z = np.array([100.0, 200.0, 30.0, 40.0], dtype=np.float32)
    ctrl.push_snapshot(s, P, mu, z, frame_idx=10)
    ctrl.on_coast_start(frame_idx=11)
    fired = ctrl.maybe_run(imm, z, conf_t2=0.9, frame_idx=20)
    assert not fired
    assert ctrl.fired_count == 0
    assert ctrl.telemetry()["snapshots_buffered"] == 0  # push was a no-op
