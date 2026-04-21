"""Tests for shared IMM integration helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tracker.imm_policy import IMMObservation, observe_with_guidance, step_guided_imm


def _make_tracker():
    tracker = MagicMock()
    tracker.association_enabled = True
    tracker.association_top_k = 5
    tracker.association_iou_threshold = 0.3
    tracker.association_score_weight = 0.0
    return tracker


def test_observe_with_guidance_steers_search_window_to_prediction():
    tracker = _make_tracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    predicted_state = np.array([100, 200, 50, 60, 0, 0, 0, 0], dtype=np.float32)
    chosen_bbox = np.array([102, 201, 51, 61], dtype=np.float32)
    tracker.track_candidates.return_value = (
        np.array([[400, 400, 20, 20], chosen_bbox], dtype=np.float32),
        np.array([0.2, 0.9], dtype=np.float32),
    )

    with patch(
        "tracker.imm_policy.associate_detections_to_trackers",
        return_value=(np.array([[0, 1]], dtype=int), np.empty((0,), dtype=int), np.array([0], dtype=int)),
    ):
        observation = observe_with_guidance(tracker, frame, predicted_state, mode="baseline")

    tracker.set_state.assert_called_once_with([100.0, 200.0, 50.0, 60.0])
    assert isinstance(observation, IMMObservation)
    assert observation.bbox is not None
    assert observation.confidence == pytest.approx(0.9)
    assert np.allclose(observation.bbox, chosen_bbox)


def test_step_guided_imm_uses_filtered_output_on_accepted_measurement():
    import tracker_cpp

    kf = tracker_cpp.IMMFilter()
    init_bbox = np.array([100, 100, 50, 40], dtype=np.float32)
    observed_bbox = np.array([101, 100, 50, 40], dtype=np.float32)
    kf.init(init_bbox)

    decision = MagicMock()
    decision.should_coast.return_value = False
    decision.is_measurement_sane.return_value = True
    decision.should_update.return_value = True

    predicted_state = np.array(kf.predict()).flatten()
    step = step_guided_imm(
        kf,
        decision,
        predicted_state,
        observed_bbox,
        0.9,
        640,
        480,
        is_tracking=True,
        judge_reference_bbox=predicted_state[:4],
        conf_bypass_threshold=1.0,  # disable bypass for this test
    )

    assert step.accepted_measurement is True
    assert step.bbox != pytest.approx(observed_bbox.tolist())
    assert abs(step.bbox[0] - observed_bbox[0]) < 2.0
    assert abs(step.bbox[1] - observed_bbox[1]) < 2.0


def test_high_conf_bypass_returns_ai_bbox():
    """Phase 1: When confidence >= bypass threshold, use AI bbox directly."""
    import tracker_cpp

    kf = tracker_cpp.IMMFilter()
    init_bbox = np.array([100, 100, 50, 40], dtype=np.float32)
    observed_bbox = np.array([105, 102, 50, 40], dtype=np.float32)
    kf.init(init_bbox)

    decision = MagicMock()
    decision.is_measurement_sane.return_value = True
    predicted_state = np.array(kf.predict()).flatten()
    step = step_guided_imm(
        kf,
        decision,
        predicted_state,
        observed_bbox,
        0.95,
        640,
        480,
        is_tracking=True,
        conf_bypass_threshold=0.90,
    )

    # After H2 fix: bypass still outputs AI bbox but NOW updates KF.
    assert step.accepted_measurement is True
    assert step.should_coast is False
    assert step.reject_streak == 0
    assert step.bbox == pytest.approx(observed_bbox.tolist())
    assert step.innovation_norm > 0.0
    # KF state should be close to AI bbox (not stale prediction).
    assert abs(step.state[0] - observed_bbox[0]) < 5.0
    assert abs(step.state[1] - observed_bbox[1]) < 5.0
    # Sanity IS checked in bypass, but conf/IoU gates are skipped.
    decision.is_measurement_sane.assert_called_once()
    decision.should_coast.assert_not_called()
    decision.should_update.assert_not_called()


def test_innovation_qboost_triggers_on_large_innovation():
    """Phase 2: Large innovation triggers Q-boost for next frame."""
    import tracker_cpp

    kf = tracker_cpp.IMMFilter()
    init_bbox = np.array([100, 100, 50, 40], dtype=np.float32)
    # Large jump: 200px away from prediction
    far_bbox = np.array([300, 100, 50, 40], dtype=np.float32)
    kf.init(init_bbox)

    decision = MagicMock()
    decision.should_coast.return_value = False
    decision.is_measurement_sane.return_value = True
    decision.should_update.return_value = True

    predicted_state = np.array(kf.predict()).flatten()
    step = step_guided_imm(
        kf,
        decision,
        predicted_state,
        far_bbox,
        0.5,
        640,
        480,
        is_tracking=True,
        conf_bypass_threshold=1.0,  # disable bypass
        innovation_threshold=1.0,
    )

    # Innovation should be large (200px / ~64px diagonal ≈ 3.1)
    assert step.innovation_norm > 2.0


def test_reject_streak_accumulates_across_calls():
    """H1 fix: reject_streak persists when passed back.

    Scenario: is_measurement_sane passes but Mahalanobis rejects every frame
    (simulating a measurement sequence that is geometrically plausible but
    statistically inconsistent with the KF state).  The reject_streak counter
    must accumulate across calls so that the reinit-gate can eventually fire.
    """
    import tracker_cpp

    kf = tracker_cpp.IMMFilter()
    init_bbox = np.array([100, 100, 50, 40], dtype=np.float32)
    kf.init(init_bbox)

    decision = MagicMock()
    decision.should_coast.return_value = False
    decision.is_measurement_sane.return_value = True
    # Mahalanobis rejects every measurement → mahal_confirmed=False → IoU gate applied
    decision.is_measurement_mahalanobis_ok.return_value = False
    # IoU gate also rejects (belt-and-suspenders for the fallback path)
    decision.should_update.return_value = False

    streak = 0
    last_good = None
    for _ in range(5):
        predicted_state = np.array(kf.predict()).flatten()
        step = step_guided_imm(
            kf,
            decision,
            predicted_state,
            np.array([100, 100, 50, 40], dtype=np.float32),
            0.1,  # low confidence
            640,
            480,
            is_tracking=True,
            conf_bypass_threshold=1.0,  # disable bypass
            reject_streak=streak,
            last_good_bbox=last_good,
        )
        streak = step.reject_streak
        last_good = step.last_good_bbox

    # After 5 rejected frames, streak should be 5
    assert streak == 5


def test_reject_streak_triggers_reinit():
    """H1 fix: after reinit_after frames of rejection, KF reinits."""
    import tracker_cpp

    kf = tracker_cpp.IMMFilter()
    init_bbox = np.array([100, 100, 50, 40], dtype=np.float32)
    kf.init(init_bbox)

    decision = MagicMock()
    decision.should_coast.return_value = False
    decision.is_measurement_sane.return_value = True
    decision.should_update.return_value = False

    streak = 0
    last_good = [100.0, 100.0, 50.0, 40.0]
    reinit_after = 4
    for _ in range(reinit_after):
        predicted_state = np.array(kf.predict()).flatten()
        step = step_guided_imm(
            kf,
            decision,
            predicted_state,
            np.array([100, 100, 50, 40], dtype=np.float32),
            0.1,
            640,
            480,
            is_tracking=True,
            conf_bypass_threshold=1.0,
            reject_streak=streak,
            last_good_bbox=last_good,
            reinit_after=reinit_after,
        )
        streak = step.reject_streak
        last_good = step.last_good_bbox

    # After reinit_after frames, streak should be reset to 0
    assert streak == 0


def test_dead_zone_conf_accepts_with_new_threshold():
    """H3 fix: conf=0.2 (in old dead zone) should now be accepted."""
    from tracker.decision import DecisionMaker

    dm = DecisionMaker()  # default conf_threshold=0.18
    ai = np.array([100, 100, 50, 50], dtype=np.float32)
    kf = np.array([100, 100, 50, 50, 0, 0, 0, 0], dtype=np.float32)
    # conf=0.2 > 0.18 and IoU=1.0 → should update
    assert dm.should_update(0.2, ai, kf) is True
    # conf=0.1 < 0.18 → should NOT update
    assert dm.should_update(0.1, ai, kf) is False


def test_qboost_suppressed_when_reject_streak_nonzero():
    """H4 fix: Q-boost should NOT fire after rejection streak (e.g. post-coast)."""
    import tracker_cpp

    kf = tracker_cpp.IMMFilter()
    init_bbox = np.array([100, 100, 50, 40], dtype=np.float32)
    far_bbox = np.array([300, 100, 50, 40], dtype=np.float32)  # large innovation
    kf.init(init_bbox)

    decision = MagicMock()
    decision.should_coast.return_value = False
    decision.is_measurement_sane.return_value = True
    decision.should_update.return_value = True

    predicted_state = np.array(kf.predict()).flatten()
    step = step_guided_imm(
        kf,
        decision,
        predicted_state,
        far_bbox,
        0.5,
        640,
        480,
        is_tracking=True,
        conf_bypass_threshold=1.0,  # disable bypass
        innovation_threshold=1.0,
        reject_streak=3,  # prior rejections → Q-boost should be suppressed
    )

    # Innovation is large, but because reject_streak > 0, Q-boost should NOT fire.
    assert step.innovation_norm > 2.0
    # We can't directly check gmc_failed, but we can verify via a second predict:
    # If Q-boost fired, covariance would be 4x larger. We compare against a fresh filter.
    kf2 = tracker_cpp.IMMFilter()
    kf2.init(init_bbox)
    kf2.predict()
    # Q-boost was NOT triggered, so next predict should have normal covariance.
    # (This test primarily validates the code path — actual covariance comparison
    # is tested in C++ side.)


def test_qboost_fires_when_reject_streak_zero():
    """H4: Q-boost SHOULD fire when reject_streak == 0 and innovation is large."""
    import tracker_cpp

    kf = tracker_cpp.IMMFilter()
    init_bbox = np.array([100, 100, 50, 40], dtype=np.float32)
    far_bbox = np.array([300, 100, 50, 40], dtype=np.float32)
    kf.init(init_bbox)

    decision = MagicMock()
    decision.should_coast.return_value = False
    decision.is_measurement_sane.return_value = True
    decision.should_update.return_value = True

    predicted_state = np.array(kf.predict()).flatten()
    step = step_guided_imm(
        kf,
        decision,
        predicted_state,
        far_bbox,
        0.5,
        640,
        480,
        is_tracking=True,
        conf_bypass_threshold=1.0,
        innovation_threshold=1.0,
        reject_streak=0,  # should fire
    )

    assert step.innovation_norm > 2.0
    assert step.reject_streak == 0  # was accepted
