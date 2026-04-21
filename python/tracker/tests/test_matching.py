"""Tests for LAPJV-based detection-to-tracker association."""

import numpy as np

from tracker.matching import (
    associate_detections_to_trackers,
    build_cost_matrix,
    compute_iou_matrix,
)


def test_associate_empty_trackers():
    detections = np.array([[10, 10, 20, 20]], dtype=np.float32)
    matches, unmatched_trackers, unmatched_detections = associate_detections_to_trackers(
        np.empty((0, 4), dtype=np.float32), detections
    )

    assert matches.shape == (0, 2)
    assert unmatched_trackers.shape == (0,)
    assert unmatched_detections.tolist() == [0]


def test_batch_iou_matches_expected_values():
    trackers = np.array([[0, 0, 10, 10], [10, 10, 10, 10]], dtype=np.float32)
    detections = np.array([[0, 0, 10, 10], [5, 5, 10, 10]], dtype=np.float32)
    iou = compute_iou_matrix(trackers, detections)

    assert iou.shape == (2, 2)
    assert np.isclose(iou[0, 0], 1.0)
    assert np.isclose(iou[0, 1], 25.0 / 175.0)
    assert np.isclose(iou[1, 1], 25.0 / 175.0)


def test_one_by_k_assignment_matches_argmin():
    tracker = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.array(
        [[100, 100, 10, 10], [1, 1, 10, 10], [0, 0, 8, 8]],
        dtype=np.float32,
    )
    cost = build_cost_matrix(tracker, detections)
    matches, _, _ = associate_detections_to_trackers(
        tracker, detections, iou_threshold=0.0
    )

    assert matches.shape == (1, 2)
    assert int(matches[0, 1]) == int(np.argmin(cost[0]))


def test_low_iou_match_is_rejected():
    tracker = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([[50, 50, 10, 10]], dtype=np.float32)
    matches, unmatched_trackers, unmatched_detections = associate_detections_to_trackers(
        tracker, detections, iou_threshold=0.3
    )

    assert matches.shape == (0, 2)
    assert unmatched_trackers.tolist() == [0]
    assert unmatched_detections.tolist() == [0]


def test_score_weight_zero_preserves_pure_iou_cost():
    trackers = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([[0, 0, 10, 10], [2, 2, 10, 10]], dtype=np.float32)
    scores = np.array([0.1, 0.9], dtype=np.float32)

    assert np.allclose(
        build_cost_matrix(trackers, detections, detection_scores=scores, score_weight=0.0),
        1.0 - compute_iou_matrix(trackers, detections),
    )


def test_score_weight_breaks_iou_tie_toward_higher_score():
    tracker = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.1, 0.9], dtype=np.float32)

    cost = build_cost_matrix(
        tracker,
        detections,
        detection_scores=scores,
        score_weight=0.5,
    )
    matches, _, _ = associate_detections_to_trackers(
        tracker,
        detections,
        detection_scores=scores,
        iou_threshold=0.0,
        score_weight=0.5,
    )

    assert cost[0, 1] < cost[0, 0]
    assert matches.shape == (1, 2)
    assert int(matches[0, 1]) == 1
