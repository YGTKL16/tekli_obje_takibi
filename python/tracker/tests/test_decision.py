"""Tests for decision maker."""

import numpy as np
from tracker.decision import DecisionMaker, compute_iou


def test_iou_identical_boxes():
    box = np.array([100, 100, 50, 50], dtype=np.float32)
    assert abs(compute_iou(box, box) - 1.0) < 1e-6


def test_iou_no_overlap():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    b = np.array([100, 100, 10, 10], dtype=np.float32)
    assert compute_iou(a, b) == 0.0


def test_should_update_high_confidence():
    dm = DecisionMaker(conf_threshold=0.3)
    ai = np.array([100, 100, 50, 50], dtype=np.float32)
    kf = np.array([100, 100, 50, 50, 0, 0, 0, 0], dtype=np.float32)
    assert dm.should_update(0.9, ai, kf) is True


def test_should_not_update_low_confidence():
    dm = DecisionMaker(conf_threshold=0.3)
    ai = np.array([100, 100, 50, 50], dtype=np.float32)
    kf = np.array([100, 100, 50, 50, 0, 0, 0, 0], dtype=np.float32)
    assert dm.should_update(0.1, ai, kf) is False


def test_should_not_update_low_iou():
    dm = DecisionMaker(conf_threshold=0.3, iou_threshold=0.2)
    ai = np.array([500, 500, 50, 50], dtype=np.float32)  # far from KF
    kf = np.array([100, 100, 50, 50, 0, 0, 0, 0], dtype=np.float32)
    assert dm.should_update(0.9, ai, kf) is False
