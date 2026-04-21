"""Tests for visualizer module."""

import numpy as np
import pytest
from unittest.mock import patch

from tracker.visualizer import draw_bbox_topleft, draw_overlay, visualize


class TestDrawBboxTopleft:
    def test_returns_same_shape(self, sample_frame, sample_bbox):
        result = draw_bbox_topleft(sample_frame.copy(), sample_bbox, (0, 255, 0))
        assert result.shape == sample_frame.shape

    def test_modifies_pixels(self, sample_frame, sample_bbox):
        original = sample_frame.copy()
        result = draw_bbox_topleft(original, sample_bbox, (0, 255, 0))
        assert not np.array_equal(result, sample_frame)

    def test_with_label(self, sample_frame, sample_bbox):
        result = draw_bbox_topleft(sample_frame.copy(), sample_bbox, (0, 255, 0), label="test")
        assert result.shape == sample_frame.shape


class TestDrawOverlay:
    def test_adds_text(self, sample_frame):
        original = sample_frame.copy()
        result = draw_overlay(original, "TRACKING", 30.0)
        assert not np.array_equal(result, sample_frame)

    def test_coasting_includes_count(self, sample_frame):
        result = draw_overlay(sample_frame.copy(), "COASTING", 25.0, coast_count=5)
        assert result.shape == sample_frame.shape


class TestVisualize:
    def test_full_pipeline(self, sample_frame, sample_bbox, sample_kf_state):
        result = visualize(
            sample_frame, sample_bbox, sample_kf_state,
            confidence=0.8, state_name="TRACKING", fps=30.0
        )
        assert result.shape == sample_frame.shape
        assert not np.array_equal(result, sample_frame)

    def test_none_ai_bbox(self, sample_frame, sample_kf_state):
        result = visualize(
            sample_frame, None, sample_kf_state,
            confidence=0.0, state_name="COASTING", fps=25.0, coast_count=3
        )
        assert result.shape == sample_frame.shape

    def test_without_cv2(self, sample_frame, sample_bbox, sample_kf_state):
        with patch("tracker.visualizer.HAS_CV2", False):
            result = visualize(
                sample_frame, sample_bbox, sample_kf_state,
                confidence=0.8, state_name="TRACKING", fps=30.0
            )
            assert np.array_equal(result, sample_frame)
