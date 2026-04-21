"""Tests for TRT wrapper (GPU-free, fully mocked)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from tracker.trt_wrapper import (
    _hann1d,
    _hann2d,
    _preprocess,
    _clip_box,
    TRTTrackWrapper,
)


class TestHelperFunctions:
    def test_hann1d_shape(self):
        h = _hann1d(16)
        assert h.shape == (16,)
        assert h[0] > 0
        assert h[7] > h[0]  # peak near center

    def test_hann2d_shape(self):
        h = _hann2d(16, 16)
        assert h.shape == (1, 1, 16, 16)

    def test_preprocess_shape(self):
        patch = np.zeros((128, 128, 3), dtype=np.uint8)
        result = _preprocess(patch)
        assert result.shape == (1, 3, 128, 128)
        assert result.dtype == np.float32

    def test_preprocess_normalization(self):
        # All-zero image should produce negative values due to ImageNet mean subtraction
        patch = np.zeros((128, 128, 3), dtype=np.uint8)
        result = _preprocess(patch)
        assert result.mean() < 0

    def test_clip_box_in_bounds(self):
        bbox = _clip_box([100, 100, 50, 50], 480, 640)
        assert bbox == [100, 100, 50, 50]

    def test_clip_box_negative(self):
        bbox = _clip_box([-10, -20, 50, 50], 480, 640)
        assert bbox[0] >= 0
        assert bbox[1] >= 0

    def test_clip_box_overflow(self):
        bbox = _clip_box([630, 470, 50, 50], 480, 640, margin=10)
        assert bbox[0] + bbox[2] <= 640
        assert bbox[1] + bbox[3] <= 480


class TestTRTTrackWrapper:
    def test_init_defers_engine_loading(self):
        wrapper = TRTTrackWrapper(engine_path="/fake/engine.engine")
        assert wrapper._engine_loaded is False
        assert wrapper.initialized is False

    def test_track_before_init_returns_zero(self):
        wrapper = TRTTrackWrapper(engine_path="/fake/engine.engine")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox, conf = wrapper.track(frame)
        assert np.allclose(bbox, 0.0)
        assert conf == 0.0

    def test_config_defaults(self):
        wrapper = TRTTrackWrapper(engine_path="/fake/engine.engine")
        assert wrapper.template_size == 128
        assert wrapper.search_size == 256
        assert wrapper.feat_sz == 16
        assert wrapper.template_factor == 2.0
        assert wrapper.search_factor == 4.0

    def test_set_state_updates_internal_bbox(self):
        wrapper = TRTTrackWrapper(engine_path="/fake/engine.engine")
        bbox = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        wrapper.set_state(bbox)
        assert wrapper._state == bbox.tolist()

    def test_track_candidates_returns_sorted_top_k_without_mutating_state(self):
        wrapper = TRTTrackWrapper(engine_path="/fake/engine.engine")
        wrapper.initialized = True
        wrapper._state = [100.0, 200.0, 50.0, 60.0]

        response = np.zeros((1, 1, 16, 16), dtype=np.float32)
        response[0, 0, 1, 1] = 0.2
        response[0, 0, 5, 2] = 0.9
        response[0, 0, 3, 4] = 0.6
        size_map = np.full((1, 2, 16, 16), 0.25, dtype=np.float32)
        offset_map = np.zeros((1, 2, 16, 16), dtype=np.float32)
        original_state = list(wrapper._state)

        with patch.object(
            wrapper,
            "_run_search",
            return_value=(response, size_map, offset_map, 1.0, 480, 640),
        ):
            bboxes, scores = wrapper.track_candidates(np.zeros((480, 640, 3), dtype=np.uint8), top_k=3)

        assert bboxes.shape == (3, 4)
        assert scores.shape == (3,)
        assert np.allclose(scores, np.array([0.9, 0.6, 0.2], dtype=np.float32))
        assert wrapper._state == original_state

    def test_track_uses_top_k_one_and_syncs_state(self):
        wrapper = TRTTrackWrapper(engine_path="/fake/engine.engine")
        expected_bbox = np.array([105.0, 205.0, 48.0, 58.0], dtype=np.float32)

        with patch.object(
            wrapper,
            "track_candidates",
            return_value=(expected_bbox.reshape(1, 4), np.array([0.85], dtype=np.float32)),
        ):
            bbox, conf = wrapper.track(np.zeros((480, 640, 3), dtype=np.uint8))

        assert np.allclose(bbox, expected_bbox)
        assert conf == pytest.approx(0.85)
        assert wrapper._state == expected_bbox.tolist()
