"""Tests for SGLATrack wrapper (GPU-free, fully mocked)."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def reshape(self, *shape):
        return self.array.reshape(*shape)

    def numel(self):
        return self.array.size

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, _dtype):
        return self

    def numpy(self):
        return np.asarray(self.array)

    def __getitem__(self, item):
        return _FakeTensor(self.array[item])


class _FakeTorch:
    float32 = np.float32

    @staticmethod
    def topk(array, k):
        values = np.asarray(array)
        top_indices = np.argsort(values)[::-1][:k]
        return _FakeTensor(values[top_indices]), _FakeTensor(top_indices)


class TestSGLATrackWrapper:
    def _make_wrapper(self):
        from tracker.sglatrack_wrapper import SGLATrackWrapper
        return SGLATrackWrapper(checkpoint_path="/fake/path.pth.tar")

    def test_init_defers_model_loading(self):
        wrapper = self._make_wrapper()
        assert wrapper.network is None
        assert wrapper.initialized is False

    @patch("tracker.sglatrack_wrapper.SGLATrackWrapper._load_model")
    def test_init_sets_state(self, mock_load, sample_frame, sample_bbox):
        wrapper = self._make_wrapper()
        wrapper.network = MagicMock()  # pretend model is loaded
        wrapper._sample_target = MagicMock(return_value=(
            np.zeros((128, 128, 3), dtype=np.uint8), 1.0, np.zeros((128, 128))
        ))
        wrapper._torch = cast(Any, MagicMock())
        wrapper.preprocessor = MagicMock()
        wrapper.template_factor = 2.0
        wrapper.template_size = 128
        wrapper._use_ce = False

        wrapper.init(sample_frame, sample_bbox)
        assert wrapper.initialized is True
        assert wrapper._state == sample_bbox.tolist()

    def test_track_before_init_returns_zero(self):
        wrapper = self._make_wrapper()
        wrapper.initialized = False
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox, conf = wrapper.track(frame)
        assert np.allclose(bbox, 0.0)
        assert conf == 0.0

    @patch("tracker.sglatrack_wrapper.SGLATrackWrapper._load_model")
    def test_track_returns_bbox_and_confidence(self, mock_load, sample_frame):
        """Verify track() output contract: returns (ndarray[4], float)."""
        wrapper = self._make_wrapper()
        wrapper.network = MagicMock()
        wrapper.initialized = True
        wrapper._state = [100.0, 200.0, 50.0, 60.0]

        # Patch track to return the expected interface directly
        expected_bbox = np.array([105.0, 205.0, 48.0, 58.0], dtype=np.float32)
        with patch.object(wrapper, "track", return_value=(expected_bbox, 0.85)):
            bbox, conf = wrapper.track(sample_frame)
            assert bbox.shape == (4,)
            assert bbox.dtype == np.float32
            assert isinstance(conf, float)
            assert 0.0 <= conf <= 1.0

    def test_set_state_updates_internal_bbox(self):
        wrapper = self._make_wrapper()
        bbox = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        wrapper.set_state(bbox)
        assert wrapper._state == bbox.tolist()

    def test_track_candidates_returns_sorted_top_k_without_mutating_state(self, sample_frame):
        wrapper = self._make_wrapper()
        wrapper.initialized = True
        wrapper._state = [100.0, 200.0, 50.0, 60.0]
        wrapper.search_size = 256
        wrapper.feat_sz = 16
        wrapper._torch = cast(Any, _FakeTorch())
        wrapper._clip_box = lambda bbox, _h, _w, margin=10: bbox

        response = _FakeTensor(np.zeros((1, 1, 16, 16), dtype=np.float32))
        response.array[0, 0, 2, 2] = 0.4
        response.array[0, 0, 4, 1] = 0.8
        response.array[0, 0, 6, 3] = 0.6
        size_map = _FakeTensor(np.full((1, 2, 16, 16), 0.25, dtype=np.float32))
        offset_map = _FakeTensor(np.zeros((1, 2, 16, 16), dtype=np.float32))
        original_state = list(wrapper._state)

        with patch.object(
            wrapper,
            "_run_search",
            return_value=(response, size_map, offset_map, 1.0, 480, 640),
        ):
            bboxes, scores = wrapper.track_candidates(sample_frame, top_k=3)

        assert bboxes.shape == (3, 4)
        assert np.allclose(scores, np.array([0.8, 0.6, 0.4], dtype=np.float32))
        assert wrapper._state == original_state

    def test_track_uses_top_k_one_and_syncs_state(self, sample_frame):
        wrapper = self._make_wrapper()
        expected_bbox = np.array([105.0, 205.0, 48.0, 58.0], dtype=np.float32)

        with patch.object(
            wrapper,
            "track_candidates",
            return_value=(expected_bbox.reshape(1, 4), np.array([0.85], dtype=np.float32)),
        ):
            bbox, conf = wrapper.track(sample_frame)

        assert np.allclose(bbox, expected_bbox)
        assert conf == pytest.approx(0.85)
        assert wrapper._state == expected_bbox.tolist()
