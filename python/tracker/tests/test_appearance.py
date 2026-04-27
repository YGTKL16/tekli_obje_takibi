"""Tests for D4: ReIDMemory appearance module."""

import numpy as np
import pytest

from tracker.appearance import ReIDMemory, _extract_histogram, ReidMatch


def _make_frame(h: int = 60, w: int = 80, color: tuple = (100, 150, 200)) -> np.ndarray:
    """Solid-colour BGR frame."""
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _bbox(x: int = 5, y: int = 5, w: int = 20, h: int = 20) -> np.ndarray:
    return np.array([x, y, w, h], dtype=np.float32)


class TestExtractHistogram:
    def test_returns_24_floats_by_default(self):
        frame = _make_frame()
        patch = frame[5:25, 5:25]
        hist = _extract_histogram(patch)
        assert hist.shape == (24,)
        assert hist.dtype == np.float32

    def test_l2_normalised(self):
        frame = _make_frame()
        patch = frame[5:25, 5:25]
        hist = _extract_histogram(patch)
        norm = float(np.linalg.norm(hist))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_patch_returns_zeros(self):
        hist = _extract_histogram(np.zeros((0, 0, 3), dtype=np.uint8))
        assert hist.shape == (24,)
        assert np.all(hist == 0)

    def test_different_colours_differ(self):
        red_patch = np.full((32, 32, 3), (0, 0, 200), dtype=np.uint8)
        blue_patch = np.full((32, 32, 3), (200, 0, 0), dtype=np.uint8)
        h1 = _extract_histogram(red_patch)
        h2 = _extract_histogram(blue_patch)
        sim = float(np.dot(h1, h2))
        assert sim < 0.99  # distinct colours → low similarity


class TestReIDMemoryUpdate:
    def test_starts_empty(self):
        mem = ReIDMemory()
        assert len(mem) == 0

    def test_update_adds_entry(self):
        mem = ReIDMemory()
        frame = _make_frame()
        mem.update(0, frame, _bbox())
        assert len(mem) == 1

    def test_update_degenerate_bbox_is_ignored(self):
        mem = ReIDMemory()
        frame = _make_frame()
        mem.update(0, frame, np.array([0, 0, 0, 0], dtype=np.float32))
        assert len(mem) == 0

    def test_update_out_of_bounds_bbox_is_ignored(self):
        mem = ReIDMemory()
        frame = _make_frame()
        mem.update(0, frame, np.array([1000, 1000, 50, 50], dtype=np.float32))
        assert len(mem) == 0

    def test_maxlen_respected(self):
        mem = ReIDMemory(maxlen=3)
        frame = _make_frame()
        for i in range(10):
            mem.update(i, frame, _bbox())
        assert len(mem) == 3

    def test_frozen_blocks_update(self):
        mem = ReIDMemory()
        frame = _make_frame()
        mem.freeze()
        mem.update(0, frame, _bbox())
        assert len(mem) == 0

    def test_unfreeze_allows_update(self):
        mem = ReIDMemory()
        frame = _make_frame()
        mem.freeze()
        mem.unfreeze()
        mem.update(0, frame, _bbox())
        assert len(mem) == 1


class TestReIDMemorySimilarity:
    def test_returns_none_when_empty(self):
        mem = ReIDMemory()
        frame = _make_frame()
        result = mem.similarity(frame, _bbox())
        assert result is None

    def test_returns_reid_match(self):
        mem = ReIDMemory()
        frame = _make_frame()
        mem.update(5, frame, _bbox())
        result = mem.similarity(frame, _bbox())
        assert isinstance(result, ReidMatch)
        assert result.frame_idx == 5
        assert 0.0 <= result.similarity <= 1.0 + 1e-6

    def test_identical_patch_high_similarity(self):
        mem = ReIDMemory()
        frame = _make_frame(color=(120, 80, 200))
        mem.update(0, frame, _bbox())
        result = mem.similarity(frame, _bbox())
        assert result is not None
        assert result.similarity > 0.95

    def test_different_colour_lower_similarity(self):
        mem = ReIDMemory()
        frame_red = _make_frame(color=(0, 0, 200))
        frame_blue = _make_frame(color=(200, 0, 0))
        mem.update(0, frame_red, _bbox())
        result = mem.similarity(frame_blue, _bbox())
        assert result is not None
        # Red vs blue should be lower than identical (no hard threshold here)
        assert result.similarity < 1.0

    def test_best_match_is_closest(self):
        """Buffer has two entries; query matches the second one better."""
        mem = ReIDMemory()
        frame_a = _make_frame(color=(0, 0, 200))   # red
        frame_b = _make_frame(color=(200, 0, 0))   # blue
        query = _make_frame(color=(200, 0, 0))      # blue query → matches b

        mem.update(1, frame_a, _bbox())
        mem.update(2, frame_b, _bbox())

        result = mem.similarity(query, _bbox())
        assert result is not None
        assert result.frame_idx == 2  # blue entry

    def test_degenerate_query_returns_none(self):
        mem = ReIDMemory()
        frame = _make_frame()
        mem.update(0, frame, _bbox())
        result = mem.similarity(frame, np.array([0, 0, 0, 0], dtype=np.float32))
        assert result is None


class TestReIDMemoryClear:
    def test_clear_empties_buffer(self):
        mem = ReIDMemory()
        frame = _make_frame()
        mem.update(0, frame, _bbox())
        mem.freeze()
        mem.clear()
        assert len(mem) == 0
        # Should be unfrozen after clear
        mem.update(1, frame, _bbox())
        assert len(mem) == 1
