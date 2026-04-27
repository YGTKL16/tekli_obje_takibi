"""Tests for GMC (Global Motion Compensation) estimator."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tracker.gmc import GMCEstimator, GMCRawStats


def _make_textured_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # High-frequency grayscale pattern so ORB has corners to lock onto.
    noise = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)


def _shift_frame(frame: np.ndarray, tx: float, ty: float) -> np.ndarray:
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def test_identity_when_frames_match():
    gmc = GMCEstimator(downsample=1.0)
    frame = _make_textured_frame(seed=1)
    bbox = np.array([200.0, 200.0, 50.0, 50.0], dtype=np.float32)
    H, quality = gmc.estimate_with_quality(frame, frame, bbox)
    assert quality.quality_state == "good"
    assert quality.should_apply is True
    assert H.shape == (3, 3)
    # Translation components should be near zero for matching frames.
    assert abs(float(H[0, 2])) < 2.0
    assert abs(float(H[1, 2])) < 2.0


def test_recovers_pure_translation():
    gmc = GMCEstimator(downsample=1.0)
    prev = _make_textured_frame(seed=7)
    tx, ty = 20.0, -10.0
    curr = _shift_frame(prev, tx, ty)
    bbox = np.array([300.0, 200.0, 60.0, 60.0], dtype=np.float32)
    H, quality = gmc.estimate_with_quality(prev, curr, bbox)
    assert quality.quality_state == "good"
    # Homography (prev → curr) should shift centre by approx (tx, ty).
    cx, cy = 400.0, 300.0
    p = np.array([cx, cy, 1.0])
    p_warp = H @ p
    p_warp = p_warp / p_warp[2]
    assert abs(p_warp[0] - (cx + tx)) < 3.0
    assert abs(p_warp[1] - (cy + ty)) < 3.0


def test_returns_failure_on_textureless_frames():
    gmc = GMCEstimator(downsample=1.0, min_matches=6)
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)
    bbox = np.array([100.0, 100.0, 40.0, 40.0], dtype=np.float32)
    H, quality = gmc.estimate_with_quality(flat, flat, bbox)
    assert quality.quality_state == "veto"
    # On failure we return identity so callers can short-circuit.
    assert np.allclose(H, np.eye(3), atol=1e-9)


def test_none_frame_returns_failure():
    gmc = GMCEstimator(downsample=1.0)
    bbox = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    H, quality = gmc.estimate_with_quality(None, _make_textured_frame(), bbox)
    assert quality.quality_state == "veto"
    assert np.allclose(H, np.eye(3))


def test_foreground_mask_excludes_object():
    """Foreground keypoints should not drive the homography, even when
    the foreground moves differently from the background."""
    gmc = GMCEstimator(downsample=1.0, foreground_dilate_factor=1.4)
    prev = _make_textured_frame(seed=3)
    # Move the background by (5, 5) — this is the ground truth the estimator
    # should recover even though we will place moving foreground content too.
    curr = _shift_frame(prev, 5.0, 5.0)

    # Stamp a high-contrast rectangle that lies INSIDE the declared foreground
    # region and moves differently (jumps 40 px).  If the mask worked, the
    # homography should still read ~(5, 5) because these foreground corners
    # were excluded.
    bbox = np.array([200.0, 200.0, 80.0, 80.0], dtype=np.float32)
    for i in range(20):
        y0 = 210 + i
        x0 = 210 + i
        prev[y0:y0 + 5, x0:x0 + 5] = 255
        # Foreground jumps in curr.
        curr[y0 + 40:y0 + 45, x0 + 40:x0 + 45] = 255

    H, quality = gmc.estimate_with_quality(prev, curr, bbox)
    assert quality.quality_state == "good"
    # The dominant motion the estimator sees should still be the 5-px
    # background shift, not the 40-px foreground jump.
    assert abs(float(H[0, 2]) - 5.0) < 10.0
    assert abs(float(H[1, 2]) - 5.0) < 10.0


def test_low_support_affine_maps_to_borderline():
    gmc = GMCEstimator(force_python=True, history_window=0)
    stats = GMCRawStats(match_count=20, inlier_count=5, inlier_ratio=0.25, has_affine=True)
    quality = gmc._classify_quality(np.eye(3, dtype=np.float64), False, stats, (240, 320))
    assert quality.quality_state == "borderline"
    assert quality.reason == "weak_support"


def test_history_jump_is_vetoed():
    gmc = GMCEstimator(
        force_python=True,
        history_window=3,
        history_outlier_mult=2.5,
        max_translation_frac_diag=0.5,
    )
    gmc._history.extend([(2.0, 0.2), (2.5, 0.1), (1.8, 0.15)])
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 12.0
    stats = GMCRawStats(match_count=40, inlier_count=30, inlier_ratio=0.75, has_affine=True)
    quality = gmc._classify_quality(H, True, stats, (240, 320))
    assert quality.quality_state == "veto"
    assert quality.reason == "history_veto"
