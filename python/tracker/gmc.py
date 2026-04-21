"""Global Motion Compensation: estimate camera ego-motion between consecutive frames.

ORB keypoints matched against a foreground-masked region produce a partial-affine
(4 DOF) transform that describes background shift.  The C++ backend (`tracker_cpp`)
is preferred for speed (~2 ms); a pure-Python fallback is used when the native
module is unavailable.

Returned to the Kalman stage to warp the state mean before prediction; on degenerate
inputs we emit `success=False` so the caller can inflate process noise instead of
falling back to identity silently.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

try:
    import cv2 as _cv2

    cv2: Any = _cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# ── Try C++ backend first ────────────────────────────────────────
try:
    import tracker_cpp as _tracker_cpp

    _cpp: Any = _tracker_cpp
    HAS_CPP_GMC = hasattr(_tracker_cpp, "GMCEstimator")
except ImportError:
    _cpp = None
    HAS_CPP_GMC = False


class GMCEstimator:
    """ORB + partial-affine Global Motion Compensation.

    Prefers the C++ ``tracker_cpp.GMCEstimator`` for performance.  Falls back
    to a pure-Python path when the native module is not built.

    Usage::

        gmc = GMCEstimator()
        H, ok = gmc.estimate(prev_bgr, curr_bgr, foreground_bbox_xywh)
    """

    def __init__(
        self,
        n_features: int = 200,
        inlier_ratio_threshold: float = 0.3,
        min_matches: int = 6,
        ransac_reproj_threshold: float = 3.0,
        foreground_dilate_factor: float = 1.4,
        downsample: float = 0.5,
    ):
        if not HAS_CV2:
            raise ImportError("OpenCV required for GMCEstimator")

        self.n_features = n_features
        self.inlier_ratio_threshold = inlier_ratio_threshold
        self.min_matches = min_matches
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.foreground_dilate_factor = foreground_dilate_factor
        self.downsample = downsample

        # ── C++ backend ──────────────────────────────────────────
        self._cpp_gmc = None
        if HAS_CPP_GMC:
            self._cpp_gmc = _cpp.GMCEstimator(
                n_features=n_features,
                min_matches=min_matches,
                inlier_ratio_thresh=inlier_ratio_threshold,
                ransac_reproj_thresh=ransac_reproj_threshold,
                dilate_factor=foreground_dilate_factor,
                downsample=downsample,
            )

        # ── Python fallback objects ──────────────────────────────
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # ── Public API ───────────────────────────────────────────────

    def estimate(
        self,
        prev_frame: np.ndarray | None,
        curr_frame: np.ndarray | None,
        foreground_bbox_xywh: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Estimate 3×3 affine-embedded homography mapping prev → curr.

        Returns ``(H, success)``.  *H* is always 3×3 float64; when
        ``success=False``, *H* is identity and the caller must boost KF
        process noise for this step.
        """
        if prev_frame is None or curr_frame is None:
            return np.eye(3, dtype=np.float64), False

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame

        # ── C++ fast path (zero-copy) ────────────────────────────
        if self._cpp_gmc is not None:
            fg = np.asarray(foreground_bbox_xywh[:4], dtype=np.float32)
            H_eigen, ok = self._cpp_gmc.estimate(
                np.ascontiguousarray(prev_gray),
                np.ascontiguousarray(curr_gray),
                fg,
            )
            return np.asarray(H_eigen, dtype=np.float64), bool(ok)

        # ── Python fallback ──────────────────────────────────────
        return self._estimate_python(prev_gray, curr_gray, foreground_bbox_xywh)

    # ── Python fallback implementation ───────────────────────────

    def _build_mask(self, shape_hw: Tuple[int, int], fg_bbox_xywh: np.ndarray) -> np.ndarray:
        h, w = shape_hw
        mask = np.full((h, w), 255, dtype=np.uint8)
        fx, fy, fw, fh = [float(v) for v in fg_bbox_xywh[:4]]
        if fw <= 0.0 or fh <= 0.0:
            return mask
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        dw, dh = fw * self.foreground_dilate_factor, fh * self.foreground_dilate_factor
        x0 = max(0, int(cx - dw / 2.0))
        y0 = max(0, int(cy - dh / 2.0))
        x1 = min(w, int(cx + dw / 2.0))
        y1 = min(h, int(cy + dh / 2.0))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
        return mask

    def _estimate_python(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        foreground_bbox_xywh: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        scale = self.downsample
        if scale < 1.0:
            prev_gray_s = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            curr_gray_s = cv2.resize(curr_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            bbox_s = foreground_bbox_xywh.astype(np.float64) * scale
        else:
            prev_gray_s = prev_gray
            curr_gray_s = curr_gray
            bbox_s = foreground_bbox_xywh.astype(np.float64)

        mask_prev = self._build_mask(prev_gray_s.shape, bbox_s)

        kp_prev, desc_prev = self._orb.detectAndCompute(prev_gray_s, mask_prev)
        kp_curr, desc_curr = self._orb.detectAndCompute(curr_gray_s, mask_prev)

        if desc_prev is None or desc_curr is None:
            return np.eye(3, dtype=np.float64), False
        if len(kp_prev) < self.min_matches or len(kp_curr) < self.min_matches:
            return np.eye(3, dtype=np.float64), False

        matches = self._matcher.match(desc_prev, desc_curr)
        if len(matches) < self.min_matches:
            return np.eye(3, dtype=np.float64), False

        pts_prev = np.asarray(
            [kp_prev[m.queryIdx].pt for m in matches],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        pts_curr = np.asarray(
            [kp_curr[m.trainIdx].pt for m in matches],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            pts_prev, pts_curr, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )

        if affine is None or inlier_mask is None:
            return np.eye(3, dtype=np.float64), False

        inliers = int(inlier_mask.sum())
        if inliers < self.min_matches:
            return np.eye(3, dtype=np.float64), False

        inlier_ratio = inliers / len(matches)
        if inlier_ratio < self.inlier_ratio_threshold:
            return np.eye(3, dtype=np.float64), False

        # Embed 2×3 affine into 3×3
        H_s = np.eye(3, dtype=np.float64)
        H_s[:2, :] = affine.astype(np.float64)

        if scale < 1.0:
            inv_scale = 1.0 / scale
            H_s[0, 2] *= inv_scale
            H_s[1, 2] *= inv_scale

        return H_s, True
