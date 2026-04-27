"""Lucas-Kanade sparse optical flow jitter smoother.

Reduces bbox boundary jitter by comparing the AI-predicted bbox center
movement against the Lucas-Kanade optical flow displacement of Shi-Tomasi
corners tracked inside the previous output bbox.

Only the center (cx, cy) is smoothed; w/h is left unchanged for the
Kalman filter to handle.

Typical call sequence::

    smoother = LKJitterSmoother(max_corners=15, lk_alpha=0.5)
    smoother.reset()   # on each new sequence / tracker re-init
    for frame_idx, (prev_gray, curr_gray, prev_bbox, ai_bbox) in ...:
        smoothed_bbox = smoother.smooth(prev_gray, curr_gray, prev_bbox, ai_bbox)
"""
from __future__ import annotations

import numpy as np
import cv2


class LKJitterSmoother:
    """Smooths AI bbox jitter using Lucas-Kanade sparse optical flow.

    Parameters
    ----------
    max_corners : int
        Maximum Shi-Tomasi corners to track inside the bbox (default 15).
    quality_level : float
        Shi-Tomasi quality level (default 0.01).
    min_dist : float
        Minimum distance between detected corners in pixels (default 5).
    win_size : tuple[int, int]
        LK patch window size (default (21, 21)).
    max_level : int
        LK pyramid levels (default 3).
    lk_alpha : float
        Blending weight: 1.0 = pure AI bbox, 0.0 = pure LK prediction.
        Recommended 0.5 for balanced de-jittering (default 0.5).
    backtrack_err_thr : float
        Maximum allowed forward-backward error in pixels (default 2.0).
    """

    _LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

    def __init__(
        self,
        max_corners: int = 15,
        quality_level: float = 0.01,
        min_dist: float = 5.0,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
        lk_alpha: float = 0.5,
        backtrack_err_thr: float = 2.0,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_dist = min_dist
        self.win_size = win_size
        self.max_level = max_level
        self.lk_alpha = float(lk_alpha)
        self.backtrack_err_thr = float(backtrack_err_thr)
        self._prev_pts: np.ndarray | None = None  # shape (N, 1, 2) float32

    def reset(self) -> None:
        """Call when tracker is re-initialized (new sequence or Great Rescue)."""
        self._prev_pts = None

    # ------------------------------------------------------------------
    def _detect_corners(
        self, gray: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray | None:
        """Detect Shi-Tomasi corners inside the bbox region."""
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(gray.shape[1], int(x + w))
        y2 = min(gray.shape[0], int(y + h))
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_dist,
            mask=mask,
        )
        return pts  # (N, 1, 2) float32 or None

    # ------------------------------------------------------------------
    def smooth(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_bbox: np.ndarray,
        ai_bbox: np.ndarray,
    ) -> np.ndarray:
        """Smooth *ai_bbox* center using LK optical flow displacement.

        Parameters
        ----------
        prev_gray : ndarray
            Grayscale frame at t-1.
        curr_gray : ndarray
            Grayscale frame at t.
        prev_bbox : ndarray
            Output bbox [x, y, w, h] reported at t-1 (used for corner detection).
        ai_bbox : ndarray
            AI prediction [x, y, w, h] for frame t.

        Returns
        -------
        ndarray
            Smoothed [x, y, w, h] float32.  Falls back to *ai_bbox* when LK
            tracking fails or there are too few reliable corners.
        """
        ai_arr = np.asarray(ai_bbox, dtype=np.float32)
        prev_arr = np.asarray(prev_bbox, dtype=np.float32)

        # Ensure corners are detected in the previous frame's bbox region
        if self._prev_pts is None or len(self._prev_pts) == 0:
            self._prev_pts = self._detect_corners(prev_gray, prev_arr)

        if self._prev_pts is None or len(self._prev_pts) == 0:
            # No corners — seed for next frame from current AI prediction
            self._prev_pts = self._detect_corners(curr_gray, ai_arr)
            return ai_arr

        # ── Forward LK tracking ──────────────────────────────────────────
        next_pts, fwd_status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            self._prev_pts,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self._LK_CRITERIA,
        )
        if next_pts is None:
            self._prev_pts = self._detect_corners(curr_gray, ai_arr)
            return ai_arr

        # ── Backward check for consistency ───────────────────────────────
        back_pts, bck_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            prev_gray,
            next_pts,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self._LK_CRITERIA,
        )

        good = fwd_status.flatten() == 1
        if back_pts is not None and bck_status is not None:
            back_err = np.linalg.norm(
                self._prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            good &= (bck_status.flatten() == 1) & (back_err < self.backtrack_err_thr)

        if good.sum() < 3:
            # Too few reliable corners — refresh and skip smoothing this frame
            self._prev_pts = self._detect_corners(curr_gray, ai_arr)
            return ai_arr

        # ── Median displacement ──────────────────────────────────────────
        prev_good = self._prev_pts.reshape(-1, 2)[good]
        next_good = next_pts.reshape(-1, 2)[good]
        flow = next_good - prev_good
        lk_dx = float(np.median(flow[:, 0]))
        lk_dy = float(np.median(flow[:, 1]))

        # ── Center-only blending ─────────────────────────────────────────
        ai_cx = ai_arr[0] + ai_arr[2] * 0.5
        ai_cy = ai_arr[1] + ai_arr[3] * 0.5
        prev_cx = prev_arr[0] + prev_arr[2] * 0.5
        prev_cy = prev_arr[1] + prev_arr[3] * 0.5
        lk_cx = prev_cx + lk_dx
        lk_cy = prev_cy + lk_dy

        sm_cx = self.lk_alpha * ai_cx + (1.0 - self.lk_alpha) * lk_cx
        sm_cy = self.lk_alpha * ai_cy + (1.0 - self.lk_alpha) * lk_cy

        # Refresh corners at new smoothed location for next frame
        self._prev_pts = self._detect_corners(curr_gray, ai_arr)

        result = ai_arr.copy()
        result[0] = sm_cx - ai_arr[2] * 0.5
        result[1] = sm_cy - ai_arr[3] * 0.5
        return result
