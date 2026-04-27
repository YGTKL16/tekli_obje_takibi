"""Global Motion Compensation: estimate camera ego-motion between consecutive frames.

ORB keypoints matched against a foreground-masked region produce a partial-affine
(4 DOF) transform that describes background shift. The C++ backend (`tracker_cpp`)
is preferred for speed (~2 ms); a pure-Python fallback is used when the native
module is unavailable.

Returned to the Kalman stage to warp the state mean before prediction; on weak or
degenerate inputs we emit a quality report so higher layers can veto GMC instead
of silently trusting a noisy transform.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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


@dataclass
class GMCRawStats:
    """Raw support statistics emitted by the estimator backend."""

    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    has_affine: bool = False


@dataclass
class GMCQualityReport:
    """Guardrail decision for one GMC estimate."""

    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    rot_deg: float = 0.0
    scale_delta: float = 0.0
    quality_state: str = "veto"
    raw_ok: bool = False
    reason: str = "uninitialized"

    @property
    def should_apply(self) -> bool:
        return self.quality_state == "good"

    @property
    def suppress_maneuver(self) -> bool:
        return self.quality_state != "good"


class GMCEstimator:
    """ORB + partial-affine Global Motion Compensation.

    Prefers the C++ ``tracker_cpp.GMCEstimator`` for performance. Falls back
    to a pure-Python path when the native module is not built.

    Usage::

        gmc = GMCEstimator()
        H, quality = gmc.estimate_with_quality(prev_bgr, curr_bgr, fg_bbox)
    """

    def __init__(
        self,
        n_features: int = 200,
        inlier_ratio_threshold: float = 0.3,
        min_matches: int = 6,
        ransac_reproj_threshold: float = 3.0,
        foreground_dilate_factor: float = 1.4,
        downsample: float = 0.5,
        quality_enabled: bool = True,
        veto_inlier_ratio: float = 0.2,
        borderline_inlier_ratio: float = 0.3,
        max_translation_frac_diag: float = 0.08,
        max_rotation_deg: float = 12.0,
        history_window: int = 5,
        history_outlier_mult: float = 3.0,
        force_python: bool = False,
    ):
        if not HAS_CV2:
            raise ImportError("OpenCV required for GMCEstimator")

        self.n_features = n_features
        self.inlier_ratio_threshold = inlier_ratio_threshold
        self.min_matches = min_matches
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.foreground_dilate_factor = foreground_dilate_factor
        self.downsample = downsample

        self.quality_enabled = bool(quality_enabled)
        self.veto_inlier_ratio = float(veto_inlier_ratio)
        self.borderline_inlier_ratio = float(borderline_inlier_ratio)
        self.max_translation_frac_diag = float(max_translation_frac_diag)
        self.max_rotation_deg = float(max_rotation_deg)
        self.history_window = max(int(history_window), 0)
        self.history_outlier_mult = float(history_outlier_mult)

        # Last-call telemetry.
        self.last_ok: bool = False
        self.last_raw_ok: bool = False
        self.last_inliers: int = 0
        self.last_quality: GMCQualityReport = GMCQualityReport()
        self.last_quality_state: str = self.last_quality.quality_state
        self._history: deque[tuple[float, float]] = deque()

        # ── C++ backend ──────────────────────────────────────────
        self._cpp_gmc = None
        if HAS_CPP_GMC and not force_python:
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
        """Backward-compatible helper returning only apply/no-apply."""
        H, quality = self.estimate_with_quality(prev_frame, curr_frame, foreground_bbox_xywh)
        return H, quality.should_apply

    def estimate_with_quality(
        self,
        prev_frame: np.ndarray | None,
        curr_frame: np.ndarray | None,
        foreground_bbox_xywh: np.ndarray,
    ) -> Tuple[np.ndarray, GMCQualityReport]:
        """Estimate 3x3 affine motion and classify it as good/borderline/veto."""
        if prev_frame is None or curr_frame is None:
            quality = GMCQualityReport(quality_state="veto", reason="missing_frame")
            self._store_last(quality, raw_ok=False)
            return np.eye(3, dtype=np.float64), quality

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame

        if self._cpp_gmc is not None and hasattr(self._cpp_gmc, "estimate_with_stats"):
            fg = np.asarray(foreground_bbox_xywh[:4], dtype=np.float32)
            H_eigen, raw_ok, stats = self._cpp_gmc.estimate_with_stats(
                np.ascontiguousarray(prev_gray),
                np.ascontiguousarray(curr_gray),
                fg,
            )
            H = np.asarray(H_eigen, dtype=np.float64)
            raw_stats = GMCRawStats(
                match_count=int(stats.match_count),
                inlier_count=int(stats.inlier_count),
                inlier_ratio=float(stats.inlier_ratio),
                has_affine=bool(stats.has_affine),
            )
        elif self._cpp_gmc is not None:
            fg = np.asarray(foreground_bbox_xywh[:4], dtype=np.float32)
            H_eigen, raw_ok = self._cpp_gmc.estimate(
                np.ascontiguousarray(prev_gray),
                np.ascontiguousarray(curr_gray),
                fg,
            )
            H = np.asarray(H_eigen, dtype=np.float64)
            if raw_ok:
                inferred_support = max(self.min_matches * 2, 12)
                raw_stats = GMCRawStats(
                    match_count=inferred_support,
                    inlier_count=inferred_support,
                    inlier_ratio=max(self.borderline_inlier_ratio, self.inlier_ratio_threshold),
                    has_affine=True,
                )
            else:
                raw_stats = GMCRawStats(has_affine=False)
        else:
            H, raw_ok, raw_stats = self._estimate_python(prev_gray, curr_gray, foreground_bbox_xywh)

        quality = self._classify_quality(H, bool(raw_ok), raw_stats, curr_gray.shape)
        if quality.should_apply and self.history_window > 0:
            self._history.append((float(np.hypot(quality.tx, quality.ty)), abs(quality.rot_deg)))
            while len(self._history) > self.history_window:
                self._history.popleft()

        self._store_last(quality, raw_ok=bool(raw_ok))
        return H, quality

    # ── Quality gating helpers ───────────────────────────────────

    def _store_last(self, quality: GMCQualityReport, *, raw_ok: bool) -> None:
        self.last_ok = quality.should_apply
        self.last_raw_ok = raw_ok
        self.last_inliers = int(quality.inlier_count)
        self.last_quality = quality
        self.last_quality_state = quality.quality_state

    def _transform_metrics(self, H: np.ndarray) -> tuple[float, float, float, float]:
        tx = float(H[0, 2])
        ty = float(H[1, 2])
        scale = float(np.sqrt(max(H[0, 0] ** 2 + H[1, 0] ** 2, 0.0)))
        rot_deg = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
        return tx, ty, rot_deg, scale - 1.0

    def _classify_quality(
        self,
        H: np.ndarray,
        raw_ok: bool,
        raw_stats: GMCRawStats,
        frame_shape: tuple[int, int],
    ) -> GMCQualityReport:
        tx, ty, rot_deg, scale_delta = self._transform_metrics(H)
        quality_state = "good" if raw_ok else "veto"
        reason = "ok" if raw_ok else "raw_fail"

        if self.quality_enabled:
            if not raw_stats.has_affine:
                quality_state = "veto"
                reason = "no_affine"
            elif raw_ok:
                min_good_inliers = max(self.min_matches * 2, 12)
                if (
                    raw_stats.inlier_ratio < self.borderline_inlier_ratio
                    or raw_stats.inlier_count < min_good_inliers
                ):
                    quality_state = "borderline"
                    reason = "low_support"
            elif raw_stats.inlier_ratio >= self.veto_inlier_ratio and raw_stats.inlier_count > 0:
                quality_state = "borderline"
                reason = "weak_support"

            frame_diag = float(np.hypot(frame_shape[0], frame_shape[1]))
            translation_mag = float(np.hypot(tx, ty))
            if (
                frame_diag > 0.0
                and translation_mag > self.max_translation_frac_diag * frame_diag
            ):
                quality_state = "veto"
                reason = "translation_cap"

            if abs(rot_deg) > self.max_rotation_deg:
                quality_state = "veto"
                reason = "rotation_cap"

            if self.history_window > 0 and len(self._history) > 0:
                hist_t = np.asarray([entry[0] for entry in self._history], dtype=np.float32)
                hist_r = np.asarray([entry[1] for entry in self._history], dtype=np.float32)
                trans_ref = max(float(np.median(hist_t)), 1.0)
                rot_ref = max(float(np.median(hist_r)), 0.5)
                history_jump = (
                    translation_mag > self.history_outlier_mult * trans_ref
                    or abs(rot_deg) > self.history_outlier_mult * rot_ref
                )
                if history_jump:
                    quality_state = "veto"
                    reason = "history_veto"

        return GMCQualityReport(
            match_count=raw_stats.match_count,
            inlier_count=raw_stats.inlier_count,
            inlier_ratio=raw_stats.inlier_ratio,
            tx=tx,
            ty=ty,
            rot_deg=rot_deg,
            scale_delta=scale_delta,
            quality_state=quality_state,
            raw_ok=raw_ok,
            reason=reason,
        )

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
    ) -> Tuple[np.ndarray, bool, GMCRawStats]:
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
            return np.eye(3, dtype=np.float64), False, GMCRawStats()
        if len(kp_prev) < self.min_matches or len(kp_curr) < self.min_matches:
            return np.eye(3, dtype=np.float64), False, GMCRawStats()

        matches = self._matcher.match(desc_prev, desc_curr)
        raw_stats = GMCRawStats(match_count=len(matches))
        if len(matches) < self.min_matches:
            return np.eye(3, dtype=np.float64), False, raw_stats

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
            return np.eye(3, dtype=np.float64), False, raw_stats

        inliers = int(inlier_mask.sum())
        raw_stats = GMCRawStats(
            match_count=len(matches),
            inlier_count=inliers,
            inlier_ratio=(inliers / len(matches)) if len(matches) > 0 else 0.0,
            has_affine=True,
        )

        H_s = np.eye(3, dtype=np.float64)
        H_s[:2, :] = affine.astype(np.float64)

        if scale < 1.0:
            inv_scale = 1.0 / scale
            H_s[0, 2] *= inv_scale
            H_s[1, 2] *= inv_scale

        raw_ok = (
            raw_stats.inlier_count >= self.min_matches
            and raw_stats.inlier_ratio >= self.inlier_ratio_threshold
        )
        return H_s, raw_ok, raw_stats
