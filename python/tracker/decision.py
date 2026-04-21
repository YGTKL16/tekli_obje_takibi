"""Decision maker: determines whether to update KF or coast based on AI confidence."""

import numpy as np

from tracker.metrics import compute_iou


class DecisionMaker:
    """Asymmetric trust: AI leads when confident, KF coasts briefly when AI is lost.

    Two thresholds + max coast:
      conf >= coast_threshold: trust AI, feed KF silently (zero smoothing)
      conf < coast_threshold AND coast_frames <= max_coast: blind flight
      conf < coast_threshold AND coast_frames > max_coast: fall back to AI
    """

    def __init__(self, conf_threshold: float = 0.18, iou_threshold: float = 0.2,
                 coast_threshold: float = 0.05, max_coast_frames: int = 5,
                 max_area_frac: float = 0.25,
                 aspect_ratio_range: tuple = (0.2, 5.0),
                 max_center_jump_frac: float = 0.30):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.coast_threshold = coast_threshold
        self.max_coast_frames = max_coast_frames
        self.max_area_frac = max_area_frac
        self.aspect_ratio_range = aspect_ratio_range
        self.max_center_jump_frac = max_center_jump_frac
        self._coast_counter = 0

    def decide(self, confidence: float) -> str:
        """Returns 'coast' or 'trust_ai'.

        'coast': discard AI, use KF predict() only.
        'trust_ai': feed AI measurement to KF, output AI bbox.
        """
        if confidence < self.coast_threshold:
            self._coast_counter += 1
            if self._coast_counter <= self.max_coast_frames:
                return "coast"
            # Exceeded max coast: KF prediction is stale, fall back to AI
            return "trust_ai"
        else:
            self._coast_counter = 0
            return "trust_ai"

    def should_coast(self, confidence: float) -> bool:
        """Convenience wrapper: returns True if decide() == 'coast'."""
        return self.decide(confidence) == "coast"

    def should_update(
        self,
        confidence: float,
        ai_bbox: np.ndarray,
        kf_predicted: np.ndarray,
    ) -> bool:
        """
        Returns True if AI measurement is trustworthy (high confidence + consistent).

        Args:
            confidence: AI detection confidence [0, 1]
            ai_bbox: AI detection [x, y, w, h] (top-left format)
            kf_predicted: KF predicted state [x, y, w, h, vx, vy, ...]
        """
        if confidence < self.conf_threshold:
            return False

        kf_bbox = kf_predicted[:4]
        iou = compute_iou(ai_bbox, kf_bbox)

        if np.any(kf_bbox != 0) and iou < self.iou_threshold:
            return False

        return True

    def is_measurement_sane(
        self,
        ai_bbox,
        confidence: float,
        frame_w: int,
        frame_h: int,
        prev_bbox=None,
    ) -> bool:
        """Hard geometric/confidence gate — rejects degenerate AI outputs.

        Rejects when any of:
          - confidence below coast_threshold (hard floor, poison-prevention)
          - width or height <= 0
          - area exceeds max_area_frac of frame area
          - aspect ratio outside valid band
          - center jumps >max_center_jump_frac of frame diagonal (if prev_bbox given)
        """
        if confidence < self.coast_threshold:
            return False

        x, y, w, h = float(ai_bbox[0]), float(ai_bbox[1]), float(ai_bbox[2]), float(ai_bbox[3])

        if w <= 0.0 or h <= 0.0:
            return False

        frame_area = float(frame_w) * float(frame_h)
        if frame_area > 0.0 and (w * h) / frame_area > self.max_area_frac:
            return False

        ar = w / h
        if ar < self.aspect_ratio_range[0] or ar > self.aspect_ratio_range[1]:
            return False

        if prev_bbox is not None:
            px, py, pw, ph = float(prev_bbox[0]), float(prev_bbox[1]), \
                             float(prev_bbox[2]), float(prev_bbox[3])
            cur_cx, cur_cy = x + w / 2.0, y + h / 2.0
            prv_cx, prv_cy = px + pw / 2.0, py + ph / 2.0
            diag = (float(frame_w) ** 2 + float(frame_h) ** 2) ** 0.5
            jump = ((cur_cx - prv_cx) ** 2 + (cur_cy - prv_cy) ** 2) ** 0.5
            if diag > 0.0 and jump / diag > self.max_center_jump_frac:
                return False

        return True

    def is_measurement_mahalanobis_ok(
        self,
        ai_bbox: np.ndarray,
        kf_filter,
        chi2_threshold: float = 23.51,
    ) -> bool:
        """Statistical innovation gate — rejects physically impossible measurements.

        Computes d² = (z - H·x̂)ᵀ S⁻¹ (z - H·x̂) via the C++ IMMFilter binding.
        Returns True (accept) when d² < chi2_threshold.

        χ²(4 dof) reference:
            p=0.10 →  7.78  (loose)
            p=0.01 → 13.28  (moderate)
            p=0.001→ 18.47  (strict, default — aerial/vehicle recommended)
            p=1e-4 → 23.51  (paranoid)

        Args:
            ai_bbox:        AI detection [x, y, w, h].
            kf_filter:      IMMFilter pybind11 instance (must expose mahalanobis_sq).
            chi2_threshold: Rejection threshold. Tune per category via imm_tuned.yaml.

        Returns:
            True if measurement is statistically consistent with predicted state.
            Degrades gracefully to True when filter is uninitialized or binding absent.
        """
        if kf_filter is None or not hasattr(kf_filter, 'mahalanobis_sq'):
            return True  # graceful degradation — no C++ binding available
        if not kf_filter.is_initialized():
            return True  # first measurement always accepted
        z = np.asarray(ai_bbox[:4], dtype=np.float32)
        if not np.all(np.isfinite(z)):
            return False  # NaN / Inf → hard reject
        d2 = float(kf_filter.mahalanobis_sq(z))
        return d2 < chi2_threshold
