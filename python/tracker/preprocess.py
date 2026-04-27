"""Frame preprocessing utilities for the tracker pipeline.

Provides ROI-based CLAHE (Contrast Limited Adaptive Histogram Equalization)
for improving AI detection quality in low-light / low-contrast frames.
Only the predicted search region is enhanced — not the whole frame — to
preserve template statistics for matched-filter tracking.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


def apply_roi_clahe(
    frame_rgb: np.ndarray,
    bbox: np.ndarray | list[float],
    *,
    clip_limit: float = 2.0,
    roi_scale: float = 3.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE to the predicted-search ROI and return enhanced frame copy.

    Enhances only the region around the predicted bbox (expanded by
    ``roi_scale`` on each side). CLAHE is applied to the L channel in LAB
    colour space so hue/saturation are preserved.  Returns the same array
    unchanged when OpenCV is unavailable, bbox is degenerate, or the ROI
    is smaller than one tile.

    Args:
        frame_rgb:  HxWx3 RGB uint8 array — not mutated.
        bbox:       [x, y, w, h] predicted bounding box (top-left + size).
        clip_limit: CLAHE contrast limiting factor [0.5, 8.0].
                    Higher → more aggressive enhancement.
        roi_scale:  Multiplier applied to bbox extent to build the search ROI
                    [1.5, 6.0]. 3.0 covers a typical 4× search window.
        tile_size:  CLAHE tile grid size (applied as NxN) [4, 16].

    Returns:
        New ``np.ndarray`` with CLAHE applied inside the ROI, or the
        original array if enhancement was skipped.
    """
    if not HAS_CV2 or frame_rgb is None:
        return frame_rgb
    assert cv2 is not None

    b = np.asarray(bbox, dtype=np.float32).ravel()
    if b.shape[0] < 4:
        return frame_rgb

    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if w < 1.0 or h < 1.0:
        return frame_rgb

    fh, fw = frame_rgb.shape[:2]
    cx, cy = x + w * 0.5, y + h * 0.5
    rw, rh = w * roi_scale, h * roi_scale

    x1 = max(0, int(cx - rw * 0.5))
    y1 = max(0, int(cy - rh * 0.5))
    x2 = min(fw, int(cx + rw * 0.5))
    y2 = min(fh, int(cy + rh * 0.5))

    # Minimum ROI must fit at least one tile
    min_px = max(tile_size, 8)
    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return frame_rgb

    result = frame_rgb.copy()
    roi = result[y1:y2, x1:x2]

    # LAB: L channel only — no colour shift
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    clahe_obj = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_size), int(tile_size)),
    )
    lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
    result[y1:y2, x1:x2] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result
