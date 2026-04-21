"""Visualization: draw bboxes, KF prediction, state, FPS on frames."""

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# Colors (BGR)
COLOR_TRACKING = (0, 255, 0)     # green
COLOR_LOW_CONF = (0, 255, 255)   # yellow
COLOR_COASTING = (0, 0, 255)     # red
COLOR_TEXT     = (255, 255, 255)  # white


def draw_bbox_topleft(frame, bbox_xywh, color, label="", thickness=2):
    """Draw a bounding box from top-left [x, y, w, h] format."""
    if not HAS_CV2:
        return frame
    x, y, w, h = bbox_xywh
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_overlay(frame, state_name, fps, coast_count=0):
    """Draw state + FPS overlay on top-left."""
    if not HAS_CV2:
        return frame
    text = f"{state_name} | FPS: {fps:.1f}"
    if state_name == "COASTING":
        text += f" | coast: {coast_count}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    return frame


def visualize(frame, ai_bbox, kf_state, confidence, state_name, fps, coast_count=0):
    """Full visualization for one frame.

    Args:
        frame: BGR image (numpy array)
        ai_bbox: AI detection [x, y, w, h] or None
        kf_state: Kalman state [x, y, w, h, vx, vy, vw, vh]
        confidence: AI confidence [0, 1]
        state_name: "TRACKING" / "COASTING" / "LOST"
        fps: current FPS
        coast_count: frames spent coasting
    """
    if not HAS_CV2:
        return frame

    vis = frame.copy()

    # Draw AI bbox
    if ai_bbox is not None and confidence > 0:
        ai_color = COLOR_TRACKING if confidence >= 0.3 else COLOR_LOW_CONF
        conf_label = f"AI {confidence:.2f}"
        draw_bbox_topleft(vis, ai_bbox, ai_color, conf_label)

    # Draw KF prediction
    if kf_state is not None:
        kf_bbox = kf_state[:4]
        draw_bbox_topleft(vis, kf_bbox, COLOR_COASTING, "KF", thickness=1)

    # Overlay
    draw_overlay(vis, state_name, fps, coast_count)

    return vis
