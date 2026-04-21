"""IoU-based detection-to-tracker association."""

from __future__ import annotations

import ctypes
import os

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ASSOCIATION_LIB = None


class _AssociationResultC(ctypes.Structure):
    _fields_ = [
        ("matched_index", ctypes.c_int32),
        ("matched_iou", ctypes.c_float),
        ("matched_score", ctypes.c_float),
        ("matched_cost", ctypes.c_float),
        ("has_match", ctypes.c_int32),
    ]


def _load_association_backend():
    """Load the C++ association backend via pybind or ctypes."""
    global _ASSOCIATION_LIB
    if _ASSOCIATION_LIB is not None:
        return _ASSOCIATION_LIB

    try:
        import tracker_cpp  # pyright: ignore[reportMissingImports]

        if hasattr(tracker_cpp, "associate_single_track"):
            _ASSOCIATION_LIB = ("pybind", tracker_cpp)
            return _ASSOCIATION_LIB
    except ImportError:
        tracker_cpp = None

    candidate_paths = []
    for build_dir in ("build", "build_debug", "build-verify"):
        candidate_paths.extend(
            [
                os.path.join(_PROJECT_ROOT, build_dir, "libtracker_association.so"),
                os.path.join(_PROJECT_ROOT, build_dir, "lib", "libtracker_association.so"),
                os.path.join(_PROJECT_ROOT, build_dir, "bin", "libtracker_association.so"),
            ]
        )

    for lib_path in candidate_paths:
        if not os.path.exists(lib_path):
            continue
        lib = ctypes.CDLL(lib_path)
        lib.tracker_associate_single_track.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.tracker_associate_single_track.restype = _AssociationResultC
        _ASSOCIATION_LIB = ("ctypes", lib)
        return _ASSOCIATION_LIB

    raise ImportError(
        "No association backend found. Build tracker_association or tracker_cpp first."
    )


def _as_box_array(boxes) -> np.ndarray:
    """Normalize boxes into a float32 array with shape (N, 4)."""
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    return arr.reshape(-1, 4)


def compute_iou_matrix(tracker_bboxes, detection_bboxes) -> np.ndarray:
    """Compute the pairwise IoU matrix for top-left [x, y, w, h] boxes."""
    trackers = _as_box_array(tracker_bboxes)
    detections = _as_box_array(detection_bboxes)

    if trackers.shape[0] == 0 or detections.shape[0] == 0:
        return np.zeros((trackers.shape[0], detections.shape[0]), dtype=np.float32)

    trk = trackers[:, None, :]
    det = detections[None, :, :]

    trk_w = np.maximum(trk[..., 2], 0.0)
    trk_h = np.maximum(trk[..., 3], 0.0)
    det_w = np.maximum(det[..., 2], 0.0)
    det_h = np.maximum(det[..., 3], 0.0)

    trk_x2 = trk[..., 0] + trk_w
    trk_y2 = trk[..., 1] + trk_h
    det_x2 = det[..., 0] + det_w
    det_y2 = det[..., 1] + det_h

    inter_x1 = np.maximum(trk[..., 0], det[..., 0])
    inter_y1 = np.maximum(trk[..., 1], det[..., 1])
    inter_x2 = np.minimum(trk_x2, det_x2)
    inter_y2 = np.minimum(trk_y2, det_y2)

    inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h

    union = trk_w * trk_h + det_w * det_h - inter
    valid_union = np.maximum(union, 1e-6)
    iou = inter / valid_union
    iou[union <= 0.0] = 0.0
    return iou.astype(np.float32, copy=False)


def build_cost_matrix(
    tracker_bboxes,
    detection_bboxes,
    detection_scores=None,
    score_weight: float = 0.0,
) -> np.ndarray:
    """Build a LAP cost matrix from IoU and optional detection scores."""
    iou_matrix = compute_iou_matrix(tracker_bboxes, detection_bboxes)
    cost_matrix = 1.0 - iou_matrix

    if detection_scores is not None and score_weight != 0.0:
        scores = np.asarray(detection_scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != iou_matrix.shape[1]:
            raise ValueError("detection_scores must match detection_bboxes length")
        cost_matrix = cost_matrix - (score_weight * scores.reshape(1, -1))

    return cost_matrix.astype(np.float32, copy=False)


def associate_detections_to_trackers(
    tracker_bboxes,
    detection_bboxes,
    detection_scores=None,
    iou_threshold: float = 0.3,
    score_weight: float = 0.0,
):
    """Associate tracker predictions and detections for the single-target flow."""
    trackers = _as_box_array(tracker_bboxes)
    detections = _as_box_array(detection_bboxes)

    if trackers.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(detections.shape[0], dtype=int),
        )
    if detections.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(trackers.shape[0], dtype=int),
            np.empty((0,), dtype=int),
        )

    if trackers.shape[0] != 1:
        raise ValueError("associate_detections_to_trackers currently supports a single tracker")

    backend_kind, backend = _load_association_backend()
    max_candidates = 32 if backend_kind == "ctypes" else int(backend.MAX_ASSOCIATION_CANDIDATES)
    if detections.shape[0] > max_candidates:
        raise ValueError(
            f"detection_bboxes exceeds MAX_ASSOCIATION_CANDIDATES={max_candidates}"
        )

    scores = None
    if detection_scores is not None:
        scores = np.asarray(detection_scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != detections.shape[0]:
            raise ValueError("detection_scores must match detection_bboxes length")

    if backend_kind == "pybind":
        matched_index, _, _, _, has_match = backend.associate_single_track(
            trackers[0],
            detections,
            scores,
            float(iou_threshold),
            float(score_weight),
        )
    else:
        tracker_arr = np.ascontiguousarray(trackers[0], dtype=np.float32)
        detection_arr = np.ascontiguousarray(detections, dtype=np.float32)
        score_arr = None if scores is None else np.ascontiguousarray(scores, dtype=np.float32)
        score_ptr = (
            None
            if score_arr is None
            else score_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        result = backend.tracker_associate_single_track(
            tracker_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            detection_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            score_ptr,
            ctypes.c_int32(detections.shape[0]),
            ctypes.c_float(float(iou_threshold)),
            ctypes.c_float(float(score_weight)),
        )
        matched_index = result.matched_index
        has_match = bool(result.has_match)

    if not has_match or matched_index < 0:
        return (
            np.empty((0, 2), dtype=int),
            np.array([0], dtype=int),
            np.arange(detections.shape[0], dtype=int),
        )

    matched_index = int(matched_index)
    unmatched_detections = np.arange(detections.shape[0], dtype=int)
    unmatched_detections = unmatched_detections[unmatched_detections != matched_index]
    return (
        np.array([[0, matched_index]], dtype=int),
        np.empty((0,), dtype=int),
        unmatched_detections,
    )
