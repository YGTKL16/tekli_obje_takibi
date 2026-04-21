"""Shared evaluation metrics for tracking: IoU, center distance, AUC, NormPrecision."""

import numpy as np


def compute_iou(box_a, box_b) -> float:
    """Compute IoU between two [x, y, w, h] boxes in top-left format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = aw * ah + bw * bh - inter

    if union <= 0:
        return 0.0
    return inter / union


def compute_center_distance(box_a, box_b) -> float:
    """Center distance between two [x, y, w, h] top-left boxes."""
    cx_a = box_a[0] + box_a[2] / 2
    cy_a = box_a[1] + box_a[3] / 2
    cx_b = box_b[0] + box_b[2] / 2
    cy_b = box_b[1] + box_b[3] / 2
    return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


def evaluate_sequence(gt_bboxes, pred_bboxes) -> tuple[float, float]:
    """Evaluate one sequence. Returns (auc, norm_precision)."""
    n = min(len(gt_bboxes), len(pred_bboxes))
    ious, dists, gt_diags = [], [], []

    for i in range(n):
        gt, pred = gt_bboxes[i], pred_bboxes[i]
        if gt[2] <= 0 or gt[3] <= 0:
            continue
        ious.append(compute_iou(gt, pred))
        dists.append(compute_center_distance(gt, pred))
        gt_diags.append(np.sqrt(gt[2] ** 2 + gt[3] ** 2))

    if not ious:
        return 0.0, 0.0

    ious_arr = np.array(ious)
    dists_arr = np.array(dists)
    gt_diags_arr = np.array(gt_diags)

    # AUC (Area Under Success Curve)
    thresholds = np.arange(0, 1.05, 0.05)
    auc = float(np.mean([np.mean(ious_arr >= t) for t in thresholds]))

    # Normalized Precision
    norm_dists = dists_arr / np.maximum(gt_diags_arr, 1e-6)
    np_thresholds = np.arange(0, 0.51, 0.01)
    norm_prec = float(np.mean([np.mean(norm_dists <= t) for t in np_thresholds]))

    return auc, norm_prec
