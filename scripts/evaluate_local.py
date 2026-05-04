#!/usr/bin/env python3
"""Local evaluation: compute AUC, NormPrecision, and simulated FinalScore.

Usage:
    python scripts/evaluate_local.py                    # Evaluate outputs/submission_train.csv
    python scripts/evaluate_local.py --pred outputs/submission_train.csv
    python scripts/evaluate_local.py --seq dataset3/car1  # Single sequence
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from tracker.data_utils import load_gt, load_manifest  # noqa: E402
from tracker.metrics import compute_center_distance, compute_iou  # noqa: E402


def load_predictions(pred_path):
    """Load prediction CSV into dict: result_id -> [x, y, w, h]."""
    preds = {}
    with open(pred_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds[row["id"]] = [
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"])
            ]
    return preds


def success_curve(ious, thresholds=np.arange(0, 1.05, 0.05)):
    """Compute success curve (fraction of frames with IoU > threshold)."""
    curve = []
    for t in thresholds:
        curve.append(np.mean(ious >= t))
    return np.array(curve), thresholds


def precision_curve(dists, thresholds=np.arange(0, 51, 1)):
    """Compute precision curve (fraction of frames with distance < threshold)."""
    curve = []
    for t in thresholds:
        curve.append(np.mean(dists <= t))
    return np.array(curve), thresholds


def normalized_precision(dists, gt_diags):
    """Compute normalized precision curve."""
    # Normalize distances by GT bbox diagonal
    norm_dists = dists / np.maximum(gt_diags, 1e-6)
    thresholds = np.arange(0, 0.51, 0.01)
    curve = []
    for t in thresholds:
        curve.append(np.mean(norm_dists <= t))
    return np.array(curve), thresholds


def evaluate_sequence(gt_bboxes, pred_bboxes):
    """Evaluate one sequence. Returns dict with metrics."""
    n = min(len(gt_bboxes), len(pred_bboxes))
    ious = []
    dists = []
    gt_diags = []

    for i in range(n):
        gt = gt_bboxes[i]
        pred = pred_bboxes[i]

        # Skip frames where GT is 0,0,0,0 (target not visible)
        if gt[2] <= 0 or gt[3] <= 0:
            continue

        iou = compute_iou(gt, pred)
        dist = compute_center_distance(gt, pred)
        diag = np.sqrt(gt[2] ** 2 + gt[3] ** 2)

        ious.append(iou)
        dists.append(dist)
        gt_diags.append(diag)

    if not ious:
        return {"auc": 0, "norm_prec": 0, "prec_20": 0, "mean_iou": 0.0, "n_valid": 0}

    ious = np.array(ious)
    dists = np.array(dists)
    gt_diags = np.array(gt_diags)

    # AUC (Area Under Success Curve)
    sc, _ = success_curve(ious)
    auc = np.mean(sc)

    # Normalized Precision (AUC of normalized precision curve)
    npc, _ = normalized_precision(dists, gt_diags)
    norm_prec = np.mean(npc)

    # Precision at 20px
    prec_20 = np.mean(dists <= 20)

    return {
        "auc": auc,
        "norm_prec": norm_prec,
        "prec_20": prec_20,
        "mean_iou": float(np.mean(ious)),
        "n_valid": len(ious),
    }


def estimate_efficiency_score(flops_g=5.81, params_m=5.7, latency_ms=5.0, model_size_mb=25.0):
    """Estimate S_eff based on known model specs."""
    # Normalize: x~ = max(0, (x - budget) / budget), capped at 1.0
    flops_n = min(1.0, max(0.0, (flops_g - 30) / 30))
    params_n = min(1.0, max(0.0, (params_m - 50) / 50))
    latency_n = min(1.0, max(0.0, (latency_ms - 30) / 30))
    size_n = min(1.0, max(0.0, (model_size_mb - 500) / 500))

    s_eff = 0.25 * flops_n + 0.15 * params_n + 0.35 * latency_n + 0.25 * size_n
    return s_eff


def main():
    parser = argparse.ArgumentParser(description="Local evaluation for MTC-AIC4")
    parser.add_argument("--pred", default=os.path.join(PROJECT_ROOT, "outputs", "submission_train.csv"),
                        help="Prediction CSV path")
    parser.add_argument("--seq", default=None, help="Evaluate single sequence")
    parser.add_argument("--split", default=None, help="Only evaluate sequences from this split (e.g. public_lb)")
    args = parser.parse_args()

    manifest = load_manifest()
    preds = load_predictions(args.pred)

    # Group predictions by sequence
    seq_preds = defaultdict(dict)
    for result_id, bbox in preds.items():
        # result_id format: "dataset3/car1_42" -> seq="dataset3/car1", frame=42
        last_underscore = result_id.rfind("_")
        seq_id = result_id[:last_underscore]
        frame_idx = int(result_id[last_underscore + 1:])
        seq_preds[seq_id][frame_idx] = bbox

    # Evaluate each sequence
    all_metrics = []
    print(f"{'Sequence':<40} {'AUC':>6} {'NormP':>6} {'P@20':>6} {'mIoU':>6} {'Frames':>7}")
    print("-" * 80)

    for seq_id in sorted(seq_preds.keys()):
        if args.seq and seq_id != args.seq:
            continue

        gt_bboxes = load_gt(seq_id, manifest)
        if gt_bboxes is None:
            continue  # Skip sequences without annotation

        if args.split:
            # Only evaluate sequences belonging to the requested split
            split_seqs = manifest.get(args.split, {})
            if seq_id not in split_seqs:
                continue

        # Convert pred dict to ordered list
        pred_dict = seq_preds[seq_id]
        max_frame = max(pred_dict.keys())
        pred_bboxes = []
        for i in range(max_frame + 1):
            pred_bboxes.append(pred_dict.get(i, [0, 0, 0, 0]))

        metrics = evaluate_sequence(gt_bboxes, pred_bboxes)
        all_metrics.append(metrics)

        print(f"{seq_id:<40} {metrics['auc']:>6.3f} {metrics['norm_prec']:>6.3f} "
              f"{metrics['prec_20']:>6.3f} {metrics['mean_iou']:>6.3f} {metrics['n_valid']:>7}")

    if not all_metrics:
        print("[ERROR] No train sequences found to evaluate")
        sys.exit(1)

    # Aggregate
    auc_mean = np.mean([m["auc"] for m in all_metrics])
    norm_prec_mean = np.mean([m["norm_prec"] for m in all_metrics])
    prec20_mean = np.mean([m["prec_20"] for m in all_metrics])
    miou_mean = np.mean([m["mean_iou"] for m in all_metrics])

    print("-" * 80)
    print(f"{'MEAN':<40} {auc_mean:>6.3f} {norm_prec_mean:>6.3f} "
          f"{prec20_mean:>6.3f} {miou_mean:>6.3f}")

    # Scoring
    s_acc = 0.6 * auc_mean + 0.4 * norm_prec_mean
    s_eff = estimate_efficiency_score()  # SGLATrack-DeiT* specs
    final_score = s_acc - 0.2 * s_eff

    print(f"\n{'='*50}")
    print(f"Scoring Summary ({len(all_metrics)} sequences)")
    print(f"{'='*50}")
    print(f"  AUC (mean):          {auc_mean:.4f}")
    print(f"  NormPrecision (mean): {norm_prec_mean:.4f}")
    print(f"  S_acc = 0.6*AUC + 0.4*NP: {s_acc:.4f}")
    print(f"  S_eff (estimated):   {s_eff:.4f}")
    print(f"  FinalScore = S_acc - 0.2*S_eff: {final_score:.4f}")


if __name__ == "__main__":
    main()
