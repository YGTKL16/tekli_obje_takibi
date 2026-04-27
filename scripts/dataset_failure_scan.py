#!/usr/bin/env python3
"""Scan dataset predictions for per-sequence collapse patterns.

This is a lightweight forensic pass: it does not run the tracker.  It compares
existing predictions against train annotations and highlights the sequences
where tracking collapses into long low-IoU runs.

Examples:
    python scripts/dataset_failure_scan.py
    python scripts/dataset_failure_scan.py --pred-dir outputs/predictions --top 30
    python scripts/dataset_failure_scan.py --out outputs/failure_scan.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from tracker.data_utils import load_gt, load_manifest, parse_bbox_line  # noqa: E402
from tracker.metrics import compute_center_distance, compute_iou  # noqa: E402


@dataclass
class SequenceScan:
    seq_id: str
    auc: float
    norm_precision: float
    mean_iou: float
    n_frames: int
    native_fps: float
    init_area: float
    init_ar: float
    size_class: str
    ar_class: str
    longest_bad_run: int
    longest_bad_start: int
    worst_chunk_start: int
    worst_chunk_iou: float
    first_collapse_start: int
    failure_tag: str


def _size_class(area: float) -> str:
    if area < 500.0:
        return "tiny"
    if area < 5000.0:
        return "medium"
    return "large"


def _ar_class(ar: float) -> str:
    return "thin" if ar > 3.0 or ar < 0.33 else "normal"


def _prediction_files(pred_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(pred_dir.glob("*.txt")):
        stem = path.stem
        if "_" not in stem:
            continue
        dataset, name = stem.split("_", 1)
        files[f"{dataset}/{name}"] = path
    return files


def _load_predictions(path: Path) -> list[list[float]]:
    preds: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                preds.append(parse_bbox_line(line))
    return preds


def _auc_from_ious(ious: np.ndarray) -> float:
    thresholds = np.arange(0.0, 1.05, 0.05)
    return float(np.mean([np.mean(ious >= t) for t in thresholds]))


def _norm_precision(dists: np.ndarray, gt_diags: np.ndarray) -> float:
    norm_dists = dists / np.maximum(gt_diags, 1e-6)
    thresholds = np.arange(0.0, 0.51, 0.01)
    return float(np.mean([np.mean(norm_dists <= t) for t in thresholds]))


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_len = 0
    best_start = -1
    cur_len = 0
    cur_start = 0
    for idx, value in enumerate(mask):
        if value:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    return best_len, best_start


def _first_collapse(mask: np.ndarray, min_run: int) -> int:
    cur_len = 0
    cur_start = -1
    for idx, value in enumerate(mask):
        if value:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len >= min_run:
                return cur_start
        else:
            cur_len = 0
            cur_start = -1
    return -1


def _worst_chunk(ious: np.ndarray, chunk: int) -> tuple[int, float]:
    worst_start = 0
    worst_mean = float("inf")
    for start in range(0, len(ious), chunk):
        end = min(start + chunk, len(ious))
        if end - start < max(5, chunk // 4):
            continue
        mean_iou = float(np.mean(ious[start:end]))
        if mean_iou < worst_mean:
            worst_start = start
            worst_mean = mean_iou
    if not np.isfinite(worst_mean):
        return 0, 0.0
    return worst_start, worst_mean


def _tag_failure(
    *,
    seq_id: str,
    auc: float,
    n_frames: int,
    native_fps: float,
    init_area: float,
    init_ar: float,
    longest_bad_run: int,
    longest_bad_start: int,
    first_collapse_start: int,
) -> str:
    if auc >= 0.55 and longest_bad_run < max(30, n_frames // 8):
        return "mostly_ok"
    if longest_bad_start >= 0 and longest_bad_start <= 30 and longest_bad_run > n_frames * 0.4:
        return "early_lock_loss"
    if init_area < 500.0:
        return "tiny_target_fragile"
    if native_fps >= 60.0 and first_collapse_start >= 0:
        return "high_fps_lag_or_template_drift"
    lower = seq_id.lower()
    if "group" in lower or "basketball" in lower or "street" in lower:
        return "occlusion_or_id_switch"
    if init_ar > 3.0 or init_ar < 0.33:
        return "aspect_ratio_fragile"
    if first_collapse_start > n_frames * 0.3:
        return "late_drift"
    return "tracker_collapse"


def scan_sequence(
    seq_id: str,
    pred_path: Path,
    manifest: dict,
    *,
    bad_iou: float,
    collapse_min_run: int,
    chunk: int,
) -> SequenceScan | None:
    gt = load_gt(seq_id, manifest)
    if not gt:
        return None
    preds = _load_predictions(pred_path)
    n = min(len(gt), len(preds))
    if n <= 0:
        return None

    ious: list[float] = []
    dists: list[float] = []
    gt_diags: list[float] = []
    for gt_box, pred_box in zip(gt[:n], preds[:n]):
        if gt_box[2] <= 0.0 or gt_box[3] <= 0.0:
            continue
        ious.append(compute_iou(gt_box, pred_box))
        dists.append(compute_center_distance(gt_box, pred_box))
        gt_diags.append(float(np.hypot(gt_box[2], gt_box[3])))
    if not ious:
        return None

    ious_arr = np.asarray(ious, dtype=np.float32)
    dists_arr = np.asarray(dists, dtype=np.float32)
    diag_arr = np.asarray(gt_diags, dtype=np.float32)
    bad_mask = ious_arr < bad_iou
    longest_bad_run, longest_bad_start = _longest_true_run(bad_mask)
    first_collapse_start = _first_collapse(bad_mask, collapse_min_run)
    worst_chunk_start, worst_chunk_iou = _worst_chunk(ious_arr, chunk)

    info = manifest["train"][seq_id]
    init = gt[0]
    init_area = float(init[2] * init[3])
    init_ar = float(init[2] / init[3]) if init[3] > 0.0 else 0.0
    native_fps = float(info.get("native_fps", 30.0))
    auc = _auc_from_ious(ious_arr)
    tag = _tag_failure(
        seq_id=seq_id,
        auc=auc,
        n_frames=len(ious_arr),
        native_fps=native_fps,
        init_area=init_area,
        init_ar=init_ar,
        longest_bad_run=longest_bad_run,
        longest_bad_start=longest_bad_start,
        first_collapse_start=first_collapse_start,
    )

    return SequenceScan(
        seq_id=seq_id,
        auc=auc,
        norm_precision=_norm_precision(dists_arr, diag_arr),
        mean_iou=float(np.mean(ious_arr)),
        n_frames=len(ious_arr),
        native_fps=native_fps,
        init_area=init_area,
        init_ar=init_ar,
        size_class=_size_class(init_area),
        ar_class=_ar_class(init_ar),
        longest_bad_run=longest_bad_run,
        longest_bad_start=longest_bad_start,
        worst_chunk_start=worst_chunk_start,
        worst_chunk_iou=worst_chunk_iou,
        first_collapse_start=first_collapse_start,
        failure_tag=tag,
    )


def _write_csv(path: Path, rows: list[SequenceScan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SequenceScan.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = row.__dict__.copy()
            for key in ("auc", "norm_precision", "mean_iou", "init_area", "init_ar", "worst_chunk_iou"):
                data[key] = round(float(data[key]), 4)
            writer.writerow(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan predictions for dataset failure patterns")
    parser.add_argument("--pred-dir", default="outputs/predictions", help="Directory with per-sequence txt predictions")
    parser.add_argument("--top", type=int, default=25, help="Number of worst sequences to print")
    parser.add_argument("--bad-iou", type=float, default=0.2, help="IoU threshold treated as collapsed")
    parser.add_argument("--collapse-min-run", type=int, default=20, help="Consecutive bad frames needed for first collapse")
    parser.add_argument("--chunk", type=int, default=30, help="Chunk size for worst-window mean IoU")
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    manifest = load_manifest()
    pred_files = _prediction_files(Path(args.pred_dir))
    rows: list[SequenceScan] = []
    for seq_id in sorted(manifest.get("train", {})):
        pred_path = pred_files.get(seq_id)
        if pred_path is None:
            continue
        row = scan_sequence(
            seq_id,
            pred_path,
            manifest,
            bad_iou=args.bad_iou,
            collapse_min_run=args.collapse_min_run,
            chunk=args.chunk,
        )
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda item: (item.auc, -item.longest_bad_run))

    print(f"Scanned {len(rows)} train sequences from {args.pred_dir}")
    print("Worst sequences:")
    header = (
        f"{'seq':<38} {'AUC':>5} {'NP':>5} {'mIoU':>5} {'bad_run':>8} "
        f"{'bad@':>6} {'worst30':>8} {'w@':>5} {'area':>7} {'fps':>4} tag"
    )
    print(header)
    print("-" * len(header))
    for row in rows[: args.top]:
        print(
            f"{row.seq_id:<38} {row.auc:>5.3f} {row.norm_precision:>5.3f} "
            f"{row.mean_iou:>5.3f} {row.longest_bad_run:>8} "
            f"{row.longest_bad_start:>6} {row.worst_chunk_iou:>8.2f} "
            f"{row.worst_chunk_start:>5} {row.init_area:>7.0f} "
            f"{row.native_fps:>4.0f} {row.failure_tag}"
        )

    if args.out:
        _write_csv(Path(args.out), rows)
        print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
