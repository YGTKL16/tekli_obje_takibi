#!/usr/bin/env python3
"""Multi-engine precision comparison on 20-sequence subset.
Compares TRT-FP16, TRT-Mixed (head FP32), and optionally PyTorch FP32."""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

from tracker.trt_wrapper import TRTTrackWrapper
from tracker.data_utils import DATA_ROOT, load_manifest, parse_bbox_line, load_gt
from tracker.metrics import compute_iou, compute_center_distance
import numpy as np
import cv2

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SUBSET = [
    "dataset1/plane", "dataset1/surfer", "dataset1/volleyball",
    "dataset2/Girl2", "dataset2/Gull1", "dataset2/Kiting",
    "dataset2/ManRunning2", "dataset2/RcCar3", "dataset2/Surfing12",
    "dataset2/Wakeboarding2", "dataset3/air_conditioning_box2",
    "dataset3/basketball_player4-n", "dataset3/duck1_1",
    "dataset3/truck_night", "dataset4/car6", "dataset5/bike3",
    "dataset5/building2", "dataset5/car1_3", "dataset5/car1_s",
    "dataset5/person2_2",
]


def run_sequence(tracker, seq_id, seq_info):
    video_path = os.path.join(DATA_ROOT, seq_info["video_path"])
    ann_path = seq_info.get("annotation_path")
    annotations = []
    if ann_path:
        full = os.path.join(DATA_ROOT, ann_path)
        with open(full) as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(parse_bbox_line(line))
    init_bbox = annotations[0]

    cap = cv2.VideoCapture(video_path)
    n_frames = seq_info["n_frames"]
    pred_bboxes = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            init_arr = np.array(init_bbox, dtype=np.float32)
            tracker.init(frame_rgb, init_arr)
            pred_bboxes.append(init_bbox)
        else:
            ai_bbox, conf = tracker.track(frame_rgb)
            bbox = ai_bbox.tolist() if isinstance(ai_bbox, np.ndarray) else list(ai_bbox)
            pred_bboxes.append(bbox)
        frame_idx += 1
    cap.release()
    return pred_bboxes


def success_auc(ious):
    thresholds = np.arange(0, 1.05, 0.05)
    curve = [np.mean(ious >= t) for t in thresholds]
    return np.mean(curve)


def norm_prec_auc(dists, gt_diags):
    nd = dists / np.maximum(gt_diags, 1e-6)
    thresholds = np.arange(0, 0.51, 0.01)
    curve = [np.mean(nd <= t) for t in thresholds]
    return np.mean(curve)


def evaluate(gt_bboxes, pred_bboxes):
    n = min(len(gt_bboxes), len(pred_bboxes))
    ious, dists, diags = [], [], []
    for i in range(n):
        g, p = gt_bboxes[i], pred_bboxes[i]
        if g[2] <= 0 or g[3] <= 0:
            continue
        ious.append(compute_iou(g, p))
        dists.append(compute_center_distance(g, p))
        diags.append(np.sqrt(g[2] ** 2 + g[3] ** 2))
    if not ious:
        return 0, 0
    return success_auc(np.array(ious)), norm_prec_auc(np.array(dists), np.array(diags))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-pytorch", action="store_true",
                    help="Also run PyTorch FP32 backend (slow)")
    ap.add_argument("--checkpoint", default=None,
                    help="SGLATrack checkpoint for --include-pytorch")
    ap.add_argument("--engines", nargs="+", default=None,
                    help="Engine paths to compare (default: fp16 + mixed)")
    args = ap.parse_args()

    manifest = load_manifest()

    # Define backends to test
    backends = {}

    engine_fp16 = os.path.join(_PROJECT_ROOT, "models", "sglatrack_fp16.engine")
    engine_mixed = os.path.join(_PROJECT_ROOT, "models", "sglatrack_mixed.engine")

    if args.engines:
        for ep in args.engines:
            label = os.path.splitext(os.path.basename(ep))[0]
            backends[label] = ("trt", ep)
    else:
        if os.path.exists(engine_fp16):
            backends["TRT_FP16"] = ("trt", engine_fp16)
        if os.path.exists(engine_mixed):
            backends["TRT_Mixed"] = ("trt", engine_mixed)

    if args.include_pytorch:
        backends["PyTorch_FP32"] = ("pytorch", None)

    if not backends:
        print("No backends available. Check engine paths.")
        return

    print("=" * 130)
    print(f"Precision Comparison — {len(backends)} backends, {len(SUBSET)} sequences, AI-only (no KF)")
    print("=" * 130)

    # Run all backends
    results = {}  # label -> {"aucs": [...], "nps": [...], "time": float}
    for label, (btype, engine_path) in backends.items():
        print(f"\n[{label}] Running...")
        if btype == "trt":
            tracker = TRTTrackWrapper(engine_path=engine_path)
        else:
            from tracker.sglatrack_wrapper import SGLATrackWrapper
            tracker = SGLATrackWrapper(checkpoint_path=args.checkpoint)

        aucs, nps = [], []
        t0 = time.time()
        for seq_id in SUBSET:
            seq_info = manifest["train"][seq_id]
            gt = load_gt(seq_id, manifest)
            preds = run_sequence(tracker, seq_id, seq_info)
            auc, np_ = evaluate(gt, preds)
            aucs.append(auc)
            nps.append(np_)
        elapsed = time.time() - t0
        results[label] = {"aucs": aucs, "nps": nps, "time": elapsed}
        print(f"  Done in {elapsed:.1f}s — AUC={np.mean(aucs):.3f}, NP={np.mean(nps):.3f}")

    # Results table
    labels = list(results.keys())
    print("\n" + "=" * 130)

    # Header
    hdr = f"{'Sequence':<35}"
    for lb in labels:
        hdr += f" {'AUC_'+lb:>12}"
    if len(labels) >= 2:
        hdr += f"  {'delta':>7}"
    print(hdr)
    print("-" * 130)

    # Per-sequence rows
    for i, seq_id in enumerate(SUBSET):
        row = f"{seq_id:<35}"
        for lb in labels:
            row += f" {results[lb]['aucs'][i]:>12.3f}"
        if len(labels) >= 2:
            d = results[labels[-1]]["aucs"][i] - results[labels[0]]["aucs"][i]
            marker = "+" if d > 0.005 else ("-" if d < -0.005 else "=")
            row += f"  {d:>+6.3f}{marker}"
        print(row)

    # Summary
    print("-" * 130)
    row = f"{'MEAN':<35}"
    for lb in labels:
        row += f" {np.mean(results[lb]['aucs']):>12.3f}"
    if len(labels) >= 2:
        d = np.mean(results[labels[-1]]["aucs"]) - np.mean(results[labels[0]]["aucs"])
        row += f"  {d:>+6.3f}"
    print(row)

    # FinalScore summary
    print()
    for lb in labels:
        auc_m = np.mean(results[lb]["aucs"])
        np_m = np.mean(results[lb]["nps"])
        fs = 0.6 * auc_m + 0.4 * np_m
        print(f"  {lb:20s}  AUC={auc_m:.3f}  NP={np_m:.3f}  FinalScore={fs:.4f}  Time={results[lb]['time']:.1f}s")

    # Pairwise improvement count
    if len(labels) >= 2:
        base_aucs = results[labels[0]]["aucs"]
        for lb in labels[1:]:
            compare_aucs = results[lb]["aucs"]
            n_better = sum(1 for a, b in zip(base_aucs, compare_aucs) if b > a + 0.005)
            n_worse = sum(1 for a, b in zip(base_aucs, compare_aucs) if b < a - 0.005)
            n_same = len(SUBSET) - n_better - n_worse
            print(f"\n  {lb} vs {labels[0]}: better={n_better}, worse={n_worse}, same={n_same}")


if __name__ == "__main__":
    main()
