#!/usr/bin/env python3
"""Cache AI (TRT) outputs for tracking sequences.

Runs TRTTrackWrapper once per sequence, saves (ai_bboxes, confs, gt, init_bbox,
frame_w, frame_h) as NPZ.  Subsequent tuning scripts replay filter logic on
cached data without TRT inference.

Usage:
    python scripts/cache_ai_outputs.py                   # cache hardcoded SUBSET (24 seq)
    python scripts/cache_ai_outputs.py --split train     # cache all 255 train sequences
    python scripts/cache_ai_outputs.py --cache-dir /tmp  # custom dir
    python scripts/cache_ai_outputs.py --force           # overwrite existing
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import cv2
import numpy as np
from tracker.data_utils import DATA_ROOT, load_gt, load_manifest, parse_bbox_line

SUBSET = [
    "dataset1/plane", "dataset1/surfer", "dataset1/volleyball",
    "dataset2/Girl2", "dataset2/Gull1", "dataset2/Kiting",
    "dataset2/ManRunning2", "dataset2/RcCar3", "dataset2/Surfing12",
    "dataset2/Wakeboarding2", "dataset3/air_conditioning_box2",
    "dataset3/basketball_player4-n", "dataset3/duck1_1",
    "dataset3/truck_night", "dataset3/uav1", "dataset4/car6",
    "dataset4/uav1", "dataset5/bike3", "dataset5/building2",
    "dataset5/car1_3", "dataset5/car1_s", "dataset5/person2_2",
    "dataset5/uav1_2", "dataset5/uav4",
]


def cache_key(seq_id: str) -> str:
    """dataset1/plane -> dataset1__plane"""
    return seq_id.replace("/", "__")


def cache_sequence(tracker, seq_id, seq_info, manifest, out_dir, force=False):
    """Run AI-only tracking on one sequence and save NPZ cache."""
    fname = cache_key(seq_id) + ".npz"
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path) and not force:
        print(f"  [SKIP] {seq_id} — cache exists")
        return out_path

    # Load annotations
    ann_path = seq_info.get("annotation_path")
    if not ann_path:
        print(f"  [WARN] {seq_id} — no annotation_path, skipping")
        return None
    ann_full = os.path.join(DATA_ROOT, ann_path)
    with open(ann_full) as f:
        first_line = f.readline().strip()
    init_bbox = parse_bbox_line(first_line)

    # Load ground truth
    gt_list = load_gt(seq_id, manifest)
    if gt_list is None:
        print(f"  [WARN] {seq_id} — no GT, skipping")
        return None

    # Open video
    video_path = os.path.join(DATA_ROOT, seq_info["video_path"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERR] {seq_id} — cannot open {video_path}")
        return None

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = seq_info["n_frames"]

    # Frame 0: init only (no AI output — use init_bbox as both ai_bbox and conf=1.0)
    ai_bboxes = [init_bbox]
    confs = [1.0]
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            init_arr = np.array(init_bbox, dtype=np.float32)
            tracker.init(frame_rgb, init_arr)
        else:
            ai_bbox, conf = tracker.track(frame_rgb)
            ai_list = ai_bbox.tolist() if isinstance(ai_bbox, np.ndarray) else list(ai_bbox)
            ai_bboxes.append(ai_list)
            confs.append(float(conf))
        frame_idx += 1

    cap.release()

    # Save
    np.savez_compressed(
        out_path,
        ai_bboxes=np.array(ai_bboxes, dtype=np.float32),
        confs=np.array(confs, dtype=np.float32),
        gt=np.array(gt_list, dtype=np.float32),
        init_bbox=np.array(init_bbox, dtype=np.float32),
        frame_w=np.int32(fw),
        frame_h=np.int32(fh),
    )
    print(f"  [OK] {seq_id} — {frame_idx} frames → {fname}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Cache AI outputs for tracking sequences")
    parser.add_argument("--cache-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "cache", "ai_outputs"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing caches")
    parser.add_argument("--engine", default=None, help="TRT engine path")
    parser.add_argument("--backend", default="tensorrt", choices=["tensorrt", "pytorch"],
                        help="Inference backend (default: tensorrt)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth.tar (for --backend pytorch)")
    parser.add_argument("--split", default=None, choices=["train", "public_lb"],
                        help="Cache entire manifest split instead of hardcoded SUBSET")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.cache_dir)
    os.makedirs(out_dir, exist_ok=True)

    manifest = load_manifest()

    if args.split in ("train", "public_lb"):
        seq_list = sorted(manifest[args.split].keys())
        print(f"[INFO] --split {args.split}: {len(seq_list)} sequences")
    else:
        seq_list = SUBSET
        print(f"[INFO] Using hardcoded SUBSET: {len(seq_list)} sequences")

    if args.backend == "pytorch":
        from tracker.sglatrack_wrapper import SGLATrackWrapper  # pyright: ignore[reportMissingImports]
        tracker = SGLATrackWrapper(checkpoint_path=args.checkpoint, association_enabled=False)
        print(f"[INFO] Backend: pytorch, checkpoint={args.checkpoint}")
    else:
        from tracker.trt_wrapper import TRTTrackWrapper  # pyright: ignore[reportMissingImports]
        tracker = TRTTrackWrapper(engine_path=args.engine)
        print(f"[INFO] Backend: tensorrt, engine={args.engine}")

    print(f"Caching {len(seq_list)} sequences → {out_dir}")
    t0 = time.time()
    cached = 0

    split_key = args.split if args.split in ("train", "public_lb") else "train"
    for seq_id in seq_list:
        seq_info = manifest[split_key].get(seq_id)
        if seq_info is None:
            # Fall back: search all splits
            seq_info = None
            for sk, ss in manifest.items():
                if isinstance(ss, dict) and seq_id in ss:
                    seq_info = ss[seq_id]
                    break
        if seq_info is None:
            print(f"  [WARN] {seq_id} not found in manifest, skipping")
            continue
        result = cache_sequence(tracker, seq_id, seq_info, manifest, out_dir,
                                force=args.force)
        if result:
            cached += 1

    elapsed = time.time() - t0
    print(f"\nDone: {cached}/{len(seq_list)} cached in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
