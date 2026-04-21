#!/usr/bin/env python3
"""Prepare MTC-AIC4 contest_release data for SGLATrack training.

Converts:
  data/contest_release/{datasetX}/{sequence}/*.mp4 + annotation.txt

Into:
  data/mtcaic4_sot/
    splits/train.txt
    splits/val.txt
    sequences/<seq_id>/groundtruth.txt
    sequences/<seq_id>/visible.txt
    sequences/<seq_id>/img/00000001.jpg
    ...
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MTC-AIC4 sequences for SGLATrack training")
    parser.add_argument(
        "--source-root",
        default="data/contest_release",
        help="Path to the contest_release dataset root",
    )
    parser.add_argument(
        "--output-root",
        default="data/mtcaic4_sot",
        help="Path to the converted image-sequence dataset root",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio taken from the train split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/val split",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Optional cap for a quick smoke conversion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already prepared sequences",
    )
    return parser.parse_args()


def load_manifest(source_root: Path):
    manifest_path = source_root / "metadata" / "contestant_manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def load_annotations(annotation_path: Path):
    rows = []
    with open(annotation_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return rows


def stratified_split(train_items, val_ratio, seed):
    grouped = defaultdict(list)
    for seq_id, seq_info in train_items:
        grouped[seq_info["dataset"]].append((seq_id, seq_info))

    rng = random.Random(seed)
    train_split = []
    val_split = []

    for dataset_name in sorted(grouped.keys()):
        items = grouped[dataset_name]
        rng.shuffle(items)
        if len(items) <= 1 or val_ratio <= 0.0:
            train_split.extend(items)
            continue

        val_count = int(round(len(items) * val_ratio))
        val_count = max(1, val_count)
        val_count = min(val_count, len(items) - 1)

        val_split.extend(items[:val_count])
        train_split.extend(items[val_count:])

    train_split.sort(key=lambda item: item[0])
    val_split.sort(key=lambda item: item[0])
    return train_split, val_split


def sequence_is_ready(seq_dir: Path, expected_frames: int):
    img_dir = seq_dir / "img"
    gt_path = seq_dir / "groundtruth.txt"
    if not img_dir.exists() or not gt_path.exists():
        return False

    frame_count = len(list(img_dir.glob("*.jpg")))
    if frame_count != expected_frames:
        return False

    with open(gt_path) as f:
        gt_count = sum(1 for line in f if line.strip())
    return gt_count == expected_frames


def write_groundtruth(seq_dir: Path, annotations):
    gt_path = seq_dir / "groundtruth.txt"
    with open(gt_path, "w", newline="") as f:
        writer = csv.writer(f)
        for bbox in annotations:
            writer.writerow([f"{bbox[0]:.2f}", f"{bbox[1]:.2f}", f"{bbox[2]:.2f}", f"{bbox[3]:.2f}"])

    visible_path = seq_dir / "visible.txt"
    with open(visible_path, "w") as f:
        for bbox in annotations:
            visible = 1 if bbox[2] > 0 and bbox[3] > 0 else 0
            f.write(f"{visible}\n")


def extract_frames(video_path: Path, seq_dir: Path, frame_limit: int, overwrite: bool):
    img_dir = seq_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for image_path in img_dir.glob("*.jpg"):
            image_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while frame_idx < frame_limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = img_dir / f"{frame_idx + 1:08}.jpg"
        if overwrite or not frame_path.exists():
            if not cv2.imwrite(str(frame_path), frame):
                cap.release()
                raise RuntimeError(f"Failed to write frame: {frame_path}")
        frame_idx += 1

    cap.release()

    if frame_idx != frame_limit:
        raise RuntimeError(
            f"Frame count mismatch for {video_path}: expected {frame_limit}, extracted {frame_idx}"
        )


def prepare_sequence(source_root: Path, output_root: Path, seq_id: str, seq_info: dict, overwrite: bool):
    video_path = source_root / seq_info["video_path"]
    annotation_path = source_root / seq_info["annotation_path"]
    annotations = load_annotations(annotation_path)

    frame_limit = min(len(annotations), int(seq_info["n_frames"]))
    annotations = annotations[:frame_limit]

    seq_dir = output_root / "sequences" / Path(seq_id)
    seq_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and sequence_is_ready(seq_dir, frame_limit):
        return frame_limit

    write_groundtruth(seq_dir, annotations)
    extract_frames(video_path, seq_dir, frame_limit, overwrite=overwrite)

    meta_path = seq_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "seq_id": seq_id,
                "dataset": seq_info["dataset"],
                "video_path": seq_info["video_path"],
                "annotation_path": seq_info["annotation_path"],
                "n_frames": frame_limit,
                "native_fps": seq_info["native_fps"],
            },
            f,
            indent=2,
        )

    return frame_limit


def write_split_file(output_root: Path, split_name: str, items):
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / f"{split_name}.txt", "w") as f:
        for seq_id, _ in items:
            f.write(f"{seq_id}\n")


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(source_root)
    train_items = sorted(manifest["train"].items(), key=lambda item: item[0])

    if args.max_seqs is not None:
        train_items = train_items[: args.max_seqs]

    train_split, val_split = stratified_split(train_items, args.val_ratio, args.seed)

    print(f"[INFO] Source train sequences: {len(train_items)}")
    print(f"[INFO] Prepared split sizes: train={len(train_split)}, val={len(val_split)}")
    print(f"[INFO] Output root: {output_root}")

    write_split_file(output_root, "train", train_split)
    write_split_file(output_root, "val", val_split)

    all_items = train_split + val_split
    total_frames = 0
    for index, (seq_id, seq_info) in enumerate(all_items, start=1):
        frame_count = prepare_sequence(source_root, output_root, seq_id, seq_info, overwrite=args.overwrite)
        total_frames += frame_count
        print(f"[{index:03d}/{len(all_items):03d}] {seq_id} -> {frame_count} frames")

    print(f"[OK] Prepared {len(all_items)} sequences and {total_frames} frames")


if __name__ == "__main__":
    main()
