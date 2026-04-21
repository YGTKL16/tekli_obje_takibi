"""Shared data loading utilities for competition sequences."""

import json
import os
import re

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(_PROJECT_ROOT, "data", "contest_release")
MANIFEST_PATH = os.path.join(DATA_ROOT, "metadata", "contestant_manifest.json")


def load_manifest(manifest_path=MANIFEST_PATH):
    with open(manifest_path) as f:
        return json.load(f)


def parse_bbox_line(line) -> list[float]:
    """Parse bbox line separated by commas or whitespace. Returns [x, y, w, h]."""
    parts = re.split(r"[\s,]+", line.strip())
    if len(parts) < 4:
        raise ValueError(f"Invalid bbox line: {line!r}")
    return [float(x) for x in parts[:4]]


def load_gt(seq_id, manifest, data_root=DATA_ROOT):
    """Load ground truth annotations for a train sequence."""
    train_seqs = manifest.get("train", {})
    if seq_id not in train_seqs:
        return None
    ann_path = os.path.join(data_root, train_seqs[seq_id]["annotation_path"])
    bboxes = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bboxes.append(parse_bbox_line(line))
    return bboxes
