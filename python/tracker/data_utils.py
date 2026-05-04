"""Shared data loading utilities for competition sequences."""

import json
import os
import re

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(_PROJECT_ROOT, "data", "contest_release")
SOT_DATA_ROOT = os.path.join(_PROJECT_ROOT, "MISIR PROJESİ", "SOT DATA")
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
    """Load ground truth annotations for a sequence (any split).

    Priority: official competition `groundtruth_rect.txt` under
    `data/contest_release/...` → SOT DATA `tracking_results.txt` (legacy) →
    manifest annotation_path (init bbox only).
    """
    # Preferred: official competition ground truth (per-frame rectangles)
    gt_path = os.path.join(data_root, seq_id, "groundtruth_rect.txt")
    if os.path.exists(gt_path):
        bboxes = []
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                bboxes.append(parse_bbox_line(line))
        return bboxes

    # Legacy fallback: SOT DATA tracking_results.txt
    sot_path = os.path.join(SOT_DATA_ROOT, seq_id, "tracking_results.txt")
    if os.path.exists(sot_path):
        bboxes = []
        with open(sot_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                bboxes.append(parse_bbox_line(line))
        return bboxes

    # Fallback: search manifest for annotation_path
    seq_info = None
    for split_seqs in manifest.values():
        if isinstance(split_seqs, dict) and seq_id in split_seqs:
            seq_info = split_seqs[seq_id]
            break
    if seq_info is None:
        return None
    ann_path = os.path.join(data_root, seq_info["annotation_path"])
    bboxes = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bboxes.append(parse_bbox_line(line))
    return bboxes
