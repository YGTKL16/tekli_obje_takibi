#!/usr/bin/env python3
"""Occlusion drill-down: frame-by-frame Raw-AI vs Hybrid (AI+IMM+KF) comparison
for a target sequence set.

For each target sequence:
  1. Run raw pipeline (use_kf=False) → pred_raw bboxes.
  2. Run hybrid pipeline (use_kf=True + tuned IMM cfg + GMC + adaptive-R) → pred_hyb.
  3. Build per-frame table (IoU, dist, event labels).
  4. Partition frames into occlusion window (occluded + ±margin) vs visible control.
  5. Compute ΔScore on each partition (raw vs hybrid).
  6. Emit CSV + markdown summary with a "surgical narrative" of each occlusion
     segment.

Usage:
    python scripts/occlusion_drill.py \
        --seq dataset5/person17_1 dataset5/group3_4 dataset4/person17 \
        --imm-config configs/imm_tuned.yaml --gmc --adaptive-r \
        --occ-margin 15 --out reports/occlusion_drill
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml

from ab_test import run_sequence  # noqa: E402  — reused

from tracker.data_utils import load_gt, load_manifest  # noqa: E402
from tracker.metrics import compute_center_distance, compute_iou  # noqa: E402
from tracker.trt_wrapper import TRTTrackWrapper  # noqa: E402


@dataclass
class FrameRow:
    idx: int
    gt: list[float]
    visible: bool
    raw_bbox: list[float]
    hyb_bbox: list[float]
    raw_iou: float
    hyb_iou: float
    raw_dist: float
    hyb_dist: float
    events: list[str] = field(default_factory=list)


def _run_on_sequence(tracker, seq_id: str, manifest: dict, use_kf: bool,
                     imm_cfg: dict | None, imm_cfg_path: str | None,
                     gmc_enabled: bool, adaptive_r_enabled: bool) -> list[list[float]]:
    """Wrapper around ab_test.run_sequence returning a list of per-frame bboxes."""
    seq_info = manifest["train"][seq_id]
    return run_sequence(
        tracker, seq_id, seq_info, manifest,
        use_kf=use_kf, kf_mode="baseline",
        gmc_enabled=gmc_enabled, adaptive_r_enabled=adaptive_r_enabled,
        imm_cfg=imm_cfg, imm_cfg_path=imm_cfg_path,
    )


def _build_rows(gt: list[list[float]], raw: list[list[float]], hyb: list[list[float]]
                ) -> list[FrameRow]:
    n = min(len(gt), len(raw), len(hyb))
    rows: list[FrameRow] = []
    for i in range(n):
        g = gt[i]
        visible = g[2] > 0 and g[3] > 0
        if visible:
            riou = float(compute_iou(g, raw[i]))
            hiou = float(compute_iou(g, hyb[i]))
            rdist = float(compute_center_distance(g, raw[i]))
            hdist = float(compute_center_distance(g, hyb[i]))
        else:
            riou = hiou = 0.0
            rdist = hdist = 0.0
        rows.append(FrameRow(
            idx=i, gt=list(g), visible=visible,
            raw_bbox=list(raw[i]), hyb_bbox=list(hyb[i]),
            raw_iou=riou, hyb_iou=hiou, raw_dist=rdist, hyb_dist=hdist,
        ))
    return rows


def _mark_events(rows: list[FrameRow], raw_lost_run: int = 3) -> None:
    n = len(rows)
    for i in range(n):
        r = rows[i]
        if i > 0:
            prev = rows[i - 1]
            if prev.visible and not r.visible:
                r.events.append("OCC_ENTER")
            if not prev.visible and r.visible:
                r.events.append("OCC_EXIT")
        if not r.visible:
            continue
        if r.raw_iou < 0.1 and r.hyb_iou >= 0.3:
            r.events.append("HYBRID_SAVE")
        if r.hyb_iou < 0.1 and r.raw_iou >= 0.3:
            r.events.append("HYBRID_MISS")
    # RAW_LOST requires runs of ≥raw_lost_run consecutive visible frames w/ iou<0.1
    run = 0
    for r in rows:
        if not r.visible:
            run = 0
            continue
        if r.raw_iou < 0.1:
            run += 1
        else:
            run = 0
        if run >= raw_lost_run:
            r.events.append("RAW_LOST")


def _occlusion_window_mask(rows: list[FrameRow], margin: int) -> list[bool]:
    n = len(rows)
    mask = [False] * n
    i = 0
    while i < n:
        if not rows[i].visible:
            j = i
            while j < n and not rows[j].visible:
                j += 1
            lo = max(0, i - margin)
            hi = min(n, j + margin)
            for k in range(lo, hi):
                mask[k] = True
            i = j
        else:
            i += 1
    return mask


def _auc_and_np(ious: np.ndarray, dists: np.ndarray, diags: np.ndarray) -> tuple[float, float]:
    if ious.size == 0:
        return 0.0, 0.0
    thr_s = np.arange(0, 1.05, 0.05)
    auc = float(np.mean([np.mean(ious >= t) for t in thr_s]))
    nd = dists / np.maximum(diags, 1e-6)
    thr_p = np.arange(0, 0.51, 0.01)
    npc = float(np.mean([np.mean(nd <= t) for t in thr_p]))
    return auc, npc


def _score_partition(rows: list[FrameRow], mask_in: list[bool], want_in_window: bool
                     ) -> dict:
    """Compute AUC/NP/S_acc for raw and hybrid restricted to the given partition.

    Only visible frames contribute.
    """
    ious_r, ious_h, dists_r, dists_h, diags = [], [], [], [], []
    for r, in_win in zip(rows, mask_in):
        if not r.visible:
            continue
        if in_win != want_in_window:
            continue
        ious_r.append(r.raw_iou)
        ious_h.append(r.hyb_iou)
        dists_r.append(r.raw_dist)
        dists_h.append(r.hyb_dist)
        diags.append(np.sqrt(r.gt[2] ** 2 + r.gt[3] ** 2))
    ir = np.array(ious_r); ih = np.array(ious_h)
    dr = np.array(dists_r); dh = np.array(dists_h)
    dg = np.array(diags)
    auc_r, np_r = _auc_and_np(ir, dr, dg)
    auc_h, np_h = _auc_and_np(ih, dh, dg)
    s_raw = 0.6 * auc_r + 0.4 * np_r
    s_hyb = 0.6 * auc_h + 0.4 * np_h
    return {
        "n_frames": len(ious_r),
        "auc_raw": auc_r, "auc_hyb": auc_h,
        "np_raw": np_r, "np_hyb": np_h,
        "s_raw": s_raw, "s_hyb": s_hyb,
        "delta_s": s_hyb - s_raw,
        "mean_iou_raw": float(np.mean(ir)) if ir.size else 0.0,
        "mean_iou_hyb": float(np.mean(ih)) if ih.size else 0.0,
    }


def _write_csv(rows: list[FrameRow], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx", "gt_x", "gt_y", "gt_w", "gt_h", "visible",
            "raw_x", "raw_y", "raw_w", "raw_h",
            "hyb_x", "hyb_y", "hyb_w", "hyb_h",
            "raw_iou", "hyb_iou", "raw_dist", "hyb_dist", "delta_iou", "events",
        ])
        for r in rows:
            w.writerow([
                r.idx, *[f"{v:.2f}" for v in r.gt], int(r.visible),
                *[f"{v:.2f}" for v in r.raw_bbox],
                *[f"{v:.2f}" for v in r.hyb_bbox],
                f"{r.raw_iou:.4f}", f"{r.hyb_iou:.4f}",
                f"{r.raw_dist:.2f}", f"{r.hyb_dist:.2f}",
                f"{r.hyb_iou - r.raw_iou:+.4f}",
                "|".join(r.events),
            ])


def _find_occ_segments(rows: list[FrameRow]) -> list[tuple[int, int]]:
    """Return (start, end) inclusive indices for each invisible run."""
    segments: list[tuple[int, int]] = []
    i = 0
    n = len(rows)
    while i < n:
        if not rows[i].visible:
            j = i
            while j < n and not rows[j].visible:
                j += 1
            segments.append((i, j - 1))
            i = j
        else:
            i += 1
    return segments


def _narrate_segment(rows: list[FrameRow], seg: tuple[int, int], margin: int) -> str:
    """One-paragraph narrative for an occlusion segment."""
    s, e = seg
    n = len(rows)
    pre_lo = max(0, s - margin)
    post_hi = min(n - 1, e + margin)

    def _bbox_iou_info(idx: int) -> str:
        r = rows[idx]
        vis = "visible" if r.visible else "occluded"
        if not r.visible:
            return f"frame {idx}: {vis}"
        return (f"frame {idx}: {vis} — raw IoU={r.raw_iou:.2f} "
                f"(dist {r.raw_dist:.0f}px) | hybrid IoU={r.hyb_iou:.2f} "
                f"(dist {r.hyb_dist:.0f}px)")

    # Last visible frame before the occlusion
    last_pre = s - 1 if s - 1 >= 0 else s
    # First visible frame after the occlusion (post_hi or earliest visible ≥ e+1)
    first_post = e + 1
    while first_post < n and not rows[first_post].visible:
        first_post += 1
    if first_post >= n:
        first_post = None

    # Post-occlusion recovery: window [first_post, first_post+margin)
    raw_iou_post = hyb_iou_post = None
    raw_dist_post = hyb_dist_post = None
    if first_post is not None:
        lo = first_post
        hi = min(n, first_post + margin + 1)
        vis_post = [rows[k] for k in range(lo, hi) if rows[k].visible]
        if vis_post:
            raw_iou_post = float(np.mean([v.raw_iou for v in vis_post]))
            hyb_iou_post = float(np.mean([v.hyb_iou for v in vis_post]))
            raw_dist_post = float(np.mean([v.raw_dist for v in vis_post]))
            hyb_dist_post = float(np.mean([v.hyb_dist for v in vis_post]))

    seg_len = e - s + 1
    parts = [f"**Occlusion segment frames {s}–{e}** ({seg_len} frame)."]
    if s > 0:
        parts.append(f"Entry ({_bbox_iou_info(last_pre)}).")
    if first_post is not None and raw_iou_post is not None:
        parts.append(
            f"Recovery (frames {first_post}..{min(n - 1, first_post + margin)}): "
            f"raw mean IoU={raw_iou_post:.2f} (mean dist {raw_dist_post:.0f}px); "
            f"hybrid mean IoU={hyb_iou_post:.2f} (mean dist {hyb_dist_post:.0f}px). "
            f"ΔIoU_recovery = {hyb_iou_post - raw_iou_post:+.3f}."
        )
    else:
        parts.append("Sequence ends in occlusion — no recovery window.")
    # HYBRID_SAVE / RAW_LOST markers within recovery window
    save_frames = []
    raw_lost_frames = []
    if first_post is not None:
        hi = min(n, first_post + margin + 1)
        for k in range(first_post, hi):
            if "HYBRID_SAVE" in rows[k].events:
                save_frames.append(k)
            if "RAW_LOST" in rows[k].events:
                raw_lost_frames.append(k)
    if save_frames:
        parts.append(f"HYBRID_SAVE frames: {save_frames[:5]}"
                     + (" …" if len(save_frames) > 5 else ""))
    if raw_lost_frames:
        parts.append(f"RAW_LOST frames: {raw_lost_frames[:5]}"
                     + (" …" if len(raw_lost_frames) > 5 else ""))
    return " ".join(parts)


def _summary_md(seq_id: str, rows: list[FrameRow], margin: int,
                part_occ: dict, part_vis: dict) -> str:
    n = len(rows)
    n_vis = sum(1 for r in rows if r.visible)
    n_occ = n - n_vis
    seg = _find_occ_segments(rows)

    n_save = sum(1 for r in rows if "HYBRID_SAVE" in r.events)
    n_miss = sum(1 for r in rows if "HYBRID_MISS" in r.events)
    n_raw_lost = sum(1 for r in rows if "RAW_LOST" in r.events)

    lines: list[str] = []
    lines.append(f"# Occlusion Drill — `{seq_id}`\n")
    lines.append(f"**Frames**: total={n}, visible={n_vis}, occluded={n_occ}, "
                 f"occlusion segments={len(seg)}")
    lines.append(f"**Margin (K)**: {margin} frame\n")

    lines.append("## ΔScore partitions\n")
    lines.append("| Partition | Visible frames | AUC_raw | AUC_hyb | NP_raw | NP_hyb | "
                 "S_raw | S_hyb | ΔScore |")
    lines.append("|:--|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| Occlusion window (occ ± {margin}) | {part_occ['n_frames']} | "
                 f"{part_occ['auc_raw']:.3f} | {part_occ['auc_hyb']:.3f} | "
                 f"{part_occ['np_raw']:.3f} | {part_occ['np_hyb']:.3f} | "
                 f"{part_occ['s_raw']:.3f} | {part_occ['s_hyb']:.3f} | "
                 f"**{part_occ['delta_s']:+.4f}** |")
    lines.append(f"| Visible control | {part_vis['n_frames']} | "
                 f"{part_vis['auc_raw']:.3f} | {part_vis['auc_hyb']:.3f} | "
                 f"{part_vis['np_raw']:.3f} | {part_vis['np_hyb']:.3f} | "
                 f"{part_vis['s_raw']:.3f} | {part_vis['s_hyb']:.3f} | "
                 f"**{part_vis['delta_s']:+.4f}** |")
    lines.append("")

    lines.append("## Event counts\n")
    lines.append(f"- `HYBRID_SAVE` (raw IoU<0.1 & hyb IoU≥0.3): **{n_save}**")
    lines.append(f"- `HYBRID_MISS` (hyb IoU<0.1 & raw IoU≥0.3): **{n_miss}**")
    lines.append(f"- `RAW_LOST` (raw IoU<0.1 ≥3 ardışık visible frame): **{n_raw_lost}**\n")

    lines.append("## Surgical narrative (per occlusion segment)\n")
    for s in seg:
        lines.append("- " + _narrate_segment(rows, s, margin))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Occlusion drill-down Raw vs Hybrid")
    parser.add_argument("--seq", nargs="+", required=True,
                        help="Target sequence ids (e.g. dataset5/person17_1)")
    parser.add_argument("--imm-config", default=None,
                        help="Tuned IMM config YAML for hybrid pipeline")
    parser.add_argument("--gmc", action="store_true")
    parser.add_argument("--adaptive-r", action="store_true")
    parser.add_argument("--occ-margin", type=int, default=15)
    parser.add_argument("--out", default="reports/occlusion_drill",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    imm_cfg = None
    if args.imm_config:
        with open(args.imm_config) as f:
            imm_cfg = yaml.safe_load(f) or {}

    manifest = load_manifest()
    tracker = TRTTrackWrapper()

    combined_summary: list[str] = []
    combined_summary.append("# Occlusion Drill — Combined Summary\n")
    combined_summary.append(f"**Targets**: {', '.join(args.seq)}  ")
    combined_summary.append(f"**IMM config**: {args.imm_config or '(default)'}  ")
    combined_summary.append(f"**GMC**: {'on' if args.gmc else 'off'}, "
                            f"**Adaptive-R**: {'on' if args.adaptive_r else 'off'}, "
                            f"**Margin**: {args.occ_margin}\n")
    combined_summary.append("| Sequence | n_vis | n_occ | ΔScore_occ | ΔScore_vis | "
                            "HYBRID_SAVE | HYBRID_MISS | RAW_LOST |")
    combined_summary.append("|:--|---:|---:|---:|---:|---:|---:|---:|")

    for seq_id in args.seq:
        print(f"[DRILL] {seq_id}: running raw pipeline...")
        gt = load_gt(seq_id, manifest)
        raw_bboxes = _run_on_sequence(
            tracker, seq_id, manifest, use_kf=False,
            imm_cfg=None, imm_cfg_path=None,
            gmc_enabled=False, adaptive_r_enabled=False,
        )
        gc.collect()
        print(f"[DRILL] {seq_id}: running hybrid pipeline...")
        hyb_bboxes = _run_on_sequence(
            tracker, seq_id, manifest, use_kf=True,
            imm_cfg=imm_cfg, imm_cfg_path=args.imm_config,
            gmc_enabled=args.gmc, adaptive_r_enabled=args.adaptive_r,
        )
        gc.collect()

        rows = _build_rows(gt, raw_bboxes, hyb_bboxes)
        _mark_events(rows)
        mask = _occlusion_window_mask(rows, args.occ_margin)
        part_occ = _score_partition(rows, mask, want_in_window=True)
        part_vis = _score_partition(rows, mask, want_in_window=False)

        safe_name = seq_id.replace("/", "_")
        csv_path = os.path.join(args.out, f"{safe_name}_frames.csv")
        md_path = os.path.join(args.out, f"{safe_name}_summary.md")
        _write_csv(rows, csv_path)
        with open(md_path, "w") as f:
            f.write(_summary_md(seq_id, rows, args.occ_margin, part_occ, part_vis))
        print(f"[DRILL] {seq_id}: wrote {csv_path} and {md_path}")

        n_vis = sum(1 for r in rows if r.visible)
        n_occ = len(rows) - n_vis
        n_save = sum(1 for r in rows if "HYBRID_SAVE" in r.events)
        n_miss = sum(1 for r in rows if "HYBRID_MISS" in r.events)
        n_raw_lost = sum(1 for r in rows if "RAW_LOST" in r.events)
        combined_summary.append(
            f"| `{seq_id}` | {n_vis} | {n_occ} | {part_occ['delta_s']:+.4f} | "
            f"{part_vis['delta_s']:+.4f} | {n_save} | {n_miss} | {n_raw_lost} |"
        )

    combined_path = os.path.join(args.out, "COMBINED_SUMMARY.md")
    with open(combined_path, "w") as f:
        f.write("\n".join(combined_summary) + "\n")
    print(f"[DRILL] wrote {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
