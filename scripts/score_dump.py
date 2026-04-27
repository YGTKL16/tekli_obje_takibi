#!/usr/bin/env python3
"""Per-sequence score dump for all 255 train sequences.

Outputs outputs/score_dump.csv with per-sequence AUC, NormPrec, delta,
and auto-derived regime tags (size_class, fps_class, ar_class).

Usage:
    python3 scripts/score_dump.py --config configs/i12_rescue_area_gate.yaml \\
        --gmc --adaptive-r --mode ai_lead

    # Smoke test: 2 sequences only
    python3 scripts/score_dump.py --config configs/i12_rescue_area_gate.yaml \\
        --gmc --adaptive-r --mode ai_lead \\
        --seq dataset3/car8 --seq dataset2/Paragliding3
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build2"))

import numpy as np
import torch

from tracker.config import load_yaml_config
from tracker.data_utils import DATA_ROOT, load_manifest, load_gt, parse_bbox_line
from tracker.trt_wrapper import TRTTrackWrapper

# Import run_sequence + evaluate from ab_test (same module, no duplication)
_AB_DIR = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, _AB_DIR)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ab_test", os.path.join(_AB_DIR, "ab_test.py"))
_ab = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ab)  # type: ignore[union-attr]
run_sequence = _ab.run_sequence
evaluate = _ab.evaluate
_apply_imm_config = _ab._apply_imm_config
_load_filter_params = _ab._load_filter_params


_CSV_FIELDS = [
    "seq_id", "dataset", "n_frames", "native_fps",
    "init_w", "init_h", "init_area", "init_ar",
    "size_class",   # small / medium / large
    "fps_class",    # high / normal
    "ar_class",     # thin / normal
    "auc_raw", "auc_imm", "delta_auc",
    "np_raw",  "np_imm",  "delta_np",
    "fs_raw",  "fs_imm",  "delta_fs",
]


def _classify_size(area: float) -> str:
    if area < 500:
        return "small"
    if area < 5000:
        return "medium"
    return "large"


def _classify_fps(fps: float) -> str:
    return "high" if fps >= 60 else "normal"


def _classify_ar(ar: float) -> str:
    return "thin" if (ar > 3.0 or ar < 0.33) else "normal"


def _fs(auc: float, np_: float) -> float:
    return 0.6 * auc + 0.4 * np_


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-sequence score dump for 255-seq train set")
    parser.add_argument("--config", default="configs/i12_rescue_area_gate.yaml",
                        help="IMM config YAML path (default: i12_rescue_area_gate.yaml)")
    parser.add_argument("--mode", default="ai_lead",
                        choices=["baseline", "coast_only", "velocity_shift", "ai_lead"],
                        help="KF mode (default: ai_lead)")
    parser.add_argument("--gmc", action="store_true", help="Enable GMC")
    parser.add_argument("--adaptive-r", action="store_true", help="Enable adaptive R")
    parser.add_argument("--seq", action="append", default=None, metavar="SEQ",
                        help="Run only these sequences. Repeatable. Default: all 255.")
    parser.add_argument("--out", default="outputs/score_dump.csv",
                        help="Output CSV path (default: outputs/score_dump.csv)")
    parser.add_argument("--no-kf-col", action="store_true",
                        help="Skip IMM column (raw-only, faster for quick tagging runs)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    imm_cfg = load_yaml_config(args.config) if args.config else {}
    manifest = load_manifest()
    all_seq_ids = sorted(manifest["train"].keys())
    seq_ids = args.seq if args.seq else all_seq_ids

    tracker = TRTTrackWrapper()

    print(f"[score_dump] Config: {args.config}")
    print(f"[score_dump] Mode: {args.mode} | GMC: {args.gmc} | AdaptiveR: {args.adaptive_r}")
    print(f"[score_dump] Sequences: {len(seq_ids)} | Output: {args.out}")
    print("-" * 90)

    rows: list[dict] = []

    for idx, seq_id in enumerate(seq_ids, 1):
        if seq_id not in manifest["train"]:
            print(f"  [{idx:>3}] {seq_id} NOT FOUND — skipped")
            continue
        seq_info = manifest["train"][seq_id]
        gt = load_gt(seq_id, manifest)
        if not gt:
            print(f"  [{idx:>3}] {seq_id} NO GT — skipped")
            continue

        init_bbox = gt[0]
        init_w = float(init_bbox[2])
        init_h = float(init_bbox[3])
        init_area = init_w * init_h
        init_ar = (init_w / init_h) if init_h > 0 else 1.0
        fps = float(seq_info.get("native_fps", 30))
        dataset = seq_info.get("dataset", seq_id.split("/")[0])
        n_frames = int(seq_info.get("n_frames", 0))

        size_cls = _classify_size(init_area)
        fps_cls = _classify_fps(fps)
        ar_cls = _classify_ar(init_ar)

        # ── Raw (AI only) ─────────────────────────────────────────────────────
        preds_raw = run_sequence(tracker, seq_id, seq_info, manifest, use_kf=False)
        auc_r, np_r = evaluate(gt, preds_raw)
        del preds_raw

        # ── IMM ───────────────────────────────────────────────────────────────
        auc_i, np_i = auc_r, np_r
        if not args.no_kf_col:
            preds_imm = run_sequence(
                tracker, seq_id, seq_info, manifest,
                use_kf=True,
                kf_mode=args.mode,
                gmc_enabled=args.gmc,
                adaptive_r_enabled=args.adaptive_r,
                imm_cfg=imm_cfg,
                imm_cfg_path=args.config,
            )
            auc_i, np_i = evaluate(gt, preds_imm)
            del preds_imm

        del gt
        gc.collect()
        torch.cuda.empty_cache()

        d_auc = auc_i - auc_r
        d_np  = np_i  - np_r
        fs_r  = _fs(auc_r, np_r)
        fs_i  = _fs(auc_i, np_i)
        d_fs  = fs_i - fs_r

        marker = "+" if d_auc > 0.005 else ("-" if d_auc < -0.005 else "=")
        print(f"[{idx:>3}/{len(seq_ids)}] {seq_id:<40} "
              f"raw={auc_r:.3f} imm={auc_i:.3f} Δ={d_auc:+.3f}{marker}  "
              f"sz={size_cls[0]} fps={fps_cls[0]} ar={ar_cls[0]}")

        rows.append({
            "seq_id": seq_id,
            "dataset": dataset,
            "n_frames": n_frames,
            "native_fps": fps,
            "init_w": round(init_w, 1),
            "init_h": round(init_h, 1),
            "init_area": round(init_area, 1),
            "init_ar": round(init_ar, 3),
            "size_class": size_cls,
            "fps_class": fps_cls,
            "ar_class": ar_cls,
            "auc_raw": round(auc_r, 4),
            "auc_imm": round(auc_i, 4),
            "delta_auc": round(d_auc, 4),
            "np_raw": round(np_r, 4),
            "np_imm": round(np_i, 4),
            "delta_np": round(d_np, 4),
            "fs_raw": round(fs_r, 4),
            "fs_imm": round(fs_i, 4),
            "delta_fs": round(d_fs, 4),
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 90)
    print(f"\n[score_dump] Wrote {len(rows)} rows → {args.out}")

    # ── Quick summary by size_class ───────────────────────────────────────────
    if rows:
        print("\n--- Summary by size_class ---")
        for cls in ("small", "medium", "large"):
            subset = [r for r in rows if r["size_class"] == cls]
            if subset:
                med_d = float(np.median([r["delta_auc"] for r in subset]))
                n_bad = sum(1 for r in subset if r["delta_auc"] < -0.02)
                print(f"  {cls:8s}: n={len(subset):>3}  median_Δ={med_d:+.3f}  "
                      f"n_bad(Δ<-0.02)={n_bad}")
        print("\n--- Summary by fps_class ---")
        for cls in ("high", "normal"):
            subset = [r for r in rows if r["fps_class"] == cls]
            if subset:
                med_d = float(np.median([r["delta_auc"] for r in subset]))
                n_bad = sum(1 for r in subset if r["delta_auc"] < -0.02)
                print(f"  {cls:8s}: n={len(subset):>3}  median_Δ={med_d:+.3f}  "
                      f"n_bad(Δ<-0.02)={n_bad}")
        print("\n--- Bottom 15 sequences (worst delta_auc) ---")
        worst = sorted(rows, key=lambda r: r["delta_auc"])[:15]
        for r in worst:
            print(f"  {r['seq_id']:<40} Δ={r['delta_auc']:+.3f}  "
                  f"sz={r['size_class']}  fps={r['fps_class']}  "
                  f"area={r['init_area']:.0f}px²")


if __name__ == "__main__":
    main()
