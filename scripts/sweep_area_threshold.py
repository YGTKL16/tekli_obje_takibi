"""sweep_area_threshold.py — Phase 1 robustness sweep.

Sweeps refresh_small_area_thr and rescue_min_area in tandem across a range of
values, running the 20-sequence A/B test for each.  Reports a FinalScore table
so we can distinguish a stable plateau from a narrow over-fit peak around 220.

Usage:
    python3 scripts/sweep_area_threshold.py \
        --imm-config configs/i12_rescue_area_gate.yaml \
        --mode ai_lead --gmc --adaptive-r

The script patches the config dict in memory — no YAML files are modified.
"""

import argparse
import copy
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from python.tracker.config import load_runtime_config          # noqa: E402
from python.tracker.data_utils import load_manifest, load_gt   # noqa: E402
from scripts.ab_test import run_sequence, evaluate             # noqa: E402

# ── shared imports that ab_test.py already pulls in ───────────────────────────
import cv2                            # noqa: E402
import torch                          # noqa: E402
import gc                             # noqa: E402
import yaml                           # noqa: E402

# ── local imports ─────────────────────────────────────────────────────────────
from python.tracker.trt_wrapper import TRTTrackWrapper         # noqa: E402

DATA_ROOT = os.path.join(_PROJECT_ROOT, "data")

THRESHOLDS = [0, 100, 150, 180, 200, 220, 260, 300, 500, 99999]

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


def _load_manifest():
    return load_manifest()


def _patched_config(base_yaml_path: str, thr: float) -> dict:
    """Load the YAML and override both area-gate thresholds in-memory."""
    with open(base_yaml_path) as f:
        cfg = yaml.safe_load(f)
    ai_block = cfg.setdefault("ai", {})
    ai_block["refresh_small_area_thr"] = thr
    ai_block["rescue_min_area"] = thr      # keep in sync
    return cfg


def _run_one_threshold(thr, args, manifest, tracker):
    """Run the full 20-seq subset with the given threshold and return FinalScore."""
    imm_cfg = _patched_config(args.imm_config, thr)

    auc_raws, auc_imms, np_raws, np_imms = [], [], [], []
    for seq_id in SUBSET:
        seq_info = manifest["train"][seq_id]
        gt = load_gt(seq_id, manifest)

        preds_raw = run_sequence(
            tracker, seq_id, seq_info, manifest, use_kf=False,
        )
        auc_r, np_r = evaluate(gt, preds_raw)
        del preds_raw

        preds_imm = run_sequence(
            tracker, seq_id, seq_info, manifest, use_kf=True,
            kf_mode=args.mode,
            gmc_enabled=args.gmc,
            adaptive_r_enabled=args.adaptive_r,
            imm_cfg=imm_cfg,
            imm_cfg_path=args.imm_config,
        )
        auc_i, np_i = evaluate(gt, preds_imm)
        del preds_imm, gt

        gc.collect()
        torch.cuda.empty_cache()

        auc_raws.append(auc_r)
        auc_imms.append(auc_i)
        np_raws.append(np_r)
        np_imms.append(np_i)

    ar = float(np.mean(auc_raws))
    ai = float(np.mean(auc_imms))
    nr = float(np.mean(np_raws))
    ni = float(np.mean(np_imms))
    final_raw = 0.6 * ar + 0.4 * nr
    final_imm = 0.6 * ai + 0.4 * ni
    return final_raw, final_imm, ar, ai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imm-config", default="configs/i12_rescue_area_gate.yaml")
    parser.add_argument("--mode", default="ai_lead")
    parser.add_argument("--gmc", action="store_true")
    parser.add_argument("--adaptive-r", action="store_true")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None,
                        help="Override default sweep values (px²)")
    args = parser.parse_args()

    thresholds = args.thresholds if args.thresholds else THRESHOLDS

    manifest = _load_manifest()
    tracker = TRTTrackWrapper()

    print(f"\n{'Threshold':>12}  {'FS_raw':>8}  {'FS_imm':>8}  {'Delta':>7}  {'AUC_raw':>8}  {'AUC_imm':>8}")
    print("-" * 68)

    results = []
    for thr in thresholds:
        thr_label = f"{int(thr)}" if thr < 99998 else "∞ (disabled)"
        fr, fi, ar, ai = _run_one_threshold(thr, args, manifest, tracker)
        delta = fi - fr
        marker = " ←" if abs(thr - 220) < 1 else ""
        print(f"{thr_label:>12}  {fr:>8.4f}  {fi:>8.4f}  {delta:>+7.4f}  {ar:>8.4f}  {ai:>8.4f}{marker}")
        results.append((thr, fr, fi))

    # Plateau analysis
    print("\n── Plateau Analysis ───────────────────────────────────────────────")
    scores = [r[2] for r in results]
    best = max(scores)
    plateau_count = sum(1 for s in scores if s >= best - 0.005)
    print(f"Peak FS_imm:   {best:.4f}")
    print(f"Plateau width: {plateau_count}/{len(thresholds)} thresholds within 0.005 of peak")
    if plateau_count >= 5:
        print("→ ROBUST: wide plateau, 220 is a safe generalisation threshold")
    elif plateau_count >= 3:
        print("→ MODERATE: medium plateau, monitor 255-seq result")
    else:
        print("→ FRAGILE: narrow peak, likely over-fit — consider relaxing threshold")


if __name__ == "__main__":
    main()
