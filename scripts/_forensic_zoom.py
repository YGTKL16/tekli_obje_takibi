"""Zoom into the AI-collapse window of one disaster sequence.

Prints frame-by-frame: AI bbox, KF prediction, FINAL bbox, GT bbox,
IoU(AI, GT), IoU(FINAL, GT), state, gate, conf, mahal, innov.
"""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

import numpy as np


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def load_gt(path):
    gt = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [float(p) for p in line.split(",")]
            gt.append(parts[:4])
    return gt


def load_telemetry(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def main(seq_id, start, end):
    slug = seq_id.replace("/", "__")
    rows = load_telemetry(Path(f"runs/forensic/golden_v1/{slug}.csv.gz"))
    gt = load_gt(Path(f"data/contest_release/{seq_id}/annotation.txt"))
    n = min(len(rows), len(gt))
    end = min(end, n)
    print(f"\n══ {seq_id}  zoom frames [{start}..{end})  (total={n}) ══")
    print(f"{'f':>4} {'state':>9} {'gate':>14} {'conf':>5} "
          f"{'AI(x,y,w,h)':>26} {'IoU_AI':>6} "
          f"{'KF_pred':>22} "
          f"{'FINAL':>26} {'IoU_F':>6} "
          f"{'mahal':>7} {'innov':>6} {'mu(C/A/S)':>15}")
    for i in range(start, end):
        r = rows[i]
        gtb = gt[i]
        ai = (f(r["ai_x"]), f(r["ai_y"]), f(r["ai_w"]), f(r["ai_h"]))
        kf = (f(r["kf_px"]), f(r["kf_py"]), f(r["kf_pw"]), f(r["kf_ph"]))
        fb = (f(r["final_x"]), f(r["final_y"]), f(r["final_w"]), f(r["final_h"]))
        i_ai = iou(ai, gtb)
        i_fb = iou(fb, gtb)
        ai_str = f"({ai[0]:5.0f},{ai[1]:5.0f},{ai[2]:4.0f},{ai[3]:4.0f})"
        kf_str = f"({kf[0]:5.0f},{kf[1]:5.0f},{kf[2]:4.0f},{kf[3]:4.0f})"
        fb_str = f"({fb[0]:5.0f},{fb[1]:5.0f},{fb[2]:4.0f},{fb[3]:4.0f})"
        mu = f"{f(r['mu_cv']):.2f}/{f(r['mu_ca']):.2f}/{f(r['mu_singer']):.2f}"
        print(f"{i:>4} {r['state']:>9} {r['gate_decision']:>14} "
              f"{f(r['ai_conf']):>5.2f} {ai_str:>26} {i_ai:>6.2f} "
              f"{kf_str:>22} {fb_str:>26} {i_fb:>6.2f} "
              f"{f(r['mahal_d2']):>7.1f} {f(r['innov_norm']):>6.2f} {mu:>15}")


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "dataset3/car8"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    main(seq, start, end)
