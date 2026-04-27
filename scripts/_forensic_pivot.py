"""Per-frame forensic pivot for one disaster sequence.

Reads telemetry CSV.gz + GT annotation.txt, emits a compact summary:
  - Frame-by-frame IoU(AI, GT), IoU(final, GT)
  - Phase markers (TRACKING/COASTING/LOST), gate decision distribution
  - First divergence frame (where final diverges from GT > 0.5 IoU drop)
  - Mode prob trace at divergence
  - Mahalanobis & innovation peaks
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


def load_gt(path):
    gt = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [float(p) for p in line.split(",")]
            gt.append(parts[:4])
    return gt


def load_telemetry(path):
    rows = []
    with gzip.open(path, "rt") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def main(seq_id):
    slug = seq_id.replace("/", "__")
    tele_path = Path(f"runs/forensic/golden_v1/{slug}.csv.gz")
    gt_path = Path(f"data/contest_release/{seq_id}/annotation.txt")
    if not tele_path.exists() or not gt_path.exists():
        print(f"MISSING: {tele_path} or {gt_path}")
        return

    rows = load_telemetry(tele_path)
    gt = load_gt(gt_path)
    n = min(len(rows), len(gt))

    print(f"\n══ {seq_id}  (telem={len(rows)}, gt={len(gt)}, common={n}) ══")

    iou_ai, iou_final, mahal, innov, mu_singer, mu_cv, mu_ca, ai_conf = [], [], [], [], [], [], [], []
    state_seq, gate_seq, refresh_seq, gmc_seq = [], [], [], []
    for i in range(n):
        r = rows[i]
        gt_box = gt[i]
        ai_box = (f(r["ai_x"]), f(r["ai_y"]), f(r["ai_w"]), f(r["ai_h"]))
        final_box = (f(r["final_x"]), f(r["final_y"]), f(r["final_w"]), f(r["final_h"]))
        iou_ai.append(iou(ai_box, gt_box))
        iou_final.append(iou(final_box, gt_box))
        mahal.append(f(r["mahal_d2"]))
        innov.append(f(r["innov_norm"]))
        mu_singer.append(f(r["mu_singer"]))
        mu_cv.append(f(r["mu_cv"]))
        mu_ca.append(f(r["mu_ca"]))
        ai_conf.append(f(r["ai_conf"]))
        state_seq.append(r.get("state", ""))
        gate_seq.append(r.get("gate_decision", ""))
        refresh_seq.append(r.get("refresh", ""))
        gmc_seq.append(r.get("gmc_ok", ""))

    iou_ai = np.array(iou_ai)
    iou_final = np.array(iou_final)
    mahal = np.array(mahal)
    innov = np.array(innov)
    mu_singer = np.array(mu_singer)
    mu_cv = np.array(mu_cv)
    mu_ca = np.array(mu_ca)
    ai_conf = np.array(ai_conf)

    # Aggregate
    print(f"  AUC-like: AI mean IoU={iou_ai.mean():.3f}, FINAL mean IoU={iou_final.mean():.3f}")
    print(f"  Drop : final - AI mean IoU = {iou_final.mean() - iou_ai.mean():+.3f}")

    # State distribution
    from collections import Counter
    print(f"  State dist : {dict(Counter(state_seq))}")
    print(f"  Gate dist  : {dict(Counter(gate_seq))}")
    print(f"  Refresh dist: {dict(Counter(refresh_seq))}")
    print(f"  GMC ok dist : {dict(Counter(gmc_seq))}")

    # First divergence (where AI is fine but FINAL collapses)
    # AI good = iou_ai > 0.5; FINAL bad = iou_final < 0.3
    div_mask = (iou_ai > 0.5) & (iou_final < 0.3)
    if div_mask.any():
        first = int(np.argmax(div_mask))
        print(f"  ▼ FIRST DIVERGENCE @ frame {first}: AI_iou={iou_ai[first]:.2f}, FINAL_iou={iou_final[first]:.2f}")
        print(f"    around frame {first}: ai_conf={ai_conf[first]:.2f}, mahal={mahal[first]:.1f}, innov={innov[first]:.2f}")
        print(f"    mu(CV/CA/Sg)={mu_cv[first]:.2f}/{mu_ca[first]:.2f}/{mu_singer[first]:.2f}, state={state_seq[first]}, gate={gate_seq[first]}")

    # Mahalanobis + innovation peaks
    mahal_top = np.argsort(mahal)[-5:][::-1]
    print(f"  Top-5 mahal_d2 frames: " + ", ".join(f"f{int(i)}={mahal[i]:.1f}" for i in mahal_top))
    innov_top = np.argsort(innov)[-5:][::-1]
    print(f"  Top-5 innov_norm frames: " + ", ".join(f"f{int(i)}={innov[i]:.2f}" for i in innov_top))

    # Confidence collapse run
    low_conf_mask = ai_conf < 0.3
    if low_conf_mask.any():
        runs = []
        cur = 0
        for v in low_conf_mask:
            if v:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        print(f"  Low-conf (<0.3) total frames={low_conf_mask.sum()}, longest run={max(runs) if runs else 0}")

    # Phase windows: chunks of 30 frames, mean IoU each
    chunk = 30
    print("  Per-chunk mean IoU (AI / FINAL):")
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        if e - s < 5:
            break
        a = iou_ai[s:e].mean()
        fn = iou_final[s:e].mean()
        marker = "***" if (a - fn) > 0.2 else ("→ KO" if fn < 0.2 else "")
        print(f"    [{s:4d}-{e:4d}]  AI={a:.2f}  FINAL={fn:.2f}  drop={a - fn:+.2f}  {marker}")


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "dataset3/car8"
    main(seq)
