#!/usr/bin/env python3
"""Quick forensics: air_conditioning_box2 IMM regression."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

from tracker.replay import evaluate_cached, raw_baseline_score, DEFAULT_PARAMS
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cache_dir = os.path.join(ROOT, "cache", "ai_outputs")

SEQS_OF_INTEREST = [
    "dataset3/air_conditioning_box2",
    "dataset1/volleyball",
    "dataset2/Girl2",
    "dataset5/car1_s",
]

for seq in SEQS_OF_INTEREST:
    cp = os.path.join(cache_dir, seq.replace("/", "__") + ".npz")
    if not os.path.exists(cp):
        print(f"MISSING: {seq}")
        continue

    data = np.load(cp, allow_pickle=True)
    print(f"\n{'='*60}")
    print(f"Sequence: {seq}")
    print(f"  Cache keys: {list(data.keys())}")
    confs = data["confs"]

    print(f"  Frames: {len(confs)}")
    print(f"  Conf: min={confs.min():.3f}  mean={confs.mean():.3f}  "
          f"p10={np.percentile(confs,10):.3f}  p25={np.percentile(confs,25):.3f}")
    print(f"  Conf<0.20: {(confs<0.20).sum()}  Conf<0.30: {(confs<0.30).sum()}")
    print(f"  Conf>0.70: {(confs>0.70).sum()}  Conf>0.85: {(confs>0.85).sum()}")

    raw_auc, raw_np = raw_baseline_score(cp)
    imm_auc, imm_np = evaluate_cached(cp, DEFAULT_PARAMS)
    print(f"  Raw:  AUC={raw_auc:.4f}  NP={raw_np:.4f}")
    print(f"  IMM:  AUC={imm_auc:.4f}  NP={imm_np:.4f}  dAUC={imm_auc-raw_auc:+.4f}")

    # Try with tighter chi2 gate
    strict_params = {**DEFAULT_PARAMS, "chi2_threshold": 9.49}  # p=0.05, 4-dof
    strict_auc, strict_np = evaluate_cached(cp, strict_params)
    print(f"  chi2=9.49: AUC={strict_auc:.4f}  NP={strict_np:.4f}  dAUC={strict_auc-raw_auc:+.4f}")

    # Try with high q_scale (trust KF more) 
    q_hi = {**DEFAULT_PARAMS, "q_scale": 5.0}
    q_hi_auc, q_hi_np = evaluate_cached(cp, q_hi)    
    print(f"  q=5.0:    AUC={q_hi_auc:.4f}  NP={q_hi_np:.4f}  dAUC={q_hi_auc-raw_auc:+.4f}")

    # Low r_pos (trust KF prediction more, measurements less)
    r_lo = {**DEFAULT_PARAMS, "r_pos_scale": 200.0}
    r_lo_auc, r_lo_np = evaluate_cached(cp, r_lo)
    print(f"  r_pos=200: AUC={r_lo_auc:.4f}  NP={r_lo_np:.4f}  dAUC={r_lo_auc-raw_auc:+.4f}")
