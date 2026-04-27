#!/usr/bin/env python3
"""Extract and evaluate top Optuna trials from C3 study DB.

Runs evaluate_cached on the top-K trials by their stored value,
then prints a ranking against the default and raw baselines.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import numpy as np
import optuna

from tracker.replay import DEFAULT_PARAMS, evaluate_cached, raw_baseline_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT, "cache", "ai_outputs")
DB_PATH = os.path.join(ROOT, "cache", "optuna_studies", "unified_tune.db")
STUDY_NAME = "c3_unified_v1"

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


def cache_path(seq_id):
    return os.path.join(CACHE_DIR, seq_id.replace("/", "__") + ".npz")


def final_score(params):
    aucs, nps = [], []
    for seq in SUBSET:
        auc, np_ = evaluate_cached(cache_path(seq), params)
        aucs.append(auc); nps.append(np_)
    return 0.6 * np.mean(aucs) + 0.4 * np.mean(nps)


# --- Baselines ---
raw_results = [raw_baseline_score(cache_path(s)) for s in SUBSET]
raw_fs = 0.6 * np.mean([r[0] for r in raw_results]) + 0.4 * np.mean([r[1] for r in raw_results])
def_fs = final_score(DEFAULT_PARAMS)
print(f"AI-only FinalScore:   {raw_fs:.4f}")
print(f"Default IMM:          {def_fs:.4f}  (delta={def_fs-raw_fs:+.4f})")

# --- Load study ---
storage = f"sqlite:///{DB_PATH}"
study = optuna.load_study(study_name=STUDY_NAME, storage=storage)

# All trials with value (both complete and pruned w/ intermediate values)
all_trials = [(t.number, t.value, t.params)
              for t in study.trials
              if t.value is not None]
all_trials.sort(key=lambda x: x[1], reverse=True)

print(f"\nStudy: {STUDY_NAME}  |  Total trials with value: {len(all_trials)}")
print(f"Best trial by stored value: #{all_trials[0][0]} = {all_trials[0][1]:.4f}")
print(f"\n{'Rank':>4}  {'Trial':>6}  {'Opt_val':>8}  {'FS_full':>8}  {'Delta':>7}  q      rp     rs    pi     conf")
print("-" * 90)

for rank, (tn, opt_val, p) in enumerate(all_trials[:15], 1):
    p_full = {**p, "aspect_ratio_lo": 0.1}
    fs = final_score(p_full)
    delta = fs - raw_fs
    marker = "**" if fs > def_fs else "  "
    print(f"{rank:>4}  #{tn:>5}  {opt_val:>8.4f}  {fs:>8.4f}  {delta:>+7.4f}  "
          f"{p.get('q_scale',0):.2f}  {p.get('r_pos_scale',0):.2f}  "
          f"{p.get('r_size_scale',0):.2f}  {p.get('pi_persist',0):.3f}  "
          f"{p.get('conf_threshold',0):.3f}  {marker}")
