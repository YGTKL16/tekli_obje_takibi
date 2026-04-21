#!/usr/bin/env python3
"""Unified Optuna IMM parameter tuning: hygiene + blend + KF physics.

Combines both search spaces (tune_filter.py 10D + tune_kalman.py 5D) into
a single 17D optimization with Leave-One-Dataset-Out cross-validation.

Search space (17D):
  KF Physics (4D):   q_scale, r_pos_scale, r_size_scale, pi_persist
  Decision  (3D):    conf_threshold, coast_threshold, max_coast_frames
  Blend     (2D):    alpha_conf_lo, alpha_conf_hi
  Hygiene   (4D):    reinit_after, max_area_frac, max_center_jump_frac, aspect_ratio_hi
  Smart     (4D):    conf_bypass_threshold, innovation_threshold,
                     sm_conf_threshold, sm_max_coast

All evaluation is cached (no TRT inference) — 500 trials in minutes.

Usage:
    python scripts/cache_ai_outputs.py --force     # prerequisite: re-cache with new engine
    python scripts/tune_unified.py                  # 500 trials
    python scripts/tune_unified.py --n-trials 100   # quick test
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import numpy as np
import optuna

from tracker.replay import DEFAULT_PARAMS, evaluate_cached, raw_baseline_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

# Leave-One-Dataset-Out folds
FOLDS = {
    "dataset1": [s for s in SUBSET if s.startswith("dataset1/")],
    "dataset2": [s for s in SUBSET if s.startswith("dataset2/")],
    "dataset3": [s for s in SUBSET if s.startswith("dataset3/")],
    "dataset4": [s for s in SUBSET if s.startswith("dataset4/")],
    "dataset5": [s for s in SUBSET if s.startswith("dataset5/")],
}


def cache_key(seq_id: str) -> str:
    return seq_id.replace("/", "__")


def cache_path(seq_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, cache_key(seq_id) + ".npz")


def _evaluate_seqs(seq_ids, cache_dir, params):
    """Return per-sequence (auc, np) and means."""
    results = []
    for seq_id in seq_ids:
        cp = cache_path(seq_id, cache_dir)
        auc, norm_prec = evaluate_cached(cp, params)
        results.append((seq_id, auc, norm_prec))
    return results


def _final_score(results):
    """Compute FinalScore from list of (seq_id, auc, np)."""
    aucs = [r[1] for r in results]
    nps = [r[2] for r in results]
    return 0.6 * np.mean(aucs) + 0.4 * np.mean(nps)


def _raw_baselines(seq_ids, cache_dir):
    """Return dict {seq_id: (auc, np)} for AI-only baseline."""
    baselines = {}
    for seq_id in seq_ids:
        cp = cache_path(seq_id, cache_dir)
        auc, norm_prec = raw_baseline_score(cp)
        baselines[seq_id] = (auc, norm_prec)
    return baselines


def create_objective(cache_dir: str, baselines: dict, max_degradation: float):
    """Create Optuna objective with LODO cross-validation.

    Objective = mean FinalScore across 5 folds (train on 4 datasets, eval on held-out).
    Penalty applied if any single sequence degrades > max_degradation vs AI-only.
    """
    fold_names = sorted(FOLDS.keys())

    def objective(trial):
        params = {
            # KF Physics (4D)
            "q_scale": trial.suggest_float("q_scale", 0.01, 100.0, log=True),
            "r_pos_scale": trial.suggest_float("r_pos_scale", 0.1, 50.0, log=True),
            "r_size_scale": trial.suggest_float("r_size_scale", 0.1, 100.0, log=True),
            "pi_persist": trial.suggest_float("pi_persist", 0.70, 0.98),
            # Decision (3D)
            "conf_threshold": trial.suggest_float("conf_threshold", 0.10, 0.50),
            "coast_threshold": trial.suggest_float("coast_threshold", 0.02, 0.15),
            "max_coast_frames": trial.suggest_int("max_coast_frames", 5, 80),
            # Blend (2D)
            "alpha_conf_lo": trial.suggest_float("alpha_conf_lo", 0.10, 0.50),
            "alpha_conf_hi": trial.suggest_float("alpha_conf_hi", 0.50, 0.95),
            # Hygiene (4D)
            "reinit_after": trial.suggest_int("reinit_after", 3, 60),
            "max_area_frac": trial.suggest_float("max_area_frac", 0.10, 0.60),
            "max_center_jump_frac": trial.suggest_float("max_center_jump_frac", 0.10, 0.60),
            "aspect_ratio_hi": trial.suggest_float("aspect_ratio_hi", 2.5, 10.0),
            # Smart intervention (4D)
            "conf_bypass_threshold": trial.suggest_float("conf_bypass_threshold", 0.70, 0.98),
            "innovation_threshold": trial.suggest_float("innovation_threshold", 0.5, 5.0, log=True),
            "sm_conf_threshold": trial.suggest_float("sm_conf_threshold", 0.15, 0.50),
            "sm_max_coast": trial.suggest_int("sm_max_coast", 10, 80),
            # Soft fusion (3D) — k capped at 3 to prevent near-binary sigmoid
            "alpha_gate_k_conf": trial.suggest_float("alpha_gate_k_conf", 0.0, 3.0),
            "alpha_gate_lambda":  trial.suggest_float("alpha_gate_lambda",  0.0, 0.3),
            "reacq_r_decay":      trial.suggest_float("reacq_r_decay",      0.0, 0.5),
            # Faz C1b: IMM skip thresholds (4D)
            "singer_skip_thr": trial.suggest_float("singer_skip_thr", 0.20, 0.55),
            "cv_stable_thr":   trial.suggest_float("cv_stable_thr",   0.50, 0.90),
            "cv_conf_min":     trial.suggest_float("cv_conf_min",     0.45, 0.80),
            # Faz C2: Singer search window boost (1D)
            "search_scale_boost": trial.suggest_float("search_scale_boost", 0.0, 3.0),
            # Fixed
            "aspect_ratio_lo": 0.1,
        }

        fold_scores = []
        worst_degradation = 0.0

        for fi, fold_name in enumerate(fold_names):
            held_out = FOLDS[fold_name]
            results = _evaluate_seqs(held_out, cache_dir, params)
            fold_fs = _final_score(results)
            fold_scores.append(fold_fs)

            # Check per-sequence degradation
            for seq_id, auc, np_ in results:
                base_auc, base_np = baselines[seq_id]
                base_fs = 0.6 * base_auc + 0.4 * base_np
                seq_fs = 0.6 * auc + 0.4 * np_
                deg = base_fs - seq_fs
                if deg > worst_degradation:
                    worst_degradation = deg

            # Intermediate pruning after each fold
            trial.report(float(np.mean(fold_scores)), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))

        # Soft penalty for excessive per-sequence degradation.
        penalty = 0.0
        if worst_degradation > max_degradation:
            penalty += (worst_degradation - max_degradation) * 2.0
        penalty += 5.0 * max(0.0, worst_degradation - 0.03) ** 2
        mean_score -= penalty

        return mean_score

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Unified Optuna IMM tuning (13D, LODO CV, cached replay)")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--cache-dir", default=os.path.join(
        _PROJECT_ROOT, "cache", "ai_outputs"))
    parser.add_argument("--max-degradation", type=float, default=0.03,
                        help="Max per-sequence FinalScore degradation vs AI-only (default: 0.03)")
    parser.add_argument("--study-name", default="unified_tune")
    parser.add_argument("--no-db", action="store_true",
                        help="Use in-memory storage (no sqlite)")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)

    # Validate cache exists
    missing = [s for s in SUBSET if not os.path.exists(cache_path(s, cache_dir))]
    if missing:
        print(f"[ERROR] Missing cache for {len(missing)} sequences:")
        for s in missing:
            print(f"  {s}")
        print("\nRun first: python scripts/cache_ai_outputs.py --force")
        sys.exit(1)

    print(f"Sequences: {len(SUBSET)}, Folds: {len(FOLDS)} (LODO)")
    print(f"Cache dir: {cache_dir}")
    print(f"Trials: {args.n_trials}, Max degradation: {args.max_degradation}")
    for fn, seqs in sorted(FOLDS.items()):
        print(f"  {fn}: {len(seqs)} seqs")

    # ── Baselines ────────────────────────────────────────────────
    print("\n--- Baselines ---")
    baselines = _raw_baselines(SUBSET, cache_dir)
    raw_results = [(s, baselines[s][0], baselines[s][1]) for s in SUBSET]
    raw_fs = _final_score(raw_results)

    default_results = _evaluate_seqs(SUBSET, cache_dir, DEFAULT_PARAMS)
    default_fs = _final_score(default_results)

    print(f"AI-only FinalScore:  {raw_fs:.4f}")
    print(f"Default IMM:         {default_fs:.4f}")

    # ── Optuna study ─────────────────────────────────────────────
    if args.no_db:
        storage = None
    else:
        db_dir = os.path.join(os.path.dirname(cache_dir), "optuna_studies")
        os.makedirs(db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(db_dir, 'unified_tune.db')}"

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15,
            n_warmup_steps=2,
        ),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = create_objective(cache_dir, baselines, args.max_degradation)

    print(f"\n{'=' * 70}")
    print(f"Starting Optuna: {args.n_trials} trials, 17D search, LODO CV")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    # ── Results ──────────────────────────────────────────────────
    best = study.best_params
    best["aspect_ratio_lo"] = 0.1  # fixed param

    tuned_results = _evaluate_seqs(SUBSET, cache_dir, best)
    tuned_fs = _final_score(tuned_results)

    print(f"\n{'=' * 70}")
    print(f"Optimization complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    print("\nBest params (17D):")
    print("  KF Physics:")
    print(f"    q_scale              = {best['q_scale']:.6f}")
    print(f"    r_pos_scale          = {best['r_pos_scale']:.6f}")
    print(f"    r_size_scale         = {best['r_size_scale']:.6f}")
    print(f"    pi_persist           = {best['pi_persist']:.4f}")
    print("  Decision:")
    print(f"    conf_threshold       = {best['conf_threshold']:.6f}")
    print(f"    coast_threshold      = {best['coast_threshold']:.6f}")
    print(f"    max_coast_frames     = {best['max_coast_frames']}")
    print("  Blend:")
    print(f"    alpha_conf_lo        = {best['alpha_conf_lo']:.6f}")
    print(f"    alpha_conf_hi        = {best['alpha_conf_hi']:.6f}")
    print("  Hygiene:")
    print(f"    reinit_after         = {best['reinit_after']}")
    print(f"    max_area_frac        = {best['max_area_frac']:.6f}")
    print(f"    max_center_jump_frac = {best['max_center_jump_frac']:.6f}")
    print(f"    aspect_ratio_hi      = {best['aspect_ratio_hi']:.6f}")
    print("  Smart Intervention:")
    print(f"    conf_bypass_threshold= {best.get('conf_bypass_threshold', 0.90):.6f}")
    print(f"    innovation_threshold = {best.get('innovation_threshold', 2.0):.6f}")
    print(f"    sm_conf_threshold    = {best.get('sm_conf_threshold', 0.25):.6f}")
    print(f"    sm_max_coast         = {best.get('sm_max_coast', 30)}")

    # ── Per-sequence comparison ──────────────────────────────────
    print(f"\n{'Sequence':<38} {'AI_FS':>7} {'IMM_FS':>7} {'delta':>7}")
    print("-" * 62)
    n_better, n_worse, n_same = 0, 0, 0
    for seq_id, auc, np_ in tuned_results:
        ba, bn = baselines[seq_id]
        bfs = 0.6 * ba + 0.4 * bn
        tfs = 0.6 * auc + 0.4 * np_
        d = tfs - bfs
        if d > 0.005:
            n_better += 1
            m = "+"
        elif d < -0.005:
            n_worse += 1
            m = "-"
        else:
            n_same += 1
            m = "="
        print(f"{seq_id:<38} {bfs:>7.3f} {tfs:>7.3f} {d:>+7.3f}{m}")

    print("-" * 62)
    print(f"{'MEAN':<38} {raw_fs:>7.3f} {tuned_fs:>7.3f} {tuned_fs-raw_fs:>+7.3f}")
    print(f"\nFinalScore: AI-only={raw_fs:.4f}  Default-IMM={default_fs:.4f}  "
          f"Tuned-IMM={tuned_fs:.4f}")
    print(f"Delta vs AI-only: {tuned_fs - raw_fs:+.4f}")
    print(f"Delta vs default: {tuned_fs - default_fs:+.4f}")
    print(f"Better: {n_better}/{len(SUBSET)}, Worse: {n_worse}/{len(SUBSET)}, "
          f"Same: {n_same}/{len(SUBSET)}")

    # ── Per-fold breakdown ───────────────────────────────────────
    print("\nPer-fold results:")
    for fn in sorted(FOLDS.keys()):
        held = FOLDS[fn]
        fold_results = [(s, a, n) for s, a, n in tuned_results if s in held]
        fold_raw = [(s, baselines[s][0], baselines[s][1]) for s in held]
        print(f"  {fn}: AI={_final_score(fold_raw):.4f} → IMM={_final_score(fold_results):.4f} "
              f"({_final_score(fold_results)-_final_score(fold_raw):+.4f})")

    # ── Top-5 trials ─────────────────────────────────────────────
    print("\nTop 5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1,
                    reverse=True)
    for t in trials[:5]:
        if t.value is not None:
            p = t.params
            print(f"  #{t.number:3d}: CV_score={t.value:.4f} | "
                  f"q={p.get('q_scale', 0):.3f} "
                  f"rp={p.get('r_pos_scale', 0):.3f} "
                  f"rs={p.get('r_size_scale', 0):.3f} "
                  f"pi={p.get('pi_persist', 0):.3f} "
                  f"conf={p.get('conf_threshold', 0):.3f}")

    # ── Save best config ─────────────────────────────────────────
    config_path = os.path.join(_PROJECT_ROOT, "configs", "imm_tuned.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        f.write("# Optuna-tuned unified IMM parameters (13D, LODO CV)\n")
        f.write(f"# FinalScore: AI-only={raw_fs:.4f}  Default-IMM={default_fs:.4f}  "
                f"Tuned-IMM={tuned_fs:.4f}\n")
        f.write(f"# Delta vs AI-only: {tuned_fs - raw_fs:+.4f}\n")
        f.write(f"# Trials: {args.n_trials}, Sequences: {len(SUBSET)}, "
                f"Folds: {len(FOLDS)}\n\n")
        f.write("imm:\n")
        f.write(f"  q_scale: {best['q_scale']:.6f}\n")
        f.write("  measurement_noise:\n")
        f.write(f"    r_pos_scale: {best['r_pos_scale']:.6f}\n")
        f.write(f"    r_size_scale: {best['r_size_scale']:.6f}\n")
        f.write("  transition_matrix:\n")
        f.write(f"    pi_persist: {best['pi_persist']:.4f}\n\n")
        f.write("decision:\n")
        f.write(f"  conf_threshold: {best['conf_threshold']:.6f}\n")
        f.write(f"  coast_threshold: {best['coast_threshold']:.6f}\n")
        f.write(f"  max_coast_frames: {best['max_coast_frames']}\n\n")
        f.write("blend:\n")
        f.write(f"  alpha_conf_lo: {best['alpha_conf_lo']:.6f}\n")
        f.write(f"  alpha_conf_hi: {best['alpha_conf_hi']:.6f}\n\n")
        f.write("hygiene:\n")
        f.write(f"  reinit_after: {best['reinit_after']}\n")
        f.write(f"  max_area_frac: {best['max_area_frac']:.6f}\n")
        f.write(f"  max_center_jump_frac: {best['max_center_jump_frac']:.6f}\n")
        f.write("  aspect_ratio_lo: 0.1\n")
        f.write(f"  aspect_ratio_hi: {best['aspect_ratio_hi']:.6f}\n")

        f.write("\nsmart_intervention:\n")
        f.write(f"  conf_bypass_threshold: {best.get('conf_bypass_threshold', 0.90):.6f}\n")
        f.write(f"  innovation_threshold: {best.get('innovation_threshold', 2.0):.6f}\n")
        f.write(f"  sm_conf_threshold: {best.get('sm_conf_threshold', 0.25):.6f}\n")
        f.write(f"  sm_max_coast: {best.get('sm_max_coast', 30)}\n")

    print(f"\n[OK] Best config saved to {config_path}")


if __name__ == "__main__":
    main()
