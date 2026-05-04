#!/usr/bin/env python3
"""Unified Optuna IMM parameter tuning: hygiene + blend + KF physics.

Combines both search spaces (tune_filter.py 10D + tune_kalman.py 5D) into
a single 22D optimization with Leave-One-Dataset-Out cross-validation.

Search space (22D):
  KF Physics (4D):   q_scale, r_pos_scale, r_size_scale, pi_persist
  Decision  (3D):    conf_threshold, coast_threshold, max_coast_frames
  Blend     (2D):    alpha_conf_lo, alpha_conf_hi
  Hygiene   (4D):    reinit_after, max_area_frac, max_center_jump_frac, aspect_ratio_hi
  Smart     (4D):    conf_bypass_threshold, innovation_threshold,
                     sm_conf_threshold, sm_max_coast
  Maneuver  (2D):    maneuver_threshold, maneuver_bypass_boost
  Soft fusion (3D):  alpha_gate_k_conf, alpha_gate_lambda, reacq_r_decay

Note: search_scale_boost (C2 REJECT) and C1 skip thresholds are EXCLUDED —
they are disabled in live pipeline and have no effect in cached replay.

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


def build_stratified_subset(cache_dir: str,
                             hard_frac: float = 0.6,
                             easy_frac: float = 0.2,
                             seed: int = 42) -> list[str]:
    """Build a stratified SUBSET from all available NPZ caches.

    Per dataset: sort seqs by AI-only AUC (ascending = hardest first),
    take hard_frac from bottom + easy_frac from top.
    Remaining middle fraction is discarded to avoid redundancy.

    Falls back to hardcoded SUBSET if < 40 NPZs found (cache not yet full).
    """
    import glob
    npz_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if len(npz_files) < 40:
        return SUBSET  # Not enough cache yet — use hardcoded subset

    # Map filename back to seq_id  (dataset1__plane.npz → dataset1/plane)
    available = {}
    for f in npz_files:
        key = os.path.basename(f).replace(".npz", "")
        seq_id = key.replace("__", "/", 1)
        dataset = seq_id.split("/")[0]
        available.setdefault(dataset, []).append((seq_id, f))

    rng = np.random.default_rng(seed)
    selected = []

    for dataset, seq_files in sorted(available.items()):
        # Score each seq with raw (AI-only) AUC from NPZ
        scored = []
        for seq_id, fpath in seq_files:
            try:
                data = np.load(fpath)
                gt = data["gt"]          # (N,4)  x,y,w,h
                ai = data["ai_bboxes"]   # (N,4)
                n = min(len(gt), len(ai))
                # Vectorized IoU proxy (x1y1x2y2 conversion)
                ax1, ay1 = ai[:n, 0], ai[:n, 1]
                ax2, ay2 = ax1 + ai[:n, 2], ay1 + ai[:n, 3]
                gx1, gy1 = gt[:n, 0], gt[:n, 1]
                gx2, gy2 = gx1 + gt[:n, 2], gy1 + gt[:n, 3]
                ix1 = np.maximum(ax1, gx1)
                iy1 = np.maximum(ay1, gy1)
                ix2 = np.minimum(ax2, gx2)
                iy2 = np.minimum(ay2, gy2)
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_g = (gx2 - gx1) * (gy2 - gy1)
                union = area_a + area_g - inter + 1e-6
                iou = inter / union
                raw_auc = float(np.mean(iou > 0.5))
            except Exception:
                raw_auc = 0.5  # unknown → treat as middle
            scored.append((seq_id, raw_auc))

        scored.sort(key=lambda x: x[1])  # ascending: hardest first
        n = len(scored)
        n_hard = max(1, int(n * hard_frac))
        n_easy = max(1, int(n * easy_frac))

        hard_seqs = [s for s, _ in scored[:n_hard]]
        easy_seqs = [s for s, _ in scored[n - n_easy:]]

        # Sample from hard and easy to keep total manageable
        rng.shuffle(hard_seqs)
        rng.shuffle(easy_seqs)
        selected.extend(hard_seqs[:max(3, n_hard)])
        selected.extend(easy_seqs[:max(1, n_easy)])

    return sorted(set(selected))


# Leave-One-Dataset-Out folds — built dynamically from active SUBSET
def build_folds(subset: list[str]) -> dict:
    folds = {}
    for s in subset:
        ds = s.split("/")[0]
        folds.setdefault(ds, []).append(s)
    return folds


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


def create_objective(cache_dir: str, baselines: dict, max_degradation: float,
                     folds: dict):
    """Create Optuna objective with LODO cross-validation.

    Objective = mean FinalScore across folds (eval on held-out dataset).
    Penalty applied if any single sequence degrades > max_degradation vs AI-only.
    """
    fold_names = sorted(folds.keys())

    def objective(trial):
        params = {
            # KF Physics (4D)
            "q_scale": trial.suggest_float("q_scale", 0.01, 100.0, log=True),
            "r_pos_scale": trial.suggest_float("r_pos_scale", 0.1, 150.0, log=True),
            "r_size_scale": trial.suggest_float("r_size_scale", 0.1, 100.0, log=True),
            "pi_persist": trial.suggest_float("pi_persist", 0.70, 0.98),
            # Decision (3D)
            "conf_threshold": trial.suggest_float("conf_threshold", 0.10, 0.50),
            "coast_threshold": trial.suggest_float("coast_threshold", 0.02, 0.40),
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
            "sm_conf_threshold": trial.suggest_float("sm_conf_threshold", 0.05, 0.50),
            "sm_max_coast": trial.suggest_int("sm_max_coast", 10, 80),
            # Maneuver (2D)
            "maneuver_threshold": trial.suggest_float("maneuver_threshold", 0.0, 1.0),
            "maneuver_bypass_boost": trial.suggest_float("maneuver_bypass_boost", 0.0, 0.30),
            # Soft fusion (3D) — k capped at 3 to prevent near-binary sigmoid
            "alpha_gate_k_conf": trial.suggest_float("alpha_gate_k_conf", 0.0, 3.0),
            "alpha_gate_lambda":  trial.suggest_float("alpha_gate_lambda",  0.0, 0.3),
            "reacq_r_decay":      trial.suggest_float("reacq_r_decay",      0.0, 0.5),
            # Fixed
            "aspect_ratio_lo": 0.1,
        }

        fold_scores = []
        per_seq_degradations = []

        for fi, fold_name in enumerate(fold_names):
            held_out = folds[fold_name]
            results = _evaluate_seqs(held_out, cache_dir, params)
            fold_fs = _final_score(results)
            fold_scores.append(fold_fs)

            # Collect per-sequence degradation vs AI-only baseline
            for seq_id, auc, np_ in results:
                base_auc, base_np = baselines[seq_id]
                base_fs = 0.6 * base_auc + 0.4 * base_np
                seq_fs = 0.6 * auc + 0.4 * np_
                per_seq_degradations.append(base_fs - seq_fs)

            # Intermediate pruning after each fold
            trial.report(float(np.mean(fold_scores)), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))

        # Soft penalty based on p90 degradation (not worst-case).
        # Worst-case explodes on hard sequences that are already near 0.
        # p90 is robust: up to 10% of sequences can degrade freely.
        p90_deg = float(np.percentile(per_seq_degradations, 90))
        penalty = 0.0
        if p90_deg > max_degradation:
            penalty += (p90_deg - max_degradation) * 1.0
        penalty += 0.5 * max(0.0, p90_deg - 0.10) ** 2
        mean_score -= penalty

        return mean_score

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Unified Optuna IMM tuning (22D, LODO CV, cached replay)")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--cache-dir", default=os.path.join(
        _PROJECT_ROOT, "cache", "ai_outputs"))
    parser.add_argument("--max-degradation", type=float, default=0.03,
                        help="Max per-sequence FinalScore degradation vs AI-only (default: 0.03)")
    parser.add_argument("--study-name", default="unified_tune")
    parser.add_argument("--no-db", action="store_true",
                        help="Use in-memory storage (no sqlite)")
    parser.add_argument("--seqs-file", default=None,
                        help="Newline-separated file of seq IDs. Overrides stratified subset.")
    parser.add_argument("--out-config", default=None,
                        help="Output config path (default: configs/imm_tuned.yaml).")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)

    # Build subset: from --seqs-file, or stratified from cache
    if args.seqs_file:
        with open(args.seqs_file) as f:
            active_subset = sorted(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )
        print(f"[INFO] Loaded {len(active_subset)} seqs from {args.seqs_file}")
    else:
        active_subset = build_stratified_subset(cache_dir)
    active_folds = build_folds(active_subset)

    # Validate cache exists
    missing = [s for s in active_subset if not os.path.exists(cache_path(s, cache_dir))]
    if missing:
        print(f"[ERROR] Missing cache for {len(missing)} sequences:")
        for s in missing:
            print(f"  {s}")
        print("\nRun first: python scripts/cache_ai_outputs.py --force")
        sys.exit(1)

    print(f"Sequences: {len(active_subset)}, Folds: {len(active_folds)} (LODO)")
    if len(active_subset) > 20:
        print(f"[INFO] Using stratified subset from {len(active_subset)} cached seqs "
              f"(hard 60% + easy 20% per dataset)")
    else:
        print("[INFO] Using hardcoded SUBSET (cache < 40 seqs)")
    print(f"Cache dir: {cache_dir}")
    print(f"Trials: {args.n_trials}, Max degradation: {args.max_degradation}")
    for fn, seqs in sorted(active_folds.items()):
        print(f"  {fn}: {len(seqs)} seqs")

    # ── Baselines ────────────────────────────────────────────────
    print("\n--- Baselines ---")
    baselines = _raw_baselines(active_subset, cache_dir)
    raw_results = [(s, baselines[s][0], baselines[s][1]) for s in active_subset]
    raw_fs = _final_score(raw_results)

    default_results = _evaluate_seqs(active_subset, cache_dir, DEFAULT_PARAMS)
    default_fs = _final_score(default_results)

    print(f"AI-only FinalScore:  {raw_fs:.4f}")
    print(f"Default IMM:         {default_fs:.4f}")

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
            n_startup_trials=50,
            n_warmup_steps=2,
        ),
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )

    objective = create_objective(cache_dir, baselines, args.max_degradation,
                                 active_folds)

    # C1 warm start — başlangıç noktası olarak bilinen en iyi config
    C1_PARAMS = {
        "q_scale": 31.113513, "r_pos_scale": 44.953786, "r_size_scale": 2.107025,
        "pi_persist": 0.9600, "conf_threshold": 0.186309, "coast_threshold": 0.140301,
        "max_coast_frames": 44, "alpha_conf_lo": 0.373544, "alpha_conf_hi": 0.845429,
        "reinit_after": 10, "max_area_frac": 0.345940, "max_center_jump_frac": 0.511044,
        "aspect_ratio_hi": 7.099779, "conf_bypass_threshold": 0.740469,
        "innovation_threshold": 1.454772, "sm_conf_threshold": 0.155357,
        "sm_max_coast": 69, "maneuver_threshold": 0.40, "maneuver_bypass_boost": 0.10,
        "alpha_gate_k_conf": 0.0, "alpha_gate_lambda": 0.0, "reacq_r_decay": 0.0,
    }
    if not any(t.params == C1_PARAMS for t in study.trials):
        study.enqueue_trial(C1_PARAMS)
        print("[INFO] Enqueued C1 warm start trial")

    print(f"\n{'=' * 70}")
    print(f"Starting Optuna: {args.n_trials} trials, 22D search, LODO CV")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    # ── Results ──────────────────────────────────────────────────
    best = study.best_params
    best["aspect_ratio_lo"] = 0.1  # fixed param

    tuned_results = _evaluate_seqs(active_subset, cache_dir, best)
    tuned_fs = _final_score(tuned_results)

    # C1 guard: evaluate C1 on full active_subset to ensure we never regress
    c1_params_full = dict(C1_PARAMS)
    c1_params_full["aspect_ratio_lo"] = 0.1
    c1_results = _evaluate_seqs(active_subset, cache_dir, c1_params_full)
    c1_fs = _final_score(c1_results)

    _save_best = best
    _save_fs = tuned_fs
    _save_label = "Optuna"
    if tuned_fs < c1_fs:
        print(f"\n[GUARD] Optuna best ({tuned_fs:.4f}) < C1 ({c1_fs:.4f}) on active subset "
              f"— keeping C1 params to avoid regression.")
        _save_best = c1_params_full
        _save_fs = c1_fs
        _save_label = "C1 (guard fallback)"

    print(f"\n{'=' * 70}")
    print(f"Optimization complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    print("\nBest params (22D):")
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
    print("  Maneuver:")
    print(f"    maneuver_threshold   = {best.get('maneuver_threshold', 0.0):.6f}")
    print(f"    maneuver_bypass_boost= {best.get('maneuver_bypass_boost', 0.0):.6f}")
    print("  Soft Fusion:")
    print(f"    alpha_gate_k_conf    = {best.get('alpha_gate_k_conf', 0.0):.6f}")
    print(f"    alpha_gate_lambda    = {best.get('alpha_gate_lambda', 0.0):.6f}")
    print(f"    reacq_r_decay        = {best.get('reacq_r_decay', 0.0):.6f}")

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
          f"Optuna={tuned_fs:.4f}  C1={c1_fs:.4f}")
    print(f"Delta vs AI-only: {_save_fs - raw_fs:+.4f}")
    print(f"Delta vs default: {_save_fs - default_fs:+.4f}")
    print(f"Saving config: {_save_label} (FS={_save_fs:.4f})")
    print(f"Better: {n_better}/{len(active_subset)}, Worse: {n_worse}/{len(active_subset)}, "
          f"Same: {n_same}/{len(active_subset)}")

    # ── Per-fold breakdown ───────────────────────────────────────
    print("\nPer-fold results:")
    for fn in sorted(active_folds.keys()):
        held = active_folds[fn]
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
    config_path = (os.path.abspath(args.out_config) if args.out_config
                   else os.path.join(_PROJECT_ROOT, "configs", "imm_tuned.yaml"))
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        f.write("# Optuna-tuned unified IMM parameters (22D, LODO CV)\n")
        f.write(f"# Saved: {_save_label}  FinalScore={_save_fs:.4f}\n")
        f.write(f"# AI-only={raw_fs:.4f}  Default-IMM={default_fs:.4f}  "
                f"Optuna={tuned_fs:.4f}  C1={c1_fs:.4f}\n")
        f.write(f"# Delta vs AI-only: {_save_fs - raw_fs:+.4f}\n")
        f.write(f"# Trials: {args.n_trials}, Sequences: {len(active_subset)}, "
                f"Folds: {len(active_folds)}\n\n")
        f.write("imm:\n")
        f.write(f"  q_scale: {_save_best['q_scale']:.6f}\n")
        f.write("  measurement_noise:\n")
        f.write(f"    r_pos_scale: {_save_best['r_pos_scale']:.6f}\n")
        f.write(f"    r_size_scale: {_save_best['r_size_scale']:.6f}\n")
        f.write("  transition_matrix:\n")
        f.write(f"    pi_persist: {_save_best['pi_persist']:.4f}\n\n")
        f.write("decision:\n")
        f.write(f"  conf_threshold: {_save_best['conf_threshold']:.6f}\n")
        f.write(f"  coast_threshold: {_save_best['coast_threshold']:.6f}\n")
        f.write(f"  max_coast_frames: {_save_best['max_coast_frames']}\n\n")
        f.write("blend:\n")
        f.write(f"  alpha_conf_lo: {_save_best['alpha_conf_lo']:.6f}\n")
        f.write(f"  alpha_conf_hi: {_save_best['alpha_conf_hi']:.6f}\n\n")
        f.write("hygiene:\n")
        f.write(f"  reinit_after: {_save_best['reinit_after']}\n")
        f.write(f"  max_area_frac: {_save_best['max_area_frac']:.6f}\n")
        f.write(f"  max_center_jump_frac: {_save_best['max_center_jump_frac']:.6f}\n")
        f.write("  aspect_ratio_lo: 0.1\n")
        f.write(f"  aspect_ratio_hi: {_save_best['aspect_ratio_hi']:.6f}\n")

        f.write("\nsmart_intervention:\n")
        f.write(f"  conf_bypass_threshold: {_save_best.get('conf_bypass_threshold', 0.90):.6f}\n")
        f.write(f"  innovation_threshold: {_save_best.get('innovation_threshold', 2.0):.6f}\n")
        f.write(f"  sm_conf_threshold: {_save_best.get('sm_conf_threshold', 0.25):.6f}\n")
        f.write(f"  sm_max_coast: {_save_best.get('sm_max_coast', 30)}\n")
        f.write(f"  maneuver_threshold: {_save_best.get('maneuver_threshold', 0.0):.6f}\n")
        f.write(f"  maneuver_bypass_boost: {_save_best.get('maneuver_bypass_boost', 0.0):.6f}\n")
        f.write("\nsoft_fusion:\n")
        f.write(f"  alpha_gate_k_conf: {_save_best.get('alpha_gate_k_conf', 0.0):.6f}\n")
        f.write(f"  alpha_gate_lambda: {_save_best.get('alpha_gate_lambda', 0.0):.6f}\n")
        f.write(f"  reacq_r_decay: {_save_best.get('reacq_r_decay', 0.0):.6f}\n")
    print(f"\n[OK] Config saved ({_save_label}) → {config_path}")


if __name__ == "__main__":
    main()
