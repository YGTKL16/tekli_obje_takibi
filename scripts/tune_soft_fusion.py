#!/usr/bin/env python3
"""Optuna Faz A: Focused 3D search for soft-fusion parameters only.

Freezes ALL existing IMM/KF/hygiene params from imm_tuned.yaml and
explores only the 3 new soft-fusion knobs:
  - alpha_gate_k_conf  [0, 15]  : sigmoid slope for confidence axis
  - alpha_gate_lambda  [0, 1]   : Mahalanobis suppression weight
  - reacq_r_decay      [0, 0.5] : R-inflation per coast frame

50 trials in 3D gives dense coverage (vs 50 in the full 20D of tune_unified).

Usage:
    python scripts/tune_soft_fusion.py --n-trials 50
    python scripts/tune_soft_fusion.py --n-trials 50 --study-name soft_fusion_v1
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import numpy as np
import optuna
import yaml

from tracker.replay import evaluate_cached, raw_baseline_score, DEFAULT_PARAMS

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_IMM_TUNED = os.path.join(_PROJECT_ROOT, "configs", "imm_tuned.yaml")

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

FOLDS = {
    "dataset1": [s for s in SUBSET if s.startswith("dataset1/")],
    "dataset2": [s for s in SUBSET if s.startswith("dataset2/")],
    "dataset3": [s for s in SUBSET if s.startswith("dataset3/")],
    "dataset4": [s for s in SUBSET if s.startswith("dataset4/")],
    "dataset5": [s for s in SUBSET if s.startswith("dataset5/")],
}


def _cache_path(seq_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, seq_id.replace("/", "__") + ".npz")


def _evaluate_seqs(seq_ids, cache_dir, params):
    results = []
    for seq_id in seq_ids:
        cp = _cache_path(seq_id, cache_dir)
        auc, np_ = evaluate_cached(cp, params)
        results.append((seq_id, auc, np_))
    return results


def _final_score(results):
    aucs = [r[1] for r in results]
    nps = [r[2] for r in results]
    return 0.6 * np.mean(aucs) + 0.4 * np.mean(nps)


def _raw_baselines(seq_ids, cache_dir):
    baselines = {}
    for seq_id in seq_ids:
        cp = _cache_path(seq_id, cache_dir)
        auc, np_ = raw_baseline_score(cp)
        baselines[seq_id] = (auc, np_)
    return baselines


def _load_imm_tuned_params(config_path: str) -> dict:
    """Parse imm_tuned.yaml into the flat dict expected by replay_sequence.

    normalize_runtime_config() doesn't handle the nested imm: section, so we
    parse it manually here to ensure q_scale, r_pos_scale, pi_persist etc. are
    correctly forwarded.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Start from the replay DEFAULT_PARAMS as base
    params = dict(DEFAULT_PARAMS)

    # KF Physics — nested under imm:
    imm = cfg.get("imm", {}) or {}
    if "q_scale" in imm:
        params["q_scale"] = float(imm["q_scale"])
    mn = imm.get("measurement_noise", {}) or {}
    if "r_pos_scale" in mn:
        params["r_pos_scale"] = float(mn["r_pos_scale"])
    if "r_size_scale" in mn:
        params["r_size_scale"] = float(mn["r_size_scale"])
    tm = imm.get("transition_matrix", {}) or {}
    if "pi_persist" in tm:
        params["pi_persist"] = float(tm["pi_persist"])

    # Decision
    decision = cfg.get("decision", {}) or {}
    for key in ("conf_threshold", "coast_threshold", "max_coast_frames",
                "conf_bypass_threshold", "innovation_threshold"):
        if key in decision:
            params[key] = decision[key]
    if "max_coast_frames" in decision:
        params["max_coast_frames"] = int(decision["max_coast_frames"])

    # Blend
    blend = cfg.get("blend", {}) or {}
    for key in ("alpha_conf_lo", "alpha_conf_hi"):
        if key in blend:
            params[key] = float(blend[key])

    # Hygiene
    hygiene = cfg.get("hygiene", {}) or {}
    for key in ("reinit_after", "max_area_frac", "max_center_jump_frac",
                "aspect_ratio_lo", "aspect_ratio_hi"):
        if key in hygiene:
            params[key] = hygiene[key]
    if "reinit_after" in hygiene:
        params["reinit_after"] = int(hygiene["reinit_after"])

    # Smart intervention
    smart = cfg.get("smart_intervention", cfg.get("smart", {})) or {}
    for key in ("conf_bypass_threshold", "innovation_threshold",
                "sm_conf_threshold", "sm_max_coast",
                "maneuver_threshold", "maneuver_bypass_boost"):
        if key in smart:
            params[key] = smart[key]
    if "sm_max_coast" in smart:
        params["sm_max_coast"] = int(smart["sm_max_coast"])

    # Adaptive R
    adaptive_r = cfg.get("adaptive_r", {}) or {}
    if "enabled" in adaptive_r:
        params["adaptive_r_enabled"] = bool(adaptive_r["enabled"])
    if "floor" in adaptive_r:
        params["adaptive_r_floor"] = float(adaptive_r["floor"])
    if "r_exponent" in adaptive_r:
        params["r_exponent"] = float(adaptive_r["r_exponent"])

    # Mahalanobis gate — replay.py uses "mahal_chi2_threshold" (separate from "mahal_chi2_gate")
    mahal = cfg.get("mahalanobis", {}) or {}
    if "chi2_threshold" in mahal:
        params["mahal_chi2_threshold"] = float(mahal["chi2_threshold"])
    if "bypass_after" in mahal:
        params["mahal_bypass_after"] = int(mahal["bypass_after"])
    if "r_pos_base" in mahal:
        params["r_pos_base"] = float(mahal["r_pos_base"])
    if "r_size_base" in mahal:
        params["r_size_base"] = float(mahal["r_size_base"])

    # Soft fusion defaults (0.0 = disabled = binary; will be overridden by Optuna)
    params["alpha_gate_k_conf"] = 0.0
    params["alpha_gate_lambda"]  = 0.0
    params["reacq_r_decay"]      = 0.0

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Faz A: 3D soft-fusion Optuna (frozen base from imm_tuned.yaml)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--cache-dir", default=os.path.join(
        _PROJECT_ROOT, "cache", "ai_outputs"))
    parser.add_argument("--base-config", default=_IMM_TUNED)
    parser.add_argument("--max-degradation", type=float, default=0.03,
                        help="Max per-sequence FinalScore regression vs AI-only baseline")
    parser.add_argument("--study-name", default="soft_fusion_v1")
    parser.add_argument("--no-db", action="store_true",
                        help="Use in-memory storage (no sqlite, results lost after run)")
    parser.add_argument("--k-max", type=float, default=15.0,
                        help="Upper bound for alpha_gate_k_conf (default 15.0; use 3.0 for Faz B2)")
    parser.add_argument("--lambda-max", type=float, default=1.0,
                        help="Upper bound for alpha_gate_lambda (default 1.0; use 0.3 for Faz B2)")
    parser.add_argument("--isolate-r-decay", action="store_true",
                        help="1D mode: fix k=0 and lambda=0, search only reacq_r_decay")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    missing = [s for s in SUBSET if not os.path.exists(_cache_path(s, cache_dir))]
    if missing:
        print(f"[ERROR] Missing cache for {len(missing)} sequences:")
        for s in missing:
            print(f"  {s}")
        print("\nRun: python scripts/cache_ai_outputs.py --force")
        sys.exit(1)

    # Load frozen base params
    base_params = _load_imm_tuned_params(args.base_config)
    print(f"Base config: {os.path.basename(args.base_config)}")
    print(f"  q_scale={base_params['q_scale']:.3f}  "
          f"r_pos={base_params['r_pos_scale']:.3f}  "
          f"r_size={base_params['r_size_scale']:.3f}  "
          f"pi={base_params['pi_persist']:.4f}")
    print(f"  adaptive_r={base_params['adaptive_r_enabled']}  "
          f"r_exp={base_params['r_exponent']:.1f}  "
          f"bypass={base_params['conf_bypass_threshold']:.4f}")
    print(f"  chi2_thr={base_params['mahal_chi2_threshold']:.2f}  "
          f"bypass_after={base_params['mahal_bypass_after']}")

    # Baselines
    print("\n--- Baselines ---")
    baselines = _raw_baselines(SUBSET, cache_dir)
    raw_results = [(s, baselines[s][0], baselines[s][1]) for s in SUBSET]
    raw_fs = _final_score(raw_results)

    base_results = _evaluate_seqs(SUBSET, cache_dir, base_params)
    base_fs = _final_score(base_results)

    print(f"AI-only FinalScore:   {raw_fs:.4f}")
    print(f"Base (imm_tuned):     {base_fs:.4f}  "
          f"(Δ vs AI-only = {base_fs - raw_fs:+.4f})")
    if args.isolate_r_decay:
        print(f"\nSearch space (1D — k=0 and lambda=0 fixed):")
        print(f"  reacq_r_decay      [0.0,  0.5]  (0 → no R-inflation)")
    else:
        print(f"\nSearch space (3D — everything else frozen):")
        print(f"  alpha_gate_k_conf  [0.0, {args.k_max:.1f}]  (0 → α=1.0 = binary, full update)")
        print(f"  alpha_gate_lambda  [0.0,  {args.lambda_max:.1f}]  (0 → no Mahal suppression)")
        print(f"  reacq_r_decay      [0.0,  0.5]  (0 → no R-inflation)")

    # Storage
    if args.no_db:
        storage = None
    else:
        db_dir = os.path.join(os.path.dirname(cache_dir), "optuna_studies")
        os.makedirs(db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(db_dir, 'soft_fusion.db')}"

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    fold_names = sorted(FOLDS.keys())

    def objective(trial):
        if args.isolate_r_decay:
            k_conf = 0.0
            lam    = 0.0
            r_dec  = trial.suggest_float("reacq_r_decay", 0.0, 0.5)
        else:
            k_conf = trial.suggest_float("alpha_gate_k_conf", 0.0, args.k_max)
            lam    = trial.suggest_float("alpha_gate_lambda",  0.0, args.lambda_max)
            r_dec  = trial.suggest_float("reacq_r_decay",      0.0, 0.5)

        params = dict(base_params)
        params["alpha_gate_k_conf"] = k_conf
        params["alpha_gate_lambda"]  = lam
        params["reacq_r_decay"]      = r_dec

        fold_scores = []
        worst_degradation = 0.0

        for fi, fold_name in enumerate(fold_names):
            held_out = FOLDS[fold_name]
            results = _evaluate_seqs(held_out, cache_dir, params)
            fold_fs = _final_score(results)
            fold_scores.append(fold_fs)

            for seq_id, auc, np_ in results:
                b_auc, b_np = baselines[seq_id]
                deg = (0.6 * b_auc + 0.4 * b_np) - (0.6 * auc + 0.4 * np_)
                if deg > worst_degradation:
                    worst_degradation = deg

            trial.report(float(np.mean(fold_scores)), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))

        # Penalty for per-sequence regression
        penalty = 0.0
        if worst_degradation > args.max_degradation:
            penalty += (worst_degradation - args.max_degradation) * 2.0
        penalty += 5.0 * max(0.0, worst_degradation - 0.03) ** 2

        # Sanity penalty: k_conf > 12 → near-binary sigmoid (defeats soft blending)
        if k_conf > 12.0:
            penalty += (k_conf - 12.0) * 0.001

        return mean_score - penalty

    dims = "1D" if args.isolate_r_decay else "3D"
    print(f"\n{'='*70}")
    print(f"Optuna: {args.n_trials} trials, {dims} soft-fusion, LODO CV")
    print(f"Study: {args.study_name}")
    print(f"{'='*70}\n")

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    # ── Results ──────────────────────────────────────────────────────────────
    best_fusion = study.best_params
    best_params = dict(base_params)
    best_params.update(best_fusion)

    tuned_results = _evaluate_seqs(SUBSET, cache_dir, best_params)
    tuned_fs = _final_score(tuned_results)

    print(f"\n{'='*70}")
    print(f"Faz A Results — {len(study.trials)} trials in {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"\nBest soft-fusion params:")
    for k in ("alpha_gate_k_conf", "alpha_gate_lambda", "reacq_r_decay"):
        print(f"  {k:30s}: {best_fusion.get(k, 0.0):.6f}")

    k_best   = best_fusion.get("alpha_gate_k_conf", 0.0)
    lam_best = best_fusion.get("alpha_gate_lambda", 0.0)
    r_best   = best_fusion["reacq_r_decay"]

    print(f"\nScores (20-seq subset):")
    print(f"  AI-only:           {raw_fs:.4f}")
    print(f"  Base (imm_tuned):  {base_fs:.4f}  (Δ vs AI  = {base_fs - raw_fs:+.4f})")
    print(f"  Faz A best:        {tuned_fs:.4f}  "
          f"(Δ vs base = {tuned_fs - base_fs:+.4f})  "
          f"(Δ vs AI  = {tuned_fs - raw_fs:+.4f})")

    print(f"\n{'Sequence':<45} {'Base':>8} {'Tuned':>8} {'Delta':>8}")
    print("-" * 73)
    base_dict = {r[0]: (r[1], r[2]) for r in base_results}
    for seq_id, auc, np_ in tuned_results:
        b_auc, b_np = base_dict[seq_id]
        b_fs = 0.6 * b_auc + 0.4 * b_np
        t_fs = 0.6 * auc + 0.4 * np_
        flag = " +" if t_fs > b_fs + 0.001 else (" -" if t_fs < b_fs - 0.001 else " =")
        print(f"  {seq_id:<43} {b_fs:8.4f} {t_fs:8.4f} {t_fs - b_fs:+8.4f}{flag}")

    # ── Sanity interpretation ─────────────────────────────────────────────────
    print(f"\n--- Interpretability ---")
    if k_best < 0.5 and lam_best < 0.05 and r_best < 0.05:
        print("  All 3 params near-zero → soft fusion brings no incremental benefit.")
        print("  Binary behavior is already well-calibrated for this dataset.")
    else:
        if k_best > 0:
            print(f"  α(conf) profile with k={k_best:.2f}:")
            for conf_ex in [0.25, 0.35, 0.50, 0.65, 0.74, 0.85]:
                alpha_ex = 1.0 / (1.0 + math.exp(-k_best * (conf_ex - 0.5)))
                print(f"    conf={conf_ex:.2f}  α={alpha_ex:.3f}")
        if lam_best > 0.05:
            print(f"  Mahalanobis suppression λ={lam_best:.3f}: "
                  f"moderate d² drives α toward 0 (reject bad AI detections).")
        if r_best > 0.05:
            r30 = 1.0 + r_best * 30.0
            print(f"  R-inflation: after 30 coast frames, r_factor={r30:.2f}x "
                  f"(eff_conf / sqrt(r_factor) = more conservative trust).")

    # ── Kaizen decision ───────────────────────────────────────────────────────
    delta = tuned_fs - base_fs
    print(f"\n--- Kaizen Decision ---")
    if delta >= 0.005:
        decision = "ACCEPT"
        action = "Commit soft_fusion_tuned.yaml to imm_tuned.yaml soft_fusion: section."
    elif delta >= -0.001:
        decision = "OBSERVE"
        action = "Run full train split to confirm: python scripts/run_competition.py --imm-config ..."
    else:
        decision = "REJECT"
        action = "Soft fusion is not beneficial on this 20-seq subset. Consider Faz B (full 20D) instead."
    print(f"  [{decision}]  Δ = {delta:+.4f}  — {action}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_yaml = os.path.join(_PROJECT_ROOT, "configs", "soft_fusion_tuned.yaml")
    with open(out_yaml, "w") as f:
        f.write(f"# Optuna Faz A: 3D soft-fusion ({args.n_trials} trials)\n")
        f.write(f"# Base: {os.path.basename(args.base_config)}\n")
        f.write(f"# FinalScore: base={base_fs:.4f}  tuned={tuned_fs:.4f}  "
                f"delta={delta:+.4f}\n")
        f.write(f"# Kaizen: {decision}\n")
        f.write(f"# Study: {args.study_name}\n\n")
        f.write("soft_fusion:\n")
        f.write(f"  alpha_gate_k_conf: {k_best:.6f}   "
                f"# sigmoid slope; 0=disabled\n")
        f.write(f"  alpha_gate_lambda:  {lam_best:.6f}   "
                f"# Mahalanobis suppression; 0=no effect\n")
        f.write(f"  reacq_r_decay:      {r_best:.6f}   "
                f"# R-inflation per coast frame; 0=disabled\n")

    print(f"\nResults saved → {os.path.relpath(out_yaml)}")
    print(f"Trial database → cache/optuna_studies/soft_fusion.db "
          f"(if not --no-db)")


if __name__ == "__main__":
    main()
