#!/usr/bin/env python3
"""Optuna study: optimize f5_alpha, q_scale, r_pos_scale against 20-seq SUBSET.

Objective: maximize FinalScore_imm on the standard 20-seq SUBSET
  (same metric as ab_test.py --mode ai_lead --gmc --adaptive-r).

Pruning strategy:
  - Run 2 "canary" sequences (car8, Paragliding3) FIRST.
  - If worst canary delta < PRUNE_THRESHOLD, kill trial immediately.
  - Then report intermediate FS after canaries for Optuna MedianPruner.

Search space:
  f5_alpha    [0.0, 1.0]          — F5 blending factor (1.0 = full KF, 0.0 = AI only)
  q_scale     [5, 200] log        — IMM process noise scale
  r_pos_scale [5, 200] log        — IMM position measurement noise scale

Usage:
  # Local (TRT):
  python3 scripts/optuna_f5alpha.py --n-trials 50

  # Colab (PyTorch/SGLATrack):
  python3 scripts/optuna_f5alpha.py --n-trials 50 --use-sgla

  # Resume a study:
  python3 scripts/optuna_f5alpha.py --n-trials 100 --study-db cache/optuna_studies/f5alpha.db

Baseline: FS_imm = 0.7139 (i12 + T1 + F5 all-frames, 20-seq)
"""
import argparse
import copy
import gc
import os
import sys
import yaml

# ---------------------------------------------------------------------------
# Path setup — works both locally and on Colab (repo root as CWD)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "build2"))   # local dev build
sys.path.insert(0, os.path.join(_REPO_ROOT, "build"))    # Colab build fallback

import numpy as np
import optuna
from optuna.pruners import MedianPruner

# Import evaluation helpers from ab_test (no side effects — guarded by __name__)
import importlib.util
_spec = importlib.util.spec_from_file_location("ab_test", os.path.join(_SCRIPT_DIR, "ab_test.py"))
_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ab)  # type: ignore[union-attr]

run_sequence = _ab.run_sequence
evaluate = _ab.evaluate
SUBSET = _ab.SUBSET

from tracker.config import load_yaml_config
from tracker.data_utils import load_manifest, load_gt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_CONFIG = os.path.join(_REPO_ROOT, "configs", "i12_rescue_area_gate.yaml")

# Canary seqs run first for cheap early pruning; NOT counted in FS_imm
CANARY_SEQS = [
    "dataset3/car8",        # known F5 casualty: raw AUC ≈ 0.857, F5-full → 0.447
    "dataset2/Paragliding3",  # known F5 casualty: raw AUC ≈ 0.752, F5-full → 0.387
]

# Prune if any canary IMM delta drops below this threshold
# (F5-full worst canary is -0.410; we tolerate up to -0.35 before pruning)
PRUNE_THRESHOLD = -0.35

# Baseline to beat (i12+T1+F5 all-frames, 20-seq)
BASELINE_FS_IMM = 0.7139


# ---------------------------------------------------------------------------
# Config builder — deep-copy base YAML, override trial params
# ---------------------------------------------------------------------------
def _build_imm_cfg(base_cfg: dict, q_scale: float, r_pos_scale: float) -> dict:
    """Return a modified copy of base_cfg with trial-specific Q/R."""
    cfg = copy.deepcopy(base_cfg)
    imm = cfg.setdefault("imm", {})
    imm["q_scale"] = q_scale
    mn = imm.setdefault("measurement_noise", {})
    mn["r_pos_scale"] = r_pos_scale
    return cfg


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def make_objective(tracker, manifest, base_cfg: dict,
                   gmc_enabled: bool, adaptive_r_enabled: bool):
    """Close over shared state; return the objective function."""

    def objective(trial: optuna.Trial) -> float:
        # ── Search space ────────────────────────────────────────────────
        f5_alpha    = trial.suggest_float("f5_alpha",    0.6,   1.0)
        q_scale     = trial.suggest_float("q_scale",     5.0, 200.0, log=True)
        r_pos_scale = trial.suggest_float("r_pos_scale", 5.0, 200.0, log=True)

        imm_cfg = _build_imm_cfg(base_cfg, q_scale, r_pos_scale)

        # ── Phase 1: Canary check (fast prune) ──────────────────────────
        canary_auc_r, canary_auc_i = [], []
        for seq_id in CANARY_SEQS:
            if seq_id not in manifest["train"]:
                continue
            seq_info = manifest["train"][seq_id]
            gt = load_gt(seq_id, manifest)

            preds_raw = run_sequence(
                tracker, seq_id, seq_info, manifest, use_kf=False)
            auc_r, _ = evaluate(gt, preds_raw);  del preds_raw

            preds_imm = run_sequence(
                tracker, seq_id, seq_info, manifest,
                use_kf=True, kf_mode="ai_lead",
                gmc_enabled=gmc_enabled, adaptive_r_enabled=adaptive_r_enabled,
                imm_cfg=imm_cfg, imm_cfg_path=BASE_CONFIG,
                f5_feedback=True, f5_alpha=f5_alpha,
            )
            auc_i, _ = evaluate(gt, preds_imm);  del preds_imm, gt

            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

            canary_auc_r.append(auc_r)
            canary_auc_i.append(auc_i)

        canary_deltas = [i - r for i, r in zip(canary_auc_i, canary_auc_r)]
        worst_canary = min(canary_deltas) if canary_deltas else 0.0
        print(f"  [T{trial.number}] canary worst Δ={worst_canary:+.3f} "
              f"(f5α={f5_alpha:.2f}, q={q_scale:.1f}, rp={r_pos_scale:.1f})")

        if worst_canary < PRUNE_THRESHOLD:
            raise optuna.TrialPruned(
                f"canary delta {worst_canary:.3f} < threshold {PRUNE_THRESHOLD}")

        # Report intermediate value for MedianPruner
        partial_fs = 0.6 * np.mean(canary_auc_i) + 0.4 * np.mean(
            [0.0] * len(canary_auc_i))  # NP unknown at canary stage — use 0
        trial.report(partial_fs, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ── Phase 2: Full SUBSET evaluation ─────────────────────────────
        auc_raws, auc_imms, np_raws, np_imms = [], [], [], []
        for seq_id in SUBSET:
            if seq_id not in manifest["train"]:
                continue
            seq_info = manifest["train"][seq_id]
            gt = load_gt(seq_id, manifest)

            preds_raw = run_sequence(
                tracker, seq_id, seq_info, manifest, use_kf=False)
            auc_r, np_r = evaluate(gt, preds_raw);  del preds_raw

            preds_imm = run_sequence(
                tracker, seq_id, seq_info, manifest,
                use_kf=True, kf_mode="ai_lead",
                gmc_enabled=gmc_enabled, adaptive_r_enabled=adaptive_r_enabled,
                imm_cfg=imm_cfg, imm_cfg_path=BASE_CONFIG,
                f5_feedback=True, f5_alpha=f5_alpha,
            )
            auc_i, np_i = evaluate(gt, preds_imm);  del preds_imm, gt

            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

            auc_raws.append(auc_r)
            auc_imms.append(auc_i)
            np_raws.append(np_r)
            np_imms.append(np_i)

        fs_imm = 0.6 * np.mean(auc_imms) + 0.4 * np.mean(np_imms)
        delta = fs_imm - BASELINE_FS_IMM
        print(f"  [T{trial.number}] FS_imm={fs_imm:.4f}  Δ={delta:+.4f}  "
              f"f5α={f5_alpha:.3f}  q={q_scale:.1f}  rp={r_pos_scale:.1f}")
        return fs_imm

    return objective


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------
def _write_checkpoint(study: optuna.Study, trial: optuna.Trial,
                      checkpoint_dir: str, base_cfg: dict) -> None:
    """Called after every completed trial. Saves best config + summary."""
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        return
    best = max(completed, key=lambda t: t.value)
    # Only write when this trial is the new best (or is the first completed)
    if trial.number != best.number:
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    study_slug = study.study_name.replace(" ", "_")

    # ── 1. Best YAML config ─────────────────────────────────────────────
    cfg = copy.deepcopy(base_cfg)
    cfg["imm"]["q_scale"] = round(best.params["q_scale"], 3)
    cfg["imm"]["measurement_noise"]["r_pos_scale"] = round(best.params["r_pos_scale"], 3)
    cfg.setdefault("ai", {})["f5_alpha"] = round(best.params["f5_alpha"], 4)
    yaml_path = os.path.join(checkpoint_dir, f"{study_slug}_best.yaml")
    with open(yaml_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

    # ── 2. Human-readable summary ───────────────────────────────────────
    summary_path = os.path.join(checkpoint_dir, f"{study_slug}_summary.txt")
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    lines = [
        f"study      : {study.study_name}",
        f"completed  : {len(completed)}  pruned: {n_pruned}",
        f"best_trial : #{best.number}",
        f"FS_imm     : {best.value:.4f}  (Δ={best.value - BASELINE_FS_IMM:+.4f})",
        f"f5_alpha   : {best.params['f5_alpha']:.4f}",
        f"q_scale    : {best.params['q_scale']:.3f}",
        f"r_pos_scale: {best.params['r_pos_scale']:.3f}",
        "",
        "Top-10:",
    ]
    top10 = sorted(completed, key=lambda t: t.value, reverse=True)[:10]
    for t in top10:
        lines.append(
            f"  #{t.number:<4} FS={t.value:.4f}"
            f"  f5α={t.params['f5_alpha']:.3f}"
            f"  q={t.params['q_scale']:.1f}"
            f"  rp={t.params['r_pos_scale']:.1f}"
        )
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  [CKPT] New best → {yaml_path}  (FS={best.value:.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna: f5_alpha + q_scale + r_pos_scale")
    parser.add_argument("--n-trials",   type=int, default=50)
    parser.add_argument("--gmc",        action="store_true", default=True,
                        help="Enable GMC (default ON, matches golden standard)")
    parser.add_argument("--adaptive-r", action="store_true", default=True,
                        help="Enable adaptive-R (default ON)")
    parser.add_argument("--no-gmc",     dest="gmc", action="store_false")
    parser.add_argument("--no-adaptive-r", dest="adaptive_r", action="store_false")
    parser.add_argument("--use-sgla",   action="store_true",
                        help="Use SGLATrackWrapper (PyTorch) instead of TRTTrackWrapper. "
                             "Required on Colab — TRT engines are device-specific.")
    parser.add_argument("--study-db",   default=None,
                        help="SQLite path for resumable study "
                             "(default: cache/optuna_studies/f5alpha.db)")
    parser.add_argument("--study-name", default="f5alpha_qscale_rpos",
                        help="Optuna study name (used with --study-db)")
    parser.add_argument("--jobs",       type=int, default=1,
                        help="Parallel jobs (default 1; >1 requires NFS-safe SQLite path)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory to save best YAML + summary after each new best trial. "
                             "Default: same dir as --study-db")
    args = parser.parse_args()

    # ── Tracker init ────────────────────────────────────────────────────
    if args.use_sgla:
        from tracker.sglatrack_wrapper import SGLATrackWrapper
        tracker = SGLATrackWrapper()
        print("Tracker: SGLATrackWrapper (PyTorch)")
    else:
        from tracker.trt_wrapper import TRTTrackWrapper
        tracker = TRTTrackWrapper()
        print("Tracker: TRTTrackWrapper (TensorRT)")

    manifest = load_manifest()
    base_cfg  = load_yaml_config(BASE_CONFIG)

    # ── Study setup ─────────────────────────────────────────────────────
    db_path = args.study_db or os.path.join(
        _REPO_ROOT, "cache", "optuna_studies", "f5alpha.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
    )

    checkpoint_dir = args.checkpoint_dir or os.path.dirname(db_path)

    print(f"Study      : {args.study_name}")
    print(f"DB         : {db_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Baseline FS_imm: {BASELINE_FS_IMM}  |  Trials: {args.n_trials}")
    print(f"Prune threshold (canary): {PRUNE_THRESHOLD}")
    print("-" * 60)

    def checkpoint_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        if trial.value is not None:  # skip pruned trials
            _write_checkpoint(study, trial, checkpoint_dir, base_cfg)

    objective = make_objective(
        tracker, manifest, base_cfg,
        gmc_enabled=args.gmc,
        adaptive_r_enabled=args.adaptive_r,
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.jobs,
                   callbacks=[checkpoint_callback])

    # ── Results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    completed = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )
    if not completed:
        n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
        print(f"  No completed trials (all {n_pruned} pruned by canary gate).")
        print(f"  Try relaxing PRUNE_THRESHOLD (currently {PRUNE_THRESHOLD}).")
        return
    bt = completed[0]
    print(f"  FS_imm : {bt.value:.4f}  (Δ vs baseline: {bt.value - BASELINE_FS_IMM:+.4f})")
    print(f"  f5_alpha    = {bt.params['f5_alpha']:.4f}")
    print(f"  q_scale     = {bt.params['q_scale']:.2f}")
    print(f"  r_pos_scale = {bt.params['r_pos_scale']:.2f}")
    print()
    print("Top-5 trials:")
    completed = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )[:5]
    print(f"  {'#':>4}  {'FS_imm':>8}  {'f5_alpha':>9}  {'q_scale':>9}  {'r_pos_scale':>11}")
    for t in completed[:5]:
        print(f"  {t.number:>4}  {t.value:>8.4f}"
              f"  {t.params['f5_alpha']:>9.4f}"
              f"  {t.params['q_scale']:>9.2f}"
              f"  {t.params['r_pos_scale']:>11.2f}")

    # ── Final save (idempotent — already written by callback if best stayed same) ──
    _write_checkpoint(study, bt, checkpoint_dir, base_cfg)
    print(f"\nFinal config saved to: {checkpoint_dir}/{args.study_name}_best.yaml")


if __name__ == "__main__":
    main()
