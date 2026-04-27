#!/usr/bin/env python3
"""Optuna canary tuner over the full tracker control surface.

This is the "take everything" tuner: it samples Kalman/IMM physics, adaptive-R,
GMC, decision gates, F5/rescue/refresh, ReID, ORU, Chaos, LK jitter smoothing,
EMA template update, particle filter, and regime-adaptive toggles.

It is intentionally canary-first because this search space is large and live
tracking is expensive.  Use it to find a promising policy, then validate with
`scripts/run_competition.py --split train`.

Examples:
    python scripts/tune_all_params_canary.py --n-trials 20
    python scripts/tune_all_params_canary.py --seq dataset5/uav4 --seq dataset3/truck_night
    python scripts/tune_all_params_canary.py --use-sgla --n-trials 50
"""

from __future__ import annotations

# ── Noise suppression (TF/CUDA/timm warnings spam) ───────────────────────────
import os as _os
import warnings as _warnings
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import copy
import gc
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial INFO dumps
except ImportError:  # keep --help usable on minimal environments
    optuna = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "build2"))
sys.path.insert(0, str(PROJECT_ROOT / "build"))

import importlib.util

_AB_SPEC = importlib.util.spec_from_file_location("ab_test", SCRIPT_DIR / "ab_test.py")
_AB = importlib.util.module_from_spec(_AB_SPEC)
assert _AB_SPEC.loader is not None
_AB_SPEC.loader.exec_module(_AB)

run_sequence = _AB.run_sequence
evaluate = _AB.evaluate

from tracker.config import load_yaml_config  # noqa: E402
from tracker.data_utils import load_gt, load_manifest  # noqa: E402


BASE_CONFIG = PROJECT_ROOT / "configs" / "i12_rescue_area_gate.yaml"

DEFAULT_CANARY = [
    # tiny target / early fragile
    "dataset5/uav4",
    "dataset5/car15",
    "dataset5/bike2",
    # early lock loss
    "dataset4/bird1",
    "dataset5/car2",
    # late drift / long coast
    "dataset3/truck_night",
    "dataset3/bus2-n",
    # occlusion / ID switch
    "dataset3/basketball_player1",
    "dataset3/group4",
    # high-fps lag / template drift
    "dataset4/group3",
    "dataset5/group3_2",
    # controls that must not be sacrificed
    "dataset3/car8",
    "dataset2/Paragliding3",
    "dataset3/uav1",
]

CONTROL_SEQS = {
    "dataset3/car8",
    "dataset2/Paragliding3",
    "dataset3/uav1",
}


def _fs(auc: float, norm_prec: float) -> float:
    return 0.6 * float(auc) + 0.4 * float(norm_prec)


def _set_enabled_block(block: dict[str, Any], enabled: bool) -> dict[str, Any]:
    block["enabled"] = bool(enabled)
    return block


def _suggest_bool(trial: optuna.Trial, name: str) -> bool:
    return bool(trial.suggest_categorical(name, [False, True]))


def suggest_trial(base_cfg: dict[str, Any], trial: optuna.Trial) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (trial_config_yaml_dict, call_args) for one Optuna trial."""
    cfg = copy.deepcopy(base_cfg)

    # ── Kalman / IMM physics ─────────────────────────────────────
    imm = cfg.setdefault("imm", {})
    meas = imm.setdefault("measurement_noise", {})
    trans = imm.setdefault("transition_matrix", {})
    imm["q_scale"] = trial.suggest_float("imm.q_scale", 0.3, 120.0, log=True)
    imm["q_size_vel_scale"] = trial.suggest_float("imm.q_size_vel_scale", 0.05, 8.0, log=True)
    meas["r_pos_scale"] = trial.suggest_float("imm.r_pos_scale", 0.5, 120.0, log=True)
    meas["r_size_scale"] = trial.suggest_float("imm.r_size_scale", 0.3, 20.0, log=True)
    trans["pi_persist"] = trial.suggest_float("imm.pi_persist", 0.80, 0.985)

    singer = cfg.setdefault("singer", {})
    singer["alpha"] = trial.suggest_float("singer.alpha", 0.5, 6.0)
    singer["sigma2_a"] = trial.suggest_float("singer.sigma2_a", 10.0, 220.0, log=True)
    singer["ar_q_sensitivity"] = trial.suggest_float("singer.ar_q_sensitivity", 0.0, 25.0)
    singer["ar_q_boost_cap"] = trial.suggest_float("singer.ar_q_boost_cap", 1.5, 6.0)

    maneuver = imm.setdefault("maneuver_detector", {})
    maneuver["enabled"] = _suggest_bool(trial, "maneuver.enabled")
    maneuver["chi2_threshold"] = trial.suggest_float("maneuver.chi2_threshold", 8.0, 35.0)
    maneuver["pi_persist"] = trial.suggest_float("maneuver.pi_persist", 0.60, 0.98)
    maneuver["pi_singer_boost"] = trial.suggest_float("maneuver.pi_singer_boost", 0.0, 0.45)
    maneuver["normal_pi_persist"] = trial.suggest_float("maneuver.normal_pi_persist", 0.85, 0.995)

    # ── Decision / hygiene / soft fusion ─────────────────────────
    decision = cfg.setdefault("decision", {})
    decision["conf_threshold"] = trial.suggest_float("decision.conf_threshold", 0.08, 0.45)
    decision["coast_threshold"] = trial.suggest_float("decision.coast_threshold", 0.01, 0.20)
    decision["max_coast_frames"] = trial.suggest_int("decision.max_coast_frames", 5, 100)
    decision["iou_threshold"] = trial.suggest_float("decision.iou_threshold", 0.05, 0.45)
    decision["conf_bypass_threshold"] = trial.suggest_float("decision.conf_bypass_threshold", 0.35, 0.95)
    decision["innovation_threshold"] = trial.suggest_float("decision.innovation_threshold", 0.3, 6.0, log=True)

    hygiene = cfg.setdefault("hygiene", {})
    hygiene["reinit_after"] = trial.suggest_int("hygiene.reinit_after", 3, 80)
    hygiene["max_area_frac"] = trial.suggest_float("hygiene.max_area_frac", 0.05, 0.70)
    hygiene["max_center_jump_frac"] = trial.suggest_float("hygiene.max_center_jump_frac", 0.05, 0.80)
    hygiene["aspect_ratio_lo"] = trial.suggest_float("hygiene.aspect_ratio_lo", 0.05, 0.40)
    hygiene["aspect_ratio_hi"] = trial.suggest_float("hygiene.aspect_ratio_hi", 2.0, 12.0)

    smart = cfg.setdefault("smart_intervention", {})
    smart["sm_conf_threshold"] = trial.suggest_float("smart.sm_conf_threshold", 0.08, 0.45)
    smart["sm_max_coast"] = trial.suggest_int("smart.sm_max_coast", 10, 100)
    smart["maneuver_threshold"] = trial.suggest_float("smart.maneuver_threshold", 0.0, 0.85)
    smart["maneuver_bypass_boost"] = trial.suggest_float("smart.maneuver_bypass_boost", 0.0, 0.35)

    soft = cfg.setdefault("soft_fusion", {})
    soft["alpha_gate_k_conf"] = trial.suggest_float("soft.alpha_gate_k_conf", 0.0, 5.0)
    soft["alpha_gate_lambda"] = trial.suggest_float("soft.alpha_gate_lambda", 0.0, 0.6)
    soft["reacq_r_decay"] = trial.suggest_float("soft.reacq_r_decay", 0.0, 0.8)

    adaptive_enabled = _suggest_bool(trial, "adaptive_r.enabled")
    adaptive = _set_enabled_block(cfg.setdefault("adaptive_r", {}), adaptive_enabled)
    adaptive["floor"] = trial.suggest_float("adaptive_r.floor", 0.005, 0.60, log=True)
    adaptive["cap"] = trial.suggest_float("adaptive_r.cap", 2.0, 30.0, log=True)
    adaptive["r_exponent"] = trial.suggest_float("adaptive_r.r_exponent", 0.6, 3.0)

    mahal = cfg.setdefault("mahalanobis", {})
    mahal["chi2_threshold"] = trial.suggest_float("mahal.chi2_threshold", 7.8, 40.0)
    mahal["bypass_after"] = trial.suggest_int("mahal.bypass_after", 1, 10)
    mahal["bypass_conf_thr"] = trial.suggest_float("mahal.bypass_conf_thr", 0.35, 0.95)
    mahal["bypass_vel_thr"] = trial.suggest_float("mahal.bypass_vel_thr", 0.5, 10.0)
    mahal["bypass_after_fast"] = trial.suggest_int("mahal.bypass_after_fast", 1, 5)
    mahal["bypass_after_slow"] = trial.suggest_int("mahal.bypass_after_slow", 3, 12)
    mahal["r_pos_base"] = trial.suggest_float("mahal.r_pos_base", 0.5, 120.0, log=True)
    mahal["r_size_base"] = trial.suggest_float("mahal.r_size_base", 2.0, 240.0, log=True)

    vel_gate = cfg.setdefault("vel_gate", {})
    vel_gate["min_speed"] = trial.suggest_float("vel_gate.min_speed", 0.0, 20.0)
    vel_gate["cos_thr"] = trial.suggest_float("vel_gate.cos_thr", 0.2, 0.9)

    vel_innov = cfg.setdefault("vel_innov", {})
    vel_innov["ratio_gate"] = trial.suggest_float("vel_innov.ratio_gate", 0.0, 8.0)
    vel_innov["min_speed"] = trial.suggest_float("vel_innov.min_speed", 0.1, 8.0)
    vel_innov["min_innov"] = trial.suggest_float("vel_innov.min_innov", 0.0, 40.0)
    cfg["accept_bbox_raw"] = _suggest_bool(trial, "accept_bbox_raw")

    # ── GMC ─────────────────────────────────────────────────────
    gmc_enabled = _suggest_bool(trial, "gmc.enabled")
    gmc = _set_enabled_block(cfg.setdefault("gmc", {}), gmc_enabled)
    gmc["n_features"] = trial.suggest_int("gmc.n_features", 150, 900)
    gmc["n_features_high"] = trial.suggest_categorical("gmc.n_features_high", [0, 500, 800, 1200])
    gmc["rot_thr_deg"] = trial.suggest_float("gmc.rot_thr_deg", 1.5, 8.0)
    gmc["high_feature_frames"] = trial.suggest_int("gmc.high_feature_frames", 3, 15)
    gmc["min_matches"] = trial.suggest_int("gmc.min_matches", 6, 24)
    gmc["inlier_ratio_threshold"] = trial.suggest_float("gmc.inlier_ratio_threshold", 0.18, 0.60)
    gmc["veto_inlier_ratio"] = trial.suggest_float("gmc.veto_inlier_ratio", 0.08, 0.40)
    gmc["borderline_inlier_ratio"] = trial.suggest_float("gmc.borderline_inlier_ratio", 0.18, 0.50)
    gmc["ransac_reproj_threshold"] = trial.suggest_float("gmc.ransac_reproj_threshold", 1.2, 7.0)
    gmc["downsample"] = trial.suggest_float("gmc.downsample", 0.35, 1.0)
    gmc["foreground_dilate_factor"] = trial.suggest_float("gmc.foreground_dilate_factor", 1.0, 2.4)
    gmc["max_translation_frac_diag"] = trial.suggest_float("gmc.max_translation_frac_diag", 0.03, 0.18)
    gmc["max_rotation_deg"] = trial.suggest_float("gmc.max_rotation_deg", 3.0, 25.0)
    gmc["history_window"] = trial.suggest_int("gmc.history_window", 0, 12)
    gmc["history_outlier_mult"] = trial.suggest_float("gmc.history_outlier_mult", 1.5, 6.0)
    gmc["freeze_maneuver_on_veto"] = _suggest_bool(trial, "gmc.freeze_maneuver_on_veto")
    gmc["freeze_frames_after_veto"] = trial.suggest_int("gmc.freeze_frames_after_veto", 0, 5)
    gmc["fail_q_boost"] = trial.suggest_float("gmc.fail_q_boost", 1.2, 12.0, log=True)

    # ── AI template / F5 / rescue policy ─────────────────────────
    ai = cfg.setdefault("ai", {})
    f5_feedback = _suggest_bool(trial, "ai.f5_feedback")
    f5_alpha = trial.suggest_float("ai.f5_alpha", 0.0, 1.0)
    ai["f5_feedback"] = f5_feedback
    ai["f5_alpha"] = f5_alpha
    ai["f5_scale_guard"] = trial.suggest_float("ai.f5_scale_guard", 0.0, 4.0)
    ai["f5_reject_window"] = trial.suggest_int("ai.f5_reject_window", 0, 80)
    ai["f5_reject_min_count"] = trial.suggest_int("ai.f5_reject_min_count", 1, 12)
    cfg["f5_obs_size"] = _suggest_bool(trial, "ai.f5_obs_size")
    cfg["f5_coast_only"] = _suggest_bool(trial, "ai.f5_coast_only")
    ai["f5_pos_only"] = _suggest_bool(trial, "ai.f5_pos_only")

    ai["search_scale_boost"] = trial.suggest_float("ai.search_scale_boost", 0.0, 1.0)
    ai["conf_refresh_low"] = trial.suggest_float("ai.conf_refresh_low", 0.15, 0.55)
    ai["conf_refresh_high"] = trial.suggest_float("ai.conf_refresh_high", 0.45, 0.85)
    ai["refresh_patience"] = trial.suggest_int("ai.refresh_patience", 0, 16)
    ai["refresh_decline_thr"] = trial.suggest_float("ai.refresh_decline_thr", 0.0, 0.15)
    ai["refresh_bypass_lookback"] = trial.suggest_int("ai.refresh_bypass_lookback", 0, 60)
    ai["refresh_min_bbox_area"] = trial.suggest_float("ai.refresh_min_bbox_area", 0.0, 1200.0)
    ai["refresh_min_interval"] = trial.suggest_int("ai.refresh_min_interval", 0, 60)
    ai["refresh_small_area_thr"] = trial.suggest_float("ai.refresh_small_area_thr", 0.0, 800.0)
    ai["refresh_small_interval"] = trial.suggest_categorical(
        "ai.refresh_small_interval", [0, 10, 30, 80, 99999]
    )
    ai["rescue_min_area"] = trial.suggest_float("ai.rescue_min_area", 0.0, 1200.0)

    ai["skip_when_confident"] = _suggest_bool(trial, "ai.skip_when_confident")
    ai["max_consecutive_skips"] = trial.suggest_int("ai.max_consecutive_skips", 1, 5)
    ai["singer_skip_thr"] = trial.suggest_float("ai.singer_skip_thr", 0.15, 0.70)
    ai["cv_stable_thr"] = trial.suggest_float("ai.cv_stable_thr", 0.45, 0.95)
    ai["cv_conf_min"] = trial.suggest_float("ai.cv_conf_min", 0.40, 0.90)

    # ── Appearance / occlusion extras ───────────────────────────
    reid = cfg.setdefault("reid", {})
    reid["enabled"] = _suggest_bool(trial, "reid.enabled")
    reid["sim_threshold"] = trial.suggest_float("reid.sim_threshold", 0.45, 0.90)
    reid["maxlen"] = trial.suggest_int("reid.maxlen", 10, 120)

    oru = cfg.setdefault("oru", {})
    oru["enabled"] = _suggest_bool(trial, "oru.enabled")
    oru["n_min"] = trial.suggest_int("oru.n_min", 2, 12)
    oru["n_max"] = trial.suggest_int("oru.n_max", 12, 60)
    oru["conf_reentry_min"] = trial.suggest_float("oru.conf_reentry_min", 0.25, 0.80)
    oru["velocity_cv_max"] = trial.suggest_float("oru.velocity_cv_max", 0.15, 1.5)
    oru["virtual_conf"] = trial.suggest_float("oru.virtual_conf", 0.1, 0.8)
    oru["skip_if_singer_dom"] = _suggest_bool(trial, "oru.skip_if_singer_dom")
    oru["singer_dom_thr"] = trial.suggest_float("oru.singer_dom_thr", 0.4, 0.95)
    oru["ring_buffer_size"] = trial.suggest_int("oru.ring_buffer_size", 20, 80)

    chaos = cfg.setdefault("chaos", {})
    chaos["enabled"] = _suggest_bool(trial, "chaos.enabled")
    chaos["window"] = trial.suggest_int("chaos.window", 5, 25)
    chaos["chaos_thr"] = trial.suggest_float("chaos.chaos_thr", 0.03, 0.16)
    chaos["chaos_max"] = trial.suggest_float("chaos.chaos_max", 0.10, 0.35)
    chaos["min_chaos_conf_factor"] = trial.suggest_float("chaos.min_chaos_conf_factor", 0.05, 0.60)
    chaos["min_samples"] = trial.suggest_int("chaos.min_samples", 3, 10)

    clahe = cfg.setdefault("clahe", {})
    clahe["enabled"] = _suggest_bool(trial, "clahe.enabled")
    clahe["clip_limit"] = trial.suggest_float("clahe.clip_limit", 0.8, 6.0)
    clahe["roi_scale"] = trial.suggest_float("clahe.roi_scale", 1.5, 6.0)
    clahe["tile_size"] = trial.suggest_categorical("clahe.tile_size", [4, 6, 8, 12, 16])

    # These are function-level experimental toggles in ab_test.py.
    call_args = {
        "gmc_enabled": gmc_enabled,
        "adaptive_r_enabled": adaptive_enabled,
        "search_scale_boost": ai["search_scale_boost"],
        "f5_feedback": f5_feedback,
        "f5_alpha": f5_alpha,
        "lk_jitter_enabled": _suggest_bool(trial, "lk_jitter.enabled"),
        "lk_alpha": trial.suggest_float("lk_jitter.alpha", 0.1, 0.9),
        "lk_max_corners": trial.suggest_int("lk_jitter.max_corners", 8, 40),
        "ema_template_enabled": _suggest_bool(trial, "ema_template.enabled"),
        "ema_alpha": trial.suggest_float("ema_template.alpha", 0.01, 0.25),
        "ema_conf_min": trial.suggest_float("ema_template.conf_min", 0.55, 0.95),
        "ema_mahal_thr_sq": trial.suggest_float("ema_template.mahal_thr_sq", 4.0, 25.0),
        "ema_interval": trial.suggest_int("ema_template.interval", 2, 15),
        "pf_enabled": _suggest_bool(trial, "particle_filter.enabled"),
        "pf_n_particles": trial.suggest_int("particle_filter.n_particles", 100, 1200),
        "pf_trust_ramp": trial.suggest_int("particle_filter.trust_ramp", 3, 25),
        "regime_adaptive": _suggest_bool(trial, "regime_adaptive.enabled"),
    }
    return cfg, call_args


def evaluate_trial_sequence(
    tracker,
    manifest: dict[str, Any],
    seq_id: str,
    cfg: dict[str, Any],
    cfg_path: str,
    call_args: dict[str, Any],
) -> tuple[float, float]:
    seq_info = manifest["train"][seq_id]
    gt = load_gt(seq_id, manifest)
    preds = run_sequence(
        tracker,
        seq_id,
        seq_info,
        manifest,
        use_kf=True,
        kf_mode="ai_lead",
        imm_cfg=cfg,
        imm_cfg_path=cfg_path,
        **call_args,
    )
    auc, norm_prec = evaluate(gt, preds)
    del preds, gt
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return float(auc), float(norm_prec)


def compute_raw_baselines(tracker, manifest: dict[str, Any], seq_ids: list[str]) -> dict[str, tuple[float, float]]:
    baselines: dict[str, tuple[float, float]] = {}
    for seq_id in seq_ids:
        seq_info = manifest["train"][seq_id]
        gt = load_gt(seq_id, manifest)
        preds = run_sequence(tracker, seq_id, seq_info, manifest, use_kf=False)
        auc, norm_prec = evaluate(gt, preds)
        baselines[seq_id] = (float(auc), float(norm_prec))
        del preds, gt
        gc.collect()
    return baselines


def write_trial_config(cfg: dict[str, Any], tmp_dir: str, trial_number: int) -> str:
    path = os.path.join(tmp_dir, f"trial_{trial_number:05d}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return path


class _ProgressCallback:
    """Per-trial progress line printed to stdout after every completed trial.

    Example output::

        Trial   7/100 | score=+0.7231 | best=0.7231 (#7) | elapsed=14m 03s | ETA=2h 45m
    """

    def __init__(self, n_trials: int, t0: float) -> None:
        self._n = n_trials
        self._t0 = t0

    @staticmethod
    def _fmt(secs: float) -> str:
        h, rem = divmod(int(max(secs, 0)), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

    def __call__(self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        complete = [t for t in study.trials if t.state.name == "COMPLETE"]
        done = len(complete)
        elapsed = time.time() - self._t0
        per_trial = elapsed / max(done, 1)
        eta = per_trial * max(self._n - done, 0)

        score = trial.value if trial.value is not None else float("nan")
        try:
            best_val = study.best_value
            best_num = study.best_trial.number
            best_str = f"{best_val:.4f} (#{best_num})"
        except ValueError:
            best_str = "n/a"

        pct = 100.0 * done / self._n if self._n else 0.0
        print(
            f"\033[1mTrial {done:4d}/{self._n}\033[0m  [{pct:5.1f}%]"
            f"  score={score:+.4f}"
            f"  best={best_str}"
            f"  elapsed={self._fmt(elapsed)}"
            f"  ETA={self._fmt(eta)}",
            flush=True,
        )


def make_objective(
    tracker,
    manifest: dict[str, Any],
    seq_ids: list[str],
    baselines: dict[str, tuple[float, float]],
    base_cfg: dict[str, Any],
    tmp_dir: str,
    max_degradation: float,
    control_max_degradation: float,
):
    def objective(trial: optuna.Trial) -> float:
        cfg, call_args = suggest_trial(base_cfg, trial)
        cfg_path = write_trial_config(cfg, tmp_dir, trial.number)
        results: list[tuple[str, float, float]] = []
        worst_deg = 0.0
        worst_control_deg = 0.0
        try:
            for idx, seq_id in enumerate(seq_ids):
                auc, norm_prec = evaluate_trial_sequence(
                    tracker, manifest, seq_id, cfg, cfg_path, call_args
                )
                results.append((seq_id, auc, norm_prec))
                base_auc, base_np = baselines[seq_id]
                deg = _fs(base_auc, base_np) - _fs(auc, norm_prec)
                worst_deg = max(worst_deg, deg)
                if seq_id in CONTROL_SEQS:
                    worst_control_deg = max(worst_control_deg, deg)

                partial = float(np.mean([_fs(a, n) for _, a, n in results]))
                trial.report(partial, idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        finally:
            try:
                os.remove(cfg_path)
            except OSError:
                pass

        mean_fs = float(np.mean([_fs(a, n) for _, a, n in results]))
        penalty = 0.0
        if worst_deg > max_degradation:
            penalty += 2.0 * (worst_deg - max_degradation)
        if worst_control_deg > control_max_degradation:
            penalty += 4.0 * (worst_control_deg - control_max_degradation)
        score = mean_fs - penalty

        trial.set_user_attr("mean_fs", mean_fs)
        trial.set_user_attr("worst_degradation", worst_deg)
        trial.set_user_attr("worst_control_degradation", worst_control_deg)
        trial.set_user_attr("config_yaml", yaml.safe_dump(cfg, sort_keys=False))
        trial.set_user_attr("call_args", call_args)
        return score

    return objective


def write_best_checkpoint(study: optuna.Study, checkpoint_dir: str) -> None:
    best = study.best_trial
    cfg_yaml = best.user_attrs.get("config_yaml")
    if not cfg_yaml:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    stem = study.study_name.replace("/", "_")
    yaml_path = os.path.join(checkpoint_dir, f"{stem}_best.yaml")
    summary_path = os.path.join(checkpoint_dir, f"{stem}_summary.txt")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(cfg_yaml)
    lines = [
        f"study: {study.study_name}",
        f"best_trial: {best.number}",
        f"objective: {best.value:.6f}",
        f"mean_fs: {best.user_attrs.get('mean_fs', 0.0):.6f}",
        f"worst_degradation: {best.user_attrs.get('worst_degradation', 0.0):.6f}",
        f"worst_control_degradation: {best.user_attrs.get('worst_control_degradation', 0.0):.6f}",
        "",
        "call_args:",
    ]
    for key, value in sorted(best.user_attrs.get("call_args", {}).items()):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("best_params:")
    for key, value in sorted(best.params.items()):
        lines.append(f"  {key}: {value}")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[CKPT] best config -> {yaml_path}")
    print(f"[CKPT] summary     -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="All-parameter Optuna tuner on canary sequences")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--seq", action="append", default=None, help="Canary sequence. Repeatable.")
    parser.add_argument("--use-sgla", action="store_true", help="Use PyTorch SGLATrack instead of TensorRT")
    parser.add_argument("--study-name", default="all_params_canary")
    parser.add_argument("--study-db", default=str(PROJECT_ROOT / "cache" / "optuna_studies" / "all_params_canary.db"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "cache" / "optuna_studies"))
    parser.add_argument("--max-degradation", type=float, default=0.08)
    parser.add_argument("--control-max-degradation", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if optuna is None:
        raise SystemExit(
            "Optuna is required for this tuner. Install it in this environment "
            "or run on the Colab/env where the existing Optuna scripts are used."
        )

    np.random.seed(args.seed)

    if args.use_sgla:
        from tracker.sglatrack_wrapper import SGLATrackWrapper

        tracker = SGLATrackWrapper()
        print("Tracker: SGLATrackWrapper")
    else:
        from tracker.trt_wrapper import TRTTrackWrapper

        tracker = TRTTrackWrapper()
        print("Tracker: TRTTrackWrapper")

    manifest = load_manifest()
    seq_ids = args.seq or DEFAULT_CANARY
    missing = [seq for seq in seq_ids if seq not in manifest.get("train", {})]
    if missing:
        raise SystemExit(f"Missing train sequences: {missing}")

    base_cfg = load_yaml_config(args.base_config)
    print(f"Base config: {args.base_config}")
    print(f"Canary sequences ({len(seq_ids)}):")
    for seq in seq_ids:
        tag = " control" if seq in CONTROL_SEQS else ""
        print(f"  - {seq}{tag}")

    print("\nComputing raw baselines...")
    baselines = compute_raw_baselines(tracker, manifest, seq_ids)
    raw_mean = float(np.mean([_fs(a, n) for a, n in baselines.values()]))
    print(f"Raw canary mean FS: {raw_mean:.4f}")

    os.makedirs(os.path.dirname(args.study_db), exist_ok=True)
    storage = f"sqlite:///{args.study_db}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    with tempfile.TemporaryDirectory(prefix="all_params_canary_") as tmp_dir:
        objective = make_objective(
            tracker,
            manifest,
            seq_ids,
            baselines,
            base_cfg,
            tmp_dir,
            max_degradation=args.max_degradation,
            control_max_degradation=args.control_max_degradation,
        )
        print(f"\nStarting {args.n_trials} trials. DB: {args.study_db}")
        t0 = time.time()
        progress_cb = _ProgressCallback(n_trials=args.n_trials, t0=t0)
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=False,  # _ProgressCallback handles progress
            callbacks=[progress_cb],
        )
        print(f"Elapsed: {time.time() - t0:.1f}s")

    if len([t for t in study.trials if t.value is not None]) == 0:
        print("No completed trials.")
        return

    print("\nBest:")
    print(f"  trial: {study.best_trial.number}")
    print(f"  objective: {study.best_value:.6f}")
    print(f"  mean_fs: {study.best_trial.user_attrs.get('mean_fs', 0.0):.6f}")
    print(f"  worst_deg: {study.best_trial.user_attrs.get('worst_degradation', 0.0):.6f}")
    write_best_checkpoint(study, args.checkpoint_dir)


if __name__ == "__main__":
    main()
