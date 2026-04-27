"""Shared config loading for runtime, replay, and evaluation scripts."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import yaml

DEFAULT_RUNTIME_PARAMS: dict[str, Any] = {
    "conf_threshold": 0.18,
    "iou_threshold": 0.2,
    "coast_threshold": 0.05,
    "max_coast_frames": 30,
    "reinit_after": 8,
    "max_area_frac": 0.25,
    "max_center_jump_frac": 0.30,
    "aspect_ratio_lo": 0.2,
    "aspect_ratio_hi": 5.0,
    "alpha_conf_lo": 0.3,
    "alpha_conf_hi": 0.7,
    "conf_bypass_threshold": 0.15,
    "innovation_threshold": 2.0,
    "sm_conf_threshold": 0.18,
    "sm_max_coast": 30,
    "adaptive_r_enabled": False,
    "adaptive_r_floor": 0.4,
    "adaptive_r_cap": 10.0,
    "gmc_enabled": False,
    "gmc_n_features": 200,
    "gmc_quality_enabled": True,
    "gmc_veto_inlier_ratio": 0.2,
    "gmc_borderline_inlier_ratio": 0.3,
    "gmc_max_translation_frac_diag": 0.08,
    "gmc_max_rotation_deg": 12.0,
    "gmc_history_window": 5,
    "gmc_history_outlier_mult": 3.0,
    "gmc_freeze_maneuver_on_veto": True,
    "gmc_freeze_frames_after_veto": 1,
    # D3: dual GMC
    "gmc_n_features_high": 0,          # 0 = disabled; try 600
    "gmc_rot_thr_deg": 3.0,            # rotation angle to trigger high-feature mode
    "gmc_high_feature_frames": 8,      # frames to stay in high-feature mode
    "gmc_inlier_ratio_threshold": 0.3,
    "gmc_min_matches": 6,
    "gmc_ransac_reproj_threshold": 3.0,
    "gmc_downsample": 0.5,
    "gmc_fail_q_boost": 4.0,
    "gmc_foreground_dilate_factor": 1.4,
    "association_enabled": True,
    "association_top_k": 5,
    "association_iou_threshold": 0.3,
    "association_score_weight": 0.0,
    "ai_skip_when_confident": False,
    "ai_skip_conf_threshold": 0.8,
    "ai_max_consecutive_skips": 2,
    # Phase 2: dynamic R exponent (1.0 = linear / legacy behaviour)
    "r_exponent": 1.0,
    # Phase 3: Mahalanobis innovation gate (0.0 = disabled)
    "mahal_chi2_gate": 0.0,
    "r_pos_base": 1.0,
    "r_size_base": 10.0,
    # Phase 4: IMM manoeuvre-probability bypass boost (0.0 = disabled)
    "maneuver_threshold": 0.0,
    "maneuver_bypass_boost": 0.0,
    # Soft fusion: α-blend sigmoid gate + reacquisition R-inflation (0.0 = disabled)
    "alpha_gate_k_conf": 0.0,
    "alpha_gate_lambda": 0.0,
    "reacq_r_decay": 0.0,
    # Faz C1b: parametrised AI-skip thresholds
    "singer_skip_thr": 0.35,   # Singer > this → always run AI
    "cv_stable_thr": 0.70,     # CV > this → allow skip
    "cv_conf_min": 0.65,       # conf > this required for CV skip
    # Faz C2: Singer-adaptive search window expansion (0.0 = disabled)
    "search_scale_boost": 0.0,
    # Faz D: Proactive template refresh (0 = disabled)
    "conf_refresh_low": 0.35,
    "conf_refresh_high": 0.65,
    "refresh_patience": 0,
    "refresh_decline_thr": 0.04,
    # Min predicted-bbox area (w*h px²) required to fire a refresh.
    # 0 = disabled. Blocks refreshes on small targets (e.g. bike: ~164 px²)
    # while keeping them for large targets (e.g. truck: ~2000 px²).
    "refresh_min_bbox_area": 0.0,
    # I3: Size-velocity gate. 0 = disabled. Blocks Faz D when |vw|+|vh| > threshold.
    "refresh_scale_vel_max": 0.0,
    # D1: IMM maneuver detector — dynamic pi injection on Mahalanobis spike
    "maneuver_pi_enabled": False,
    "maneuver_pi_thr": 16.0,
    "maneuver_pi_persist": 0.72,
    "maneuver_pi_singer_boost": 0.20,
    "normal_pi_persist": 0.96,
    # D2B: velocity anchor — max px/frame allowed in coast search window (0 = disabled)
    "velocity_anchor_max": 0.0,
    # D5: ReID-gated rescue
    "reid_enabled": False,
    "reid_sim_threshold": 0.70,
    "reid_maxlen": 50,
    # ROI CLAHE: contrast enhancement in search region (false = disabled)
    "clahe_enabled": False,
    "clahe_clip_limit": 2.0,   # CLAHE clip limit [0.5, 8.0]
    "clahe_roi_scale": 3.0,    # ROI expansion factor around predicted bbox [1.5, 6.0]
    "clahe_tile_size": 8,      # tile grid size (NxN) [4, 16]
    # F8-guard: cap w/h in F5 set_state to init_w/h * f5_scale_guard (0.0 = disabled)
    "f5_scale_guard": 0.0,
    # Dynamic Q from AR rate (0.0 = disabled)
    "ar_q_sensitivity": 0.0,   # q_boost = 1 + |delta_AR| * sensitivity [0, 30]
    "ar_q_boost_cap": 3.0,     # max Q multiplier per frame [1.5, 8.0]
    # Velocity direction gate: reject 180° ID-switch detections (0 = disabled)
    "vel_gate_min_speed": 0.0,  # min KF speed px/frame to arm gate [2, 20]
    "vel_gate_cos_thr": 0.5,    # reject when cos(angle) < -thr (>120°) [0.3, 0.8]
    # Velocity-relative innovation gate: force coast when center_innov / kf_speed > thr
    # 0 = disabled.  Fires only in normal (non-bypass) path to block ID-switch on
    # stationary targets (truck_night style).
    "vel_innov_ratio_gate": 0.0,  # ratio threshold; 0 = disabled
    "vel_innov_min_speed": 1.0,   # floor for kf_speed denominator (px/frame)
    "vel_innov_min_innov": 0.0,   # minimum absolute center innov (px) to fire gate; 0=no min
    "accept_bbox_raw": False,     # output raw AI obs (not KF state) on accepted frames
    "f5_obs_size": False,         # f5 feedback uses AI obs size + KF position (fixes size-lag)
    "f5_coast_only": False,       # f5 feedback only fires during coasting (reject_streak > 0)
    # ORU backfill: raw sub-dict forwarded to OruConfig.from_dict() (empty = defaults)
    "oru_config": {},
    # Chaos trigger: raw sub-dict forwarded to ChaosConfig.from_dict() (empty = disabled)
    "chaos_config": {},
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def load_yaml_config(path: str | None) -> dict[str, Any]:
    """Load a YAML mapping or return an empty dict when absent."""
    if path is None or not os.path.exists(path):
        return {}

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, Mapping):
        raise TypeError(f"config at {path!r} must be a mapping")

    return dict(data)


def normalize_runtime_config(
    cfg: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map mixed config schemas onto the canonical runtime param shape."""
    params = dict(DEFAULT_RUNTIME_PARAMS)
    if defaults is not None:
        params.update(defaults)

    if not cfg:
        return params

    decision = _as_mapping(cfg.get("decision"))
    hygiene = _as_mapping(cfg.get("hygiene"))
    blend = _as_mapping(cfg.get("blend"))
    smart = _as_mapping(cfg.get("smart_intervention")) or _as_mapping(cfg.get("smart"))
    adaptive_r = _as_mapping(cfg.get("adaptive_r"))
    gmc = _as_mapping(cfg.get("gmc"))
    association = _as_mapping(cfg.get("association"))
    ai = _as_mapping(cfg.get("ai"))
    coasting = _as_mapping(cfg.get("coasting"))

    canonical_conf_threshold = _first_non_none(
        decision.get("confidence_threshold"),
        decision.get("conf_threshold"),
        hygiene.get("conf_threshold"),
    )
    if canonical_conf_threshold is not None:
        params["conf_threshold"] = float(canonical_conf_threshold)
        params["sm_conf_threshold"] = float(canonical_conf_threshold)
    else:
        legacy_sm_conf_threshold = _first_non_none(
            smart.get("sm_conf_threshold"),
            cfg.get("sm_conf_threshold"),
        )
        if legacy_sm_conf_threshold is not None:
            params["sm_conf_threshold"] = float(legacy_sm_conf_threshold)

    coast_threshold = _first_non_none(
        decision.get("coast_threshold"),
        hygiene.get("coast_threshold"),
    )
    if coast_threshold is not None:
        params["coast_threshold"] = float(coast_threshold)

    max_coast_frames = _first_non_none(
        decision.get("max_coast_frames"),
        hygiene.get("max_coast_frames"),
        coasting.get("max_frames"),
    )
    if max_coast_frames is not None:
        params["max_coast_frames"] = int(max_coast_frames)

    sm_max_coast = _first_non_none(
        smart.get("sm_max_coast"),
        cfg.get("sm_max_coast"),
        coasting.get("max_frames"),
        decision.get("max_coast_frames"),
        hygiene.get("max_coast_frames"),
    )
    if sm_max_coast is not None:
        params["sm_max_coast"] = int(sm_max_coast)

    iou_threshold = decision.get("iou_threshold")
    if iou_threshold is not None:
        params["iou_threshold"] = float(iou_threshold)

    conf_bypass_threshold = _first_non_none(
        decision.get("conf_bypass_threshold"),
        smart.get("conf_bypass_threshold"),
        cfg.get("conf_bypass_threshold"),
    )
    if conf_bypass_threshold is not None:
        params["conf_bypass_threshold"] = float(conf_bypass_threshold)

    innovation_threshold = _first_non_none(
        decision.get("innovation_threshold"),
        smart.get("innovation_threshold"),
        cfg.get("innovation_threshold"),
    )
    if innovation_threshold is not None:
        params["innovation_threshold"] = float(innovation_threshold)

    reinit_after = hygiene.get("reinit_after")
    if reinit_after is not None:
        params["reinit_after"] = int(reinit_after)

    max_area_frac = hygiene.get("max_area_frac")
    if max_area_frac is not None:
        params["max_area_frac"] = float(max_area_frac)

    max_center_jump_frac = hygiene.get("max_center_jump_frac")
    if max_center_jump_frac is not None:
        params["max_center_jump_frac"] = float(max_center_jump_frac)

    aspect_ratio_lo = hygiene.get("aspect_ratio_lo")
    if aspect_ratio_lo is not None:
        params["aspect_ratio_lo"] = float(aspect_ratio_lo)

    aspect_ratio_hi = hygiene.get("aspect_ratio_hi")
    if aspect_ratio_hi is not None:
        params["aspect_ratio_hi"] = float(aspect_ratio_hi)

    alpha_conf_lo = blend.get("alpha_conf_lo")
    if alpha_conf_lo is not None:
        params["alpha_conf_lo"] = float(alpha_conf_lo)

    alpha_conf_hi = blend.get("alpha_conf_hi")
    if alpha_conf_hi is not None:
        params["alpha_conf_hi"] = float(alpha_conf_hi)

    adaptive_r_enabled = adaptive_r.get("enabled")
    if adaptive_r_enabled is not None:
        params["adaptive_r_enabled"] = bool(adaptive_r_enabled)

    adaptive_r_floor = adaptive_r.get("floor")
    if adaptive_r_floor is not None:
        params["adaptive_r_floor"] = float(adaptive_r_floor)

    # D2A: R inflation cap
    adaptive_r_cap = adaptive_r.get("cap")
    if adaptive_r_cap is not None:
        params["adaptive_r_cap"] = float(adaptive_r_cap)

    # D2B: velocity anchor
    velocity_anchor_max = adaptive_r.get("velocity_anchor_max")
    if velocity_anchor_max is not None:
        params["velocity_anchor_max"] = float(velocity_anchor_max)

    # Phase 2: dynamic R exponent
    r_exponent = adaptive_r.get("r_exponent")
    if r_exponent is not None:
        params["r_exponent"] = float(r_exponent)

    # Phase 3: Mahalanobis innovation gate
    mahal = _as_mapping(cfg.get("mahalanobis"))
    mahal_chi2 = _first_non_none(mahal.get("chi2_threshold"), mahal.get("chi2_gate"))
    if mahal_chi2 is not None:
        params["mahal_chi2_gate"] = float(mahal_chi2)
    r_pos_base_ = mahal.get("r_pos_base")
    if r_pos_base_ is not None:
        params["r_pos_base"] = float(r_pos_base_)
    r_size_base_ = mahal.get("r_size_base")
    if r_size_base_ is not None:
        params["r_size_base"] = float(r_size_base_)
    mahal_bypass_after_ = mahal.get("bypass_after")
    if mahal_bypass_after_ is not None:
        params["mahal_bypass_after"] = int(mahal_bypass_after_)

    # Dynamic bypass: confidence + velocity based adaptive mahal_bypass_after
    for _k, _conv in (
        ("bypass_conf_thr",   float),
        ("bypass_vel_thr",    float),
        ("bypass_after_fast", int),
        ("bypass_after_slow", int),
    ):
        _v = mahal.get(_k)
        if _v is not None:
            params[f"mahal_{_k}"] = _conv(_v)

    # Phase 4: IMM manoeuvre-probability bypass boost
    maneuver_threshold = smart.get("maneuver_threshold")
    if maneuver_threshold is not None:
        params["maneuver_threshold"] = float(maneuver_threshold)
    maneuver_bypass_boost = smart.get("maneuver_bypass_boost")
    if maneuver_bypass_boost is not None:
        params["maneuver_bypass_boost"] = float(maneuver_bypass_boost)

    # Soft fusion: α-blend sigmoid gate + reacquisition R-inflation ramp
    soft_fusion = _as_mapping(cfg.get("soft_fusion"))
    for _sf_key in ("alpha_gate_k_conf", "alpha_gate_lambda", "reacq_r_decay"):
        _sf_val = soft_fusion.get(_sf_key)
        if _sf_val is not None:
            params[_sf_key] = float(_sf_val)

    gmc_enabled = gmc.get("enabled")
    if gmc_enabled is not None:
        params["gmc_enabled"] = bool(gmc_enabled)

    gmc_n_features = gmc.get("n_features")
    if gmc_n_features is not None:
        params["gmc_n_features"] = int(gmc_n_features)

    v = gmc.get("quality_enabled")
    if v is not None:
        params["gmc_quality_enabled"] = bool(v)
    v = gmc.get("veto_inlier_ratio")
    if v is not None:
        params["gmc_veto_inlier_ratio"] = float(v)
    v = gmc.get("borderline_inlier_ratio")
    if v is not None:
        params["gmc_borderline_inlier_ratio"] = float(v)
    v = gmc.get("max_translation_frac_diag")
    if v is not None:
        params["gmc_max_translation_frac_diag"] = float(v)
    v = gmc.get("max_rotation_deg")
    if v is not None:
        params["gmc_max_rotation_deg"] = float(v)
    v = gmc.get("history_window")
    if v is not None:
        params["gmc_history_window"] = int(v)
    v = gmc.get("history_outlier_mult")
    if v is not None:
        params["gmc_history_outlier_mult"] = float(v)
    v = gmc.get("freeze_maneuver_on_veto")
    if v is not None:
        params["gmc_freeze_maneuver_on_veto"] = bool(v)
    v = gmc.get("freeze_frames_after_veto")
    if v is not None:
        params["gmc_freeze_frames_after_veto"] = int(v)

    gmc_inlier_ratio_threshold = gmc.get("inlier_ratio_threshold")
    if gmc_inlier_ratio_threshold is not None:
        params["gmc_inlier_ratio_threshold"] = float(gmc_inlier_ratio_threshold)

    gmc_min_matches = gmc.get("min_matches")
    if gmc_min_matches is not None:
        params["gmc_min_matches"] = int(gmc_min_matches)

    gmc_ransac_reproj_threshold = gmc.get("ransac_reproj_threshold")
    if gmc_ransac_reproj_threshold is not None:
        params["gmc_ransac_reproj_threshold"] = float(gmc_ransac_reproj_threshold)

    gmc_downsample = gmc.get("downsample")
    if gmc_downsample is not None:
        params["gmc_downsample"] = float(gmc_downsample)

    gmc_fail_q_boost = gmc.get("fail_q_boost")
    if gmc_fail_q_boost is not None:
        params["gmc_fail_q_boost"] = float(gmc_fail_q_boost)

    gmc_foreground_dilate_factor = gmc.get("foreground_dilate_factor")
    if gmc_foreground_dilate_factor is not None:
        params["gmc_foreground_dilate_factor"] = float(gmc_foreground_dilate_factor)

    # D3: dual GMC params
    v = gmc.get("n_features_high")
    if v is not None:
        params["gmc_n_features_high"] = int(v)
    v = gmc.get("rot_thr_deg")
    if v is not None:
        params["gmc_rot_thr_deg"] = float(v)
    v = gmc.get("high_feature_frames")
    if v is not None:
        params["gmc_high_feature_frames"] = int(v)

    association_enabled = association.get("enabled")
    if association_enabled is not None:
        params["association_enabled"] = bool(association_enabled)

    association_top_k = association.get("top_k")
    if association_top_k is not None:
        params["association_top_k"] = int(association_top_k)

    association_iou_threshold = association.get("iou_threshold")
    if association_iou_threshold is not None:
        params["association_iou_threshold"] = float(association_iou_threshold)

    association_score_weight = association.get("score_weight")
    if association_score_weight is not None:
        params["association_score_weight"] = float(association_score_weight)

    ai_skip_when_confident = ai.get("skip_when_confident")
    if ai_skip_when_confident is not None:
        params["ai_skip_when_confident"] = bool(ai_skip_when_confident)

    ai_skip_conf_threshold = ai.get("skip_conf_threshold")
    if ai_skip_conf_threshold is not None:
        params["ai_skip_conf_threshold"] = float(ai_skip_conf_threshold)

    ai_max_consecutive_skips = ai.get("max_consecutive_skips")
    if ai_max_consecutive_skips is not None:
        params["ai_max_consecutive_skips"] = int(ai_max_consecutive_skips)

    singer_skip_thr = ai.get("singer_skip_thr")
    if singer_skip_thr is not None:
        params["singer_skip_thr"] = float(singer_skip_thr)

    cv_stable_thr = ai.get("cv_stable_thr")
    if cv_stable_thr is not None:
        params["cv_stable_thr"] = float(cv_stable_thr)

    cv_conf_min = ai.get("cv_conf_min")
    if cv_conf_min is not None:
        params["cv_conf_min"] = float(cv_conf_min)

    search_scale_boost = ai.get("search_scale_boost")
    if search_scale_boost is not None:
        params["search_scale_boost"] = float(search_scale_boost)

    conf_refresh_low = ai.get("conf_refresh_low")
    if conf_refresh_low is not None:
        params["conf_refresh_low"] = float(conf_refresh_low)

    conf_refresh_high = ai.get("conf_refresh_high")
    if conf_refresh_high is not None:
        params["conf_refresh_high"] = float(conf_refresh_high)

    refresh_patience = ai.get("refresh_patience")
    if refresh_patience is not None:
        params["refresh_patience"] = int(refresh_patience)

    conf_streak_mean_max = ai.get("conf_streak_mean_max")
    if conf_streak_mean_max is not None:
        params["conf_streak_mean_max"] = float(conf_streak_mean_max)

    refresh_decline_thr = ai.get("refresh_decline_thr")
    if refresh_decline_thr is not None:
        params["refresh_decline_thr"] = float(refresh_decline_thr)

    refresh_min_bbox_area = ai.get("refresh_min_bbox_area")
    if refresh_min_bbox_area is not None:
        params["refresh_min_bbox_area"] = float(refresh_min_bbox_area)

    refresh_scale_vel_max = ai.get("refresh_scale_vel_max")
    if refresh_scale_vel_max is not None:
        params["refresh_scale_vel_max"] = float(refresh_scale_vel_max)

    # i11: Smart Cooldown — minimum interval between template re-inits.
    refresh_min_interval = ai.get("refresh_min_interval")
    if refresh_min_interval is not None:
        params["refresh_min_interval"] = int(refresh_min_interval)

    refresh_small_area_thr = ai.get("refresh_small_area_thr")
    if refresh_small_area_thr is not None:
        params["refresh_small_area_thr"] = float(refresh_small_area_thr)

    refresh_small_interval = ai.get("refresh_small_interval")
    if refresh_small_interval is not None:
        params["refresh_small_interval"] = int(refresh_small_interval)

    # i12: Great Rescue boyut barajı — parsed from ai: block
    rescue_min_area = ai.get("rescue_min_area")
    if rescue_min_area is not None:
        params["rescue_min_area"] = float(rescue_min_area)

    # F5: Closed-loop feedback — per-frame tracker.set_state(kf_output)
    f5_feedback_v = ai.get("f5_feedback")
    if f5_feedback_v is not None:
        params["f5_feedback"] = bool(f5_feedback_v)

    f5_scale_guard_v = ai.get("f5_scale_guard")
    if f5_scale_guard_v is not None:
        params["f5_scale_guard"] = float(f5_scale_guard_v)

    # F5-ObsSize: use AI obs size + KF position in set_state (avoids KF size-lag)
    f5_obs_size_v = ai.get("f5_obs_size")
    if f5_obs_size_v is not None:
        params["f5_obs_size"] = bool(f5_obs_size_v)

    # F5-CoastOnly: only fire f5 during coasting (reject_streak > 0)
    f5_coast_only_v = ai.get("f5_coast_only")
    if f5_coast_only_v is not None:
        params["f5_coast_only"] = bool(f5_coast_only_v)

    # N5: f5 rolling-reject gate
    f5_reject_window_v = ai.get("f5_reject_window")
    if f5_reject_window_v is not None:
        params["f5_reject_window"] = int(f5_reject_window_v)
    f5_reject_min_count_v = ai.get("f5_reject_min_count")
    if f5_reject_min_count_v is not None:
        params["f5_reject_min_count"] = int(f5_reject_min_count_v)

    # N7: f5 position-only — use KF position + last_good_bbox size
    f5_pos_only_v = ai.get("f5_pos_only")
    if f5_pos_only_v is not None:
        params["f5_pos_only"] = bool(f5_pos_only_v)

    # IMM physics params — extracted from nested imm: block in imm_tuned.yaml.
    imm_block = _as_mapping(cfg.get("imm"))
    q_scale_v = imm_block.get("q_scale")
    if q_scale_v is not None:
        params["q_scale"] = float(q_scale_v)
    mn_block = _as_mapping(imm_block.get("measurement_noise"))
    r_pos_scale_v = mn_block.get("r_pos_scale")
    if r_pos_scale_v is not None:
        params["r_pos_scale"] = float(r_pos_scale_v)
    r_size_scale_v = mn_block.get("r_size_scale")
    if r_size_scale_v is not None:
        params["r_size_scale"] = float(r_size_scale_v)
    tm_block = _as_mapping(imm_block.get("transition_matrix"))
    pi_persist_v = tm_block.get("pi_persist")
    if pi_persist_v is not None:
        params["pi_persist"] = float(pi_persist_v)

    # D1: maneuver detector sub-block
    md_block = _as_mapping(imm_block.get("maneuver_detector"))
    v = md_block.get("enabled")
    if v is not None:
        params["maneuver_pi_enabled"] = bool(v)
    v = md_block.get("chi2_threshold")
    if v is not None:
        params["maneuver_pi_thr"] = float(v)
    v = md_block.get("pi_persist")
    if v is not None:
        params["maneuver_pi_persist"] = float(v)
    v = md_block.get("pi_singer_boost")
    if v is not None:
        params["maneuver_pi_singer_boost"] = float(v)
    v = md_block.get("normal_pi_persist")
    if v is not None:
        params["normal_pi_persist"] = float(v)

    # D5: ReID rescue
    reid_block = _as_mapping(cfg.get("reid"))
    v = reid_block.get("enabled")
    if v is not None:
        params["reid_enabled"] = bool(v)
    v = reid_block.get("sim_threshold")
    if v is not None:
        params["reid_sim_threshold"] = float(v)
    v = reid_block.get("maxlen")
    if v is not None:
        params["reid_maxlen"] = int(v)

    # ROI CLAHE
    clahe_block = _as_mapping(cfg.get("clahe"))
    v = clahe_block.get("enabled")
    if v is not None:
        params["clahe_enabled"] = bool(v)
    v = clahe_block.get("clip_limit")
    if v is not None:
        params["clahe_clip_limit"] = float(v)
    v = clahe_block.get("roi_scale")
    if v is not None:
        params["clahe_roi_scale"] = float(v)
    v = clahe_block.get("tile_size")
    if v is not None:
        params["clahe_tile_size"] = int(v)

    # Dynamic Q from AR rate
    singer_block = _as_mapping(cfg.get("singer"))
    v = singer_block.get("ar_q_sensitivity")
    if v is not None:
        params["ar_q_sensitivity"] = float(v)
    v = singer_block.get("ar_q_boost_cap")
    if v is not None:
        params["ar_q_boost_cap"] = float(v)

    # Velocity direction gate
    vel_gate = _as_mapping(cfg.get("vel_gate"))
    v = vel_gate.get("min_speed")
    if v is not None:
        params["vel_gate_min_speed"] = float(v)
    v = vel_gate.get("cos_thr")
    if v is not None:
        params["vel_gate_cos_thr"] = float(v)

    # Velocity-relative innovation gate (vel_innov: block or top-level keys)
    vel_innov_block = _as_mapping(cfg.get("vel_innov"))
    v = _first_non_none(vel_innov_block.get("ratio_gate"), cfg.get("vel_innov_ratio_gate"))
    if v is not None:
        params["vel_innov_ratio_gate"] = float(v)
    v = _first_non_none(vel_innov_block.get("min_speed"), cfg.get("vel_innov_min_speed"))
    if v is not None:
        params["vel_innov_min_speed"] = float(v)
    v = _first_non_none(vel_innov_block.get("min_innov"), cfg.get("vel_innov_min_innov"))
    if v is not None:
        params["vel_innov_min_innov"] = float(v)
    v = cfg.get("accept_bbox_raw")
    if v is not None:
        params["accept_bbox_raw"] = bool(v)
    v = cfg.get("f5_obs_size")
    if v is not None:
        params["f5_obs_size"] = bool(v)
    v = cfg.get("f5_coast_only")
    if v is not None:
        params["f5_coast_only"] = bool(v)

    # ORU backfill — forward raw sub-dict; parsed by OruConfig.from_dict() at runtime
    oru_block = _as_mapping(cfg.get("oru"))
    if oru_block:
        params["oru_config"] = dict(oru_block)

    # Chaos trigger — forward raw sub-dict; parsed by ChaosConfig.from_dict() at runtime
    chaos_block = _as_mapping(cfg.get("chaos"))
    if chaos_block:
        params["chaos_config"] = dict(chaos_block)

    return params


def load_runtime_config(
    path: str | None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and normalize a runtime config file."""
    return normalize_runtime_config(load_yaml_config(path), defaults=defaults)
