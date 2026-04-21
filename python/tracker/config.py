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
    "gmc_enabled": False,
    "gmc_n_features": 200,
    "gmc_inlier_ratio_threshold": 0.3,
    "gmc_min_matches": 6,
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

    gmc_inlier_ratio_threshold = gmc.get("inlier_ratio_threshold")
    if gmc_inlier_ratio_threshold is not None:
        params["gmc_inlier_ratio_threshold"] = float(gmc_inlier_ratio_threshold)

    gmc_min_matches = gmc.get("min_matches")
    if gmc_min_matches is not None:
        params["gmc_min_matches"] = int(gmc_min_matches)

    gmc_downsample = gmc.get("downsample")
    if gmc_downsample is not None:
        params["gmc_downsample"] = float(gmc_downsample)

    gmc_fail_q_boost = gmc.get("fail_q_boost")
    if gmc_fail_q_boost is not None:
        params["gmc_fail_q_boost"] = float(gmc_fail_q_boost)

    gmc_foreground_dilate_factor = gmc.get("foreground_dilate_factor")
    if gmc_foreground_dilate_factor is not None:
        params["gmc_foreground_dilate_factor"] = float(gmc_foreground_dilate_factor)

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

    return params


def load_runtime_config(
    path: str | None,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and normalize a runtime config file."""
    return normalize_runtime_config(load_yaml_config(path), defaults=defaults)
