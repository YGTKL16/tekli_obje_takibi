"""Shared IMM integration helpers.

This module implements a single-target protocol:

1. Predict with the motion model.
2. Steer the SGLATrack search window toward that prediction.
3. Observe one or more AI candidates in that guided region.
4. Judge whether the observation is physically consistent.
5. Update the filter if accepted, otherwise coast on prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tracker.matching import associate_detections_to_trackers


@dataclass
class IMMObservation:
    """Observed AI candidate(s) for the current frame."""

    bbox: np.ndarray | None
    confidence: float
    search_bbox: list[float]
    candidate_bboxes: np.ndarray
    candidate_scores: np.ndarray


@dataclass
class IMMPolicyStep:
    """One-step result after judging and applying the observation."""

    bbox: list[float]
    state: np.ndarray
    predicted_state: np.ndarray
    accepted_measurement: bool
    measurement_sane: bool
    should_coast: bool
    reject_streak: int
    last_good_bbox: list[float]
    innovation_norm: float = 0.0
    # Integration-audit telemetry (populated by step_guided_imm; default values
    # preserve backward compatibility for callers that construct the step
    # directly in tests).
    gate_decision: str = ""
    mahal_d2: float = 0.0
    alpha: float = 1.0
    # D1: Maneuver Detector — True when dynamic pi was injected this frame.
    maneuver_fired: bool = False


def _as_bbox_list(bbox) -> list[float]:
    bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(4)
    return bbox_arr.tolist()


def _mahalanobis_sq(
    innov: np.ndarray,
    P_arr: np.ndarray,
    r_pos: float,
    r_size: float,
) -> float:
    """Squared Mahalanobis distance for a 4D innovation: d² = vᵀ S⁻¹ v.

    S = P[:4,:4] + diag([r_pos, r_pos, r_size, r_size])
    Returns 0.0 on numerical failure (singular S).
    """
    S = P_arr[:4, :4].astype(np.float64) + np.diag(
        np.array([r_pos, r_pos, r_size, r_size], dtype=np.float64)
    )
    try:
        v = innov.astype(np.float64)
        return float(v @ np.linalg.solve(S, v))
    except np.linalg.LinAlgError:
        return 0.0


def _compute_update_alpha(
    conf: float,
    d2_mahal: float,
    k_conf: float,
    lambda_d2: float,
    chi2_max: float,
    singer_prob: float,
    maneuver_threshold: float,
    coast_threshold: float,
) -> float:
    """Compute soft update weight α ∈ [0, 1] for innovation blending.

    α = sigmoid(k·(conf − 0.5)) × exp(−λ·d²/χ²_max)
    α_eff = max(α, μ_Singer) when Singer dominates [escape hatch: prevents lag
    on genuine sharp manoeuvres even if conf is partial].

    Short-circuits to 1.0 when k≤0 AND λ≤0 — preserves binary behavior
    (backward-compatible default: alpha_gate_k_conf=0, alpha_gate_lambda=0).
    """
    if k_conf <= 0.0 and lambda_d2 <= 0.0:
        return 1.0  # disabled — full update, existing behavior

    # Sigmoid along the confidence axis
    sig = 1.0 / (1.0 + math.exp(-k_conf * (conf - 0.5))) if k_conf > 0.0 else 1.0

    # Mahalanobis suppression term
    mahal_term = math.exp(-lambda_d2 * d2_mahal / max(chi2_max, 1e-9)) if lambda_d2 > 0.0 else 1.0

    alpha = sig * mahal_term

    # Singer escape hatch: when Singer model is dominant and AI still sees
    # the target (conf > coast_threshold), floor alpha at Singer probability
    # so genuine sharp manoeuvres are not lag-penalised.
    if singer_prob > maneuver_threshold > 0.0 and conf > coast_threshold:
        alpha = max(alpha, singer_prob)

    return float(min(max(alpha, 0.0), 1.0))


def clamp_coast_velocity(predicted_state, max_speed: float) -> np.ndarray:
    """D2B: Return a copy of predicted_state with velocity clamped to max_speed px/frame.

    Only translational velocity (vx, vy) is clamped; size-rate (vw, vh) and
    acceleration components are untouched.  Clamping is per-axis (L-inf norm) to
    keep direction. Used to prevent the search window drifting arbitrarily far
    during coast when the KF velocity estimate grows unrealistically.
    """
    st = np.asarray(predicted_state, dtype=np.float32).flatten().copy()
    if st.shape[0] > 5:
        st[4] = float(np.clip(st[4], -max_speed, max_speed))
        st[5] = float(np.clip(st[5], -max_speed, max_speed))
    return st


def choose_search_bbox(
    predicted_state,
    *,
    mode: str = "baseline",
    last_output_bbox=None,
    singer_prob: float = 0.0,
    search_scale_boost: float = 0.0,
) -> list[float] | None:
    """Choose the search window that the AI should inspect next.

    When search_scale_boost > 0 and singer_prob is elevated (manoeuvre detected),
    the predicted bbox is expanded proportionally so the AI search window is wider
    and can catch sharp turns that carry the target outside the nominal search area.
    """
    pred = np.asarray(predicted_state, dtype=np.float32).flatten()
    pred_bbox = pred[:4].tolist()
    # Clamp dimensions to prevent NaN in search window
    pred_bbox[2] = max(pred_bbox[2], 1.0)
    pred_bbox[3] = max(pred_bbox[3], 1.0)

    if mode in ("open_loop", "ai_lead"):
        # Apply Singer-adaptive expansion even in ai_lead so the expanded
        # search window is set via tracker.set_state() before track() is called.
        if search_scale_boost > 0.0 and singer_prob > 0.0:
            scale = 1.0 + search_scale_boost * singer_prob
            cx = pred_bbox[0] + pred_bbox[2] * 0.5
            cy = pred_bbox[1] + pred_bbox[3] * 0.5
            nw = pred_bbox[2] * scale
            nh = pred_bbox[3] * scale
            return [cx - nw * 0.5, cy - nh * 0.5, nw, nh]
        return None
    if mode == "velocity_shift":
        return [
            float(pred[0] + pred[4]),
            float(pred[1] + pred[5]),
            max(float(pred[2]), 1.0),
            max(float(pred[3]), 1.0),
        ]
    if mode == "coast_only":
        if last_output_bbox is None:
            return None
        return _as_bbox_list(last_output_bbox)

    # baseline / default: use predicted position; optionally expand for manoeuvres
    if search_scale_boost > 0.0 and singer_prob > 0.0:
        scale = 1.0 + search_scale_boost * singer_prob
        cx = pred_bbox[0] + pred_bbox[2] * 0.5
        cy = pred_bbox[1] + pred_bbox[3] * 0.5
        nw = pred_bbox[2] * scale
        nh = pred_bbox[3] * scale
        return [cx - nw * 0.5, cy - nh * 0.5, nw, nh]
    return pred_bbox


def observe_with_guidance(
    tracker,
    frame_rgb,
    predicted_state,
    *,
    mode: str = "baseline",
    last_output_bbox=None,
    singer_prob: float = 0.0,
    search_scale_boost: float = 0.0,
) -> IMMObservation:
    """Observe candidates after optionally steering the AI search window."""
    search_bbox = choose_search_bbox(
        predicted_state,
        mode=mode,
        last_output_bbox=last_output_bbox,
        singer_prob=singer_prob,
        search_scale_boost=search_scale_boost,
    )
    if search_bbox is not None:
        tracker.set_state(search_bbox)

    # ai_lead: let AI track normally (track() updates internal state)
    if mode == "ai_lead":
        ai_bbox, ai_conf = tracker.track(frame_rgb)
        return IMMObservation(
            bbox=np.asarray(ai_bbox, dtype=np.float32),
            confidence=float(ai_conf),
            search_bbox=pred_bbox(predicted_state),
            candidate_bboxes=np.asarray(ai_bbox, dtype=np.float32).reshape(1, 4),
            candidate_scores=np.array([float(ai_conf)], dtype=np.float32),
        )

    if getattr(tracker, "association_enabled", False):
        candidate_bboxes, candidate_scores = tracker.track_candidates(
            frame_rgb,
            top_k=tracker.association_top_k,
        )
        if candidate_bboxes.shape[0] == 0:
            return IMMObservation(
                bbox=None,
                confidence=0.0,
                search_bbox=search_bbox or pred_bbox(predicted_state),
                candidate_bboxes=candidate_bboxes,
                candidate_scores=candidate_scores,
            )

        matches, _, _ = associate_detections_to_trackers(
            np.asarray(predicted_state[:4], dtype=np.float32).reshape(1, 4),
            candidate_bboxes,
            detection_scores=candidate_scores,
            iou_threshold=tracker.association_iou_threshold,
            score_weight=tracker.association_score_weight,
        )
        if matches.shape[0] == 0:
            return IMMObservation(
                bbox=None,
                confidence=0.0,
                search_bbox=search_bbox or pred_bbox(predicted_state),
                candidate_bboxes=candidate_bboxes,
                candidate_scores=candidate_scores,
            )

        det_idx = int(matches[0, 1])
        return IMMObservation(
            bbox=np.asarray(candidate_bboxes[det_idx], dtype=np.float32),
            confidence=float(candidate_scores[det_idx]),
            search_bbox=search_bbox or pred_bbox(predicted_state),
            candidate_bboxes=np.asarray(candidate_bboxes, dtype=np.float32),
            candidate_scores=np.asarray(candidate_scores, dtype=np.float32),
        )

    candidate_bboxes, candidate_scores = tracker.track_candidates(frame_rgb, top_k=1)
    if candidate_bboxes.shape[0] == 0:
        return IMMObservation(
            bbox=None,
            confidence=0.0,
            search_bbox=search_bbox or pred_bbox(predicted_state),
            candidate_bboxes=np.asarray(candidate_bboxes, dtype=np.float32),
            candidate_scores=np.asarray(candidate_scores, dtype=np.float32),
        )

    return IMMObservation(
        bbox=np.asarray(candidate_bboxes[0], dtype=np.float32),
        confidence=float(candidate_scores[0]),
        search_bbox=search_bbox or pred_bbox(predicted_state),
        candidate_bboxes=np.asarray(candidate_bboxes, dtype=np.float32),
        candidate_scores=np.asarray(candidate_scores, dtype=np.float32),
    )


def pred_bbox(predicted_state) -> list[float]:
    """Return the top-left bbox slice of a predicted KF state."""
    return np.asarray(predicted_state, dtype=np.float32).flatten()[:4].tolist()


def step_guided_imm(
    kf,
    decision,
    predicted_state,
    observation_bbox,
    confidence: float,
    frame_w: int,
    frame_h: int,
    *,
    is_tracking: bool,
    judge_reference_bbox=None,
    last_good_bbox=None,
    reject_streak: int = 0,
    reinit_after: int = 8,
    adaptive_r_enabled: bool = False,
    r_exponent: float = 1.0,
    conf_bypass_threshold: float = 0.15,
    innovation_threshold: float = 2.0,
    mahal_chi2_gate: float = 0.0,
    r_pos_base: float = 1.0,
    r_size_base: float = 10.0,
    maneuver_probability: float = 0.0,
    maneuver_threshold: float = 0.0,
    maneuver_bypass_boost: float = 0.0,
    mahal_chi2_threshold: float = 23.51,
    mahal_bypass_after: int = 2,
    coast_count: int = 0,
    alpha_gate_k_conf: float = 0.0,
    alpha_gate_lambda: float = 0.0,
    reacq_r_decay: float = 0.0,
    # D1: Maneuver Detector — dynamic pi injection
    maneuver_pi_enabled: bool = False,
    maneuver_pi_thr: float = 16.0,
    maneuver_pi_persist: float = 0.72,
    maneuver_pi_singer_boost: float = 0.20,
    normal_pi_persist: float = 0.96,
    # Velocity direction gate: reject 180° antipode detections (0 = disabled)
    vel_gate_min_speed: float = 0.0,   # min px/frame for KF velocity to arm gate
    vel_gate_cos_thr: float = 0.5,     # reject when cos(angle) < -thr (>120°)
    # Dynamic bypass: adapt mahal_bypass_after based on AI confidence + KF speed.
    # Discriminates between maneuver (high conf + fast → open gate quickly) and
    # ID-switch risk (low conf or stable → keep gate armored longer).
    # Set mahal_bypass_conf_thr > 0 or mahal_bypass_vel_thr > 0 to enable.
    # When disabled (both = 0), falls back to static mahal_bypass_after.
    mahal_bypass_conf_thr: float = 0.0,   # AI conf ≥ thr → fast bypass eligible; 0=disabled
    mahal_bypass_vel_thr: float = 0.0,    # KF speed ≥ thr (px/frame) → fast bypass eligible; 0=disabled
    mahal_bypass_after_fast: int = 2,     # bypass_after when conf+vel conditions both met
    mahal_bypass_after_slow: int = 5,     # bypass_after otherwise (ID-switch protection)
    # Velocity-relative innovation gate: blocks wrong-target ID switches when
    # the AI observation is implausibly far from the KF prediction relative to
    # the current KF velocity.  Ratio = center_innov_px / max(kf_speed, min_speed).
    # High ratio (>>1) signals the AI jumped to a stationary wrong target.
    # Set vel_innov_ratio_gate > 0 to enable; 0 = disabled (default).
    vel_innov_ratio_gate: float = 0.0,    # force coast when ratio > this; 0=disabled
    vel_innov_min_speed: float = 1.0,     # floor for kf_speed denominator (px/frame)
    vel_innov_min_innov: float = 0.0,     # minimum absolute center innov (px) to fire gate; 0=no min
    accept_bbox_raw: bool = False,        # output raw AI obs (not KF state) on accepted frames
) -> IMMPolicyStep:
    """Judge the observed bbox and produce the single final filter output."""
    predicted = np.asarray(predicted_state, dtype=np.float32).flatten()
    state = predicted
    bbox = pred_bbox(predicted)
    judge_bbox = pred_bbox(predicted) if judge_reference_bbox is None else _as_bbox_list(judge_reference_bbox)
    last_good = bbox if last_good_bbox is None else _as_bbox_list(last_good_bbox)

    # Dynamic bypass: select effective bypass_after based on scene conditions.
    # If both conf_thr=0 and vel_thr=0, fall back to static mahal_bypass_after.
    if mahal_bypass_conf_thr > 0.0 or mahal_bypass_vel_thr > 0.0:
        _speed = math.sqrt(float(predicted[4]) ** 2 + float(predicted[5]) ** 2)
        _cond_conf = confidence >= mahal_bypass_conf_thr if mahal_bypass_conf_thr > 0.0 else True
        _cond_vel  = _speed >= mahal_bypass_vel_thr  if mahal_bypass_vel_thr  > 0.0 else True
        _eff_bypass_after = mahal_bypass_after_fast if (_cond_conf and _cond_vel) else mahal_bypass_after_slow
    else:
        _eff_bypass_after = mahal_bypass_after  # static mode (backward compat)

    # ── Parse observation & compute innovation ───────────────────
    observation_arr = None
    innovation_norm = 0.0
    innov = np.zeros(4, dtype=np.float32)
    if observation_bbox is not None:
        observation_arr = np.asarray(observation_bbox, dtype=np.float32).reshape(4)
        innov = observation_arr - predicted[:4]
        diag = np.sqrt(max(predicted[2], 1.0) ** 2 + max(predicted[3], 1.0) ** 2)
        innovation_norm = float(np.linalg.norm(innov) / max(diag, 1.0))

    # ── Early Mahalanobis distance (shared by bypass guard + α-blend) ────
    # Computed once here via the C++ binding so bypass and normal path both
    # use the same d² without redundant solves.
    d2_raw = 0.0
    if (
        observation_arr is not None
        and kf is not None
        and hasattr(kf, "mahalanobis_sq")
        and kf.is_initialized()
    ):
        d2_raw = float(kf.mahalanobis_sq(observation_arr))

    # ── D1: Maneuver Detector — dynamic pi injection ──────────────
    # When Mahalanobis distance indicates a sudden maneuver (d² > thr),
    # temporarily lower pi_persist so Singer can dominate faster, then
    # restore the normal pi after the update.
    _maneuver_fired = False
    _pi_was_overridden = False
    if (
        maneuver_pi_enabled
        and kf is not None
        and hasattr(kf, "set_transition_matrix")
        and kf.is_initialized()
        and d2_raw > maneuver_pi_thr
        and observation_arr is not None
    ):
        # Build agile pi: reduced self-persistence, boosted Singer channel.
        # off_cv_ca = share of non-Singer probability allocated to CV and CA
        off_total = 1.0 - maneuver_pi_persist         # e.g. 0.28 when persist=0.72
        singer_off = maneuver_pi_singer_boost          # e.g. 0.20
        # other_off = full cross-term from CV↔CA (rows 0/1 each have one cross slot)
        other_off = max(off_total - singer_off, 0.0)  # e.g. 0.08 → row sum: 0.72+0.08+0.20=1.00
        agile_pi = np.array(
            [
                [maneuver_pi_persist, other_off, singer_off],
                [other_off, maneuver_pi_persist, singer_off],
                [other_off / 2.0, other_off / 2.0, 1.0 - other_off],
            ],
            dtype=np.float32,
        )
        kf.set_transition_matrix(agile_pi)
        _pi_was_overridden = True
        _maneuver_fired = True

    # ── High-confidence bypass: trust AI directly ────────────────
    # Bypass skips conf/IoU gates but STILL updates KF to keep state in sync
    # with the AI output.  Without this, the search window diverges from the
    # displayed bbox on every bypass frame.
    # Phase 4: lower bypass threshold when IMM manoeuvre mode is active.
    eff_bypass_thr = conf_bypass_threshold - (
        maneuver_bypass_boost
        if maneuver_bypass_boost > 0.0 and maneuver_probability > maneuver_threshold > 0.0
        else 0.0
    )
    # Bypass Mahalanobis guard: reject bypass when the AI measurement is
    # statistically inconsistent with the KF prediction (teleportation case).
    # Skip the guard when KF is uninitialised (d2_raw=0) or when reject_streak
    # already reached _eff_bypass_after — at that point KF has diverged and
    # blocking bypass would make the failure permanent.
    _bypass_d2_ok = (
        d2_raw < mahal_chi2_threshold
        or not (kf is not None and kf.is_initialized())
        or reject_streak >= _eff_bypass_after
    )
    if (
        eff_bypass_thr > 0
        and observation_arr is not None
        and confidence >= eff_bypass_thr
        and _bypass_d2_ok
        and bool(
            decision.is_measurement_sane(
                observation_arr.tolist(),
                confidence,
                frame_w,
                frame_h,
                prev_bbox=judge_bbox,
            )
        )
    ):
        ai_bbox = _as_bbox_list(observation_arr)
        if innovation_threshold > 0 and innovation_norm > innovation_threshold and reject_streak == 0:
            kf.set_gmc_failed(True)
        # R-inflation ramp: newly re-acquired target gets conservative R even
        # in bypass — coast streak inflates R so KF doesn't snap violently.
        r_factor_bypass = 1.0 + reacq_r_decay * min(float(coast_count), 30.0) if reacq_r_decay > 0.0 else 1.0
        eff_conf_bypass = confidence / math.sqrt(r_factor_bypass)
        # Update KF so state stays aligned with the AI output.
        # Phase 2: r_exponent scales confidence before adaptive-R → R ∝ 1/conf^n.
        if adaptive_r_enabled:
            bypass_state = np.array(kf.update(observation_arr, float(eff_conf_bypass ** r_exponent))).flatten()
        else:
            bypass_state = np.array(kf.update(observation_arr)).flatten()
        # D1: restore normal pi after bypass update
        if _pi_was_overridden:
            _off = (1.0 - normal_pi_persist) / 2.0
            _normal_pi = np.array(
                [[normal_pi_persist, _off, _off],
                 [_off, normal_pi_persist, _off],
                 [_off, _off, normal_pi_persist]],
                dtype=np.float32,
            )
            kf.set_transition_matrix(_normal_pi)
            _pi_was_overridden = False
        return IMMPolicyStep(
            bbox=ai_bbox,
            state=bypass_state,
            predicted_state=predicted,
            accepted_measurement=True,
            measurement_sane=True,
            should_coast=False,
            reject_streak=0,
            last_good_bbox=ai_bbox,
            innovation_norm=innovation_norm,
            gate_decision="bypass",
            mahal_d2=d2_raw,
            alpha=1.0,
            maneuver_fired=_maneuver_fired,
        )

    # ── Normal path: sanity + gating ─────────────────────────────
    measurement_sane = False
    mahal_confirmed = False   # True when chi² gate explicitly accepted the measurement
    sanity_reason = "no_obs" if observation_arr is None else "ok"

    # ── Velocity direction gate: reject 180° ID-switch candidates ──
    # If the observation implies a direction nearly opposite to the KF velocity,
    # it's almost certainly a different object. Only fires when KF has meaningful
    # velocity (> vel_gate_min_speed px/frame) and the implied displacement also
    # has sufficient magnitude. Does NOT fire at bypass level (handled above).
    if (
        vel_gate_min_speed > 0.0
        and observation_arr is not None
        and predicted[4] ** 2 + predicted[5] ** 2 > vel_gate_min_speed ** 2
    ):
        pred_vx, pred_vy = float(predicted[4]), float(predicted[5])
        pred_speed_sq = pred_vx ** 2 + pred_vy ** 2
        obs_cx = float(observation_arr[0]) + float(observation_arr[2]) * 0.5
        obs_cy = float(observation_arr[1]) + float(observation_arr[3]) * 0.5
        pred_cx = float(predicted[0]) + float(predicted[2]) * 0.5
        pred_cy = float(predicted[1]) + float(predicted[3]) * 0.5
        disp_x, disp_y = obs_cx - pred_cx, obs_cy - pred_cy
        disp_speed_sq = disp_x ** 2 + disp_y ** 2
        if disp_speed_sq > vel_gate_min_speed ** 2:
            dot = pred_vx * disp_x + pred_vy * disp_y
            cos_angle = dot / (math.sqrt(pred_speed_sq) * math.sqrt(disp_speed_sq))
            if cos_angle < -vel_gate_cos_thr:
                observation_arr = None   # force coast — antipode ID switch
                sanity_reason = "vel_gate"

    if observation_arr is None:
        sanity_reason = "no_obs"
    if observation_arr is not None:
        measurement_sane = bool(
            decision.is_measurement_sane(
                observation_arr.tolist(),
                confidence,
                frame_w,
                frame_h,
                prev_bbox=judge_bbox,
            )
        )
        if not measurement_sane:
            sanity_reason = "sanity"
        # Mahalanobis gate: reject physically impossible measurements
        # even when they pass geometric sanity (e.g. UAV sudden teleport).
        # BYPASS when reject_streak >= _eff_bypass_after: after N frames of
        # consecutive rejection the KF prediction has likely already diverged
        # from the target — continuing to gate makes the failure permanent.
        if measurement_sane and kf is not None and reject_streak < _eff_bypass_after:
            mahal_ok = bool(
                decision.is_measurement_mahalanobis_ok(
                    observation_arr,
                    kf,
                    chi2_threshold=mahal_chi2_threshold,
                )
            )
            measurement_sane = mahal_ok
            mahal_confirmed = mahal_ok   # chi² explicitly confirmed → skip IoU below
            if not mahal_ok:
                sanity_reason = "mahal"

    should_coast = bool(decision.should_coast(confidence))

    # ── Phase 3: Mahalanobis innovation gate ─────────────────────
    # If the chi²-weighted innovation exceeds the gate threshold, the
    # measurement is physically implausible → force coast this frame.
    # Bypass when reject_streak >= _eff_bypass_after (same logic as above).
    phase3_forced_coast = False
    if mahal_chi2_gate > 0.0 and observation_arr is not None and not should_coast \
            and reject_streak < _eff_bypass_after:
        P_arr = np.array(kf.get_covariance(), dtype=np.float64)
        d2 = _mahalanobis_sq(innov, P_arr, r_pos_base, r_size_base)
        if d2 > mahal_chi2_gate:
            should_coast = True
            phase3_forced_coast = True

    # ── Phase 3b: Velocity-relative innovation gate ───────────────
    # Blocks wrong-target ID switches when the AI observation is implausibly
    # far from the KF prediction relative to the current KF velocity.  A
    # stationary target (vx≈vy≈0) that suddenly appears 60+ px away is almost
    # certainly a different object.  Fast-moving targets (large KF speed) are
    # immune because the ratio stays small.  Only fires in the NORMAL path
    # (non-bypass), so high-confidence re-acquisitions are not blocked.
    vel_innov_fired = False
    if (vel_innov_ratio_gate > 0.0
            and observation_arr is not None
            and not should_coast
            and kf is not None
            and kf.is_initialized()
            and reject_streak < _eff_bypass_after):
        _obs_cx = float(observation_arr[0]) + float(observation_arr[2]) * 0.5
        _obs_cy = float(observation_arr[1]) + float(observation_arr[3]) * 0.5
        _pred_cx = float(predicted[0]) + float(predicted[2]) * 0.5
        _pred_cy = float(predicted[1]) + float(predicted[3]) * 0.5
        _center_innov_px = math.sqrt(
            (_obs_cx - _pred_cx) ** 2 + (_obs_cy - _pred_cy) ** 2
        )
        _kf_speed = math.sqrt(float(predicted[4]) ** 2 + float(predicted[5]) ** 2)
        _eff_speed = max(_kf_speed, vel_innov_min_speed)
        if _center_innov_px / _eff_speed > vel_innov_ratio_gate:
            if vel_innov_min_innov <= 0.0 or _center_innov_px >= vel_innov_min_innov:
                should_coast = True
                sanity_reason = "vel_innov_gate"
                vel_innov_fired = True

    accepted_measurement = bool(
        observation_arr is not None
        and (not should_coast)
        and is_tracking
        and measurement_sane
        and (
            # Mahalanobis confirmed statistical consistency → IoU gate is redundant
            # and harmful for fast-moving targets (Gull1, surfer, Surfing12).
            # When chi² explicitly accepted the measurement, trust it.
            mahal_confirmed
            or decision.should_update(confidence, observation_arr, predicted)
        )
    )

    next_reject_streak = reject_streak
    next_last_good = last_good
    alpha = 1.0
    reinit_fired = False

    if accepted_measurement:
        # ── α-Soft Innovation Blending ────────────────────────────────────
        # Blend the AI measurement toward the KF prediction using α ∈ [0,1].
        # α=1.0 (default when k_conf=0, lambda=0) → full update, binary behavior.
        # α<1.0 → soft blend → eliminates chatter at gate boundary.
        # Singer escape hatch inside _compute_update_alpha prevents lag on
        # genuine sharp manoeuvres.
        coast_thr = getattr(decision, "coast_threshold", 0.05)
        alpha = _compute_update_alpha(
            confidence, d2_raw,
            alpha_gate_k_conf, alpha_gate_lambda,
            mahal_chi2_threshold,
            maneuver_probability, maneuver_threshold, coast_thr,
        )
        z_soft = (predicted[:4] + alpha * innov).astype(np.float32)

        # ── Reacquisition R-Inflation Ramp ───────────────────────────────
        # After a coast streak the first accepted measurement is uncertain;
        # inflate R proportionally to coast length so KF doesn't snap hard.
        # reacq_r_decay=0 (default) → r_factor=1 → eff_conf=confidence.
        r_factor = 1.0 + reacq_r_decay * min(float(coast_count), 30.0) if reacq_r_decay > 0.0 else 1.0
        eff_conf = confidence / math.sqrt(r_factor)

        if adaptive_r_enabled:
            state = np.array(kf.update(z_soft, float(eff_conf ** r_exponent))).flatten()
        else:
            state = np.array(kf.update(z_soft)).flatten()
        # accept_bbox_raw: output the raw AI observation directly instead of the
        # Kalman-smoothed state.  KF is still updated (state used for prediction).
        # This eliminates size-smoothing lag when the target changes apparent size
        # rapidly (e.g. car approaching camera), without affecting coasting benefit.
        if accept_bbox_raw and observation_arr is not None:
            bbox = _as_bbox_list(observation_arr)
        else:
            bbox = pred_bbox(state)
        next_last_good = bbox
        next_reject_streak = 0
    else:
        next_reject_streak += 1
        if next_reject_streak >= reinit_after:
            # Re-init at the current AI position if available — tracking was
            # lost and KF has diverged; reiniting at last_good (old position)
            # would restart the same failure loop.
            reinit_pos = observation_arr if observation_arr is not None else np.asarray(last_good, dtype=np.float32)
            kf.init(np.asarray(reinit_pos, dtype=np.float32))
            state = np.array(kf.get_state()).flatten()
            bbox = pred_bbox(state)
            next_reject_streak = 0
            reinit_fired = True
        else:
            # AI-fallback output: when coasting (should_coast=True) or the
            # measurement passed sanity but was rejected for another reason,
            # prefer AI bbox over extrapolating KF prediction — supports F5
            # closed-loop feedback so the search window follows the detection.
            # SAFETY: do NOT use AI bbox when it was rejected by sanity/mahal
            # gate AND we are not already coasting — that would output a
            # physically implausible bbox (e.g. 400px teleport with conf=0.95).
            # KF state is NOT updated — gate decision stands.
            #
            # vel_innov_fired exception: the observation IS the wrong target,
            # so output the KF prediction to keep f5-feedback on the correct
            # last-known-good location.  Do not follow the wrong object.
            if observation_arr is not None and (should_coast or measurement_sane) and not vel_innov_fired:
                bbox = _as_bbox_list(observation_arr)
            else:
                bbox = pred_bbox(state)

    if any(np.isnan(v) for v in bbox) or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        fallback = pred_bbox(predicted)
        kf.init(np.asarray(fallback, dtype=np.float32))
        state = np.array(kf.get_state()).flatten()
        bbox = pred_bbox(state)
        next_last_good = bbox
        next_reject_streak = 0

    # ── Innovation-based Q-boost for next frame ──────────────────
    # Only fire when actively tracking (reject_streak == 0) to avoid
    # false Q-boost after coasting or prolonged measurement rejection.
    if innovation_threshold > 0 and innovation_norm > innovation_threshold and next_reject_streak == 0 and is_tracking:
        kf.set_gmc_failed(True)

    # D1: restore normal pi after all update paths
    if _pi_was_overridden and kf is not None and hasattr(kf, "set_transition_matrix"):
        _off = (1.0 - normal_pi_persist) / 2.0
        _normal_pi = np.array(
            [[normal_pi_persist, _off, _off],
             [_off, normal_pi_persist, _off],
             [_off, _off, normal_pi_persist]],
            dtype=np.float32,
        )
        kf.set_transition_matrix(_normal_pi)

    # ── Telemetry: classify which gate path this frame took ──────
    if reinit_fired:
        gate_decision = "reinit"
    elif accepted_measurement:
        gate_decision = "accept"
    elif observation_arr is None:
        gate_decision = "no_obs"
    elif phase3_forced_coast:
        gate_decision = "reject_mahal"
    elif should_coast:
        gate_decision = "coast"
    elif not is_tracking:
        gate_decision = "lost"
    elif sanity_reason == "sanity":
        gate_decision = "reject_sanity"
    elif sanity_reason == "mahal":
        gate_decision = "reject_mahal"
    else:
        gate_decision = "reject_iou"

    return IMMPolicyStep(
        bbox=bbox,
        state=state,
        predicted_state=predicted,
        accepted_measurement=accepted_measurement,
        measurement_sane=measurement_sane,
        should_coast=should_coast,
        reject_streak=next_reject_streak,
        last_good_bbox=next_last_good,
        innovation_norm=innovation_norm,
        gate_decision=gate_decision,
        mahal_d2=d2_raw,
        alpha=alpha,
        maneuver_fired=_maneuver_fired,
    )
