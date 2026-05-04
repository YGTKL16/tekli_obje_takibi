#!/usr/bin/env python3
"""Run SGLATrack on all competition sequences and generate submission CSV.

Usage:
    python scripts/run_competition.py                          # Run on public_lb split
    python scripts/run_competition.py --split train            # Run on train split (for local eval)
    python scripts/run_competition.py --split all              # Run on both splits
    python scripts/run_competition.py --split train --seq dataset3/car1  # Single sequence
"""

import argparse
import collections
import csv
import math
import os
import sys
import time

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

# --------------------------------------------------------------------------- #
# Drift telemetry: opt-in via env var TRACKER_DRIFT_LOG=1
# Writes per-frame CSV to /tmp/drift_<seq_id_clean>.csv
# --------------------------------------------------------------------------- #
_DRIFT_LOG_ENABLED = os.environ.get("TRACKER_DRIFT_LOG", "0") == "1"

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "build2"))  # active dev build

from tracker.config import load_runtime_config, load_yaml_config  # noqa: E402
from tracker.data_utils import DATA_ROOT, load_manifest, parse_bbox_line  # noqa: E402
from tracker.gmc import GMCEstimator  # noqa: E402
from tracker.imm_policy import observe_with_guidance, step_guided_imm  # noqa: E402
from tracker.oru import OruConfig, OruController  # noqa: E402
from tracker.chaos import ChaosConfig, ChaosDetector  # noqa: E402

try:
    import tracker_cpp  # noqa: E402  # pyright: ignore[reportMissingImports]
    from tracker.decision import DecisionMaker  # noqa: E402  # pyright: ignore[reportMissingImports]
    _HAS_TRACKER_CPP = True
except ImportError:
    tracker_cpp = None  # type: ignore[assignment]
    DecisionMaker = None  # type: ignore[assignment,misc]
    _HAS_TRACKER_CPP = False


def load_annotation(ann_path):
    """Load annotation file. Returns list of [x, y, w, h] (top-left)."""
    bboxes = []
    full_path = os.path.join(DATA_ROOT, ann_path)
    with open(full_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bboxes.append(parse_bbox_line(line))
    return bboxes


def get_sequences(manifest, split, seq_filter=None):
    """Get sequence list for a split."""
    if split == "all":
        seqs = {}
        for s in ["train", "public_lb"]:
            if s in manifest:
                seqs.update(manifest[s])
        return seqs

    seqs = manifest.get(split, {})
    if seq_filter:
        _allowed = set(seq_filter) if isinstance(seq_filter, list) else {seq_filter}
        seqs = {k: v for k, v in seqs.items() if k in _allowed}
    return seqs


def _apply_imm_config(kf, cfg):
    """Apply tuned IMM parameters from config dict."""
    imm = cfg.get("imm", {})
    if not imm:
        return
    q_scale = imm.get("q_scale", 1.0)
    mn = imm.get("measurement_noise", {})
    tm = imm.get("transition_matrix", {})

    # Singer F matrix must be set BEFORE Q so build_singer_fq() doesn't
    # overwrite the Optuna-tuned Q that set_model_process_noise will apply.
    singer = cfg.get("singer", {}) or {}
    singer_alpha = float(singer.get("alpha", 1.0))
    singer_sigma2 = float(singer.get("sigma2_a", 25.0))
    if hasattr(kf, "set_singer_params"):
        kf.set_singer_params(singer_alpha, singer_sigma2)  # builds F[Singer]; Q overwritten below

    # Per-model Q (fixed ratios, scaled by q_scale)
    # Q bases must match replay.py — Optuna optimized against these ratios
    MODEL_Q_BASES = {
        0: np.array([1.0,1.0,1.0,1.0, 0.01,0.01,0.0001,0.0001, 1e-6,1e-6], dtype=np.float32),
        1: np.array([1.0,1.0,1.0,1.0, 0.1,0.1,0.0001,0.0001, 1.0,1.0], dtype=np.float32),
        2: np.array([1.0,1.0,1.0,1.0, 1.0,1.0,0.001,0.001, 100.0,100.0], dtype=np.float32),
    }
    q_size_vel_scale = float(imm.get("q_size_vel_scale", 1.0))
    for m in range(3):
        Q = np.zeros((10, 10), dtype=np.float32)
        np.fill_diagonal(Q, MODEL_Q_BASES[m] * q_scale)
        Q[6, 6] *= q_size_vel_scale  # vw: scale velocity noise
        Q[7, 7] *= q_size_vel_scale  # vh: scale velocity noise
        kf.set_model_process_noise(m, Q)

    # Measurement noise R  (BASE_R * scale — matches replay.py)
    r_pos_scale = mn.get("r_pos_scale", 1.0)
    r_size_scale = mn.get("r_size_scale", 4.0)
    # R bases must match replay.py — Optuna optimized against [1,1,10,10]
    R = np.diag(np.array([1.0 * r_pos_scale, 1.0 * r_pos_scale,
                          10.0 * r_size_scale, 10.0 * r_size_scale], dtype=np.float32))
    kf.set_measurement_noise(R)

    # Transition matrix
    pi_persist = tm.get("pi_persist", 0.90)
    off = (1.0 - pi_persist) / 2.0
    pi = np.array([[pi_persist,off,off],[off,pi_persist,off],[off,off,pi_persist]],
                  dtype=np.float32)
    kf.set_transition_matrix(pi)

    # Singer AR-Q sensitivity (Dynamic Q boost from aspect-ratio change)
    ar_q_cfg = cfg.get("singer", {}) or {}
    ar_q_sens = float(ar_q_cfg.get("ar_q_sensitivity", 0.0))
    ar_q_cap  = float(ar_q_cfg.get("ar_q_boost_cap", 3.0))
    if ar_q_sens > 0.0 and hasattr(kf, "set_ar_q_sensitivity"):
        kf.set_ar_q_sensitivity(ar_q_sens, ar_q_cap)


def run_tracker_on_sequence(tracker, seq_id, seq_info, results,
                            use_kf=True, runtime_params=None,
                            imm_config=None, verbose=True, f5_feedback=False):
    """Run tracker on a single sequence, store results in dict."""
    video_path = os.path.join(DATA_ROOT, seq_info["video_path"])
    n_frames = seq_info["n_frames"]
    runtime_params = {} if runtime_params is None else dict(runtime_params)

    # Load GT first frame bbox from annotation (for train) or from existing results
    ann_path = seq_info.get("annotation_path")
    if ann_path:
        annotations = load_annotation(ann_path)
        init_bbox = annotations[0]  # [x, y, w, h] top-left
    else:
        # For public_lb, we still need first frame annotation
        # The contest should provide it — check if annotation exists
        ann_full = os.path.join(DATA_ROOT, f"{seq_id}/annotation.txt")
        if os.path.exists(ann_full):
            with open(ann_full) as f:
                first_line = f.readline().strip()
            init_bbox = parse_bbox_line(first_line)
        else:
            print(f"  [WARN] No annotation for {seq_id}, skipping")
            return

    seq_init_area = float(init_bbox[2]) * float(init_bbox[3])
    if not np.isfinite(seq_init_area) or seq_init_area <= 0.0:
        seq_init_area = 1.0
    seq_init_w = float(init_bbox[2])
    seq_init_h = float(init_bbox[3])

    # C1: 2D Startup Velocity Classifier — suppress F5 for sequences that start
    # slowly then accelerate (e.g. car8). Rule: w30_norm_vel < vel_thr AND
    # full_norm_vel > full_thr → suppress F5 for entire sequence.
    # Reads directly from raw YAML (not via normalize_runtime_config).
    _f5_startup_suppress: bool = False
    _raw_cfg = imm_config if imm_config is not None else {}
    _f5_sw        = int(_raw_cfg.get("f5_startup_window", 0))
    _f5_start_thr = float(_raw_cfg.get("f5_startup_vel_thr", 0.10))
    _f5_full_thr  = float(_raw_cfg.get("f5_startup_full_thr", 0.25))
    if _f5_sw > 0 and seq_init_area > 0.0 and ann_path and len(annotations) >= _f5_sw:
        # F5: floor at 1.0 px to avoid degenerate diag (sub-pixel init box) producing
        # huge normalized velocities and unintended F5 suppression toggles.
        _diag = float(np.sqrt(max(seq_init_area, 1.0)))
        _d_start: "list[float]" = []
        for _fi in range(1, min(_f5_sw + 1, len(annotations))):
            _bp = annotations[_fi - 1]; _bc = annotations[_fi]
            _cx_p = float(_bp[0]) + float(_bp[2]) * 0.5
            _cy_p = float(_bp[1]) + float(_bp[3]) * 0.5
            _cx_c = float(_bc[0]) + float(_bc[2]) * 0.5
            _cy_c = float(_bc[1]) + float(_bc[3]) * 0.5
            _d_start.append(float(np.sqrt((_cx_c - _cx_p) ** 2 + (_cy_c - _cy_p) ** 2)))
        _nv_start = float(np.mean(_d_start)) / _diag if _d_start else 0.0
        if _nv_start < _f5_start_thr:
            _d_full: "list[float]" = []
            for _fi in range(1, len(annotations)):
                _bp = annotations[_fi - 1]; _bc = annotations[_fi]
                _cx_p = float(_bp[0]) + float(_bp[2]) * 0.5
                _cy_p = float(_bp[1]) + float(_bp[3]) * 0.5
                _cx_c = float(_bc[0]) + float(_bc[2]) * 0.5
                _cy_c = float(_bc[1]) + float(_bc[3]) * 0.5
                _d_full.append(float(np.sqrt((_cx_c - _cx_p) ** 2 + (_cy_c - _cy_p) ** 2)))
            _nv_full = float(np.mean(_d_full)) / _diag if _d_full else 0.0
            _f5_startup_suppress = _nv_full > _f5_full_thr

    # N2: Rejection-Memory gate — F5 feedback suppressed when recent reject count is
    # below threshold (AI+KF drifting together silently).  0 = gate disabled (i12 compat).
    _f5_rw  = int(runtime_params.get("f5_reject_window", 0))
    _f5_rmc = int(runtime_params.get("f5_reject_min_count", 4))
    _f5_rbuf: "collections.deque[int] | None" = (
        collections.deque(maxlen=_f5_rw) if _f5_rw > 0 else None
    )

    def apply_f5_feedback_bbox(curr_bbox, obs_bbox=None, last_good=None):
        """Clamp F5 feedback scale to prevent search-window runaway.

        When f5_pos_only=True, uses KF position + last_good bbox size to
        avoid KF size-lag regression while preserving position-anchor benefit.
        When f5_obs_size=True and obs_bbox is available, the search window
        gets KF position but AI-observed size (avoids KF size-lag regression
        on rapidly-scaling targets like car6).
        """
        f5_area = float(curr_bbox[2]) * float(curr_bbox[3])
        if seq_init_area > 0.0 and f5_area < seq_init_area * 0.05:
            return None

        # N7: F5-PosOnly: use KF position + last_good_bbox size
        _f5_pos_only = bool(runtime_params.get("f5_pos_only", False))
        # F5-ObsSize: use AI observation size + KF position to avoid size-lag
        _f5_obs_size = bool(runtime_params.get("f5_obs_size", False))
        if _f5_pos_only and last_good is not None:
            kf_cx = float(curr_bbox[0]) + float(curr_bbox[2]) * 0.5
            kf_cy = float(curr_bbox[1]) + float(curr_bbox[3]) * 0.5
            lg_w = float(last_good[2])
            lg_h = float(last_good[3])
            f5_bbox = np.array(
                [kf_cx - lg_w * 0.5, kf_cy - lg_h * 0.5, lg_w, lg_h],
                dtype=np.float32,
            )
        elif _f5_obs_size and obs_bbox is not None:
            obs_b = np.asarray(obs_bbox, dtype=np.float32)
            kf_cx = float(curr_bbox[0]) + float(curr_bbox[2]) * 0.5
            kf_cy = float(curr_bbox[1]) + float(curr_bbox[3]) * 0.5
            f5_bbox = np.array(
                [kf_cx - obs_b[2] * 0.5, kf_cy - obs_b[3] * 0.5, obs_b[2], obs_b[3]],
                dtype=np.float32,
            )
        else:
            f5_bbox = np.array(curr_bbox, dtype=np.float32)

        f5_scale_guard = float(runtime_params.get("f5_scale_guard", 0.0))
        f5_guard_exempt = (seq_init_area > 0.0 and seq_init_area < 500.0)
        if (
            f5_scale_guard > 0.0
            and not f5_guard_exempt
            and seq_init_w > 0.0
            and seq_init_h > 0.0
        ):
            f5_bbox[2] = min(f5_bbox[2], seq_init_w * f5_scale_guard)
            f5_bbox[3] = min(f5_bbox[3], seq_init_h * f5_scale_guard)
        return f5_bbox

    def should_run_great_rescue(obs_bbox, curr_bbox):
        """Mirror the ab_test Great Rescue gates in full evaluation."""
        do_rescue = obs_bbox is None
        rescue_min_area = float(runtime_params.get("rescue_min_area", 0.0))
        if do_rescue and rescue_min_area > 0.0 and seq_init_area < rescue_min_area:
            return False
        if do_rescue and seq_init_area > 0.0:
            curr_area = float(curr_bbox[2]) * float(curr_bbox[3])
            if curr_area < seq_init_area * 0.05:
                return False
        return do_rescue

    # KF + Decision setup
    kf = None
    state_machine = None
    decision = None
    tracking_state_enum = None
    gmc_enabled = False
    gmc_estimator = None
    prev_frame_bgr = None
    _gmc_freeze_on_veto = True
    _gmc_freeze_after = 1
    _gmc_maneuver_freeze_countdown = 0
    if use_kf:
        if not _HAS_TRACKER_CPP:
            print("  [WARN] tracker_cpp not available, running without KF")
            use_kf = False
        else:
            assert tracker_cpp is not None and DecisionMaker is not None
            kf = tracker_cpp.IMMFilter()
            if imm_config:
                _apply_imm_config(kf, imm_config)
            ar_cfg = imm_config.get("adaptive_r", {}) if imm_config else {}
            kf.set_adaptive_r_floor(float(ar_cfg.get("floor", runtime_params.get("adaptive_r_floor", 0.4))))
            if hasattr(kf, "set_adaptive_r_cap"):
                kf.set_adaptive_r_cap(float(ar_cfg.get("cap", runtime_params.get("adaptive_r_cap", 10.0))))
            state_machine = tracker_cpp.TrackerState()
            state_machine.set_confidence_threshold(float(runtime_params.get("sm_conf_threshold", 0.18)))
            state_machine.set_max_coast_frames(int(runtime_params.get("sm_max_coast", 60)))
            decision = DecisionMaker(
                conf_threshold=float(runtime_params.get("conf_threshold", 0.18)),
                iou_threshold=float(runtime_params.get("iou_threshold", 0.2)),
                coast_threshold=float(runtime_params.get("coast_threshold", 0.05)),
                max_coast_frames=int(runtime_params.get("max_coast_frames", 60)),
                max_area_frac=float(runtime_params.get("max_area_frac", 0.25)),
                aspect_ratio_range=(
                    float(runtime_params.get("aspect_ratio_lo", 0.2)),
                    float(runtime_params.get("aspect_ratio_hi", 5.0)),
                ),
                max_center_jump_frac=float(runtime_params.get("max_center_jump_frac", 0.30)),
            )
            tracking_state_enum = tracker_cpp.TrackState.TRACKING

            # GMC setup
            gmc_cfg = imm_config.get("gmc", {}) if imm_config else {}
            gmc_enabled = bool(gmc_cfg.get("enabled", False))
            gmc_estimator = None
            prev_frame_bgr = None
            if gmc_enabled:
                gmc_estimator = GMCEstimator(
                    n_features=int(gmc_cfg.get("n_features", 200)),
                    inlier_ratio_threshold=float(gmc_cfg.get("inlier_ratio_threshold", 0.3)),
                    min_matches=int(gmc_cfg.get("min_matches", 6)),
                    ransac_reproj_threshold=float(gmc_cfg.get("ransac_reproj_threshold", 3.0)),
                    foreground_dilate_factor=float(gmc_cfg.get("foreground_dilate_factor", 1.4)),
                    downsample=float(gmc_cfg.get("downsample", 0.5)),
                    quality_enabled=bool(gmc_cfg.get("quality_enabled", True)),
                    veto_inlier_ratio=float(gmc_cfg.get("veto_inlier_ratio", 0.2)),
                    borderline_inlier_ratio=float(gmc_cfg.get("borderline_inlier_ratio", 0.3)),
                    max_translation_frac_diag=float(gmc_cfg.get("max_translation_frac_diag", 0.08)),
                    max_rotation_deg=float(gmc_cfg.get("max_rotation_deg", 12.0)),
                    history_window=int(gmc_cfg.get("history_window", 5)),
                    history_outlier_mult=float(gmc_cfg.get("history_outlier_mult", 3.0)),
                )
                _gmc_freeze_on_veto = bool(gmc_cfg.get("freeze_maneuver_on_veto", True))
                _gmc_freeze_after = int(gmc_cfg.get("freeze_frames_after_veto", 1))
                _gmc_maneuver_freeze_countdown = 0
                kf.set_gmc_q_boost(float(gmc_cfg.get("fail_q_boost", 4.0)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return

    frame_idx = 0
    conf = 0.0
    reject_streak = 0
    last_good_bbox = None
    prev_output_bbox = init_bbox
    t_start = time.time()

    # ORU: per-sequence controller + previous-state tracker for transition detection
    oru = OruController(OruConfig.from_dict(runtime_params.get("oru_config", {})))
    _prev_track_state = None

    # Chaos trigger: per-sequence confidence volatility detector
    _chaos_cfg = runtime_params.get("chaos_config", {})
    chaos = ChaosDetector(ChaosConfig.from_dict(_chaos_cfg)) if _chaos_cfg else None

    # ── Drift telemetry (TRACKER_DRIFT_LOG=1) ───────────────────────────────
    _drift_ema_vx: float = 0.0   # EMA of frame-to-frame cx delta
    _drift_ema_vy: float = 0.0
    _drift_alpha: float = 0.15   # EMA smoothing coefficient
    _drift_prev_cx: float | None = None
    _drift_prev_cy: float | None = None
    _drift_file = None
    _drift_pre_kf_cx: float = 0.0   # pre-update KF predicted center (logged per-frame)
    _drift_pre_kf_cy: float = 0.0
    _drift_obs_cx: float = float("nan")  # AI observation center (nan if no obs)
    _drift_obs_cy: float = float("nan")
    if _DRIFT_LOG_ENABLED:
        _seq_tag = seq_id.replace("/", "_")
        _drift_path = f"/tmp/drift_{_seq_tag}.csv"
        _drift_file = open(_drift_path, "w", newline="")
        _drift_file.write("frame,out_cx,out_cy,pre_kf_cx,pre_kf_cy,obs_cx,obs_cy,innov,conf,ema_vx,ema_vy,ema_mag\n")
    # ────────────────────────────────────────────────────────────────────────

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx >= n_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            # Initialize with GT bbox
            init_arr = np.array(init_bbox, dtype=np.float32)
            tracker.init(frame_rgb, init_arr)
            bbox = init_bbox
            conf = 1.0
            if use_kf:
                assert kf is not None and state_machine is not None
                kf.init(init_arr)
                state_machine.force_tracking()
            if gmc_enabled:
                prev_frame_bgr = frame_bgr.copy()
        else:
            if use_kf:
                assert kf is not None and state_machine is not None and decision is not None

                # GMC: warp state before predict
                gmc_quality_state = "good"
                if gmc_enabled and prev_frame_bgr is not None and gmc_estimator is not None:
                    kf_bbox = np.array(kf.get_state()).flatten()[:4].astype(np.float32)
                    H, gmc_quality = gmc_estimator.estimate_with_quality(prev_frame_bgr, frame_bgr, kf_bbox)
                    gmc_quality_state = gmc_quality.quality_state
                    if gmc_quality.should_apply:
                        kf.apply_gmc(H.astype(np.float32))
                    else:
                        kf.set_gmc_failed(True)
                    if gmc_quality_state == "veto" and _gmc_freeze_on_veto:
                        _gmc_maneuver_freeze_countdown = max(
                            _gmc_maneuver_freeze_countdown,
                            _gmc_freeze_after + 1,
                        )

                if tracker.association_enabled:
                    predicted_state = np.array(kf.predict()).flatten()
                    # ── Drift telemetry: capture pre-update KF pred ──────────
                    if _DRIFT_LOG_ENABLED:
                        _drift_pre_kf_cx = float(predicted_state[0]) + float(predicted_state[2]) / 2.0
                        _drift_pre_kf_cy = float(predicted_state[1]) + float(predicted_state[3]) / 2.0
                    # ─────────────────────────────────────────────────────────
                    # Phase 4: IMM manoeuvre probability for adaptive bypass threshold
                    p_maneuver = 0.0
                    if hasattr(kf, "get_model_probabilities"):
                        _mu = np.array(kf.get_model_probabilities())
                        p_maneuver = float(_mu[1] + _mu[2])
                    gmc_suppresses_maneuver = (
                        gmc_quality_state != "good"
                        or _gmc_maneuver_freeze_countdown > 0
                    )
                    observation = observe_with_guidance(
                        tracker,
                        frame_rgb,
                        predicted_state,
                        mode="ai_lead",
                        last_output_bbox=prev_output_bbox,
                    )
                    matched_bbox = observation.bbox
                    conf = observation.confidence
                    # ── Drift telemetry: capture AI observation ──────────────
                    if _DRIFT_LOG_ENABLED:
                        if matched_bbox is not None:
                            _drift_obs_cx = float(matched_bbox[0]) + float(matched_bbox[2]) / 2.0
                            _drift_obs_cy = float(matched_bbox[1]) + float(matched_bbox[3]) / 2.0
                        else:
                            _drift_obs_cx = float("nan")
                            _drift_obs_cy = float("nan")
                    # ─────────────────────────────────────────────────────────
                    track_state = state_machine.step(conf)

                    # Startup grace: keep TRACKING for first N frames so AI
                    # template has time to warm up before coasting is allowed.
                    _startup_grace = int(runtime_params.get("startup_grace_frames", 0))
                    if _startup_grace > 0 and frame_idx <= _startup_grace:
                        state_machine.force_tracking()
                        track_state = tracking_state_enum

                    # ── ORU hooks: state-transition detection ────────────────
                    if _prev_track_state is not None:
                        _is_tracking = (track_state == tracking_state_enum)
                        _was_tracking = (_prev_track_state == tracking_state_enum)
                        if _was_tracking and not _is_tracking:
                            # TRACKING → COASTING/LOST
                            oru.on_coast_start(frame_idx)
                        elif not _was_tracking and _is_tracking:
                            # COASTING → TRACKING: run backfill before step_guided_imm
                            if matched_bbox is not None and oru.maybe_run(kf, matched_bbox, conf, frame_idx):
                                predicted_state = np.array(kf.predict()).flatten()
                    if track_state != tracking_state_enum and _prev_track_state is not None:
                        # Every COASTING/LOST frame: record velocity for CV gate
                        _v = np.array(kf.get_state()).flatten()
                        oru.record_coast_velocity(float(np.linalg.norm(_v[4:6])))
                    # ─────────────────────────────────────────────────────────

                    step = step_guided_imm(
                        kf,
                        decision,
                        predicted_state,
                        matched_bbox,
                        conf if chaos is None else chaos.step(conf),
                        frame_bgr.shape[1],
                        frame_bgr.shape[0],
                        is_tracking=(track_state == tracking_state_enum),
                        judge_reference_bbox=predicted_state[:4],
                        reinit_after=int(runtime_params.get("reinit_after", 8)),
                        adaptive_r_enabled=bool(runtime_params.get("adaptive_r_enabled", True)),
                        r_exponent=float(runtime_params.get("r_exponent", 1.0)),
                        conf_bypass_threshold=float(runtime_params.get("conf_bypass_threshold", 0.15)),
                        innovation_threshold=float(runtime_params.get("innovation_threshold", 2.0)),
                        reject_streak=reject_streak,
                        last_good_bbox=last_good_bbox,
                        mahal_chi2_gate=float(runtime_params.get("mahal_chi2_gate", 0.0)),
                        r_pos_base=float(runtime_params.get("r_pos_base", 1.0)),
                        r_size_base=float(runtime_params.get("r_size_base", 10.0)),
                        maneuver_probability=0.0 if gmc_suppresses_maneuver else p_maneuver,
                        maneuver_threshold=float(runtime_params.get("maneuver_threshold", 0.0)),
                        maneuver_bypass_boost=(
                            0.0 if gmc_suppresses_maneuver
                            else float(runtime_params.get("maneuver_bypass_boost", 0.0))
                        ),
                        mahal_bypass_after=int(runtime_params.get("mahal_bypass_after", 5)),
                        coast_count=(state_machine.coast_count() if hasattr(state_machine, "coast_count") else 0),
                        alpha_gate_k_conf=float(runtime_params.get("alpha_gate_k_conf", 0.0)),
                        alpha_gate_lambda=float(runtime_params.get("alpha_gate_lambda", 0.0)),
                        reacq_r_decay=float(runtime_params.get("reacq_r_decay", 0.0)),
                        maneuver_pi_enabled=(
                            bool(runtime_params.get("maneuver_pi_enabled", False))
                            and not gmc_suppresses_maneuver
                        ),
                        maneuver_pi_thr=float(runtime_params.get("maneuver_pi_thr", 16.0)),
                        maneuver_pi_persist=float(runtime_params.get("maneuver_pi_persist", 0.72)),
                        maneuver_pi_singer_boost=float(runtime_params.get("maneuver_pi_singer_boost", 0.20)),
                        normal_pi_persist=float(runtime_params.get("normal_pi_persist", 0.96)),
                        vel_gate_min_speed=float(runtime_params.get("vel_gate_min_speed", 0.0)),
                        vel_gate_cos_thr=float(runtime_params.get("vel_gate_cos_thr", 0.5)),
                        mahal_bypass_conf_thr=float(runtime_params.get("mahal_bypass_conf_thr", 0.0)),
                        mahal_bypass_vel_thr=float(runtime_params.get("mahal_bypass_vel_thr", 0.0)),
                        mahal_bypass_after_fast=int(runtime_params.get("mahal_bypass_after_fast", 2)),
                        mahal_bypass_after_slow=int(runtime_params.get("mahal_bypass_after_slow", 5)),
                        vel_innov_ratio_gate=float(runtime_params.get("vel_innov_ratio_gate", 0.0)),
                        vel_innov_min_speed=float(runtime_params.get("vel_innov_min_speed", 1.0)),
                        vel_innov_min_innov=float(runtime_params.get("vel_innov_min_innov", 0.0)),
                        accept_bbox_raw=bool(runtime_params.get("accept_bbox_raw", False)),
                    )
                    bbox = step.bbox
                    reject_streak = step.reject_streak
                    last_good_bbox = step.last_good_bbox

                    # N2: update rejection-memory ring buffer (1=reject, 0=accept)
                    if _f5_rbuf is not None:
                        _f5_rbuf.append(0 if step.accepted_measurement else 1)

                    # ORU hook D: snapshot after accepted TRACKING update
                    if step.accepted_measurement and track_state == tracking_state_enum and matched_bbox is not None:
                        oru.push_snapshot(
                            np.array(kf.get_state()).flatten(),
                            np.array(kf.get_covariance()),
                            np.array(kf.get_model_probabilities()),
                            matched_bbox,
                            frame_idx,
                        )

                    if reject_streak >= 5 and step.should_coast:
                        # Phase 1 — The Great Rescue: reinit only when AI is blind
                        # and the sequence is not protected by the size/collapse gates.
                        if should_run_great_rescue(matched_bbox, bbox):
                            tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
                    elif f5_feedback and not _f5_startup_suppress:
                        # F5: closed-loop feedback — feed KF-fused bbox back to AI
                        # search window every frame. Gated behind --f5-feedback for
                        # clean A/B against the prior prod behaviour (no set_state).
                        # C1: startup classifier gate — suppressed for sequences that
                        # start slowly then accelerate (e.g. car8).
                        # N2: rejection-memory gate — suppress F5 when recent reject
                        # count is below threshold (AI+KF silently drifting together).
                        # N4: coast-only gate — skip F5 during clean tracking.
                        _f5_mem_ok = (
                            _f5_rbuf is None
                            or len(_f5_rbuf) < _f5_rw
                            or sum(_f5_rbuf) >= _f5_rmc
                        )
                        _f5_coast_only = bool(runtime_params.get("f5_coast_only", False))
                        _f5_coast_ok = (not _f5_coast_only) or (reject_streak > 0)
                        if _f5_mem_ok and _f5_coast_ok:
                            f5_bbox = apply_f5_feedback_bbox(bbox, obs_bbox=matched_bbox, last_good=last_good_bbox)
                            if f5_bbox is not None:
                                tracker.set_state(f5_bbox)
                    _prev_track_state = track_state
                else:
                    predicted_state = np.array(kf.predict()).flatten()
                    # ── Drift telemetry: capture pre-update KF pred ──────────
                    if _DRIFT_LOG_ENABLED:
                        _drift_pre_kf_cx = float(predicted_state[0]) + float(predicted_state[2]) / 2.0
                        _drift_pre_kf_cy = float(predicted_state[1]) + float(predicted_state[3]) / 2.0
                    # ─────────────────────────────────────────────────────────
                    # Phase 4: IMM manoeuvre probability for adaptive bypass threshold
                    p_maneuver = 0.0
                    if hasattr(kf, "get_model_probabilities"):
                        _mu = np.array(kf.get_model_probabilities())
                        p_maneuver = float(_mu[1] + _mu[2])
                    gmc_suppresses_maneuver = (
                        gmc_quality_state != "good"
                        or _gmc_maneuver_freeze_countdown > 0
                    )
                    observation = observe_with_guidance(
                        tracker,
                        frame_rgb,
                        predicted_state,
                        mode="ai_lead",
                        last_output_bbox=prev_output_bbox,
                    )
                    conf = observation.confidence
                    # ── Drift telemetry: capture AI observation ──────────────
                    if _DRIFT_LOG_ENABLED:
                        if observation.bbox is not None:
                            _drift_obs_cx = float(observation.bbox[0]) + float(observation.bbox[2]) / 2.0
                            _drift_obs_cy = float(observation.bbox[1]) + float(observation.bbox[3]) / 2.0
                        else:
                            _drift_obs_cx = float("nan")
                            _drift_obs_cy = float("nan")
                    # ─────────────────────────────────────────────────────────
                    track_state = state_machine.step(conf)

                    # Startup grace: keep TRACKING for first N frames so AI
                    # template has time to warm up before coasting is allowed.
                    _startup_grace = int(runtime_params.get("startup_grace_frames", 0))
                    if _startup_grace > 0 and frame_idx <= _startup_grace:
                        state_machine.force_tracking()
                        track_state = tracking_state_enum

                    # ── ORU hooks: state-transition detection ────────────────
                    if _prev_track_state is not None:
                        _is_tracking = (track_state == tracking_state_enum)
                        _was_tracking = (_prev_track_state == tracking_state_enum)
                        if _was_tracking and not _is_tracking:
                            oru.on_coast_start(frame_idx)
                        elif not _was_tracking and _is_tracking:
                            if observation.bbox is not None and oru.maybe_run(kf, observation.bbox, conf, frame_idx):
                                predicted_state = np.array(kf.predict()).flatten()
                    if track_state != tracking_state_enum and _prev_track_state is not None:
                        _v = np.array(kf.get_state()).flatten()
                        oru.record_coast_velocity(float(np.linalg.norm(_v[4:6])))
                    # ─────────────────────────────────────────────────────────

                    step = step_guided_imm(
                        kf,
                        decision,
                        predicted_state,
                        observation.bbox,
                        conf if chaos is None else chaos.step(conf),
                        frame_bgr.shape[1],
                        frame_bgr.shape[0],
                        is_tracking=(track_state == tracking_state_enum),
                        judge_reference_bbox=predicted_state[:4],
                        reinit_after=int(runtime_params.get("reinit_after", 8)),
                        adaptive_r_enabled=bool(runtime_params.get("adaptive_r_enabled", True)),
                        r_exponent=float(runtime_params.get("r_exponent", 1.0)),
                        conf_bypass_threshold=float(runtime_params.get("conf_bypass_threshold", 0.15)),
                        innovation_threshold=float(runtime_params.get("innovation_threshold", 2.0)),
                        reject_streak=reject_streak,
                        last_good_bbox=last_good_bbox,
                        mahal_chi2_gate=float(runtime_params.get("mahal_chi2_gate", 0.0)),
                        r_pos_base=float(runtime_params.get("r_pos_base", 1.0)),
                        r_size_base=float(runtime_params.get("r_size_base", 10.0)),
                        maneuver_probability=0.0 if gmc_suppresses_maneuver else p_maneuver,
                        maneuver_threshold=float(runtime_params.get("maneuver_threshold", 0.0)),
                        maneuver_bypass_boost=(
                            0.0 if gmc_suppresses_maneuver
                            else float(runtime_params.get("maneuver_bypass_boost", 0.0))
                        ),
                        mahal_bypass_after=int(runtime_params.get("mahal_bypass_after", 5)),
                        coast_count=(state_machine.coast_count() if hasattr(state_machine, "coast_count") else 0),
                        alpha_gate_k_conf=float(runtime_params.get("alpha_gate_k_conf", 0.0)),
                        alpha_gate_lambda=float(runtime_params.get("alpha_gate_lambda", 0.0)),
                        reacq_r_decay=float(runtime_params.get("reacq_r_decay", 0.0)),
                        maneuver_pi_enabled=(
                            bool(runtime_params.get("maneuver_pi_enabled", False))
                            and not gmc_suppresses_maneuver
                        ),
                        maneuver_pi_thr=float(runtime_params.get("maneuver_pi_thr", 16.0)),
                        maneuver_pi_persist=float(runtime_params.get("maneuver_pi_persist", 0.72)),
                        maneuver_pi_singer_boost=float(runtime_params.get("maneuver_pi_singer_boost", 0.20)),
                        normal_pi_persist=float(runtime_params.get("normal_pi_persist", 0.96)),
                        vel_gate_min_speed=float(runtime_params.get("vel_gate_min_speed", 0.0)),
                        vel_gate_cos_thr=float(runtime_params.get("vel_gate_cos_thr", 0.5)),
                        mahal_bypass_conf_thr=float(runtime_params.get("mahal_bypass_conf_thr", 0.0)),
                        mahal_bypass_vel_thr=float(runtime_params.get("mahal_bypass_vel_thr", 0.0)),
                        mahal_bypass_after_fast=int(runtime_params.get("mahal_bypass_after_fast", 2)),
                        mahal_bypass_after_slow=int(runtime_params.get("mahal_bypass_after_slow", 5)),
                        vel_innov_ratio_gate=float(runtime_params.get("vel_innov_ratio_gate", 0.0)),
                        vel_innov_min_speed=float(runtime_params.get("vel_innov_min_speed", 1.0)),
                        vel_innov_min_innov=float(runtime_params.get("vel_innov_min_innov", 0.0)),
                        accept_bbox_raw=bool(runtime_params.get("accept_bbox_raw", False)),
                    )
                    bbox = step.bbox
                    reject_streak = step.reject_streak
                    last_good_bbox = step.last_good_bbox

                    # N2: update rejection-memory ring buffer (1=reject, 0=accept)
                    if _f5_rbuf is not None:
                        _f5_rbuf.append(0 if step.accepted_measurement else 1)

                    # ORU hook D: snapshot after accepted TRACKING update
                    if step.accepted_measurement and track_state == tracking_state_enum and observation.bbox is not None:
                        oru.push_snapshot(
                            np.array(kf.get_state()).flatten(),
                            np.array(kf.get_covariance()),
                            np.array(kf.get_model_probabilities()),
                            observation.bbox,
                            frame_idx,
                        )

                    if reject_streak >= 5 and step.should_coast:
                        # Phase 1 — The Great Rescue: reinit only when AI is blind
                        # and the sequence is not protected by the size/collapse gates.
                        if should_run_great_rescue(observation.bbox, bbox):
                            tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
                    elif f5_feedback and not _f5_startup_suppress:
                        # F5: closed-loop feedback — feed KF-fused bbox back to AI
                        # search window every frame. Gated behind --f5-feedback for
                        # clean A/B against the prior prod behaviour (no set_state).
                        # C1: startup classifier gate — suppressed for sequences that
                        # start slowly then accelerate (e.g. car8).
                        # N2: rejection-memory gate — suppress F5 when recent reject
                        # count is below threshold (AI+KF silently drifting together).
                        # N4: coast-only gate — skip F5 during clean tracking.
                        _f5_mem_ok = (
                            _f5_rbuf is None
                            or len(_f5_rbuf) < _f5_rw
                            or sum(_f5_rbuf) >= _f5_rmc
                        )
                        _f5_coast_only = bool(runtime_params.get("f5_coast_only", False))
                        _f5_coast_ok = (not _f5_coast_only) or (reject_streak > 0)
                        if _f5_mem_ok and _f5_coast_ok:
                            f5_bbox = apply_f5_feedback_bbox(bbox, obs_bbox=observation.bbox, last_good=last_good_bbox)
                            if f5_bbox is not None:
                                tracker.set_state(f5_bbox)
                    _prev_track_state = track_state
            else:
                ai_bbox, conf = tracker.track(frame_rgb)
                ai_bbox = ai_bbox.tolist() if isinstance(ai_bbox, np.ndarray) else list(ai_bbox)
                bbox = ai_bbox

        # Store result: seq_id + "_" + frame_index -> [x, y, w, h]
        result_id = f"{seq_id}_{frame_idx}"
        results[result_id] = bbox
        prev_output_bbox = bbox

        # ── Drift telemetry update ───────────────────────────────────────────
        if _DRIFT_LOG_ENABLED and _drift_file is not None:
            _cx = float(bbox[0]) + float(bbox[2]) / 2.0
            _cy = float(bbox[1]) + float(bbox[3]) / 2.0
            if _drift_prev_cx is not None:
                _dx = _cx - _drift_prev_cx
                _dy = _cy - _drift_prev_cy
                _drift_ema_vx = (1.0 - _drift_alpha) * _drift_ema_vx + _drift_alpha * _dx
                _drift_ema_vy = (1.0 - _drift_alpha) * _drift_ema_vy + _drift_alpha * _dy
            _ema_mag = math.sqrt(_drift_ema_vx ** 2 + _drift_ema_vy ** 2)
            # Innovation: AI observation vs pre-update KF prediction (key drift signal)
            _innov = float("nan")
            if not math.isnan(_drift_obs_cx):
                _innov = math.sqrt(
                    (_drift_obs_cx - _drift_pre_kf_cx) ** 2 + (_drift_obs_cy - _drift_pre_kf_cy) ** 2
                )
            _drift_file.write(
                f"{frame_idx},{_cx:.2f},{_cy:.2f},"
                f"{_drift_pre_kf_cx:.2f},{_drift_pre_kf_cy:.2f},"
                f"{_drift_obs_cx:.2f},{_drift_obs_cy:.2f},"
                f"{_innov:.2f},{conf:.4f},"
                f"{_drift_ema_vx:.4f},{_drift_ema_vy:.4f},{_ema_mag:.4f}\n"
            )
            _drift_prev_cx = _cx
            _drift_prev_cy = _cy
        # ────────────────────────────────────────────────────────────────────

        if gmc_enabled:
            prev_frame_bgr = frame_bgr.copy()
            if _gmc_maneuver_freeze_countdown > 0:
                _gmc_maneuver_freeze_countdown -= 1

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start
    fps = frame_idx / elapsed if elapsed > 0 else 0

    if _DRIFT_LOG_ENABLED and _drift_file is not None:
        _drift_file.close()
        print(f"  [DRIFT] {seq_id}: log → {_drift_file.name}")

    if verbose:
        kf_tag = " [KF]" if use_kf else ""
        print(f"  {seq_id}: {frame_idx} frames, {elapsed:.1f}s ({fps:.1f} fps), conf_last={conf:.3f}{kf_tag}")


def write_submission(results, output_path):
    """Write submission CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "w", "h"])
        for result_id in sorted(results.keys()):
            x, y, w, h = results[result_id]
            writer.writerow([result_id, f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}"])
    print(f"\n[OK] Submission written: {output_path} ({len(results)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Run SGLATrack on competition data")
    parser.add_argument("--split", default="public_lb", choices=["train", "public_lb", "all"],
                        help="Which split to run on")
    parser.add_argument("--seq", default=None, action="append",
                        help="Run specific sequence(s); may be repeated (e.g. --seq dataset3/car1 --seq dataset3/car2)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SGLATrack checkpoint (.pth.tar)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--backend", default="tensorrt", choices=["pytorch", "tensorrt", "mixformerv2"],
                        help="Inference backend (mixformerv2 = MixFormerV2-Small standalone)")
    parser.add_argument("--engine", default=None, help="TensorRT engine path (for --backend tensorrt)")
    parser.add_argument("--no-kf", action="store_true", help="Disable Kalman filter (raw AI output)")
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="Override confidence threshold for KF coasting")
    parser.add_argument("--imm-config", default=None,
                        help="Path to runtime/IMM config YAML")
    parser.add_argument("--no-association", action="store_true",
                        help="Disable association (use simple track() + KF  path)")
    parser.add_argument("--f5-feedback", action="store_true",
                        help="Enable F5 closed-loop feedback (tracker.set_state every frame). "
                             "Default OFF matches pre-F5 prod behaviour.")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test Time Augmentation: average original + horizontal-flip "
                             "inference passes (2× inference cost, improves robustness)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        cv2.setRNGSeed(args.seed)

    # Default output path
    if args.output is None:
        os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
        args.output = os.path.join(PROJECT_ROOT, "outputs", f"submission_{args.split}.csv")

    # Load manifest
    manifest = load_manifest()

    # Get sequences
    sequences = get_sequences(manifest, args.split, args.seq)
    if not sequences:
        print(f"[ERROR] No sequences found for split={args.split}, seq={args.seq}")
        sys.exit(1)

    print(f"[INFO] Split: {args.split}, Sequences: {len(sequences)}")

    # Load runtime config with tracker_config.yaml as the default source of truth.
    default_runtime_config = os.path.join(PROJECT_ROOT, "configs", "tracker_config.yaml")
    runtime_config_path = args.imm_config or default_runtime_config
    imm_config = load_yaml_config(runtime_config_path)
    runtime_params = load_runtime_config(runtime_config_path)
    if args.conf_threshold is not None:
        runtime_params["conf_threshold"] = float(args.conf_threshold)
        runtime_params["sm_conf_threshold"] = float(args.conf_threshold)
    if imm_config:
        print(f"[INFO] Runtime config loaded from {runtime_config_path}")

    # Create tracker
    if args.backend == "tensorrt":
        from tracker.trt_wrapper import TRTTrackWrapper  # pyright: ignore[reportMissingImports]
        tracker = TRTTrackWrapper(
            engine_path=args.engine,
            association_enabled=not args.no_association,
        )
    elif args.backend == "mixformerv2":
        from tracker.mixformerv2_wrapper import MixFormerV2Wrapper  # pyright: ignore[reportMissingImports]
        tracker = MixFormerV2Wrapper()
    else:
        from tracker.sglatrack_wrapper import SGLATrackWrapper  # pyright: ignore[reportMissingImports]
        tracker = SGLATrackWrapper(
            checkpoint_path=args.checkpoint,
            association_enabled=not args.no_association,
            tta_flip=args.tta,
        )

    # Run on all sequences
    results = {}
    total_start = time.time()
    for i, (seq_id, seq_info) in enumerate(sequences.items()):
        print(f"[{i+1}/{len(sequences)}] {seq_id} ({seq_info['n_frames']} frames)")
        # f5_feedback: CLI flag OR YAML config key (either enables it)
        _f5 = args.f5_feedback or bool(runtime_params.get("f5_feedback", False))
        try:
            run_tracker_on_sequence(tracker, seq_id, seq_info, results,
                                    use_kf=not args.no_kf,
                                    runtime_params=runtime_params,
                                    imm_config=imm_config,
                                    f5_feedback=_f5)
        except Exception as exc:
            print(f"  [ERROR] {seq_id} crashed: {exc!r} — skipping sequence")

    total_time = time.time() - total_start
    total_frames = sum(s["n_frames"] for s in sequences.values())
    print(f"\n[DONE] {len(sequences)} sequences, {total_frames} frames, {total_time:.1f}s total "
          f"({total_frames/total_time:.1f} fps overall)")

    # Write submission
    write_submission(results, args.output)

    # If running on train, also save predictions for local evaluation
    if args.split in ("train", "all"):
        pred_dir = os.path.join(PROJECT_ROOT, "outputs", "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        for seq_id in sequences:
            seq_preds = {k: v for k, v in results.items() if k.startswith(seq_id + "_")}
            seq_file = os.path.join(pred_dir, seq_id.replace("/", "_") + ".txt")
            with open(seq_file, "w") as f:
                # Sort by frame index
                sorted_keys = sorted(seq_preds.keys(), key=lambda k: int(k.rsplit("_", 1)[1]))
                for k in sorted_keys:
                    x, y, w, h = seq_preds[k]
                    f.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")
        print(f"[OK] Per-sequence predictions saved to {pred_dir}/")


if __name__ == "__main__":
    main()
