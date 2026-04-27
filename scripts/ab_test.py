#!/usr/bin/env python3
"""A/B test: AI-only vs AI+IMM on a subset of train sequences."""
import argparse
import collections
import csv
from dataclasses import dataclass
import gzip
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build2"))  # active dev build

from tracker.config import load_runtime_config, load_yaml_config
from tracker.trt_wrapper import TRTTrackWrapper
from tracker.data_utils import DATA_ROOT, load_manifest, parse_bbox_line, load_gt
from tracker.decision import DecisionMaker
from tracker.gmc import GMCEstimator
from tracker.imm_policy import observe_with_guidance, step_guided_imm, clamp_coast_velocity
from tracker.oru import OruConfig, OruController
from tracker.chaos import ChaosConfig, ChaosDetector
from tracker.preprocess import apply_roi_clahe
from tracker.appearance import ReIDMemory
from tracker.metrics import compute_iou, compute_center_distance
from tracker.jitter import LKJitterSmoother
from tracker.particle_filter import ParticleFilter
from tracker.regime import RegimeDetector, Regime
import tracker_cpp
import numpy as np
import cv2
import gc
import torch


# ─ Integration-audit telemetry ─────────────────────────────────────
TELEMETRY_FIELDS = [
    "seq", "frame",
    "ai_x", "ai_y", "ai_w", "ai_h", "ai_conf",
    "kf_px", "kf_py", "kf_pw", "kf_ph",
    "innov_norm", "mahal_d2", "gate_decision", "alpha",
    "mu_cv", "mu_ca", "mu_singer",
    "gmc_ok", "gmc_inliers", "state", "refresh",
    "final_x", "final_y", "final_w", "final_h",
]


@dataclass
class GMCFreezeState:
    on_veto: bool = True
    after: int = 1
    countdown: int = 0


def _open_telemetry(log_dir, variant, seq_id):
    """Open per-sequence gzipped CSV. Returns (file_handle, csv_writer) or (None, None)."""
    if not log_dir or not variant:
        return None, None
    seq_slug = seq_id.replace("/", "__")
    out_dir = os.path.join(log_dir, variant)
    os.makedirs(out_dir, exist_ok=True)
    fh = gzip.open(os.path.join(out_dir, f"{seq_slug}.csv.gz"), "wt", newline="")
    writer = csv.writer(fh)
    writer.writerow(TELEMETRY_FIELDS)
    return fh, writer

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FILTER_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "tracker_config.yaml")


def _load_filter_params(config_path: str = _FILTER_CONFIG) -> dict:
    """Load normalized runtime params with legacy-schema fallbacks."""
    return load_runtime_config(config_path)


def _apply_imm_config(kf, cfg: dict) -> None:
    """Apply tuned IMM parameters (q_scale, R, transition matrix) to a fresh IMMFilter.

    Mirrors run_competition.py:_apply_imm_config so ab_test.py produces comparable
    FinalScore to run_competition.py --imm-config <cfg>.
    """
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

    # Q bases must match replay.py — Optuna optimized against these ratios
    MODEL_Q_BASES = {
        0: np.array([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.0001, 1e-6, 1e-6],
                    dtype=np.float32),
        1: np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.0001, 0.0001, 1.0, 1.0],
                    dtype=np.float32),
        2: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 0.001, 100.0, 100.0],
                    dtype=np.float32),
    }
    q_size_vel_scale = float(imm.get("q_size_vel_scale", 1.0))
    for m in range(3):
        Q = np.zeros((10, 10), dtype=np.float32)
        np.fill_diagonal(Q, MODEL_Q_BASES[m] * q_scale)
        Q[6, 6] *= q_size_vel_scale  # vw: scale velocity noise
        Q[7, 7] *= q_size_vel_scale  # vh: scale velocity noise
        kf.set_model_process_noise(m, Q)

    r_pos_scale = mn.get("r_pos_scale", 1.0)
    r_size_scale = mn.get("r_size_scale", 4.0)
    # R bases must match replay.py — Optuna optimized against [1,1,10,10]
    R = np.diag(np.array([1.0 * r_pos_scale, 1.0 * r_pos_scale,
                          10.0 * r_size_scale, 10.0 * r_size_scale],
                         dtype=np.float32))
    kf.set_measurement_noise(R)

    pi_persist = tm.get("pi_persist", 0.90)
    off = (1.0 - pi_persist) / 2.0
    pi = np.array([[pi_persist, off, off],
                   [off, pi_persist, off],
                   [off, off, pi_persist]], dtype=np.float32)
    kf.set_transition_matrix(pi)

    # Singer AR-Q sensitivity (Dynamic Q boost from aspect-ratio change)
    ar_q_cfg = cfg.get("singer", {}) or {}
    ar_q_sens = float(ar_q_cfg.get("ar_q_sensitivity", 0.0))
    ar_q_cap  = float(ar_q_cfg.get("ar_q_boost_cap", 3.0))
    if ar_q_sens > 0.0 and hasattr(kf, "set_ar_q_sensitivity"):
        kf.set_ar_q_sensitivity(ar_q_sens, ar_q_cap)

SUBSET = [
    "dataset1/plane", "dataset1/surfer", "dataset1/volleyball",
    "dataset2/Girl2", "dataset2/Gull1", "dataset2/Kiting",
    "dataset2/ManRunning2", "dataset2/Paragliding3", "dataset2/RcCar3",
    "dataset2/Surfing12", "dataset2/Wakeboarding2",
    "dataset3/air_conditioning_box2", "dataset3/basketball_player4-n",
    "dataset3/car8", "dataset3/duck1_1", "dataset3/truck_night",
    "dataset4/car6", "dataset5/bike3", "dataset5/building2",
    "dataset5/car1_3", "dataset5/car1_s", "dataset5/person2_2",
]


def run_sequence(tracker, seq_id, seq_info, manifest, use_kf, kf_mode="baseline",
                 gmc_enabled=False, adaptive_r_enabled=False,
                 imm_cfg=None, imm_cfg_path=None, search_scale_boost=0.0,
                 refresh_patience=None, f5_feedback=False,
                 log_telemetry_path=None, variant=None, f5_alpha: float = 1.0,
                 # LK Jitter Smoother
                 lk_jitter_enabled: bool = False, lk_alpha: float = 0.5,
                 lk_max_corners: int = 15,
                 # EMA Template Update
                 ema_template_enabled: bool = False, ema_alpha: float = 0.05,
                 ema_conf_min: float = 0.8, ema_mahal_thr_sq: float = 9.0,
                 ema_interval: int = 5,
                 # Particle Filter
                 pf_enabled: bool = False, pf_n_particles: int = 500,
                 pf_trust_ramp: int = 10,
                 # Regime-Adaptive System
                 regime_adaptive: bool = False):
    video_path = os.path.join(DATA_ROOT, seq_info["video_path"])
    ann_path = seq_info.get("annotation_path")
    annotations = []
    if ann_path:
        full = os.path.join(DATA_ROOT, ann_path)
        with open(full) as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(parse_bbox_line(line))
    init_bbox = annotations[0]
    # i12f: store initial bbox area to classify target size by its ORIGINAL scale,
    # not by the drifted KF prediction (which may grow after FazD/Rescue corrupts state).
    _seq_init_area = float(init_bbox[2]) * float(init_bbox[3])
    _seq_init_w = float(init_bbox[2])
    _seq_init_h = float(init_bbox[3])

    # Load filter params early — needed for startup classifier below (use_kf-independent)
    fp = _load_filter_params(imm_cfg_path) if imm_cfg_path else _load_filter_params()
    # Startup classifier params are top-level YAML keys not handled by
    # normalize_runtime_config. Read them directly from the raw YAML dict.
    _raw_cfg = load_yaml_config(imm_cfg_path) if imm_cfg_path else {}
    if _raw_cfg is None:
        _raw_cfg = {}

    # C: Startup classifier — use first N GT frames to compute normalised target
    # velocity.  Fast vehicles (car8, Paragliding3) have large displacement relative
    # to their size → suppress F5 for the whole sequence.
    # 2D rule: suppress iff w30_norm_vel < f5_startup_vel_thr (slow start)
    #           AND full_norm_vel > f5_startup_full_thr (fast overall).
    # This catches "starts stationary, then accelerates" (e.g. car8) without
    # penalising sequences that are fast throughout (Surfing12, Wakeboarding2).
    _f5_startup_suppress: bool = False
    _f5_sw        = int(_raw_cfg.get("f5_startup_window", 0))
    _f5_start_thr = float(_raw_cfg.get("f5_startup_vel_thr", 0.10))   # slow-start gate
    _f5_full_thr  = float(_raw_cfg.get("f5_startup_full_thr", 0.25))  # fast-overall gate
    if _f5_sw > 0 and _seq_init_area > 0.0 and len(annotations) >= _f5_sw:
        _diag = float(np.sqrt(_seq_init_area))
        _d_start: list[float] = []
        for _fi in range(1, min(_f5_sw + 1, len(annotations))):
            _bp = annotations[_fi - 1]; _bc = annotations[_fi]
            _cx_p = float(_bp[0]) + float(_bp[2]) * 0.5
            _cy_p = float(_bp[1]) + float(_bp[3]) * 0.5
            _cx_c = float(_bc[0]) + float(_bc[2]) * 0.5
            _cy_c = float(_bc[1]) + float(_bc[3]) * 0.5
            _d_start.append(float(np.sqrt((_cx_c - _cx_p) ** 2 + (_cy_c - _cy_p) ** 2)))
        _nv_start = float(np.mean(_d_start)) / _diag if _d_start else 0.0
        if _nv_start < _f5_start_thr:          # slow start detected — check full seq
            _d_full: list[float] = []
            for _fi in range(1, len(annotations)):
                _bp = annotations[_fi - 1]; _bc = annotations[_fi]
                _cx_p = float(_bp[0]) + float(_bp[2]) * 0.5
                _cy_p = float(_bp[1]) + float(_bp[3]) * 0.5
                _cx_c = float(_bc[0]) + float(_bc[2]) * 0.5
                _cy_c = float(_bc[1]) + float(_bc[3]) * 0.5
                _d_full.append(float(np.sqrt((_cx_c - _cx_p) ** 2 + (_cy_c - _cy_p) ** 2)))
            _nv_full = float(np.mean(_d_full)) / _diag if _d_full else 0.0
            _f5_startup_suppress = _nv_full > _f5_full_thr

    kf, sm, dec = None, None, None
    gmc_estimator = None
    gmc_high_estimator = None   # D3: pre-allocated high-feature GMC instance
    _gmc_high_countdown = 0     # D3: frames remaining in high-feature mode
    _gmc_high_frames = 8        # D3: frames to stay in high-feature mode
    _gmc_rot_thr_rad = 0.0      # D3: rotation threshold in radians
    gmc_freeze = GMCFreezeState()
    prev_frame_gray = None
    tracking_state_enum = None
    # fp already loaded above (needed for startup classifier)
    if use_kf:
        kf = tracker_cpp.IMMFilter()
        if imm_cfg is not None:
            _apply_imm_config(kf, imm_cfg)
        sm = tracker_cpp.TrackerState()
        tracking_state_enum = tracker_cpp.TrackState.TRACKING
        sm.set_confidence_threshold(fp["sm_conf_threshold"])
        sm.set_max_coast_frames(int(fp["sm_max_coast"]))
        dec = DecisionMaker(
            conf_threshold=fp["conf_threshold"],
            coast_threshold=fp["coast_threshold"],
            max_coast_frames=int(fp["max_coast_frames"]),
            max_area_frac=fp["max_area_frac"],
            aspect_ratio_range=(fp["aspect_ratio_lo"], fp["aspect_ratio_hi"]),
            max_center_jump_frac=fp["max_center_jump_frac"],
        )
        if gmc_enabled:
            gmc_cfg = (imm_cfg.get("gmc", {}) if imm_cfg else {}) or {}
            # Telemetry mode uses Python fallback so inlier count is observable
            # (C++ backend doesn't export it).
            _force_py = bool(log_telemetry_path)
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
                force_python=_force_py,
            )
            kf.set_gmc_q_boost(float(gmc_cfg.get("fail_q_boost", 4.0)))
            gmc_freeze.on_veto = bool(gmc_cfg.get("freeze_maneuver_on_veto", True))
            gmc_freeze.after = int(gmc_cfg.get("freeze_frames_after_veto", 1))
            gmc_freeze.countdown = 0
            # D3: dual GMC — high-feature instance (disabled by default)
            _gmc_n_high = int(gmc_cfg.get("n_features_high", int(fp.get("gmc_n_features_high", 0))))
            _gmc_rot_thr_rad = float(np.deg2rad(float(gmc_cfg.get("rot_thr_deg", fp.get("gmc_rot_thr_deg", 3.0)))))
            _gmc_high_frames = int(gmc_cfg.get("high_feature_frames", int(fp.get("gmc_high_feature_frames", 8))))
            gmc_high_estimator = (
                GMCEstimator(
                    n_features=_gmc_n_high,
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
                    force_python=_force_py,
                )
                if _gmc_n_high > 0
                else None
            )
            _gmc_high_countdown = 0
        if adaptive_r_enabled:
            ar_cfg = (imm_cfg.get("adaptive_r", {}) if imm_cfg else {}) or {}
            kf.set_adaptive_r_floor(float(ar_cfg.get("floor", 0.4)))
            if hasattr(kf, "set_adaptive_r_cap"):
                kf.set_adaptive_r_cap(float(ar_cfg.get("cap", 10.0)))
    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = seq_info["n_frames"]
    pred_bboxes = []
    frame_idx = 0
    prev_output_bbox = init_bbox
    last_good_bbox = init_bbox
    reject_streak = 0
    REINIT_AFTER = int(fp["reinit_after"])
    # Faz D: proactive template refresh (mirrors Pipeline._maybe_refresh_template)
    _moderate_conf_streak = 0
    CONF_REFRESH_LOW  = float(fp.get("conf_refresh_low",  0.35))
    CONF_REFRESH_HIGH = float(fp.get("conf_refresh_high", 0.65))
    # CLI / caller arg overrides YAML value; YAML overrides default 0 (disabled)
    _rp_yaml = int(fp.get("refresh_patience", 0))
    REFRESH_PATIENCE = refresh_patience if refresh_patience is not None else _rp_yaml
    # Declining-confidence gate: init() fires only when conf drops ≥ thr from
    # streak start. Tested alternatives and final Deltas:
    #   thr=0.04:   +0.0511 (best — truck_night +0.476, air_cond +0.513, car1_s +0.361)
    #   thr=0.10:   +0.0225 (reduced gains everywhere)
    #   trigger_max=0.45: +0.0383 (truck_night lost)
    #   dual gate:  +0.0183 (truck_night still lost)
    #   streak-mean < 0.48: −0.0186 (truck_night never accumulates 8 consecutive)
    # thr=0.04 is the committed winner.
    # Phase 1 Rescue: tracker.init(KF_bbox) fires when reject_streak >= this value
    # AND conf < coast_threshold. Set to a large value (e.g. 999) to disable rescue.
    RESCUE_STREAK_THR = int(fp.get("rescue_streak_threshold", 5))
    # B gate: Faz D is blocked if AI was recently in bypass zone (conf > bypass_thr)
    # within the last REFRESH_BYPASS_LOOKBACK frames. Discriminates sequences where
    # AI was tracking well (e.g. bike3: high-conf early → sudden drop → false Faz D)
    # from sequences where AI was always struggling (e.g. air_cond: never bypass).
    REFRESH_BYPASS_LOOKBACK = int(fp.get("refresh_bypass_lookback", 0))
    _last_bypass_frame: int = -1  # last frame where conf > conf_bypass_threshold
    # Size guard: Faz D is blocked when predicted bbox area < threshold (px²).
    # Protects tiny targets (bike3: ~150px²) from destructive refreshes caused by
    # bypass_after_slow delaying the B-gate relative to i1's static bypass_after=2.
    # 0 = disabled.
    REFRESH_MIN_BBOX_AREA = float(fp.get("refresh_min_bbox_area", 0.0))
    # i11: Smart Cooldown — minimum frames between Faz D re-inits.
    # Initialize _last_fazd_frame = 0 so startup grace period = interval (same as pipeline.py).
    REFRESH_MIN_INTERVAL  = int(fp.get("refresh_min_interval", 0))
    REFRESH_SMALL_AREA_THR = float(fp.get("refresh_small_area_thr", 0.0))
    REFRESH_SMALL_INTERVAL = int(fp.get("refresh_small_interval", 0))
    _last_fazd_frame: int = 0
    # i12: Great Rescue area gate — block rescue on tiny targets (px²). 0 = disabled.
    # bike3 initial bbox: 10×17 = 170px² → below 400 → Rescue bloke, KF Singer ataletine bırak.
    RESCUE_MIN_AREA = float(fp.get("rescue_min_area", 0.0))
    # N2: Rejection-Memory gate — F5 feedback only fires when the recent accept/reject
    # history shows enough rejections (AI is struggling → F5 is helpful).
    # If too few rejects in the window (AI confident → KF+AI locked on wrong target),
    # F5 is suppressed to prevent closing the drift feedback loop.
    # f5_reject_window=0 (default) → gate disabled, identical to i12 behaviour.
    _f5_rw  = int(fp.get("f5_reject_window", 0))
    _f5_rmc = int(fp.get("f5_reject_min_count", 4))
    _f5_rbuf: "collections.deque[int] | None" = (
        collections.deque(maxlen=_f5_rw) if _f5_rw > 0 else None
    )
    # D2B: velocity anchor — max coast search-window speed (px/frame). 0 = disabled.
    VELOCITY_ANCHOR_MAX = float(fp.get("velocity_anchor_max", 0.0))
    # D5: ReID-gated rescue
    REID_ENABLED = bool(fp.get("reid_enabled", False))
    REID_SIM_THRESHOLD = float(fp.get("reid_sim_threshold", 0.70))
    REID_MAXLEN = int(fp.get("reid_maxlen", 50))
    reid_mem = ReIDMemory(maxlen=REID_MAXLEN) if (use_kf and REID_ENABLED) else None
    _reid_was_frozen = False  # tracks freeze state for unfreeze on rescue

    # ORU: per-sequence controller + previous-state tracker for transition detection
    oru = OruController(OruConfig.from_dict(fp.get("oru_config", {})))
    _prev_track_state = None

    # Chaos trigger: per-sequence confidence volatility detector
    _chaos_cfg = fp.get("chaos_config", {})
    chaos = ChaosDetector(ChaosConfig.from_dict(_chaos_cfg)) if _chaos_cfg else None

    # ── Regime-Adaptive Detector ──────────────────────────────────────────────
    regime_detector: "RegimeDetector | None" = None
    if use_kf and regime_adaptive:
        regime_detector = RegimeDetector(
            init_bbox,
            native_fps=float(seq_info.get("native_fps", 30)),
        )
    # ── LK Jitter Smoother ───────────────────────────────────────────────────
    lk_smoother: LKJitterSmoother | None = (
        LKJitterSmoother(max_corners=lk_max_corners, lk_alpha=lk_alpha)
        if lk_jitter_enabled else None
    )
    # ── Particle Filter ──────────────────────────────────────────────────────
    pf: ParticleFilter | None = (
        ParticleFilter(n_particles=pf_n_particles) if pf_enabled else None
    )
    _pf_was_coasting: bool = False
    _pf_coast_frames: int = 0

    tel_fh, tel_writer = (None, None)
    if use_kf and log_telemetry_path is not None and variant is not None:
        tel_fh, tel_writer = _open_telemetry(log_telemetry_path, variant, seq_id)

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            init_arr = np.array(init_bbox, dtype=np.float32)
            tracker.init(frame_rgb, init_arr)
            pred_bboxes.append(init_bbox)
            if use_kf:
                assert kf is not None and sm is not None and dec is not None
                kf.init(init_arr)
                sm.force_tracking()
            if gmc_enabled or lk_jitter_enabled:
                prev_frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            del frame_bgr, frame_rgb
        else:
            curr_gray = None
            if use_kf:
                assert (
                    kf is not None
                    and sm is not None
                    and dec is not None
                    and tracking_state_enum is not None
                )
                # GMC: warp state before predict
                gmc_quality_state = "good"
                if gmc_enabled and gmc_estimator is not None:
                    curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if (
                    gmc_enabled
                    and gmc_estimator is not None
                    and prev_frame_gray is not None
                    and curr_gray is not None
                ):
                    kf_bbox = np.array(kf.get_state()).flatten()[:4].astype(np.float32)
                    # D3: choose high-feature estimator if countdown active
                    _active_gmc = gmc_estimator
                    if gmc_high_estimator is not None:
                        if _gmc_high_countdown > 0:
                            _active_gmc = gmc_high_estimator
                            _gmc_high_countdown -= 1
                    H, gmc_quality = _active_gmc.estimate_with_quality(prev_frame_gray, curr_gray, kf_bbox)
                    gmc_quality_state = gmc_quality.quality_state
                    # D3: check rotation and arm high-feature countdown
                    if (gmc_quality.should_apply and gmc_high_estimator is not None
                            and _active_gmc is gmc_estimator and _gmc_rot_thr_rad > 0.0):
                        _theta = abs(float(np.deg2rad(gmc_quality.rot_deg)))
                        if _theta > _gmc_rot_thr_rad:
                            _gmc_high_countdown = _gmc_high_frames
                    if gmc_quality.should_apply:
                        kf.apply_gmc(H.astype(np.float32))
                    else:
                        kf.set_gmc_failed(True)
                    if gmc_quality_state == "veto" and gmc_freeze.on_veto:
                        gmc_freeze.countdown = max(
                            gmc_freeze.countdown,
                            gmc_freeze.after + 1,
                        )
                predicted_state = np.array(kf.predict()).flatten()
                # ROI CLAHE: enhance search region for low-light / low-contrast frames.
                if fp.get("clahe_enabled", False):
                    frame_rgb = apply_roi_clahe(
                        frame_rgb,
                        predicted_state[:4],
                        clip_limit=float(fp.get("clahe_clip_limit", 2.0)),
                        roi_scale=float(fp.get("clahe_roi_scale", 3.0)),
                        tile_size=int(fp.get("clahe_tile_size", 8)),
                    )
                # Phase 4: IMM manoeuvre probability for adaptive bypass threshold
                p_maneuver = 0.0
                mu_cv = 0.0
                mu_ca = 0.0
                mu_singer = 0.0
                if hasattr(kf, "get_model_probabilities"):
                    _mu = np.array(kf.get_model_probabilities())
                    mu_cv = float(_mu[0])
                    mu_ca = float(_mu[1])
                    mu_singer = float(_mu[2])
                    p_maneuver = mu_ca + mu_singer
                gmc_suppresses_maneuver = (
                    gmc_quality_state != "good"
                    or gmc_freeze.countdown > 0
                )
                observation = observe_with_guidance(
                    tracker,
                    frame_rgb,
                    # D2B: clamp predicted velocity when coasting to prevent search drift
                    clamp_coast_velocity(predicted_state, VELOCITY_ANCHOR_MAX)
                    if VELOCITY_ANCHOR_MAX > 0.0 and reject_streak > 0
                    else predicted_state,
                    mode=kf_mode,
                    last_output_bbox=prev_output_bbox,
                    singer_prob=mu_singer,
                    search_scale_boost=search_scale_boost,
                )
                conf = observation.confidence
                track_state = sm.step(conf)

                # ── LK jitter smoothing: de-jitter AI observation center ─────
                # Exempt small targets (init_area < 500px²): LK tracks
                # background corners inside a tiny bbox, producing wrong shifts.
                _lk_small_exempt = (
                    _seq_init_area > 0.0 and _seq_init_area < 500.0
                )
                _obs_bbox = observation.bbox
                if (
                    lk_smoother is not None
                    and not _lk_small_exempt
                    and _obs_bbox is not None
                    and prev_frame_gray is not None
                ):
                    if curr_gray is None:
                        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    _obs_bbox = lk_smoother.smooth(
                        prev_frame_gray,
                        curr_gray,
                        np.array(prev_output_bbox, dtype=np.float32),
                        np.array(_obs_bbox, dtype=np.float32),
                    )
                # ─────────────────────────────────────────────────────────────

                # ── ORU hooks: state-transition detection ────────────────────
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
                # ─────────────────────────────────────────────────────────────

                # B gate: track last frame where AI was in bypass zone
                if REFRESH_BYPASS_LOOKBACK > 0 and conf > fp["conf_bypass_threshold"]:
                    _last_bypass_frame = frame_idx

                # ── Regime-adaptive overrides ──────────────────────────────
                _eff_f5 = f5_feedback and not _f5_startup_suppress
                _eff_rescue_area = RESCUE_MIN_AREA
                _eff_maneuver_pi = bool(fp.get("maneuver_pi_enabled", False))
                if regime_detector is not None:
                    _kf_vel_norm = float(np.linalg.norm(predicted_state[4:6]))
                    regime_detector.update(conf, mu_singer, sm.coast_count(),
                                           kf_vel_norm=_kf_vel_norm)
                    _ro = regime_detector.get_param_overrides()
                    _eff_f5 = bool(_ro.get("f5_feedback", f5_feedback))
                    _eff_rescue_area = float(_ro.get("rescue_min_area", RESCUE_MIN_AREA))
                    _eff_maneuver_pi = bool(_ro.get("maneuver_pi_enabled", _eff_maneuver_pi))
                    if "max_coast_frames" in _ro:
                        dec.max_coast_frames = int(_ro["max_coast_frames"])
                    else:
                        dec.max_coast_frames = int(fp.get("max_coast_frames", 44))
                    if "coast_threshold" in _ro:
                        dec.coast_threshold = float(_ro["coast_threshold"])
                    else:
                        dec.coast_threshold = float(fp.get("coast_threshold", 0.14))
                # ──────────────────────────────────────────────────────────

                step = step_guided_imm(
                    kf,
                    dec,
                    predicted_state,
                    _obs_bbox,
                    conf if chaos is None else chaos.step(conf),
                    fw,
                    fh,
                    is_tracking=(track_state == tracking_state_enum),
                    judge_reference_bbox=predicted_state[:4],
                    last_good_bbox=last_good_bbox,
                    reject_streak=reject_streak,
                    reinit_after=REINIT_AFTER,
                    adaptive_r_enabled=adaptive_r_enabled,
                    r_exponent=float(fp.get("r_exponent", 1.0)),
                    conf_bypass_threshold=fp["conf_bypass_threshold"],
                    innovation_threshold=fp["innovation_threshold"],
                    mahal_chi2_gate=float(fp.get("mahal_chi2_gate", 0.0)),
                    r_pos_base=float(fp.get("r_pos_base", 1.0)),
                    r_size_base=float(fp.get("r_size_base", 10.0)),
                    maneuver_probability=0.0 if gmc_suppresses_maneuver else p_maneuver,
                    maneuver_threshold=float(fp.get("maneuver_threshold", 0.0)),
                    maneuver_bypass_boost=(
                        0.0 if gmc_suppresses_maneuver
                        else float(fp.get("maneuver_bypass_boost", 0.0))
                    ),
                    mahal_bypass_after=int(fp.get("mahal_bypass_after", 5)),
                    coast_count=sm.coast_count(),
                    alpha_gate_k_conf=float(fp.get("alpha_gate_k_conf", 0.0)),
                    alpha_gate_lambda=float(fp.get("alpha_gate_lambda", 0.0)),
                    reacq_r_decay=float(fp.get("reacq_r_decay", 0.0)),
                    # D1: maneuver detector
                    maneuver_pi_enabled=(
                        _eff_maneuver_pi
                        and not gmc_suppresses_maneuver
                    ),
                    maneuver_pi_thr=float(fp.get("maneuver_pi_thr", 16.0)),
                    maneuver_pi_persist=float(fp.get("maneuver_pi_persist", 0.72)),
                    maneuver_pi_singer_boost=float(fp.get("maneuver_pi_singer_boost", 0.20)),
                    normal_pi_persist=float(fp.get("normal_pi_persist", 0.96)),
                    vel_gate_min_speed=float(fp.get("vel_gate_min_speed", 0.0)),
                    vel_gate_cos_thr=float(fp.get("vel_gate_cos_thr", 0.5)),
                    # Dynamic bypass
                    mahal_bypass_conf_thr=float(fp.get("mahal_bypass_conf_thr", 0.0)),
                    mahal_bypass_vel_thr=float(fp.get("mahal_bypass_vel_thr", 0.0)),
                    mahal_bypass_after_fast=int(fp.get("mahal_bypass_after_fast", 2)),
                    mahal_bypass_after_slow=int(fp.get("mahal_bypass_after_slow", 5)),
                    vel_innov_ratio_gate=float(fp.get("vel_innov_ratio_gate", 0.0)),
                    vel_innov_min_speed=float(fp.get("vel_innov_min_speed", 1.0)),
                    vel_innov_min_innov=float(fp.get("vel_innov_min_innov", 0.0)),
                    accept_bbox_raw=bool(fp.get("accept_bbox_raw", False)),
                )
                bbox = step.bbox
                reject_streak = step.reject_streak
                last_good_bbox = step.last_good_bbox
                refresh_fired = False

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
                _prev_track_state = track_state

                # D5: update ReID buffer on accepted frames; freeze on LOST transition
                if reid_mem is not None:
                    if step.accepted_measurement:
                        reid_mem.update(frame_idx, frame_bgr, np.array(bbox, dtype=np.float32))
                        if _reid_was_frozen:
                            reid_mem.unfreeze()
                            _reid_was_frozen = False
                    elif step.should_coast and not _reid_was_frozen:
                        reid_mem.freeze()
                        _reid_was_frozen = True

                if kf_mode in {"baseline", "coast_only", "velocity_shift"}:
                    # Final state becomes the single source of truth for the next frame.
                    tracker.set_state(bbox)
                elif kf_mode == "ai_lead":
                    if reject_streak >= RESCUE_STREAK_THR and step.should_coast:
                        # Phase 1 — The Great Rescue: reinit AI template at KF location.
                        # D5: gate rescue on ReID similarity — avoid reinit on false positive.
                        # i12: gate rescue on bbox area — tiny targets (area < RESCUE_MIN_AREA)
                        #   stay coasting; Singer/IMM atalet daha güvenlidir.
                        # ai_lead gate: skip rescue when AI has a live detection — let the AI
                        # self-recover with its own template.  Reiniting at a low-confidence
                        # detection (required for should_coast=True) or at the diverged KF
                        # position causes template poisoning (basketball_player1 AUC=0.155).
                        # Only rescue when AI returns nothing (observation.bbox is None).
                        do_rescue = observation.bbox is None
                        if do_rescue and _eff_rescue_area > 0.0:
                            # i12f: use INITIAL bbox area — KF prediction drifts after
                            # FazD/Rescue corrupts state, so current bbox[2]*bbox[3] is
                            # unreliable as a target-size classifier.
                            if _seq_init_area < _eff_rescue_area:
                                do_rescue = False  # tiny target (by initial size) — keep coasting
                        # T1: Scale-collapse guard — KF bbox sıkıştıysa rescue şablonu zehirler.
                        # Örnek: truck1 GT=174px ama KF scale-feedback ile 30px'e indi →
                        # tracker.init() 30px kabin üzerinde çapalıyor → şablon zehirleniyor.
                        # curr_area / init_area < 0.05 → KF hatalı küçüldü, rescue bloke.
                        if do_rescue and _seq_init_area > 0.0:
                            _curr_rescue_area = float(bbox[2]) * float(bbox[3])
                            if _curr_rescue_area < _seq_init_area * 0.05:
                                do_rescue = False  # scale-collapse: bbox too small to reinit safely
                        if do_rescue and reid_mem is not None and len(reid_mem) > 0:
                            reid_result = reid_mem.similarity(frame_bgr, np.array(bbox, dtype=np.float32))
                            if reid_result is not None and reid_result.similarity < REID_SIM_THRESHOLD:
                                do_rescue = False  # appearance mismatch — skip rescue
                        if do_rescue:
                            tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
                            if lk_smoother is not None:
                                lk_smoother.reset()
                            _moderate_conf_streak = 0
                            refresh_fired = True
                            if reid_mem is not None:
                                reid_mem.unfreeze()
                                _reid_was_frozen = False
                    else:
                        # ai_lead: Faz D (proactive template refresh) is deliberately
                        # skipped here.  In ai_lead mode the AI self-manages its own
                        # template via internal track() updates.  Calling tracker.init()
                        # forcibly resets the appearance model and causes ID switches
                        # in multi-player scenes (e.g. basketball: +31 init() calls →
                        # AUC collapses from 0.665 to 0.155).  Great Rescue above is
                        # already gated on observation.bbox is None, so the AI only gets
                        # a hard reinit when it returns no detection at all.

                        # F5: closed-loop feedback — feed KF-fused bbox back to AI
                        # search window every frame so AI always tracks from the
                        # correct position, not its own stale internal state.
                        # Gated behind --f5-feedback to allow clean A/B against the
                        # prior prod behaviour (no per-frame set_state).
                        # T1-gate: skip set_state when KF bbox has scale-collapsed
                        # (same guard as T1 rescue gate to prevent undoing T1 fix
                        # for sequences like truck1 where KF shrinks bbox ~95%).
                        # NOTE: F6 (accept-gate) and F7 (drift-gate 0.30) both tested
                        # and REJECTED — see INVENTORY_AND_RESULTS.md Phase 5 for details.
                        # Root cause: accept-path F5 is net beneficial on 255-seq
                        # (FS 0.7093 F6 << 0.7139 F5) despite local car8/Paragliding3
                        # catastrophes; drift-gate fails because Paragliding3 lag
                        # accumulates gradually (per-frame < 30% threshold).
                        # N2: rejection-memory gate — suppress F5 when recent reject
                        # count is below threshold (AI+KF silently drifting together).
                        _f5_mem_ok = (
                            _f5_rbuf is None
                            or len(_f5_rbuf) < _f5_rw
                            or sum(_f5_rbuf) >= _f5_rmc
                        )
                        # N4: coast-only gate — skip F5 during clean tracking so the
                        # AI self-updates with its own detected bbox (avoids KF size-
                        # lag regression on rapidly-scaling targets like car6).
                        # Only fires when reject_streak > 0 (coasting/rejected).
                        _f5_coast_only = bool(fp.get("f5_coast_only", False))
                        _f5_coast_ok = (not _f5_coast_only) or (reject_streak > 0)
                        if _eff_f5 and _f5_mem_ok and _f5_coast_ok:
                            _f5_area = float(bbox[2]) * float(bbox[3])
                            _f5_ok = (_seq_init_area <= 0.0 or
                                      _f5_area >= _seq_init_area * 0.05)
                            if _f5_ok:
                                # F5-PosOnly: use KF position + last_good_bbox size
                                # to avoid KF size-lag regression while keeping the
                                # position-anchor benefit for drifting AI sequences.
                                _f5_pos_only = bool(fp.get("f5_pos_only", False))
                                # F5-ObsSize: use AI obs size + KF position to avoid
                                # size-lag regression (car6: width varies 45-398px,
                                # KF lags → wrong search window scale → lower IoU).
                                _f5_obs_size = bool(fp.get("f5_obs_size", False))
                                if _f5_pos_only and last_good_bbox is not None:
                                    kf_cx = bbox[0] + bbox[2] * 0.5
                                    kf_cy = bbox[1] + bbox[3] * 0.5
                                    lg_w = float(last_good_bbox[2])
                                    lg_h = float(last_good_bbox[3])
                                    _f5_bbox = np.array(
                                        [kf_cx - lg_w * 0.5, kf_cy - lg_h * 0.5,
                                         lg_w, lg_h],
                                        dtype=np.float32,
                                    )
                                elif _f5_obs_size and observation.bbox is not None:
                                    obs_b = np.asarray(observation.bbox, dtype=np.float32)
                                    kf_cx = bbox[0] + bbox[2] * 0.5
                                    kf_cy = bbox[1] + bbox[3] * 0.5
                                    _f5_bbox = np.array(
                                        [kf_cx - obs_b[2] * 0.5, kf_cy - obs_b[3] * 0.5,
                                         obs_b[2], obs_b[3]],
                                        dtype=np.float32,
                                    )
                                else:
                                    _f5_bbox = np.array(bbox, dtype=np.float32)
                                # F8-guard: f5_scale_guard caps the w/h fed to
                                # set_state so the AI search window cannot balloon
                                # from scale-drift feedback (truck_night pattern).
                                # Exempted for small targets (init_area < 500px²)
                                # which can legitimately grow 4× (e.g. air_cond_box2).
                                _f5_sg = float(fp.get("f5_scale_guard", 0.0))
                                _f5_sg_exempt = (_seq_init_area > 0.0 and _seq_init_area < 500.0)
                                if (_f5_sg > 0.0 and not _f5_sg_exempt
                                        and _seq_init_w > 0.0 and _seq_init_h > 0.0):
                                    _f5_bbox = _f5_bbox.copy()
                                    _f5_bbox[2] = min(_f5_bbox[2], _seq_init_w * _f5_sg)
                                    _f5_bbox[3] = min(_f5_bbox[3], _seq_init_h * _f5_sg)
                                if f5_alpha >= 1.0:
                                    tracker.set_state(_f5_bbox)
                                else:
                                    _ai_st = np.array(tracker._state, dtype=np.float32)
                                    _blended = (f5_alpha * _f5_bbox
                                                + (1.0 - f5_alpha) * _ai_st)
                                    tracker.set_state(_blended)

                if tel_writer is not None:
                    ai_obs = observation.bbox
                    ai_row = ([float(ai_obs[0]), float(ai_obs[1]),
                               float(ai_obs[2]), float(ai_obs[3])]
                              if ai_obs is not None else [float("nan")] * 4)
                    pred4 = [float(predicted_state[0]), float(predicted_state[1]),
                             float(predicted_state[2]), float(predicted_state[3])]
                    gmc_ok = int(gmc_estimator.last_ok) if gmc_estimator is not None else -1
                    gmc_inl = int(gmc_estimator.last_inliers) if gmc_estimator is not None else -1
                    state_name = str(track_state).split(".")[-1]
                    tel_writer.writerow([
                        seq_id, frame_idx,
                        *ai_row, float(conf),
                        *pred4,
                        float(step.innovation_norm), float(step.mahal_d2),
                        step.gate_decision, float(step.alpha),
                        mu_cv, mu_ca, mu_singer,
                        gmc_ok, gmc_inl, state_name, int(refresh_fired),
                        float(bbox[0]), float(bbox[1]),
                        float(bbox[2]), float(bbox[3]),
                    ])

                # ── EMA template update ──────────────────────────────────────
                if (
                    ema_template_enabled
                    and not refresh_fired
                    and step.accepted_measurement
                    and conf >= ema_conf_min
                    and step.mahal_d2 <= ema_mahal_thr_sq
                    and frame_idx % ema_interval == 0
                    and hasattr(tracker, "update_template_ema")
                ):
                    tracker.update_template_ema(
                        frame_rgb,
                        np.array(step.bbox, dtype=np.float32),
                        ema_alpha,
                    )
                # ── Particle Filter blend ────────────────────────────────────
                if pf is not None:
                    if step.should_coast:
                        if not _pf_was_coasting:
                            # Transition into coasting — initialize PF
                            _kf_vel = (
                                np.array(step.state, dtype=np.float32).flatten()[4:6]
                                if step.state is not None
                                else np.zeros(2, dtype=np.float32)
                            )
                            pf.init(last_good_bbox, velocity=_kf_vel)
                            _pf_coast_frames = 0
                        _pf_coast_frames += 1
                        pf.predict(process_noise_std=2.0)
                        if _obs_bbox is not None:
                            pf.update(_obs_bbox)
                            pf.resample()
                        # Gradually blend KF → PF as coasting deepens
                        _pf_w = min(
                            1.0, float(_pf_coast_frames) / float(max(pf_trust_ramp, 1))
                        )
                        _pf_est = pf.estimate()
                        bbox = (
                            (1.0 - _pf_w) * np.array(bbox, dtype=np.float32)
                            + _pf_w * _pf_est
                        ).tolist()
                        _pf_was_coasting = True
                    else:
                        _pf_was_coasting = False
                        _pf_coast_frames = 0
                # ─────────────────────────────────────────────────────────────
            else:
                ai_bbox, conf = tracker.track(frame_rgb)
                bbox = ai_bbox.tolist() if isinstance(ai_bbox, np.ndarray) else list(ai_bbox)
            pred_bboxes.append(bbox)
            prev_output_bbox = bbox
            if gmc_enabled or lk_jitter_enabled:
                if curr_gray is None:
                    curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                prev_frame_gray = curr_gray
            if gmc_enabled and gmc_freeze.countdown > 0:
                gmc_freeze.countdown -= 1
            del frame_bgr, frame_rgb
        frame_idx += 1

    cap.release()
    if tel_fh is not None:
        tel_fh.close()
    # Agresif bellek temizliği — her sekans sonrası
    del prev_frame_gray
    gc.collect()
    return pred_bboxes


def success_auc(ious):
    thresholds = np.arange(0, 1.05, 0.05)
    curve = [np.mean(ious >= t) for t in thresholds]
    return np.mean(curve)


def norm_prec_auc(dists, gt_diags):
    nd = dists / np.maximum(gt_diags, 1e-6)
    thresholds = np.arange(0, 0.51, 0.01)
    curve = [np.mean(nd <= t) for t in thresholds]
    return np.mean(curve)


def evaluate(gt_bboxes, pred_bboxes):
    n = min(len(gt_bboxes), len(pred_bboxes))
    ious, dists, diags = [], [], []
    for i in range(n):
        g, p = gt_bboxes[i], pred_bboxes[i]
        if g[2] <= 0 or g[3] <= 0:
            continue
        ious.append(compute_iou(g, p))
        dists.append(compute_center_distance(g, p))
        diags.append(np.sqrt(g[2] ** 2 + g[3] ** 2))
    if not ious:
        return 0, 0
    return success_auc(np.array(ious)), norm_prec_auc(np.array(dists), np.array(diags))


def main():
    parser = argparse.ArgumentParser(description="A/B test: AI-only vs AI+IMM")
    parser.add_argument("--all", action="store_true",
                        help="Run on ALL 255 train sequences (default: 20 subset)")
    parser.add_argument("--mode", default="baseline",
                        choices=["baseline", "open_loop", "coast_only", "velocity_shift", "ai_lead"],
                        help="KF feedback mode")
    parser.add_argument("--gmc", action="store_true",
                        help="Enable Global Motion Compensation")
    parser.add_argument("--adaptive-r", action="store_true",
                        help="Enable adaptive measurement noise scaling")
    parser.add_argument("--imm-config", default=None,
                        help="Path to IMM tuned config YAML (e.g. configs/imm_tuned.yaml). "
                             "Applies q_scale/R/π and decision.* overrides to the IMM column.")
    parser.add_argument("--search-scale-boost", type=float, default=None,
                        help="Singer-adaptive search window boost factor (overrides YAML value). "
                             "0.0 = disabled (default), >0 expands bbox by (1 + boost * singer_prob).")
    parser.add_argument("--refresh-patience", type=int, default=None,
                        help="Proactive template refresh patience (Faz D). "
                             "0 = disabled, N = refresh after N consecutive moderate-conf frames.")
    parser.add_argument("--seq", action="append", default=None, metavar="SEQ",
                        help="Run only this sequence ID (e.g. dataset5/bike3). Repeatable. Overrides --all.")
    parser.add_argument("--f5-feedback", action="store_true",
                        help="Enable F5 closed-loop feedback (tracker.set_state every frame "
                             "in ai_lead mode). Default OFF matches pre-F5 prod behaviour.")
    parser.add_argument("--log-telemetry", default=None,
                        help="Write per-frame integration telemetry CSV.gz into DIR/<variant>/<seq>.csv.gz.")
    parser.add_argument("--variant", default=None,
                        help="Variant tag for telemetry output subdir (e.g. V1, V2, V3).")
    # ── LK Jitter Smoother ───────────────────────────────────────────────────
    parser.add_argument("--lk-jitter", action="store_true",
                        help="Enable LK optical flow jitter smoother for AI observation.")
    parser.add_argument("--lk-alpha", type=float, default=0.5,
                        help="LK smoother center blend weight (1.0=all-AI, 0.0=all-LK; default 0.5).")
    parser.add_argument("--lk-max-corners", type=int, default=15,
                        help="Max Shi-Tomasi corners tracked inside bbox (default 15).")
    # ── EMA Template Update ──────────────────────────────────────────────────
    parser.add_argument("--ema-template", action="store_true",
                        help="Enable EMA (slow leaky) template update on high-confidence frames.")
    parser.add_argument("--ema-alpha", type=float, default=0.05,
                        help="EMA new-frame weight per accepted frame (default 0.05).")
    parser.add_argument("--ema-conf-min", type=float, default=0.8,
                        help="Min AI confidence required for EMA update (default 0.8).")
    parser.add_argument("--ema-mahal-thr", type=float, default=3.0,
                        help="Max Mahalanobis distance (sqrt) for EMA gate (default 3.0).")
    parser.add_argument("--ema-interval", type=int, default=5,
                        help="EMA update every N accepted frames (default 5).")
    # ── Particle Filter ──────────────────────────────────────────────────────
    parser.add_argument("--particle-filter", action="store_true",
                        help="Enable Particle Filter blend during coasting/occlusion.")
    parser.add_argument("--pf-particles", type=int, default=500,
                        help="Number of PF particles (default 500).")
    parser.add_argument("--pf-trust-ramp", type=int, default=10,
                        help="Frames to ramp PF trust from 0→1 during coasting (default 10).")
    # ── Regime-Adaptive System ───────────────────────────────────────────────────────────────────
    parser.add_argument("--regime-adaptive", action="store_true",
                        help="Enable runtime regime detection for sequence-adaptive parameters. "
                             "4 regimes: SMALL_TARGET / FAST_MANEUVER / OCCLUSION / DEFAULT.")
    args = parser.parse_args()
    tags = [f"Mode: {args.mode}"]
    if args.gmc:
        tags.append("GMC: ON")
    if args.adaptive_r:
        tags.append("AdaptiveR: ON")
    imm_cfg = None
    if args.imm_config:
        imm_cfg = load_yaml_config(args.imm_config)
        tags.append(f"IMM: {os.path.basename(args.imm_config)}")
    # search_scale_boost: CLI overrides YAML; YAML overrides default 0.0
    _fp_preview = _load_filter_params(args.imm_config) if args.imm_config else _load_filter_params()
    _boost_from_yaml = float(_fp_preview.get("search_scale_boost", 0.0))
    search_scale_boost = args.search_scale_boost if args.search_scale_boost is not None else _boost_from_yaml
    if search_scale_boost > 0:
        tags.append(f"SearchBoost: {search_scale_boost:.2f}")
    # refresh_patience: CLI overrides YAML; YAML overrides default 0
    _refresh_from_yaml = int(_fp_preview.get("refresh_patience", 0))
    refresh_patience = args.refresh_patience if args.refresh_patience is not None else _refresh_from_yaml
    if refresh_patience > 0:
        tags.append(f"RefreshPatience: {refresh_patience}")
    # f5_feedback: CLI --f5-feedback enables it; YAML f5_feedback: true also enables it
    _f5_from_yaml = bool(_fp_preview.get("f5_feedback", False))
    f5_feedback = args.f5_feedback or _f5_from_yaml
    if f5_feedback:
        tags.append("F5: ON")
    # ── new feature flags ────────────────────────────────────────────────────
    lk_jitter_enabled = args.lk_jitter
    if lk_jitter_enabled:
        tags.append(f"LK-Jitter: alpha={args.lk_alpha:.2f}")
    ema_template_enabled = args.ema_template
    if ema_template_enabled:
        tags.append(f"EMA: alpha={args.ema_alpha:.3f}")
    pf_enabled = args.particle_filter
    if pf_enabled:
        tags.append(f"PF: n={args.pf_particles} ramp={args.pf_trust_ramp}")
    regime_adaptive = args.regime_adaptive
    if regime_adaptive:
        tags.append("RegimeAdaptive: ON")
    print(", ".join(tags))

    manifest = load_manifest()
    tracker = TRTTrackWrapper()

    if args.seq:
        seq_ids = args.seq
    elif args.all:
        seq_ids = sorted(manifest["train"].keys())
    else:
        seq_ids = SUBSET

    header = (f"{'Sequence':<38} {'AUC_raw':>8} {'AUC_imm':>8} {'dAUC':>7} "
              f"{'NP_raw':>8} {'NP_imm':>8} {'dNP':>7}")
    print(header)
    print("-" * 100)

    auc_raws, auc_imms, np_raws, np_imms = [], [], [], []

    total = len(seq_ids)
    for idx, seq_id in enumerate(seq_ids, 1):
        seq_info = manifest["train"][seq_id]
        gt = load_gt(seq_id, manifest)

        preds_raw = run_sequence(tracker, seq_id, seq_info, manifest, use_kf=False)
        auc_r, np_r = evaluate(gt, preds_raw)
        del preds_raw

        preds_imm = run_sequence(tracker, seq_id, seq_info, manifest, use_kf=True,
                                  kf_mode=args.mode, gmc_enabled=args.gmc,
                                  adaptive_r_enabled=args.adaptive_r,
                                  imm_cfg=imm_cfg, imm_cfg_path=args.imm_config,
                                  search_scale_boost=search_scale_boost,
                                  refresh_patience=refresh_patience,
                                  f5_feedback=f5_feedback,
                                  log_telemetry_path=args.log_telemetry,
                                  variant=args.variant,
                                  lk_jitter_enabled=lk_jitter_enabled,
                                  lk_alpha=args.lk_alpha,
                                  lk_max_corners=args.lk_max_corners,
                                  ema_template_enabled=ema_template_enabled,
                                  ema_alpha=args.ema_alpha,
                                  ema_conf_min=args.ema_conf_min,
                                  ema_mahal_thr_sq=args.ema_mahal_thr ** 2,
                                  ema_interval=args.ema_interval,
                                  pf_enabled=pf_enabled,
                                  pf_n_particles=args.pf_particles,
                                  pf_trust_ramp=args.pf_trust_ramp,
                                  regime_adaptive=regime_adaptive)
        auc_i, np_i = evaluate(gt, preds_imm)
        del preds_imm, gt

        # Her sekans sonrası RAM + VRAM temizliği
        gc.collect()
        torch.cuda.empty_cache()

        d_auc = auc_i - auc_r
        d_np = np_i - np_r

        auc_raws.append(auc_r)
        auc_imms.append(auc_i)
        np_raws.append(np_r)
        np_imms.append(np_i)

        marker = "+" if d_auc > 0.005 else ("-" if d_auc < -0.005 else "=")
        print(f"[{idx:>3}/{total}] {seq_id:<38} {auc_r:>8.3f} {auc_i:>8.3f} {d_auc:>+7.3f}{marker} "
              f"{np_r:>8.3f} {np_i:>8.3f} {d_np:>+7.3f}")

    print("-" * 100)
    ar = np.mean(auc_raws)
    ai = np.mean(auc_imms)
    nr = np.mean(np_raws)
    ni = np.mean(np_imms)
    print(f"{'MEAN':<38} {ar:>8.3f} {ai:>8.3f} {ai - ar:>+7.3f}  "
          f"{nr:>8.3f} {ni:>8.3f} {ni - nr:>+7.3f}")
    print()
    sr = 0.6 * ar + 0.4 * nr
    si = 0.6 * ai + 0.4 * ni
    print(f"FinalScore (raw):  {sr:.4f}")
    print(f"FinalScore (IMM):  {si:.4f}")
    print(f"Delta:             {si - sr:+.4f}")
    n_better = sum(1 for a, b in zip(auc_raws, auc_imms) if b > a + 0.005)
    n_worse = sum(1 for a, b in zip(auc_raws, auc_imms) if b < a - 0.005)
    n_same = len(auc_raws) - n_better - n_worse
    print(f"\nIMM better: {n_better}/{total}, "
          f"worse: {n_worse}/{total}, same: {n_same}/{total}")


if __name__ == "__main__":
    main()
