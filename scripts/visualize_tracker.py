#!/usr/bin/env python3
"""Visualize tracker output on contest_release videos.

Each output video shows:
  - GT bbox (grey, dashed)
  - KF predicted bbox (cyan)
  - AI observation bbox (green=bypass / yellow=moderate / red=low)
  - Final fused output bbox (white, thick)
  - Trajectory trail: last 80 output-bbox centers
  - Telemetry panel (top-right): conf, state, μ_CV/CA/Singer, gate, α, IoU, live AUC
  - Refresh-fired flash indicator

Usage:
    # Single sequence
    python3 scripts/visualize_tracker.py --seq dataset5/bike3

    # Multiple
    python3 scripts/visualize_tracker.py --seq dataset5/bike3 --seq dataset3/air_conditioning_box2

    # 20-seq subset
    python3 scripts/visualize_tracker.py --all

    # All 255 train sequences
    python3 scripts/visualize_tracker.py --all-full

    # With custom config
    python3 scripts/visualize_tracker.py --seq dataset5/bike3 \\
        --imm-config configs/h3_singer_uav.yaml --output-dir output_viz
"""

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build2"))  # active dev build

from tracker.config import load_runtime_config, load_yaml_config
from tracker.trt_wrapper import TRTTrackWrapper
from tracker.data_utils import DATA_ROOT, load_manifest, parse_bbox_line
from tracker.decision import DecisionMaker
from tracker.gmc import GMCEstimator
from tracker.imm_policy import (
    clamp_coast_velocity,
    observe_with_guidance,
    step_guided_imm,
)
from tracker.metrics import compute_iou
import tracker_cpp
import numpy as np
import cv2
import gc
import torch

# ─── colours (BGR) ───────────────────────────────────────────────────────────
C_GT          = (160, 160, 160)   # grey        — ground truth
C_KF          = (255, 200,  50)   # cyan-gold   — KF predicted
C_AI_BYPASS   = ( 40, 255,  60)   # bright green — conf > bypass_thr
C_AI_TRACK    = ( 60, 180,  60)   # green        — normal accepted
C_AI_MOD      = (  0, 200, 255)   # yellow       — moderate zone (Faz D)
C_AI_LOW      = (  0,  80, 220)   # red-orange   — low / coasting
C_OUTPUT      = (255, 255, 255)   # white        — final fused bbox
C_TRAIL_BASE  = ( 60, 255, 160)   # mint         — trajectory trail tip
C_TEXT        = (255, 255, 255)
C_TEXT_SHADOW = (  0,   0,   0)
C_REFRESH_BG  = (  0,  40, 220)   # red bg flash on refresh

TRAIL_LEN = 80  # history points to draw as trajectory
PANEL_W   = 270  # telemetry panel width (px)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FILTER_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "tracker_config.yaml")

SUBSET_20 = [
    "dataset1/plane", "dataset1/surfer", "dataset1/volleyball",
    "dataset2/Girl2", "dataset2/Gull1", "dataset2/Kiting",
    "dataset2/ManRunning2", "dataset2/RcCar3", "dataset2/Surfing12",
    "dataset2/Wakeboarding2", "dataset3/air_conditioning_box2",
    "dataset3/basketball_player4-n", "dataset3/duck1_1",
    "dataset3/truck_night", "dataset4/car6", "dataset5/bike3",
    "dataset5/building2", "dataset5/car1_3", "dataset5/car1_s",
    "dataset5/person2_2",
]


# ─── drawing helpers ─────────────────────────────────────────────────────────

def _txt(frame, text, xy, scale=0.48, thickness=1, color=C_TEXT):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, C_TEXT_SHADOW, thickness + 2)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


def draw_bbox(frame, xywh, color, label="", thickness=2):
    x, y, w, h = int(xywh[0]), int(xywh[1]), int(xywh[2]), int(xywh[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        _txt(frame, label, (x, max(y - 5, 14)), scale=0.43, color=color)


def draw_dashed_rect(frame, xywh, color, gap=6, thickness=1):
    """Draw a dashed rectangle (GT visualization)."""
    x, y, w, h = int(xywh[0]), int(xywh[1]), int(xywh[2]), int(xywh[3])
    segments = [
        ((x, y),     (x + w, y)),
        ((x + w, y), (x + w, y + h)),
        ((x + w, y + h), (x, y + h)),
        ((x, y + h), (x, y)),
    ]
    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        dist = int(np.hypot(dx, dy))
        if dist == 0:
            continue
        for i in range(0, dist, gap * 2):
            t0 = i / dist
            t1 = min((i + gap) / dist, 1.0)
            p0 = (int(x1 + dx * t0), int(y1 + dy * t0))
            p1 = (int(x1 + dx * t1), int(y1 + dy * t1))
            cv2.line(frame, p0, p1, color, thickness)


def draw_trail(frame, trail):
    """Draw trajectory as circles fading from dim (old) to bright (new)."""
    n = len(trail)
    for i, (cx, cy) in enumerate(trail):
        frac = (i + 1) / max(n, 1)
        r = max(1, int(4 * frac))
        c = tuple(int(v * frac) for v in C_TRAIL_BASE)
        cv2.circle(frame, (int(cx), int(cy)), r, c, -1)


def draw_telemetry_panel(frame, *, frame_idx, n_frames, seq_name,
                          conf, conf_bypass_thr, bypass_active, state_name,
                          mu_cv, mu_ca, mu_singer, gate, alpha_val,
                          coast_count, refresh_fired, iou_now, running_auc,
                          ai_ms, kf_ms, total_ms, fps_live, gflops):
    fh, fw = frame.shape[:2]
    ph = 260  # taller to fit timing rows
    px = fw - PANEL_W - 6
    py = 6

    # semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + PANEL_W, py + ph), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    state_clr = {
        "TRACKING": (80, 255, 80),
        "COASTING": (0, 200, 255),
        "LOST":     (0, 60, 220),
    }.get(state_name, C_TEXT)

    conf_clr = (C_AI_BYPASS if bypass_active
                else C_AI_MOD if conf >= 0.35
                else C_AI_LOW)

    # FPS color: green ≥30, yellow ≥15, red <15
    fps_clr = ((80, 255, 80) if fps_live >= 30 else
               (0, 200, 255) if fps_live >= 15 else
               (0, 80, 220))

    x0, y0, dy = px + 8, py + 18, 21
    _txt(frame, seq_name,                          (x0, y0),       scale=0.50, color=(210, 230, 255))
    _txt(frame, f"Frame {frame_idx} / {n_frames}", (x0, y0+dy),   scale=0.43, color=(180, 180, 180))
    _txt(frame, f"Conf:  {conf:.3f}{'  BYPASS' if bypass_active else ''}",
                                                    (x0, y0+2*dy), scale=0.46, color=conf_clr)
    _txt(frame, f"State: {state_name}  coast={coast_count}",
                                                    (x0, y0+3*dy), scale=0.46, color=state_clr)
    _txt(frame, f"CV={mu_cv:.2f}  CA={mu_ca:.2f}  S={mu_singer:.2f}",
                                                    (x0, y0+4*dy), scale=0.43, color=(210, 170, 255))
    _txt(frame, f"Gate: {gate}   alpha={alpha_val:.2f}",
                                                    (x0, y0+5*dy), scale=0.43, color=(190, 220, 200))
    _txt(frame, f"IoU={iou_now:.3f}   AUC≈{running_auc:.3f}",
                                                    (x0, y0+6*dy), scale=0.46, color=(255, 220, 80))

    # ── timing / performance rows ────────────────────────────────────────────
    _txt(frame, f"AI  {ai_ms:5.1f} ms   KF  {kf_ms:4.1f} ms",
                                                    (x0, y0+7*dy), scale=0.43, color=(170, 215, 255))
    _txt(frame, f"Frame {total_ms:5.1f} ms",        (x0, y0+8*dy), scale=0.43, color=(170, 215, 255))
    _txt(frame, f"FPS {fps_live:5.1f}   {gflops:.2f} GFLOPs",
                                                    (x0, y0+9*dy), scale=0.46, color=fps_clr)

    # Refresh flash
    if refresh_fired:
        cv2.rectangle(frame, (px, py + ph - 22), (px + PANEL_W, py + ph), C_REFRESH_BG, -1)
        _txt(frame, "  REFRESH FIRED", (px + 8, py + ph - 6), scale=0.48, color=(255, 255, 255))

    # Confidence bar
    bar_y = py + ph - (24 if refresh_fired else 10)
    bar_x0, bar_x1 = px + 8, px + PANEL_W - 8
    bw = bar_x1 - bar_x0
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + 9), (50, 50, 50), -1)
    fill = int(bw * min(max(conf, 0.0), 1.0))
    if fill > 0:
        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + fill, bar_y + 9), conf_clr, -1)
    # bypass threshold line
    bt_x = bar_x0 + int(bw * conf_bypass_thr)
    cv2.line(frame, (bt_x, bar_y - 1), (bt_x, bar_y + 10), (255, 255, 0), 1)


def draw_legend(frame):
    """Fixed legend in bottom-left corner."""
    items = [
        (C_GT,        "GT (dashed)"),
        (C_KF,        "KF predicted"),
        (C_AI_BYPASS, "AI (bypass)"),
        (C_AI_MOD,    "AI (moderate)"),
        (C_AI_LOW,    "AI (low/coast)"),
        (C_OUTPUT,    "Output (final)"),
        (C_TRAIL_BASE,"Trajectory"),
    ]
    fh = frame.shape[0]
    y0 = fh - len(items) * 20 - 8
    for i, (clr, label) in enumerate(items):
        y = y0 + i * 20
        cv2.rectangle(frame, (8, y - 10), (22, y + 2), clr, -1)
        _txt(frame, label, (26, y), scale=0.42, color=C_TEXT)


# ─── main per-sequence visualizer ────────────────────────────────────────────

def _load_fp(imm_cfg_path):
    return load_runtime_config(imm_cfg_path) if imm_cfg_path else load_runtime_config(_FILTER_CONFIG)


def _apply_imm_config(kf, cfg):
    imm = cfg.get("imm", {})
    if not imm:
        return
    q_scale = imm.get("q_scale", 1.0)
    mn = imm.get("measurement_noise", {})
    tm = imm.get("transition_matrix", {})

    # Singer F matrix must be set BEFORE Q so build_singer_fq() doesn't
    # overwrite the Optuna-tuned Q that set_model_process_noise will apply.
    singer_pre = cfg.get("singer", {}) or {}
    if hasattr(kf, "set_singer_params"):
        kf.set_singer_params(
            float(singer_pre.get("alpha", 1.0)),
            float(singer_pre.get("sigma2_a", 25.0))
        )  # builds F[Singer]; Q overwritten by set_model_process_noise below

    MODEL_Q_BASES = {
        0: np.array([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.0001, 1e-6, 1e-6],   dtype=np.float32),
        1: np.array([1.0, 1.0, 1.0, 1.0, 0.1,  0.1,  0.0001, 0.0001, 1.0,  1.0],    dtype=np.float32),
        2: np.array([1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  0.001,  0.001, 100.0, 100.0],  dtype=np.float32),
    }
    for m in range(3):
        Q = np.zeros((10, 10), dtype=np.float32)
        np.fill_diagonal(Q, MODEL_Q_BASES[m] * q_scale)
        kf.set_model_process_noise(m, Q)
    r_pos_scale  = mn.get("r_pos_scale",  1.0)
    r_size_scale = mn.get("r_size_scale", 4.0)
    R = np.diag(np.array([1.0 * r_pos_scale, 1.0 * r_pos_scale,
                          10.0 * r_size_scale, 10.0 * r_size_scale], dtype=np.float32))
    kf.set_measurement_noise(R)
    pi_persist = tm.get("pi_persist", 0.90)
    off = (1.0 - pi_persist) / 2.0
    pi  = np.array([[pi_persist, off, off],
                    [off, pi_persist, off],
                    [off, off, pi_persist]], dtype=np.float32)
    kf.set_transition_matrix(pi)

    # Singer AR-Q sensitivity (Dynamic Q boost from aspect-ratio change)
    ar_q_cfg = cfg.get("singer", {}) or {}
    ar_q_sens = float(ar_q_cfg.get("ar_q_sensitivity", 0.0))
    ar_q_cap  = float(ar_q_cfg.get("ar_q_boost_cap", 3.0))
    if ar_q_sens > 0.0 and hasattr(kf, "set_ar_q_sensitivity"):
        kf.set_ar_q_sensitivity(ar_q_sens, ar_q_cap)


def success_auc(ious):
    thr = np.arange(0, 1.05, 0.05)
    return float(np.mean([np.mean(ious >= t) for t in thr]))


def visualize_sequence(tracker, seq_id, seq_info, manifest,
                        imm_cfg, imm_cfg_path, output_dir,
                        gmc_enabled=True, adaptive_r_enabled=True):
    """Run full IMM pipeline on one sequence and write annotated video."""
    fp = _load_fp(imm_cfg_path)
    video_path = os.path.join(DATA_ROOT, seq_info["video_path"])

    # Load GT annotations
    ann_path = seq_info.get("annotation_path")
    annotations = []
    if ann_path:
        full_ann = ann_path if os.path.isabs(ann_path) else os.path.join(DATA_ROOT, ann_path)
        with open(full_ann) as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(parse_bbox_line(line))
    if not annotations:
        print(f"  [SKIP] No annotations for {seq_id}")
        return

    init_bbox = annotations[0]

    # ── build filter stack ──────────────────────────────────────────────────
    kf = tracker_cpp.IMMFilter()
    if imm_cfg:
        _apply_imm_config(kf, imm_cfg)

    if adaptive_r_enabled and imm_cfg:
        ar = imm_cfg.get("adaptive_r", {})
        kf.set_adaptive_r_floor(float(ar.get("floor", 0.4)))
        if hasattr(kf, "set_adaptive_r_cap"):
            kf.set_adaptive_r_cap(float(ar.get("cap", 10.0)))

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

    gmc_estimator = None
    prev_frame_gray = None
    _gmc_freeze_on_veto = True
    _gmc_freeze_after = 1
    _gmc_maneuver_freeze_countdown = 0
    if gmc_enabled:
        gmc_cfg = (imm_cfg.get("gmc", {}) if imm_cfg else {}) or {}
        gmc_estimator = GMCEstimator(
            n_features=int(gmc_cfg.get("n_features", 200)),
            inlier_ratio_threshold=float(gmc_cfg.get("inlier_ratio_threshold", 0.3)),
            min_matches=int(gmc_cfg.get("min_matches", 6)),
            downsample=float(gmc_cfg.get("downsample", 0.5)),
            quality_enabled=bool(gmc_cfg.get("quality_enabled", True)),
            veto_inlier_ratio=float(gmc_cfg.get("veto_inlier_ratio", 0.2)),
            borderline_inlier_ratio=float(gmc_cfg.get("borderline_inlier_ratio", 0.3)),
            max_translation_frac_diag=float(gmc_cfg.get("max_translation_frac_diag", 0.08)),
            max_rotation_deg=float(gmc_cfg.get("max_rotation_deg", 12.0)),
            history_window=int(gmc_cfg.get("history_window", 5)),
            history_outlier_mult=float(gmc_cfg.get("history_outlier_mult", 3.0)),
        )
        kf.set_gmc_q_boost(float(gmc_cfg.get("fail_q_boost", 4.0)))
        _gmc_freeze_on_veto = bool(gmc_cfg.get("freeze_maneuver_on_veto", True))
        _gmc_freeze_after = int(gmc_cfg.get("freeze_frames_after_veto", 1))
        _gmc_maneuver_freeze_countdown = 0

    CONF_BYPASS_THR = float(fp.get("conf_bypass_threshold", 0.74))
    CONF_REFRESH_LOW  = float(fp.get("conf_refresh_low",  0.35))
    CONF_REFRESH_HIGH = float(fp.get("conf_refresh_high", 0.65))
    REFRESH_PATIENCE  = int(fp.get("refresh_patience", 0))
    REFRESH_DECLINE   = float(fp.get("refresh_decline_thr", 0.04))
    VELOCITY_ANCHOR   = float(fp.get("velocity_anchor_max", 0.0))
    REINIT_AFTER      = int(fp["reinit_after"])
    search_scale_boost = float(fp.get("search_scale_boost", 0.0))

    # ── video I/O ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(seq_info.get("native_fps", cap.get(cv2.CAP_PROP_FPS) or 30))
    n_frames = seq_info["n_frames"]

    seq_slug = seq_id.replace("/", "__")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{seq_slug}.mp4")
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    # ── per-frame state ──────────────────────────────────────────────────────
    trail = collections.deque(maxlen=TRAIL_LEN)
    ious  = []
    frame_idx = 0
    prev_output_bbox  = init_bbox
    last_good_bbox    = init_bbox
    reject_streak     = 0
    _moderate_conf_streak = 0
    _streak_start_conf    = 1.0
    # timing accumulators (moving window for stable FPS display)
    _fps_window: collections.deque = collections.deque(maxlen=30)  # last 30 frame durations (ms)
    _ai_ms_last   = 0.0
    _kf_ms_last   = 0.0
    _total_ms_last = 0.0
    # SGLATrack GFLOPs: search 256×256 + template 128×128 → ~3.6 GFLOPs per frame
    # (measured with fvcore on the PyTorch model; TRT FP16 uses same graph)
    _GFLOPS_MODEL = 3.63

    print(f"  [{seq_id}]  {n_frames} frames → {out_path}")

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= n_frames:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── get GT for this frame ────────────────────────────────────────────
        gt_bbox = annotations[frame_idx] if frame_idx < len(annotations) else None

        # ── init frame ───────────────────────────────────────────────────────
        if frame_idx == 0:
            init_arr = np.array(init_bbox, dtype=np.float32)
            tracker.init(frame_rgb, init_arr)
            kf.init(init_arr)
            sm.force_tracking()
            if gmc_enabled:
                prev_frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # draw init frame
            vis = frame_bgr.copy()
            if gt_bbox:
                draw_dashed_rect(vis, gt_bbox, C_GT)
            draw_bbox(vis, init_bbox, C_OUTPUT, "INIT", thickness=2)
            cx = init_bbox[0] + init_bbox[2] / 2
            cy = init_bbox[1] + init_bbox[3] / 2
            trail.append((cx, cy))
            draw_trail(vis, trail)
            draw_telemetry_panel(vis, frame_idx=0, n_frames=n_frames,
                seq_name=seq_id.split("/")[-1],
                conf=1.0, conf_bypass_thr=CONF_BYPASS_THR, bypass_active=True,
                state_name="TRACKING", mu_cv=1.0, mu_ca=0.0, mu_singer=0.0,
                gate="INIT", alpha_val=1.0, coast_count=0, refresh_fired=False,
                iou_now=1.0, running_auc=1.0,
                ai_ms=0.0, kf_ms=0.0, total_ms=0.0, fps_live=0.0,
                gflops=_GFLOPS_MODEL)
            draw_legend(vis)
            writer.write(vis)

            del frame_bgr, frame_rgb
            frame_idx += 1
            continue

        # ── GMC ──────────────────────────────────────────────────────────────
        curr_gray = None
        gmc_quality_state = "good"
        if gmc_enabled and gmc_estimator is not None:
            curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if (gmc_enabled and gmc_estimator is not None
                and prev_frame_gray is not None and curr_gray is not None):
            kf_bbox = np.array(kf.get_state()).flatten()[:4].astype(np.float32)
            H, gmc_quality = gmc_estimator.estimate_with_quality(prev_frame_gray, curr_gray, kf_bbox)
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

        # ── KF predict ───────────────────────────────────────────────────────
        _t_kf0 = time.perf_counter()
        predicted_state = np.array(kf.predict()).flatten()

        # ── IMM model probabilities ───────────────────────────────────────────
        mu_cv = mu_ca = mu_singer = p_maneuver = 0.0
        if hasattr(kf, "get_model_probabilities"):
            _mu    = np.array(kf.get_model_probabilities())
            mu_cv  = float(_mu[0])
            mu_ca  = float(_mu[1])
            mu_singer = float(_mu[2])
            p_maneuver = mu_ca + mu_singer
        gmc_suppresses_maneuver = (
            gmc_quality_state != "good"
            or _gmc_maneuver_freeze_countdown > 0
        )
        _kf_predict_ms = (time.perf_counter() - _t_kf0) * 1000.0

        # ── AI inference ─────────────────────────────────────────────────────
        _t_ai0 = time.perf_counter()
        observation = observe_with_guidance(
            tracker, frame_rgb,
            clamp_coast_velocity(predicted_state, VELOCITY_ANCHOR)
            if VELOCITY_ANCHOR > 0 and reject_streak > 0 else predicted_state,
            mode="ai_lead",
            last_output_bbox=prev_output_bbox,
            singer_prob=mu_singer,
            search_scale_boost=search_scale_boost,
        )
        conf = observation.confidence
        track_state = sm.step(conf)
        _ai_ms_last = (time.perf_counter() - _t_ai0) * 1000.0

        # ── decision + KF update ─────────────────────────────────────────────
        _t_kf1 = time.perf_counter()
        step = step_guided_imm(
            kf, dec, predicted_state,
            observation.bbox, conf, fw, fh,
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
            maneuver_pi_enabled=(
                bool(fp.get("maneuver_pi_enabled", False))
                and not gmc_suppresses_maneuver
            ),
            maneuver_pi_thr=float(fp.get("maneuver_pi_thr", 16.0)),
            maneuver_pi_persist=float(fp.get("maneuver_pi_persist", 0.72)),
            maneuver_pi_singer_boost=float(fp.get("maneuver_pi_singer_boost", 0.20)),
            normal_pi_persist=float(fp.get("normal_pi_persist", 0.96)),
        )
        bbox          = step.bbox
        reject_streak = step.reject_streak
        last_good_bbox = step.last_good_bbox
        _kf_ms_last = _kf_predict_ms + (time.perf_counter() - _t_kf1) * 1000.0

        # ── Faz D: proactive refresh ──────────────────────────────────────────
        refresh_fired = False
        if reject_streak >= 5 and step.should_coast:
            tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
            _moderate_conf_streak = 0
            refresh_fired = True
        else:
            if REFRESH_PATIENCE > 0:
                if step.accepted_measurement and CONF_REFRESH_LOW <= conf <= CONF_REFRESH_HIGH:
                    if _moderate_conf_streak == 0:
                        _streak_start_conf = conf
                    _moderate_conf_streak += 1
                    if (_moderate_conf_streak >= REFRESH_PATIENCE
                            and conf < _streak_start_conf - REFRESH_DECLINE):
                        tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
                        refresh_fired = True
                        _moderate_conf_streak = 0
                else:
                    _moderate_conf_streak = 0

        # ── metrics ──────────────────────────────────────────────────────────
        iou_now = compute_iou(gt_bbox, bbox) if gt_bbox and gt_bbox[2] > 0 else 0.0
        ious.append(iou_now)
        running_auc = success_auc(np.array(ious)) if len(ious) >= 5 else float(np.mean(ious))

        # ── timing ───────────────────────────────────────────────────────────
        _total_ms_last = _ai_ms_last + _kf_ms_last  # GMC included implicitly in AI observe
        _fps_window.append(_total_ms_last)
        _fps_avg_ms = float(np.mean(_fps_window)) if _fps_window else 1.0
        _fps_live   = 1000.0 / max(_fps_avg_ms, 0.1)

        # ── state strings ─────────────────────────────────────────────────────
        state_str = str(track_state).split(".")[-1]
        bypass_active = conf > CONF_BYPASS_THR

        # ── ai bbox colour ────────────────────────────────────────────────────
        ai_clr = (C_AI_BYPASS if bypass_active
                  else C_AI_MOD if CONF_REFRESH_LOW <= conf <= CONF_REFRESH_HIGH
                  else C_AI_TRACK if step.accepted_measurement
                  else C_AI_LOW)

        # ── render frame ─────────────────────────────────────────────────────
        vis = frame_bgr.copy()

        # trail first (background)
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        trail.append((cx, cy))
        draw_trail(vis, trail)

        # GT
        if gt_bbox and gt_bbox[2] > 0:
            draw_dashed_rect(vis, gt_bbox, C_GT)

        # KF predicted
        draw_bbox(vis, predicted_state[:4], C_KF, "KF", thickness=1)

        # AI observation
        if observation.bbox is not None:
            conf_label = f"AI {conf:.2f}"
            draw_bbox(vis, observation.bbox, ai_clr, conf_label, thickness=2)

        # Final output (thick white)
        draw_bbox(vis, bbox, C_OUTPUT, "", thickness=3)

        # Telemetry panel
        draw_telemetry_panel(vis,
            frame_idx=frame_idx, n_frames=n_frames,
            seq_name=seq_id.split("/")[-1],
            conf=conf, conf_bypass_thr=CONF_BYPASS_THR,
            bypass_active=bypass_active, state_name=state_str,
            mu_cv=mu_cv, mu_ca=mu_ca, mu_singer=mu_singer,
            gate=step.gate_decision, alpha_val=float(step.alpha),
            coast_count=sm.coast_count(), refresh_fired=refresh_fired,
            iou_now=iou_now, running_auc=running_auc,
            ai_ms=_ai_ms_last, kf_ms=_kf_ms_last,
            total_ms=_total_ms_last, fps_live=_fps_live,
            gflops=_GFLOPS_MODEL)

        draw_legend(vis)
        writer.write(vis)

        # ── bookkeeping ───────────────────────────────────────────────────────
        prev_output_bbox = bbox
        if gmc_enabled and curr_gray is not None:
            prev_frame_gray = curr_gray
            if _gmc_maneuver_freeze_countdown > 0:
                _gmc_maneuver_freeze_countdown -= 1
        del frame_bgr, frame_rgb
        frame_idx += 1

    cap.release()
    writer.release()

    final_auc = success_auc(np.array(ious)) if ious else 0.0
    print(f"    AUC={final_auc:.4f}   saved → {out_path}")

    del prev_frame_gray
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracker: AI + KF + telemetry overlay on video")
    parser.add_argument("--seq", action="append", default=None,
                        help="Sequence ID (e.g. dataset5/bike3). Repeatable.")
    parser.add_argument("--all", action="store_true",
                        help="Run on 20-seq evaluation subset")
    parser.add_argument("--all-full", action="store_true",
                        help="Run on all 255 train sequences")
    parser.add_argument("--public-lb", action="store_true",
                        help="Run on all 89 public_lb sequences (uses MISIR GT annotations)")
    parser.add_argument("--imm-config", default=None,
                        help="IMM YAML config (default: configs/h3_singer_uav.yaml)")
    parser.add_argument("--output-dir", default="output_viz",
                        help="Directory to write annotated MP4 files (default: output_viz/)")
    parser.add_argument("--engine", default=None,
                        help="TensorRT engine path (default: models/sglatrack_fp16.engine)")
    parser.add_argument("--no-gmc", action="store_true",
                        help="Disable GMC")
    parser.add_argument("--no-adaptive-r", action="store_true",
                        help="Disable adaptive-R")
    args = parser.parse_args()

    # Default config
    imm_cfg_path = args.imm_config or os.path.join(
        _PROJECT_ROOT, "configs", "h3_singer_uav.yaml")
    imm_cfg = load_yaml_config(imm_cfg_path)

    manifest = load_manifest()

    MISIR_GT_ROOT = os.path.join(_PROJECT_ROOT, "MISIR PROJESİ", "SOT DATA")

    if args.seq:
        seq_ids = list(args.seq)
    elif args.all_full:
        seq_ids = sorted(manifest["train"].keys())
    elif args.all:
        seq_ids = SUBSET_20
    elif args.public_lb:
        seq_ids = sorted(manifest["public_lb"].keys())
    else:
        # Default: prompt user
        print("No sequence specified. Use --seq SEQID, --all, --all-full, or --public-lb.")
        print("Example:  python3 scripts/visualize_tracker.py --seq dataset5/bike3")
        sys.exit(1)

    gmc_enabled       = not args.no_gmc
    adaptive_r_enabled = not args.no_adaptive_r

    print(f"Config:    {imm_cfg_path}")
    print(f"GMC:       {'ON' if gmc_enabled else 'OFF'}")
    print(f"AdaptiveR: {'ON' if adaptive_r_enabled else 'OFF'}")
    print(f"Output:    {os.path.abspath(args.output_dir)}/")
    print(f"Sequences: {len(seq_ids)}")
    print()

    tracker = TRTTrackWrapper(engine_path=args.engine)

    for i, seq_id in enumerate(seq_ids, 1):
        if seq_id in manifest["train"]:
            seq_info = dict(manifest["train"][seq_id])
        elif seq_id in manifest["public_lb"]:
            seq_info = dict(manifest["public_lb"][seq_id])
            # Override annotation with MISIR full GT
            misir_ann = os.path.join(MISIR_GT_ROOT, seq_id, "tracking_results.txt")
            if os.path.exists(misir_ann):
                seq_info["annotation_path"] = misir_ann
                seq_info["_misir_abs"] = True
            else:
                print(f"  [WARN] No MISIR GT for {seq_id}, using init-only")
        else:
            print(f"  [WARN] {seq_id} not in manifest, skipping")
            continue
        print(f"[{i}/{len(seq_ids)}] {seq_id}")
        try:
            visualize_sequence(
                tracker, seq_id, seq_info, manifest,
                imm_cfg=imm_cfg, imm_cfg_path=imm_cfg_path,
                output_dir=args.output_dir,
                gmc_enabled=gmc_enabled,
                adaptive_r_enabled=adaptive_r_enabled,
            )
        except Exception as e:
            print(f"  [ERROR] {seq_id}: {e}")
            import traceback

            traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Videos saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
