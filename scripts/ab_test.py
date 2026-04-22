#!/usr/bin/env python3
"""A/B test: AI-only vs AI+IMM on a subset of train sequences."""
import argparse
import csv
import gzip
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

from tracker.config import load_runtime_config, load_yaml_config
from tracker.trt_wrapper import TRTTrackWrapper
from tracker.data_utils import DATA_ROOT, load_manifest, parse_bbox_line, load_gt
from tracker.decision import DecisionMaker
from tracker.gmc import GMCEstimator
from tracker.imm_policy import observe_with_guidance, step_guided_imm
from tracker.metrics import compute_iou, compute_center_distance
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

    # Q bases must match replay.py — Optuna optimized against these ratios
    MODEL_Q_BASES = {
        0: np.array([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.0001, 1e-6, 1e-6],
                    dtype=np.float32),
        1: np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.0001, 0.0001, 1.0, 1.0],
                    dtype=np.float32),
        2: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 0.001, 100.0, 100.0],
                    dtype=np.float32),
    }
    for m in range(3):
        Q = np.zeros((10, 10), dtype=np.float32)
        np.fill_diagonal(Q, MODEL_Q_BASES[m] * q_scale)
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


def run_sequence(tracker, seq_id, seq_info, manifest, use_kf, kf_mode="baseline",
                 gmc_enabled=False, adaptive_r_enabled=False,
                 imm_cfg=None, imm_cfg_path=None, search_scale_boost=0.0,
                 refresh_patience=None, f5_feedback=False,
                 log_telemetry_path=None, variant=None):
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

    kf, sm, dec = None, None, None
    gmc_estimator = None
    prev_frame_gray = None
    tracking_state_enum = None
    fp = _load_filter_params(imm_cfg_path) if imm_cfg_path else _load_filter_params()
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
                downsample=float(gmc_cfg.get("downsample", 0.5)),
                force_python=_force_py,
            )
            kf.set_gmc_q_boost(float(gmc_cfg.get("fail_q_boost", 4.0)))
        if adaptive_r_enabled:
            ar_cfg = (imm_cfg.get("adaptive_r", {}) if imm_cfg else {}) or {}
            kf.set_adaptive_r_floor(float(ar_cfg.get("floor", 0.4)))

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
    _streak_start_conf = 1.0   # confidence when the current streak began
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
    REFRESH_DECLINE_THR = float(fp.get("refresh_decline_thr", 0.04))
    # Bbox-area gate: only refresh when predicted bbox area ≥ threshold.
    # Blocks refreshes on small fast targets (e.g. bike: area≈164–230 px²)
    # while allowing them on large targets (e.g. truck: area≈2000–3000 px²).
    # 0 = disabled (backward-compatible default).
    REFRESH_MIN_AREA = float(fp.get("refresh_min_bbox_area", 0.0))

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
            if gmc_enabled:
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
                if gmc_enabled and gmc_estimator is not None:
                    curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if (
                    gmc_enabled
                    and gmc_estimator is not None
                    and prev_frame_gray is not None
                    and curr_gray is not None
                ):
                    kf_bbox = np.array(kf.get_state()).flatten()[:4].astype(np.float32)
                    H, ok = gmc_estimator.estimate(prev_frame_gray, curr_gray, kf_bbox)
                    if ok:
                        kf.apply_gmc(H.astype(np.float32))
                    else:
                        kf.set_gmc_failed(True)
                predicted_state = np.array(kf.predict()).flatten()
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
                observation = observe_with_guidance(
                    tracker,
                    frame_rgb,
                    predicted_state,
                    mode=kf_mode,
                    last_output_bbox=prev_output_bbox,
                    singer_prob=mu_singer,
                    search_scale_boost=search_scale_boost,
                )
                conf = observation.confidence
                track_state = sm.step(conf)
                step = step_guided_imm(
                    kf,
                    dec,
                    predicted_state,
                    observation.bbox,
                    conf,
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
                    maneuver_probability=p_maneuver,
                    maneuver_threshold=float(fp.get("maneuver_threshold", 0.0)),
                    maneuver_bypass_boost=float(fp.get("maneuver_bypass_boost", 0.0)),
                    mahal_bypass_after=int(fp.get("mahal_bypass_after", 5)),
                    coast_count=sm.coast_count(),
                    alpha_gate_k_conf=float(fp.get("alpha_gate_k_conf", 0.0)),
                    alpha_gate_lambda=float(fp.get("alpha_gate_lambda", 0.0)),
                    reacq_r_decay=float(fp.get("reacq_r_decay", 0.0)),
                )
                bbox = step.bbox
                reject_streak = step.reject_streak
                last_good_bbox = step.last_good_bbox
                refresh_fired = False

                if kf_mode in {"baseline", "coast_only", "velocity_shift"}:
                    # Final state becomes the single source of truth for the next frame.
                    tracker.set_state(bbox)
                elif kf_mode == "ai_lead":
                    if reject_streak >= 5 and step.should_coast:
                        # Phase 1 — The Great Rescue: reinit AI template at KF location.
                        # init() renews template + centre (set_state only moved centre).
                        tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
                        _moderate_conf_streak = 0
                        refresh_fired = True
                    else:
                        if REFRESH_PATIENCE > 0:
                            # Faz D — Proactive refresh with declining-confidence gate.
                            # init() fires only when conf declined ≥ REFRESH_DECLINE_THR
                            # from streak start over REFRESH_PATIENCE consecutive accepted
                            # frames. Best result at thr=0.04: Delta=+0.0511.
                            if step.accepted_measurement and CONF_REFRESH_LOW <= conf <= CONF_REFRESH_HIGH:
                                if _moderate_conf_streak == 0:
                                    _streak_start_conf = conf
                                _moderate_conf_streak += 1
                                if (_moderate_conf_streak >= REFRESH_PATIENCE and
                                        conf < _streak_start_conf - REFRESH_DECLINE_THR):
                                    _bboxarea = float(predicted_state[2]) * float(predicted_state[3])
                                    if REFRESH_MIN_AREA > 0 and _bboxarea < REFRESH_MIN_AREA:
                                        pass  # area gate: skip refresh for small targets
                                    else:
                                        tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
                                        refresh_fired = True
                                    _moderate_conf_streak = 0
                            else:
                                _moderate_conf_streak = 0
                        # F5: closed-loop feedback — feed KF-fused bbox back to AI
                        # search window every frame so AI always tracks from the
                        # correct position, not its own stale internal state.
                        # Gated behind --f5-feedback to allow clean A/B against the
                        # prior prod behaviour (no per-frame set_state).
                        if f5_feedback:
                            tracker.set_state(np.array(bbox, dtype=np.float32))

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
            else:
                ai_bbox, conf = tracker.track(frame_rgb)
                bbox = ai_bbox.tolist() if isinstance(ai_bbox, np.ndarray) else list(ai_bbox)
            pred_bboxes.append(bbox)
            prev_output_bbox = bbox
            if gmc_enabled:
                if curr_gray is None:
                    curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                prev_frame_gray = curr_gray
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
    parser.add_argument("--seq", default=None,
                        help="Run only this sequence ID (e.g. dataset5/bike3). Overrides --all.")
    parser.add_argument("--f5-feedback", action="store_true",
                        help="Enable F5 closed-loop feedback (tracker.set_state every frame "
                             "in ai_lead mode). Default OFF matches pre-F5 prod behaviour.")
    parser.add_argument("--log-telemetry", default=None,
                        help="Write per-frame integration telemetry CSV.gz into DIR/<variant>/<seq>.csv.gz.")
    parser.add_argument("--variant", default=None,
                        help="Variant tag for telemetry output subdir (e.g. V1, V2, V3).")
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
    print(", ".join(tags))

    manifest = load_manifest()
    tracker = TRTTrackWrapper()

    if args.seq:
        seq_ids = [args.seq]
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
                                  f5_feedback=args.f5_feedback,
                                  log_telemetry_path=args.log_telemetry,
                                  variant=args.variant)
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
