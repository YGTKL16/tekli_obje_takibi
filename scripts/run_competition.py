#!/usr/bin/env python3
"""Run SGLATrack on all competition sequences and generate submission CSV.

Usage:
    python scripts/run_competition.py                          # Run on public_lb split
    python scripts/run_competition.py --split train            # Run on train split (for local eval)
    python scripts/run_competition.py --split all              # Run on both splits
    python scripts/run_competition.py --split train --seq dataset3/car1  # Single sequence
"""

import argparse
import csv
import os
import sys
import time

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "build"))

from tracker.config import load_runtime_config, load_yaml_config  # noqa: E402
from tracker.data_utils import DATA_ROOT, load_manifest, parse_bbox_line  # noqa: E402
from tracker.gmc import GMCEstimator  # noqa: E402
from tracker.imm_policy import observe_with_guidance, step_guided_imm  # noqa: E402

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
        seqs = {k: v for k, v in seqs.items() if k == seq_filter}
    return seqs


def _apply_imm_config(kf, cfg):
    """Apply tuned IMM parameters from config dict."""
    imm = cfg.get("imm", {})
    if not imm:
        return
    q_scale = imm.get("q_scale", 1.0)
    mn = imm.get("measurement_noise", {})
    tm = imm.get("transition_matrix", {})

    # Per-model Q (fixed ratios, scaled by q_scale)
    # Q bases must match replay.py — Optuna optimized against these ratios
    MODEL_Q_BASES = {
        0: np.array([1.0,1.0,1.0,1.0, 0.01,0.01,0.0001,0.0001, 1e-6,1e-6], dtype=np.float32),
        1: np.array([1.0,1.0,1.0,1.0, 0.1,0.1,0.0001,0.0001, 1.0,1.0], dtype=np.float32),
        2: np.array([1.0,1.0,1.0,1.0, 1.0,1.0,0.001,0.001, 100.0,100.0], dtype=np.float32),
    }
    for m in range(3):
        Q = np.zeros((10, 10), dtype=np.float32)
        np.fill_diagonal(Q, MODEL_Q_BASES[m] * q_scale)
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

    # KF + Decision setup
    kf = None
    state_machine = None
    decision = None
    tracking_state_enum = None
    gmc_enabled = False
    gmc_estimator = None
    prev_frame_bgr = None
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
                    downsample=float(gmc_cfg.get("downsample", 0.5)),
                )
                kf.set_gmc_q_boost(float(gmc_cfg.get("fail_q_boost", 4.0)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return

    frame_idx = 0
    conf = 0.0
    reject_streak = 0
    last_good_bbox = None
    t_start = time.time()

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
                if gmc_enabled and prev_frame_bgr is not None and gmc_estimator is not None:
                    kf_bbox = np.array(kf.get_state()).flatten()[:4].astype(np.float32)
                    H, ok = gmc_estimator.estimate(prev_frame_bgr, frame_bgr, kf_bbox)
                    if ok:
                        kf.apply_gmc(H.astype(np.float32))
                    else:
                        kf.set_gmc_failed(True)

                if tracker.association_enabled:
                    predicted_state = np.array(kf.predict()).flatten()
                    # Phase 4: IMM manoeuvre probability for adaptive bypass threshold
                    p_maneuver = 0.0
                    if hasattr(kf, "get_model_probabilities"):
                        _mu = np.array(kf.get_model_probabilities())
                        p_maneuver = float(_mu[1] + _mu[2])
                    observation = observe_with_guidance(
                        tracker,
                        frame_rgb,
                        predicted_state,
                        mode="ai_lead",
                        last_output_bbox=predicted_state[:4],
                    )
                    matched_bbox = observation.bbox
                    conf = observation.confidence
                    track_state = state_machine.step(conf)
                    step = step_guided_imm(
                        kf,
                        decision,
                        predicted_state,
                        matched_bbox,
                        conf,
                        frame_bgr.shape[1],
                        frame_bgr.shape[0],
                        is_tracking=(track_state == tracking_state_enum),
                        judge_reference_bbox=predicted_state[:4],
                        adaptive_r_enabled=bool(runtime_params.get("adaptive_r_enabled", True)),
                        r_exponent=float(runtime_params.get("r_exponent", 1.0)),
                        conf_bypass_threshold=float(runtime_params.get("conf_bypass_threshold", 0.15)),
                        innovation_threshold=float(runtime_params.get("innovation_threshold", 2.0)),
                        reject_streak=reject_streak,
                        last_good_bbox=last_good_bbox,
                        mahal_chi2_gate=float(runtime_params.get("mahal_chi2_gate", 0.0)),
                        r_pos_base=float(runtime_params.get("r_pos_base", 1.0)),
                        r_size_base=float(runtime_params.get("r_size_base", 10.0)),
                        maneuver_probability=p_maneuver,
                        maneuver_threshold=float(runtime_params.get("maneuver_threshold", 0.0)),
                        maneuver_bypass_boost=float(runtime_params.get("maneuver_bypass_boost", 0.0)),
                    )
                    bbox = step.bbox
                    reject_streak = step.reject_streak
                    last_good_bbox = step.last_good_bbox
                    if reject_streak >= 5 and step.should_coast:
                        # Phase 1 — The Great Rescue: reinit AI template at KF location.
                        tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
                    elif f5_feedback:
                        # F5: closed-loop feedback — feed KF-fused bbox back to AI
                        # search window every frame. Gated behind --f5-feedback for
                        # clean A/B against the prior prod behaviour (no set_state).
                        tracker.set_state(np.array(bbox, dtype=np.float32))
                else:
                    predicted_state = np.array(kf.predict()).flatten()
                    # Phase 4: IMM manoeuvre probability for adaptive bypass threshold
                    p_maneuver = 0.0
                    if hasattr(kf, "get_model_probabilities"):
                        _mu = np.array(kf.get_model_probabilities())
                        p_maneuver = float(_mu[1] + _mu[2])
                    observation = observe_with_guidance(
                        tracker,
                        frame_rgb,
                        predicted_state,
                        mode="ai_lead",
                        last_output_bbox=predicted_state[:4],
                    )
                    conf = observation.confidence
                    track_state = state_machine.step(conf)
                    step = step_guided_imm(
                        kf,
                        decision,
                        predicted_state,
                        observation.bbox,
                        conf,
                        frame_bgr.shape[1],
                        frame_bgr.shape[0],
                        is_tracking=(track_state == tracking_state_enum),
                        judge_reference_bbox=predicted_state[:4],
                        adaptive_r_enabled=bool(runtime_params.get("adaptive_r_enabled", True)),
                        r_exponent=float(runtime_params.get("r_exponent", 1.0)),
                        conf_bypass_threshold=float(runtime_params.get("conf_bypass_threshold", 0.15)),
                        innovation_threshold=float(runtime_params.get("innovation_threshold", 2.0)),
                        reject_streak=reject_streak,
                        last_good_bbox=last_good_bbox,
                        mahal_chi2_gate=float(runtime_params.get("mahal_chi2_gate", 0.0)),
                        r_pos_base=float(runtime_params.get("r_pos_base", 1.0)),
                        r_size_base=float(runtime_params.get("r_size_base", 10.0)),
                        maneuver_probability=p_maneuver,
                        maneuver_threshold=float(runtime_params.get("maneuver_threshold", 0.0)),
                        maneuver_bypass_boost=float(runtime_params.get("maneuver_bypass_boost", 0.0)),
                    )
                    bbox = step.bbox
                    reject_streak = step.reject_streak
                    last_good_bbox = step.last_good_bbox
                    if reject_streak >= 5 and step.should_coast:
                        # Phase 1 — The Great Rescue: reinit AI template at KF location.
                        tracker.init(frame_rgb, np.array(bbox, dtype=np.float32))
                    elif f5_feedback:
                        # F5: closed-loop feedback — feed KF-fused bbox back to AI
                        # search window every frame. Gated behind --f5-feedback for
                        # clean A/B against the prior prod behaviour (no set_state).
                        tracker.set_state(np.array(bbox, dtype=np.float32))
            else:
                ai_bbox, conf = tracker.track(frame_rgb)
                ai_bbox = ai_bbox.tolist() if isinstance(ai_bbox, np.ndarray) else list(ai_bbox)
                bbox = ai_bbox

        # Store result: seq_id + "_" + frame_index -> [x, y, w, h]
        result_id = f"{seq_id}_{frame_idx}"
        results[result_id] = bbox

        if gmc_enabled:
            prev_frame_bgr = frame_bgr.copy()

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start
    fps = frame_idx / elapsed if elapsed > 0 else 0

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
    parser.add_argument("--seq", default=None, help="Run a single sequence (e.g. dataset3/car1)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SGLATrack checkpoint (.pth.tar)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "tensorrt"],
                        help="Inference backend")
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
    args = parser.parse_args()

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
    else:
        from tracker.sglatrack_wrapper import SGLATrackWrapper  # pyright: ignore[reportMissingImports]
        tracker = SGLATrackWrapper(checkpoint_path=args.checkpoint)

    # Run on all sequences
    results = {}
    total_start = time.time()
    for i, (seq_id, seq_info) in enumerate(sequences.items()):
        print(f"[{i+1}/{len(sequences)}] {seq_id} ({seq_info['n_frames']} frames)")
        run_tracker_on_sequence(tracker, seq_id, seq_info, results,
                                use_kf=not args.no_kf,
                                runtime_params=runtime_params,
                                imm_config=imm_config,
                                f5_feedback=args.f5_feedback)

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
