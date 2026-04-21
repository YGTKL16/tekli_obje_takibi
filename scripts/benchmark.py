#!/usr/bin/env python3
"""End-to-end benchmark and CI guardrails for the tracker stack.

Reports micro-benchmark latency for the Kalman filter, IMM filter, and tracker
state machine. Optional full-pipeline benchmarking is also available on a real
sequence. CI mode adds replay-based RSS plateau checks and emits JSON results.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "contest_release")
CACHE_ROOT = os.path.join(PROJECT_ROOT, "cache", "ai_outputs")

try:
    import tracker_cpp
except ImportError:
    print("[ERROR] tracker_cpp not found. Build first:")
    print("  cmake -B build && cmake --build build")
    sys.exit(1)

from tracker.config import load_runtime_config  # noqa: E402
from tracker.replay import DEFAULT_PARAMS, replay_sequence  # noqa: E402

MODEL_FLOPS_G = 5.81
MODEL_PARAMS_M = 5.7
CI_THRESHOLDS = {
    "kf_update_p99_ms": 0.5,
    "imm_update_p99_ms": 1.0,
    "rss_plateau_spread_kb": 2048,
    "rss_final_delta_kb": 2048,
}
CI_SMOKE_SEQS = [
    "dataset1/surfer",
    "dataset3/truck_night",
    "dataset5/car1_s",
    "dataset5/car1_3",
]


def _summarize_ms(samples) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "avg": float(arr.mean()),
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _rss_kb() -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not available in /proc/self/status")


def _print_row(name: str, stats: dict[str, float]) -> None:
    print(f"{name:>12} {stats['avg']:>8.4f} {stats['p50']:>8.4f} "
          f"{stats['p95']:>8.4f} {stats['p99']:>8.4f} {stats['max']:>8.4f} ms")


def benchmark_kalman(n_frames: int = 1000, *, emit: bool = True) -> dict[str, dict[str, float]]:
    """Benchmark pure Kalman filter predict/update cycle."""
    kf = tracker_cpp.KalmanFilter()
    z = np.array([100, 100, 50, 50], dtype=np.float32)
    kf.init(z)
    times_update, times_predict = [], []
    for _ in range(n_frames):
        z_noisy = z + np.random.randn(4).astype(np.float32) * 2
        t0 = time.perf_counter_ns()
        kf.update(z_noisy)
        times_update.append((time.perf_counter_ns() - t0) / 1e6)
        t0 = time.perf_counter_ns()
        kf.predict()
        times_predict.append((time.perf_counter_ns() - t0) / 1e6)

    stats = {
        "update": _summarize_ms(times_update),
        "predict": _summarize_ms(times_predict),
    }
    if emit:
        print(f"\nKalman Filter Benchmark ({n_frames} frames)")
        print(f"{'':>12} {'avg':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
        _print_row("update", stats["update"])
        _print_row("predict", stats["predict"])
        print(f"  KF budget (0.5ms): {'PASS' if stats['update']['p99'] < 0.5 else 'FAIL'}")
    return stats


def benchmark_imm(n_frames: int = 1000, *, emit: bool = True) -> dict[str, dict[str, float]]:
    """Benchmark IMM filter update cycle (3 models)."""
    kf = tracker_cpp.IMMFilter()
    z = np.array([100, 100, 50, 50], dtype=np.float32)
    kf.init(z)
    times = []
    for _ in range(n_frames):
        z_noisy = z + np.random.randn(4).astype(np.float32) * 2
        t0 = time.perf_counter_ns()
        kf.update(z_noisy)
        times.append((time.perf_counter_ns() - t0) / 1e6)

    stats = {"update": _summarize_ms(times)}
    if emit:
        print(f"\nIMM Filter Benchmark ({n_frames} frames)")
        print(f"{'':>12} {'avg':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
        _print_row("update", stats["update"])
        print(f"  IMM budget (1.0ms): {'PASS' if stats['update']['p99'] < 1.0 else 'FAIL'}")
    return stats


def benchmark_state_machine(n_frames: int = 1000, *, emit: bool = True) -> dict[str, float]:
    """Benchmark state machine transitions."""
    ts = tracker_cpp.TrackerState()
    ts.force_tracking()
    times = []
    for i in range(n_frames):
        conf = 0.9 if i % 100 < 70 else 0.1
        t0 = time.perf_counter_ns()
        ts.step(conf)
        times.append((time.perf_counter_ns() - t0) / 1e6)

    stats = _summarize_ms(times)
    if emit:
        print(f"\nState Machine Benchmark ({n_frames} frames)")
        print(f"  avg={stats['avg']:.4f}ms  p99={stats['p99']:.4f}ms")
    return stats


def _cache_path(seq_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, seq_id.replace("/", "__") + ".npz")


def _ci_cache_paths(cache_dir: str) -> list[str]:
    paths = [_cache_path(seq_id, cache_dir) for seq_id in CI_SMOKE_SEQS]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing cached replay inputs: {missing}")
    return paths


def _replay_smoke_subset(cache_paths: list[str]) -> None:
    for cache_path in cache_paths:
        replay_sequence(cache_path, DEFAULT_PARAMS)


def benchmark_ci(
    *,
    frames: int = 1000,
    cache_dir: str = CACHE_ROOT,
    rss_iters: int = 8,
    emit: bool = True,
) -> dict[str, object]:
    """Run fast CI guardrails and emit a machine-readable result."""
    kf_stats = benchmark_kalman(frames, emit=False)["update"]
    imm_stats = benchmark_imm(frames, emit=False)["update"]
    cache_paths = _ci_cache_paths(cache_dir)

    rss_start = _rss_kb()
    _replay_smoke_subset(cache_paths)
    gc.collect()
    rss_after_warmup = _rss_kb()

    plateau_samples = []
    for _ in range(rss_iters):
        _replay_smoke_subset(cache_paths)
        gc.collect()
        plateau_samples.append(_rss_kb())

    stable_samples = plateau_samples[:-1] or plateau_samples or [rss_after_warmup]
    stable_ref = int(round(statistics.median(stable_samples)))
    plateau_spread = (max(plateau_samples) - min(plateau_samples)) if plateau_samples else 0
    final_delta = abs((plateau_samples[-1] if plateau_samples else rss_after_warmup) - stable_ref)

    metrics = {
        "kf_update_p99_ms": round(kf_stats["p99"], 6),
        "imm_update_p99_ms": round(imm_stats["p99"], 6),
        "rss_warmup_delta_kb": int(rss_after_warmup - rss_start),
        "rss_plateau_spread_kb": int(plateau_spread),
        "rss_final_delta_kb": int(final_delta),
    }
    failures = []
    for key, limit in CI_THRESHOLDS.items():
        if metrics[key] >= limit:
            failures.append(f"{key}={metrics[key]} exceeds {limit}")

    payload: dict[str, object] = {
        "status": "pass" if not failures else "fail",
        "metrics": metrics,
        "thresholds": CI_THRESHOLDS,
        "failures": failures,
    }
    if emit:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def benchmark_full_pipeline(seq_path, with_imm: bool = False, engine_path: str | None = None):
    """Full pipeline benchmark on a real video sequence."""
    import cv2
    from tracker.decision import DecisionMaker
    from tracker.matching import associate_detections_to_trackers
    from tracker.trt_wrapper import TRTTrackWrapper

    runtime_params = load_runtime_config(os.path.join(PROJECT_ROOT, "configs", "tracker_config.yaml"))

    seq_dir = os.path.join(DATA_ROOT, seq_path)
    video_path = None
    for filename in os.listdir(seq_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(seq_dir, filename)
            break
    if not video_path:
        print(f"[ERROR] No .mp4 in {seq_dir}")
        return

    ann_path = os.path.join(seq_dir, "annotation.txt")
    with open(ann_path, encoding="utf-8") as handle:
        parts = handle.readline().strip().replace("\t", ",").split(",")
        init_bbox = [float(x) for x in parts[:4]]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    eng_path = engine_path or os.path.join(PROJECT_ROOT, "models", "sglatrack_fp16.engine")
    engine_size_mb = os.path.getsize(eng_path) / (1024 * 1024) if os.path.exists(eng_path) else 0.0

    tracker = TRTTrackWrapper(
        engine_path=eng_path,
        association_enabled=bool(runtime_params["association_enabled"]),
        association_top_k=int(runtime_params["association_top_k"]),
        association_iou_threshold=float(runtime_params["association_iou_threshold"]),
        association_score_weight=float(runtime_params["association_score_weight"]),
    )
    kf = tracker_cpp.IMMFilter() if with_imm else None
    sm = tracker_cpp.TrackerState() if with_imm else None
    dec = None
    if with_imm:
        assert sm is not None and kf is not None
        sm.set_confidence_threshold(float(runtime_params["sm_conf_threshold"]))
        sm.set_max_coast_frames(int(runtime_params["sm_max_coast"]))
        dec = DecisionMaker(
            conf_threshold=float(runtime_params["conf_threshold"]),
            iou_threshold=float(runtime_params["iou_threshold"]),
            coast_threshold=float(runtime_params["coast_threshold"]),
            max_coast_frames=int(runtime_params["max_coast_frames"]),
        )
        if runtime_params["adaptive_r_enabled"]:
            kf.set_adaptive_r_floor(float(runtime_params["adaptive_r_floor"]))

    t_infer, t_candidate, t_matching, t_kf_list, t_total = [], [], [], [], []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            tracker.init(frame_rgb, init_bbox)
            if with_imm:
                assert kf is not None and sm is not None
                kf.init(np.array(init_bbox, dtype=np.float32))
                sm.force_tracking()
        else:
            t0 = time.perf_counter()
            ai_bbox = None
            conf = 0.0
            predicted_state = None
            ai_elapsed_ms = 0.0
            candidate_elapsed_ms = 0.0
            matching_elapsed_ms = 0.0

            if with_imm and tracker.association_enabled:
                assert kf is not None
                predicted_state = np.array(kf.predict()).flatten()
                t_candidate_start = time.perf_counter()
                candidate_bboxes, candidate_scores = tracker.track_candidates(
                    frame_rgb,
                    top_k=tracker.association_top_k,
                )
                t_candidate_end = time.perf_counter()
                ai_elapsed_ms = (t_candidate_end - t_candidate_start) * 1000
                candidate_elapsed_ms = ai_elapsed_ms

                t_match_start = time.perf_counter()
                matches, _, _ = associate_detections_to_trackers(
                    predicted_state[:4].reshape(1, 4),
                    candidate_bboxes,
                    detection_scores=candidate_scores,
                    iou_threshold=tracker.association_iou_threshold,
                    score_weight=tracker.association_score_weight,
                )
                t_match_end = time.perf_counter()
                matching_elapsed_ms = (t_match_end - t_match_start) * 1000

                if matches.shape[0] > 0:
                    det_idx = int(matches[0, 1])
                    ai_bbox = candidate_bboxes[det_idx]
                    conf = float(candidate_scores[det_idx])
            else:
                ai_bbox, conf = tracker.track(frame_rgb)
                ai_elapsed_ms = (time.perf_counter() - t0) * 1000

            t_kf_start = time.perf_counter()
            if with_imm:
                assert sm is not None and dec is not None and kf is not None
                if tracker.association_enabled:
                    track_state = sm.step(conf)
                    if (
                        ai_bbox is not None
                        and dec.should_update(conf, ai_bbox, predicted_state)
                        and track_state == tracker_cpp.TrackState.TRACKING
                    ):
                        state = np.array(kf.update(np.asarray(ai_bbox, dtype=np.float32))).flatten()
                        tracker.set_state(state[:4])
                    else:
                        tracker.set_state(predicted_state[:4])
                else:
                    ai_list = ai_bbox.tolist() if isinstance(ai_bbox, np.ndarray) else list(ai_bbox)
                    kf_state = np.array(kf.get_state()).flatten()
                    sm.step(conf)
                    if dec.should_update(conf, np.array(ai_list), kf_state):
                        np.array(kf.update(np.array(ai_list, dtype=np.float32))).flatten()
                    else:
                        kf.update(np.array(ai_list, dtype=np.float32))
            t_kf_end = time.perf_counter()

            t_infer.append(ai_elapsed_ms)
            t_candidate.append(candidate_elapsed_ms)
            t_matching.append(matching_elapsed_ms)
            t_kf_list.append((t_kf_end - t_kf_start) * 1000)
            t_total.append((t_kf_end - t0) * 1000)

        frame_idx += 1
    cap.release()

    t_infer_arr = np.asarray(t_infer, dtype=np.float64)
    t_candidate_arr = np.asarray(t_candidate, dtype=np.float64)
    t_matching_arr = np.asarray(t_matching, dtype=np.float64)
    t_kf_arr = np.asarray(t_kf_list, dtype=np.float64)
    t_total_arr = np.asarray(t_total, dtype=np.float64)

    def _row(arr, name):
        print(f"  {name:<20} {arr.mean():>8.2f} {np.median(arr):>8.2f} "
              f"{np.percentile(arr,95):>8.2f} {np.percentile(arr,99):>8.2f} {arr.max():>8.2f}")

    avg_ms = float(t_total_arr.mean()) if t_total_arr.size else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"  FULL PIPELINE BENCHMARK: {seq_path}")
    print(f"{'='*70}")
    print(f"\n  Sequence Info")
    print(f"    Frames:     {frame_idx} ({total_frames} in file)")
    print(f"    Resolution: {frame_w} x {frame_h}")
    print(f"    Init bbox:  {init_bbox}")
    print(f"\n  Model Specs")
    print(f"    Architecture: SGLATrack (DeiT-tiny)")
    print(f"    Parameters:   {MODEL_PARAMS_M:.1f} M")
    print(f"    GFLOPs:       {MODEL_FLOPS_G:.2f}")
    print(f"    Engine size:  {engine_size_mb:.1f} MB (FP16)")
    print(f"    Inputs:       template (1,3,128,128) + search (1,3,256,256)")
    print(f"\n  Latency (ms)")
    print(f"  {'Component':<20} {'avg':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    if t_infer_arr.size:
        _row(t_infer_arr, "AI inference")
    if with_imm and t_candidate_arr.size:
        _row(t_candidate_arr, "candidate_extraction")
        _row(t_matching_arr, "matching")
        _row(t_kf_arr, "IMM filter")
    if t_total_arr.size:
        _row(t_total_arr, "Total / frame")
    print(f"\n  Throughput")
    print(f"    Avg latency:  {avg_ms:.2f} ms/frame")
    print(f"    FPS:          {fps:.1f}")
    print(f"    Realtime:     {'YES (>= 30 fps)' if fps >= 30 else 'NO (< 30 fps)'}")

    s_flops = min(1.0, max(0.0, (MODEL_FLOPS_G - 30) / 30))
    s_params = min(1.0, max(0.0, (MODEL_PARAMS_M - 50) / 50))
    s_latency = min(1.0, max(0.0, (avg_ms - 30) / 30))
    s_size = min(1.0, max(0.0, (engine_size_mb - 500) / 500))
    s_eff = 0.25 * s_flops + 0.15 * s_params + 0.35 * s_latency + 0.25 * s_size

    print(f"\n  Competition Efficiency Score")
    print(f"    S_flops   = {s_flops:.4f}  ({MODEL_FLOPS_G:.2f} / 30 GFLOPs budget)")
    print(f"    S_params  = {s_params:.4f}  ({MODEL_PARAMS_M:.1f}M / 50M budget)")
    print(f"    S_latency = {s_latency:.4f}  ({avg_ms:.2f}ms / 30ms budget)")
    print(f"    S_size    = {s_size:.4f}  ({engine_size_mb:.1f}MB / 500MB budget)")
    print(f"    ────────────────────────")
    print(f"    S_eff     = {s_eff:.4f}  (0.0 = perfect)")
    print(f"{'='*70}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tracker benchmark")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--video", type=str, default=None,
                        help="Sequence path (e.g. dataset1/basketball)")
    parser.add_argument("--with-imm", action="store_true",
                        help="Include IMM filter in pipeline")
    parser.add_argument("--engine", default=None, help="TRT engine path")
    parser.add_argument("--ci", action="store_true",
                        help="Run CI guardrails and emit machine-readable JSON")
    parser.add_argument("--cache-dir", default=CACHE_ROOT,
                        help="Cache directory for replay-based RSS checks")
    parser.add_argument("--rss-iters", type=int, default=8,
                        help="Number of stabilized RSS replay iterations in --ci mode")
    args = parser.parse_args()

    if args.ci:
        payload = benchmark_ci(frames=args.frames, cache_dir=args.cache_dir, rss_iters=args.rss_iters)
        return 0 if payload["status"] == "pass" else 1

    benchmark_kalman(args.frames)
    benchmark_imm(args.frames)
    benchmark_state_machine(args.frames)

    if args.video:
        benchmark_full_pipeline(args.video, with_imm=args.with_imm,
                                engine_path=args.engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
