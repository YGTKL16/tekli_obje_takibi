---
description: "Python tracker module conventions: closed-loop feedback, pybind11 hot-path, performance budget, evaluation metrics. Use when: writing or reviewing Python tracker code, modifying pipeline/decision/IMM policy."
applyTo: "python/tracker/**/*.py"
---

# Python Tracker Conventions

## Closed-Loop Feedback (Critical)

`tracker.set_state(bbox)` **must** be called after every KF/IMM update. This feeds the fused bbox back to the AI search window. Without it, performance is catastrophic (Δ FinalScore = -0.129).

## Performance Budget

Total frame ≤ 30ms. Python overhead must be minimal:
- Prefer C++ pybind11 bindings (`tracker_cpp.KalmanFilter`, `tracker_cpp.IMMFilter`) over pure Python reimplementation
- No per-frame object creation in hot path — pre-allocate and reuse
- NumPy vectorized ops over Python loops in any batch computation
- GMC: use C++ `tracker_cpp.GMCEstimator` when available, Python fallback only as backup

## Evaluation

Always report these metrics when comparing configurations:
- **AUC** (IoU thresholds 0–1.0)
- **NormPrecision** (normalized distance curve)
- **FinalScore** (AUC + NormPrecision + efficiency penalty)
- **Delta** (new − baseline) — mandatory for every comparison

## Config Pattern

```python
# Load YAML → dataclass → pass to C++
config = load_yaml("configs/imm_tuned.yaml")
kf = tracker_cpp.KalmanFilter()
kf.set_params(config["q_scale"], config["r_pos_scale"], ...)
```

## State Machine

`DecisionMaker` in `decision.py` controls TRACKING/COASTING/LOST transitions:
- `conf >= coast_threshold` → TRACKING (update KF with AI bbox)
- `coast_threshold > conf >= conf_threshold` → COASTING (blind-fly on KF predict)
- `conf < conf_threshold` → hard reject (never update)
- Exceeding `max_coast_frames` → LOST (fallback to raw AI)

Do not bypass or duplicate this logic — all state transitions go through `DecisionMaker`.
