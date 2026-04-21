# Tracker Project — Workspace Instructions

## Architecture

Single-object visual tracker: AI inference (SGLATrack / TensorRT) fused with IMM-Kalman filtering.

- **10D state vector**: `[x, y, w, h, vx, vy, vw, vh, ax, ay]`
- **IMMFilter**: 3-model (CV / CA / Singer), linear F, differ only in Q
- **KalmanFilter**: standalone 10D CV, `S.llt().solve()` for numerical stability
- **TrackerState**: 3-state FSM — `TRACKING → COASTING → LOST`
- **Pipeline stages**: Read Frame → GMC → AI Inference → Decision → KF/IMM Update → Output Bbox
- **Closed-loop feedback** is **critical**: `tracker.set_state(bbox)` feeds KF-blended bbox back to AI search window. Without it, performance is catastrophic.

## Build & Test

All C++ builds run inside distrobox `tracker-dev` (Ubuntu 24.04, GCC 13.3.0):

```bash
# C++ build
distrobox enter tracker-dev -- bash -c "cd /home/ykula/tracker && cmake --build build2 -j$(nproc)"

# C++ tests
distrobox enter tracker-dev -- bash -c "cd /home/ykula/tracker/build2 && ctest --output-on-failure"

# Python tests
python3 -m pytest python/tracker/tests/ -v

# A/B test (20-sequence subset)
python3 scripts/ab_test.py --imm --gmc --adaptive-r

# Full evaluation
python3 scripts/run_competition.py --split train --imm-config configs/imm_tuned.yaml

# Benchmarks
python3 scripts/benchmark.py
```

## Critical Constraints

- **EIGEN_MAX_ALIGN_BYTES=0** in CMakeLists.txt — pybind11 CPython allocator doesn't guarantee Eigen alignment
- **No dynamic allocation** after construction (JSF AV C++ Rev C)
- **No exceptions** in core C++ (`-fno-exceptions`), **no RTTI** (`-fno-rtti`)
- **Fixed-size Eigen matrices** on stack — never `MatrixXd`
- **POSITION_INDEPENDENT_CODE ON** on tracker_core for pybind11 .so linking

## Performance Budgets

| Component | p99 Target |
|-----------|-----------|
| KF predict/update | < 0.5 ms |
| IMM full cycle | < 1.0 ms |
| AI inference (SGLATrack) | 10–15 ms |
| AI inference (TensorRT) | 3–5 ms |
| **Total frame** | **≤ 30 ms** |

## Evaluation Metrics

- **AUC**: area under success curve (IoU thresholds 0–1.0)
- **NormPrecision**: area under normalized distance curve
- **FinalScore**: blend of AUC + NormPrecision + efficiency penalty
- Always report **Delta** (new − baseline) when comparing configurations

## Key Files

- Configs: `configs/tracker_config.yaml`, `configs/imm_tuned.yaml`
- Evaluation: `scripts/evaluate_local.py`, `scripts/ab_test.py`
- Pipeline: `python/tracker/pipeline.py`, `python/tracker/decision.py`
- C++ core: `cpp/include/imm_filter.h`, `cpp/include/kalman_filter.h`
- Status: `INVENTORY_AND_RESULTS.md`
