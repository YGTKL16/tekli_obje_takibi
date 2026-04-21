// ---------------------------------------------------------------------------
// Project : Tracker
// File    : kalman_types.h
// Purpose : Fixed-size Eigen type aliases for Kalman / IMM filter
// Standard: JSF AV C++ Rev C (AV Rule 209 – fixed-width types)
// ---------------------------------------------------------------------------
#ifndef TRACKER_KALMAN_TYPES_H
#define TRACKER_KALMAN_TYPES_H

#include <cmath>
#include <cstdint>
#include <Eigen/Dense>

namespace tracker {

/// State vector dimension: [x, y, w, h, vx, vy, vw, vh, ax, ay]
///   x,y   – bounding-box centre
///   w,h   – bounding-box width / height
///   vx,vy – velocity of centre
///   vw,vh – rate of change of size
///   ax,ay – acceleration of centre (used by CA / Singer models)
constexpr int32_t kStateDim  = 10;

/// Measurement vector dimension: [x, y, w, h]
constexpr int32_t kMeasDim   = 4;

/// Number of models in the IMM estimator (CV, CA, Singer)
constexpr int32_t kNumModels = 3;

// ── Single-filter aliases ───────────────────────────────────────
using StateVec   = Eigen::Matrix<float, kStateDim, 1>;
using MeasVec    = Eigen::Matrix<float, kMeasDim, 1>;
using StateMat   = Eigen::Matrix<float, kStateDim, kStateDim>;
using MeasMat    = Eigen::Matrix<float, kMeasDim, kStateDim>;
using MeasCovMat = Eigen::Matrix<float, kMeasDim, kMeasDim>;
using KalmanGain = Eigen::Matrix<float, kStateDim, kMeasDim>;

// ── IMM-specific aliases ────────────────────────────────────────
using ModelProb = Eigen::Matrix<float, kNumModels, 1>;
using TransMat  = Eigen::Matrix<float, kNumModels, kNumModels>;

// ── GMC: 3×3 homography mapping prev-frame background → curr-frame ─
using HomMat = Eigen::Matrix<float, 3, 3>;

/// Check if any element in a measurement vector is NaN or Inf.
inline bool meas_has_nan_or_inf(const MeasVec& z) noexcept {
    for (int32_t i = 0; i < kMeasDim; ++i) {
        if (std::isnan(z(i)) || std::isinf(z(i))) {
            return true;
        }
    }
    return false;
}

// AV Rule 145 – compile-time guarantee: no hidden heap allocation
static_assert(sizeof(StateVec)   == kStateDim  * sizeof(float),
              "StateVec must be fixed-size (no heap)");
static_assert(sizeof(MeasVec)    == kMeasDim   * sizeof(float),
              "MeasVec must be fixed-size (no heap)");
static_assert(sizeof(ModelProb)  == kNumModels * sizeof(float),
              "ModelProb must be fixed-size (no heap)");

}  // namespace tracker

#endif  // TRACKER_KALMAN_TYPES_H
