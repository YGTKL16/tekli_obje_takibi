// ---------------------------------------------------------------------------
// Project : Tracker
// File    : kalman_filter.h
// Purpose : JSF-compliant constant-velocity Kalman filter (no heap, no throw)
//           10D state: [x,y,w,h, vx,vy,vw,vh, ax,ay] (ax/ay idle in CV)
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#ifndef TRACKER_KALMAN_FILTER_H
#define TRACKER_KALMAN_FILTER_H

#include "kalman_types.h"

namespace tracker {

/// @brief Constant-velocity Kalman filter for bounding-box tracking.
///
/// Design constraints (JSF AV):
///   - No dynamic memory allocation (new / malloc)
///   - No exceptions  (-fno-exceptions)
///   - No RTTI        (-fno-rtti)
///   - All matrices are fixed-size, stack-allocated via Eigen
class KalmanFilter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // Eigen fixed-size alignment for pybind11

    KalmanFilter() noexcept;
    ~KalmanFilter()                              noexcept = default;
    KalmanFilter(const KalmanFilter&)            noexcept = default;
    KalmanFilter& operator=(const KalmanFilter&) noexcept = default;
    KalmanFilter(KalmanFilter&&)                 noexcept = default;
    KalmanFilter& operator=(KalmanFilter&&)      noexcept = default;

    /// Initialise state from first measurement @p z0 = [x, y, w, h].
    void init(const MeasVec& z0) noexcept;

    /// Predict next state (constant-velocity model, dt = 1).
    /// @return Reference to predicted state vector.
    [[nodiscard]] const StateVec& predict() noexcept;

    /// Predict then correct with measurement @p z = [x, y, w, h].
    /// If not yet initialised, calls init(z) instead.
    /// @return Reference to corrected state vector.
    [[nodiscard]] const StateVec& update(const MeasVec& z) noexcept;

    /// Adaptive-R variant: scales R by 1 / max(confidence, adaptive_r_floor_)
    /// before a single update. Higher confidence → tighter R → KF trusts z more.
    [[nodiscard]] const StateVec& update(const MeasVec& z, float confidence) noexcept;

    /// Warp state mean by a 3×3 homography (prev → curr). Size left unchanged;
    /// centre and velocity transformed. No-op if filter not initialised.
    void apply_gmc(const HomMat& H) noexcept;

    /// Flag that GMC failed this frame. The next predict() applies Q*gmc_q_boost;
    /// the flag auto-clears after that predict call.
    void set_gmc_failed(bool failed) noexcept;

    /// Set Q multiplier used on frames where GMC failed (default 4.0).
    void set_gmc_q_boost(float boost) noexcept;

    /// Set confidence floor for adaptive R (default 0.4).
    void set_adaptive_r_floor(float floor) noexcept;

    /// @return Current state estimate [x, y, w, h, vx, vy, vw, vh, ax, ay].
    [[nodiscard]] const StateVec& get_state() const noexcept { return x_; }

    /// @return Current error-covariance matrix (10×10).
    [[nodiscard]] const StateMat& get_covariance() const noexcept { return P_; }

    /// @return True once init() or update() has been called at least once.
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// Reset filter to uninitialised state.
    void reset() noexcept;

    /// Override default process-noise matrix Q (8×8).
    void set_process_noise(const StateMat& Q) noexcept;

    /// Override default measurement-noise matrix R (4×4).
    void set_measurement_noise(const MeasCovMat& R) noexcept;

#ifdef TRACKER_RUST_BRIDGE_ENABLED
    /// Set frame dimensions for Rust safety bridge bbox clamping.
    void set_frame_bounds(float width, float height, float margin = 0.0F) noexcept;
#endif

private:
    void build_matrices() noexcept;

    StateVec   x_;    ///< state estimate
    StateMat   P_;    ///< error covariance
    StateMat   F_;    ///< state transition
    MeasMat    H_;    ///< measurement matrix (4×8)
    StateMat   Q_;    ///< process noise
    MeasCovMat R_;    ///< measurement noise
    bool       initialized_;
    bool       predicted_        = false;  ///< true after predict(), cleared by update()

    // ── GMC / adaptive-R ──────────────────────────────────────────
    bool  gmc_failed_       = false;   ///< flag consumed by next predict()
    float gmc_q_boost_      = 4.0F;    ///< Q multiplier when GMC failed
    float adaptive_r_floor_ = 0.4F;    ///< lower bound on confidence for R scale

#ifdef TRACKER_RUST_BRIDGE_ENABLED
    float frame_width_  = 1920.0F;
    float frame_height_ = 1080.0F;
    float frame_margin_ = 0.0F;
#endif
};

}  // namespace tracker

#endif  // TRACKER_KALMAN_FILTER_H
