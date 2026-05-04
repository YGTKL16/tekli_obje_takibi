// ---------------------------------------------------------------------------
// Project : Tracker
// File    : kalman_filter.cpp
// Purpose : Kalman filter implementation (constant-velocity, JSF-compliant)
//           10D state: [x,y,w,h, vx,vy,vw,vh, ax,ay] (ax/ay idle in CV)
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#include "kalman_filter.h"

#include <algorithm>
#include <cmath>

#ifdef TRACKER_RUST_BRIDGE_ENABLED
#include "tracker_safety.h"
#endif

namespace tracker {

namespace {

/// Reject homography matrices with NaN / Inf / zero last row entries.
bool hom_is_safe(const HomMat& H) noexcept {
    for (int32_t r = 0; r < 3; ++r) {
        for (int32_t c = 0; c < 3; ++c) {
            if (std::isnan(H(r, c)) || std::isinf(H(r, c))) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

KalmanFilter::KalmanFilter() noexcept : initialized_(false) {
    build_matrices();
}

void KalmanFilter::build_matrices() noexcept {
    // State transition F (constant velocity, dt=1)
    //   x' = x + vx,  y' = y + vy,  w' = w + vw,  h' = h + vh
    //   vx'= vx,       vy'= vy,      vw'= vw,      vh'= vh
    //   ax'= ax,       ay'= ay       (idle in CV mode)
    F_ = StateMat::Identity();
    F_(0, 4) = 1.0f;  // x += vx
    F_(1, 5) = 1.0f;  // y += vy
    F_(2, 6) = 1.0f;  // w += vw
    F_(3, 7) = 1.0f;  // h += vh
    F_(4, 8) = 1.0f;  // vx += ax·dt
    F_(5, 9) = 1.0f;  // vy += ay·dt

    // Measurement matrix H: observe [x, y, w, h] from 10D state
    H_ = MeasMat::Zero();
    H_(0, 0) = 1.0f;
    H_(1, 1) = 1.0f;
    H_(2, 2) = 1.0f;
    H_(3, 3) = 1.0f;

    // Process noise Q (tuned for moderate maneuvering)
    Q_ = StateMat::Zero();
    Q_(0, 0) = 0.1f;      // x
    Q_(1, 1) = 0.1f;      // y
    Q_(2, 2) = 1.0f;      // w
    Q_(3, 3) = 1.0f;      // h
    Q_(4, 4) = 0.01f;     // vx
    Q_(5, 5) = 0.01f;     // vy
    Q_(6, 6) = 0.0001f;   // vw (size change is slow)
    Q_(7, 7) = 0.0001f;   // vh
    Q_(8, 8) = 0.001f;    // ax (nearly idle in CV)
    Q_(9, 9) = 0.001f;    // ay

    // Measurement noise R (AI detector noise)
    R_ = MeasCovMat::Zero();
    R_(0, 0) = 1.0f;   // x
    R_(1, 1) = 1.0f;   // y
    R_(2, 2) = 4.0f;   // w (size less precise)
    R_(3, 3) = 4.0f;   // h

    // State and covariance initialized to zero until init() called
    x_ = StateVec::Zero();
    P_ = StateMat::Identity() * 10.0f;
}

void KalmanFilter::init(const MeasVec& z0) noexcept {
    if (meas_has_nan_or_inf(z0)) { return; }

    x_ = StateVec::Zero();
    x_(0) = z0(0);  // x
    x_(1) = z0(1);  // y
    x_(2) = z0(2);  // w
    x_(3) = z0(3);  // h
    // velocities and accelerations start at zero

    // Initial covariance: high uncertainty on velocity / acceleration
    P_ = StateMat::Zero();
    P_(0, 0) = 2.0f;     // x
    P_(1, 1) = 2.0f;     // y
    P_(2, 2) = 10.0f;    // w
    P_(3, 3) = 10.0f;    // h
    P_(4, 4) = 25.0f;    // vx
    P_(5, 5) = 25.0f;    // vy
    P_(6, 6) = 25.0f;    // vw
    P_(7, 7) = 25.0f;    // vh
    P_(8, 8) = 100.0f;   // ax (unknown)
    P_(9, 9) = 100.0f;   // ay

    initialized_ = true;
}

const StateVec& KalmanFilter::predict() noexcept {
    // AV Rule 113 – single exit point
    if (initialized_) {
        // x = F * x
        x_ = F_ * x_;

        // P = F * P * F^T + Q (boosted if GMC failed this frame)
        const float q_scale = gmc_failed_ ? gmc_q_boost_ : 1.0F;
        P_ = F_ * P_ * F_.transpose() + Q_ * q_scale;

        // Flag auto-clears — the boost is a one-shot for this predict only.
        gmc_failed_ = false;
        predicted_ = true;
    }
    return x_;
}

const StateVec& KalmanFilter::update(const MeasVec& z) noexcept {
    if (meas_has_nan_or_inf(z)) { return x_; }

#ifdef TRACKER_RUST_BRIDGE_ENABLED
    const MeasVec z_safe = safety::clamp_measurement(
        z, frame_width_, frame_height_, frame_margin_);
#else
    const MeasVec& z_safe = z;
#endif

    // AV Rule 113 – single exit point
    if (!initialized_) {
        init(z_safe);
    } else {
        // Predict step — skip if predict() was already called externally.
        if (!predicted_) {
            static_cast<void>(predict());
        }
        predicted_ = false;

        // Innovation: y = z - H * x
        const MeasVec y = z_safe - H_ * x_;

        // Innovation covariance: S = H * P * H^T + R
        MeasCovMat S = H_ * P_ * H_.transpose() + R_;
        S = 0.5F * (S + S.transpose());

        // Kalman gain: K = P * H^T * S^(-1)  (via Cholesky for stability)
        // Trap: if S is not PD, add regularisation and retry; if still
        // singular, skip the update and keep the predicted state.
        const Eigen::LLT<MeasCovMat> llt(S);
        KalmanGain K;
        if (llt.info() == Eigen::Success) {
            K = P_ * H_.transpose() * llt.solve(MeasCovMat::Identity());
        } else {
            S += 1e-6F * MeasCovMat::Identity();
            const Eigen::LLT<MeasCovMat> llt2(S);
            if (llt2.info() != Eigen::Success) {
                // Both attempts failed — keep predicted state, skip update.
                return x_;
            }
            K = P_ * H_.transpose() * llt2.solve(MeasCovMat::Identity());
        }

        // State update: x = x + K * y
        x_ = x_ + K * y;

        // Covariance update: P = (I - K*H) * P
        // Joseph form for numerical stability:
        // P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        const StateMat I_KH = StateMat::Identity() - K * H_;
        P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
    }
    return x_;
}

const StateVec& KalmanFilter::update(const MeasVec& z, float confidence) noexcept {
    if (!initialized_) {
        return update(z);
    }

    // Bug 3 fix: clamp multiplier to [1.0, cap]. Never deflate R below baseline.
    const float eff_conf = std::max(confidence, 0.001F);

    const MeasCovMat R_saved = R_;
    const float raw = adaptive_r_floor_ / eff_conf;
    const float multiplier = std::min(std::max(raw, 1.0F), adaptive_r_cap_);
    R_ = R_saved * multiplier;
    static_cast<void>(update(z));
    R_ = R_saved;

    return x_;
}

void KalmanFilter::apply_gmc(const HomMat& H) noexcept {
    if (!initialized_) { return; }
    if (!hom_is_safe(H)) { return; }

    // Affine 2×2 block.  Bug 1b fix: scale w/h by sqrt(|det|) so the bbox
    // tracks zoom-in/zoom-out camera motion.  Velocities and accelerations
    // transform via the same 2×2 block.
    const float a = H(0, 0), b = H(0, 1);
    const float c = H(1, 0), d = H(1, 1);
    const float det_abs = std::abs(a * d - b * c);
    // F9: floor at 1e-6 (was 1e-12). 1e-12 sqrt -> 1e-6 scale -> bbox shrinks to
    // sub-pixel and cannot recover. 1e-6 sqrt -> 1e-3 scale, still degenerate but
    // bounded; near-singular GMC is detected upstream and should bypass anyway.
    const float s = std::sqrt(std::max(det_abs, 1e-6F));

    // ── 1. Warp centre projectively (handles affine + perspective) ─
    const float w_half = 0.5F * x_(2);
    const float h_half = 0.5F * x_(3);
    const float cx = x_(0) + w_half;
    const float cy = x_(1) + h_half;
    const float pz = H(2, 0) * cx + H(2, 1) * cy + H(2, 2);
    if (std::abs(pz) < 1e-9F) { return; }
    const float cx_n = (a * cx + b * cy + H(0, 2)) / pz;
    const float cy_n = (c * cx + d * cy + H(1, 2)) / pz;

    // ── 2. State warp: pos via centre+size, size by s, vel/acc via 2×2 block ─
    const float w_new = s * x_(2);
    const float h_new = s * x_(3);
    const float vx_old = x_(4), vy_old = x_(5);
    const float ax_old = x_(8), ay_old = x_(9);

    x_(0) = cx_n - 0.5F * w_new;
    x_(1) = cy_n - 0.5F * h_new;
    x_(2) = w_new;
    x_(3) = h_new;
    x_(4) = a * vx_old + b * vy_old;
    x_(5) = c * vx_old + d * vy_old;
    x_(6) *= s;
    x_(7) *= s;
    x_(8) = a * ax_old + b * ay_old;
    x_(9) = c * ax_old + d * ay_old;

    // ── 3. Bug 1 fix: propagate covariance — P ← J · P · Jᵀ ─
    // Jacobian of the affine warp on the state vector.  For (x_new,y_new) we
    // use the centre+size derivation: x_new = a·(x+w/2) + b·(y+h/2) + tx − s·w/2.
    StateMat J = StateMat::Zero();
    // Position rows
    J(0, 0) = a;  J(0, 1) = b;  J(0, 2) = 0.5F * (a - s);  J(0, 3) = 0.5F * b;
    J(1, 0) = c;  J(1, 1) = d;  J(1, 2) = 0.5F * c;        J(1, 3) = 0.5F * (d - s);
    // Size
    J(2, 2) = s;
    J(3, 3) = s;
    // Velocity
    J(4, 4) = a;  J(4, 5) = b;
    J(5, 4) = c;  J(5, 5) = d;
    // Size velocity
    J(6, 6) = s;
    J(7, 7) = s;
    // Acceleration
    J(8, 8) = a;  J(8, 9) = b;
    J(9, 8) = c;  J(9, 9) = d;

    P_ = J * P_ * J.transpose();
    // Symmetrise after multiplication (Trap 5).
    P_ = 0.5F * (P_ + P_.transpose());
}

void KalmanFilter::set_gmc_failed(bool failed) noexcept {
    gmc_failed_ = failed;
}

void KalmanFilter::set_gmc_q_boost(float boost) noexcept {
    // Guard against zero/negative to keep Q positive definite.
    if (boost > 0.0F) { gmc_q_boost_ = boost; }
}

void KalmanFilter::set_adaptive_r_floor(float floor) noexcept {
    // Clamp to (0, 1] — confidence is a probability.
    if (floor > 0.0F && floor <= 1.0F) { adaptive_r_floor_ = floor; }
}

void KalmanFilter::set_adaptive_r_cap(float cap) noexcept {
    if (cap > 1.0F) { adaptive_r_cap_ = cap; }
}

void KalmanFilter::reset() noexcept {
    initialized_ = false;
    predicted_ = false;
    x_ = StateVec::Zero();
    P_ = StateMat::Identity() * 10.0f;
    gmc_failed_ = false;
}

void KalmanFilter::set_process_noise(const StateMat& Q) noexcept {
    Q_ = Q;
}

void KalmanFilter::set_measurement_noise(const MeasCovMat& R) noexcept {
    R_ = R;
}

#ifdef TRACKER_RUST_BRIDGE_ENABLED
void KalmanFilter::set_frame_bounds(float width, float height, float margin) noexcept {
    frame_width_  = width;
    frame_height_ = height;
    frame_margin_ = margin;
}
#endif

}  // namespace tracker
