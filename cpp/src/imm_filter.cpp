// ---------------------------------------------------------------------------
// Project : Tracker
// File    : imm_filter.cpp
// Purpose : IMM estimator — 3 linear models (CV + CA + Singer)
//           Full numerical-stability armour (log-likelihood, μ flooring,
//           covariance symmetry enforcement, LLT-based Kalman gain).
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#include "imm_filter.h"

#include <algorithm>  // std::max
#include <cmath>      // std::log, std::exp, std::abs

namespace tracker {

// ── Mathematical constants (AV Rule 151) ────────────────────────
static constexpr float kPi          = 3.14159265358979323846f;
static constexpr float kLogTwoPi    = 1.8378770664093453f;   // log(2π)
static constexpr float kMinModelProb = 1e-6f;   // floor for model probabilities (Trap 4)
static constexpr float kLikelihoodEps = 1e-15f;  // total-likelihood guard (Trap 3)
static constexpr float kMaxChi2     = 1e4f;      // Mahalanobis clip (Trap 8)
static constexpr float kCovRegEps   = 1e-6f;     // PD regularisation (Trap 7)

// ─────────────────────────────────────────────────────────────────
// Construction / Reset
// ─────────────────────────────────────────────────────────────────

IMMFilter::IMMFilter() noexcept : initialized_(false) {
    build_matrices();
}

void IMMFilter::build_matrices() noexcept {
    // ── Measurement matrix H (shared, 4×10): observe [x,y,w,h] ─
    H_ = MeasMat::Zero();
    H_(0, 0) = 1.0f;
    H_(1, 1) = 1.0f;
    H_(2, 2) = 1.0f;
    H_(3, 3) = 1.0f;

    // ── Measurement noise R (shared, 4×4) ───────────────────────
    R_ = MeasCovMat::Zero();
    R_(0, 0) = 1.0f;
    R_(1, 1) = 1.0f;
    R_(2, 2) = 4.0f;
    R_(3, 3) = 4.0f;

    // ── State transition F — all models share the linear 10D structure,
    //    with acceleration-to-velocity coupling (dt=1) ───────────
    for (int32_t m = 0; m < kNumModels; ++m) {
        F_[m] = StateMat::Identity();
        F_[m](0, 4) = 1.0f;  // x  += vx
        F_[m](1, 5) = 1.0f;  // y  += vy
        F_[m](2, 6) = 1.0f;  // w  += vw
        F_[m](3, 7) = 1.0f;  // h  += vh
        F_[m](4, 8) = 1.0f;  // vx += ax·dt
        F_[m](5, 9) = 1.0f;  // vy += ay·dt
    }

    // ── Process noise Q — the PRIMARY differentiator ────────────
    // Model 0 (CV): acceleration states nearly inert, size near-static
    Q_[kModelCV] = StateMat::Zero();
    Q_[kModelCV](0, 0) = 0.1f;       // x
    Q_[kModelCV](1, 1) = 0.1f;       // y
    Q_[kModelCV](2, 2) = 1.0f;       // w
    Q_[kModelCV](3, 3) = 1.0f;       // h
    Q_[kModelCV](4, 4) = 0.01f;      // vx
    Q_[kModelCV](5, 5) = 0.01f;      // vy
    Q_[kModelCV](6, 6) = 0.0001f;    // vw  (size almost frozen)
    Q_[kModelCV](7, 7) = 0.0001f;    // vh
    Q_[kModelCV](8, 8) = 1e-6f;      // ax  (nearly frozen)
    Q_[kModelCV](9, 9) = 1e-6f;      // ay

    // Model 1 (CA): moderate motion + moderate scale change (approach/recede)
    Q_[kModelCA] = StateMat::Zero();
    Q_[kModelCA](0, 0) = 0.1f;
    Q_[kModelCA](1, 1) = 0.1f;
    Q_[kModelCA](2, 2) = 1.0f;
    Q_[kModelCA](3, 3) = 1.0f;
    Q_[kModelCA](4, 4) = 0.1f;
    Q_[kModelCA](5, 5) = 0.1f;
    Q_[kModelCA](6, 6) = 0.01f;      // vw  (scale drift)
    Q_[kModelCA](7, 7) = 0.01f;      // vh
    Q_[kModelCA](8, 8) = 1.0f;       // ax  (active)
    Q_[kModelCA](9, 9) = 1.0f;       // ay

    // Model 2 (Singer): physically derived maneuver model.
    // F and Q are computed from correlation time alpha and acceleration variance sigma2.
    // build_singer_fq() overwrites F_[kModelSinger] and Q_[kModelSinger] in place.
    build_singer_fq();

    // ── Markov transition matrix π (3×3) ────────────────────────
    pi_ << 0.90f, 0.05f, 0.05f,   // CV  → CV/CA/Singer
           0.05f, 0.90f, 0.05f,   // CA  → CV/CA/Singer
           0.10f, 0.10f, 0.80f;   // Singer → CV/CA/Singer

    // ── Equal initial model probabilities ───────────────────────
    mu_ << 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f;

    // ── Zero-init workspace ─────────────────────────────────────
    for (int32_t m = 0; m < kNumModels; ++m) {
        x_[m]       = StateVec::Zero();
        P_[m]       = StateMat::Identity() * 10.0f;
        x_mixed_[m] = StateVec::Zero();
        P_mixed_[m] = StateMat::Identity() * 10.0f;
        log_likelihood_[m] = 0.0f;
        c_bar_[m]   = 1.0f / static_cast<float>(kNumModels);
        for (int32_t i = 0; i < kNumModels; ++i) {
            mu_mix_[i][m] = 1.0f / static_cast<float>(kNumModels);
        }
    }

    x_combined_ = StateVec::Zero();
    P_combined_ = StateMat::Identity() * 10.0f;
}

// ─────────────────────────────────────────────────────────────────
// init
// ─────────────────────────────────────────────────────────────────

void IMMFilter::init(const MeasVec& z0) noexcept {
    if (meas_has_nan_or_inf(z0)) { return; }

    for (int32_t m = 0; m < kNumModels; ++m) {
        x_[m] = StateVec::Zero();
        x_[m](0) = z0(0);
        x_[m](1) = z0(1);
        x_[m](2) = z0(2);
        x_[m](3) = z0(3);

        P_[m] = StateMat::Zero();
        P_[m](0, 0) = 2.0f;
        P_[m](1, 1) = 2.0f;
        P_[m](2, 2) = 10.0f;
        P_[m](3, 3) = 10.0f;
        P_[m](4, 4) = 25.0f;
        P_[m](5, 5) = 25.0f;
        P_[m](6, 6) = 25.0f;
        P_[m](7, 7) = 25.0f;
        P_[m](8, 8) = 100.0f;
        P_[m](9, 9) = 100.0f;
    }

    mu_ << 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f;

    // Set combined output to initial measurement
    x_combined_ = x_[0];
    P_combined_ = P_[0];

    initialized_ = true;
}

// ─────────────────────────────────────────────────────────────────
// IMM Step 1: Compute mixing probabilities  μ_{i|j}
//
//   c̄_j = Σ_i  π_{i,j} · μ_i
//   μ_{i|j} = π_{i,j} · μ_i  /  c̄_j
// ─────────────────────────────────────────────────────────────────

void IMMFilter::compute_mixing_probabilities() noexcept {
    for (int32_t j = 0; j < kNumModels; ++j) {
        c_bar_[j] = 0.0f;
        for (int32_t i = 0; i < kNumModels; ++i) {
            c_bar_[j] += pi_(i, j) * mu_(i);
        }
        // Guard against zero normaliser
        if (c_bar_[j] < kLikelihoodEps) {
            c_bar_[j] = kLikelihoodEps;
        }
        for (int32_t i = 0; i < kNumModels; ++i) {
            mu_mix_[i][j] = pi_(i, j) * mu_(i) / c_bar_[j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// IMM Step 2: Mix states and covariances
//
//   x̄_j = Σ_i  μ_{i|j} · x_i
//   P̄_j = Σ_i  μ_{i|j} · [ P_i + (x_i − x̄_j)(x_i − x̄_j)^T ]
// ─────────────────────────────────────────────────────────────────

void IMMFilter::mix_states() noexcept {
    for (int32_t j = 0; j < kNumModels; ++j) {
        // Mixed state
        x_mixed_[j] = StateVec::Zero();
        for (int32_t i = 0; i < kNumModels; ++i) {
            x_mixed_[j] += mu_mix_[i][j] * x_[i];
        }

        // Mixed covariance (with spread-of-means)
        P_mixed_[j] = StateMat::Zero();
        for (int32_t i = 0; i < kNumModels; ++i) {
            const StateVec dx = x_[i] - x_mixed_[j];
            P_mixed_[j] += mu_mix_[i][j] * (P_[i] + dx * dx.transpose());
        }

        // Trap 5: Enforce symmetry after mixing
        P_mixed_[j] = 0.5f * (P_mixed_[j] + P_mixed_[j].transpose());
    }
}

// ─────────────────────────────────────────────────────────────────
// IMM Step 3: Per-model predict
//
//   x_j = F_j · x̄_j
//   P_j = F_j · P̄_j · F_j^T + Q_j
// ─────────────────────────────────────────────────────────────────

void IMMFilter::predict_all() noexcept {
    const float q_scale = gmc_failed_ ? gmc_q_boost_ : 1.0F;
    for (int32_t m = 0; m < kNumModels; ++m) {
        x_[m] = F_[m] * x_mixed_[m];
        P_[m] = F_[m] * P_mixed_[m] * F_[m].transpose() + Q_[m] * q_scale;

        // Trap 5: Enforce symmetry after predict
        P_[m] = 0.5f * (P_[m] + P_[m].transpose());
    }
    // Boost is one-shot — clear after all models consumed it.
    gmc_failed_ = false;
}

// ─────────────────────────────────────────────────────────────────
// IMM Step 4: Compute log-likelihoods (Traps 1, 2, 8)
//
//   y_j = z − H · x_j
//   S_j = H · P_j · H^T + R
//   log Λ_j = −0.5 · (χ² + log|S_j| + k · log(2π))
//
// Uses log-determinant (sum of log pivots) to avoid |S| underflow.
// Clips Mahalanobis distance to prevent overflow.
// ─────────────────────────────────────────────────────────────────

void IMMFilter::compute_likelihoods(const MeasVec& z) noexcept {
    for (int32_t m = 0; m < kNumModels; ++m) {
        // Innovation
        const MeasVec y = z - H_ * x_[m];

        // Innovation covariance
        MeasCovMat S = H_ * P_[m] * H_.transpose() + R_;
        // Symmetrise S (paranoia — inherited rounding from P)
        S = 0.5f * (S + S.transpose());

        // Trap 2: Log-determinant via LU pivots (avoids raw |S| underflow)
        const auto lu = S.partialPivLu();
        float log_abs_det = 0.0f;
        bool singular = false;
        for (int32_t i = 0; i < kMeasDim; ++i) {
            const float diag_abs = std::abs(lu.matrixLU()(i, i));
            if (diag_abs < 1e-35f) {
                singular = true;
                break;
            }
            log_abs_det += std::log(diag_abs);
        }

        if (singular) {
            // Near-singular S → this model is degenerate; give worst log-likelihood
            log_likelihood_[m] = -1e10f;
        } else {
            // Trap 6: Solve via LU (numerically stable)
            const MeasVec S_inv_y = lu.solve(y);

            // Mahalanobis distance χ² = yᵀ S⁻¹ y
            float chi2 = y.dot(S_inv_y);

            // Trap 8: Clip χ² to prevent extreme values
            chi2 = std::min(chi2, kMaxChi2);

            // log Λ = −0.5 · (χ² + log|S| + k · log(2π))
            log_likelihood_[m] = -0.5f * (chi2 + log_abs_det
                                           + static_cast<float>(kMeasDim) * kLogTwoPi);
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// IMM Step 5: Per-model Kalman update (Joseph form)
// ─────────────────────────────────────────────────────────────────

void IMMFilter::update_all(const MeasVec& z) noexcept {
    for (int32_t m = 0; m < kNumModels; ++m) {
        const MeasVec y = z - H_ * x_[m];

        MeasCovMat S = H_ * P_[m] * H_.transpose() + R_;
        S = 0.5f * (S + S.transpose());

        // Trap 6: Kalman gain via Cholesky (LLT)
        const Eigen::LLT<MeasCovMat> llt(S);
        if (llt.info() != Eigen::Success) {
            // Trap 7: S not PD — add regularisation and retry
            S += kCovRegEps * MeasCovMat::Identity();
            const Eigen::LLT<MeasCovMat> llt2(S);
            const KalmanGain K = P_[m] * H_.transpose()
                * llt2.solve(MeasCovMat::Identity());
            x_[m] = x_[m] + K * y;
            const StateMat I_KH = StateMat::Identity() - K * H_;
            P_[m] = I_KH * P_[m] * I_KH.transpose() + K * R_ * K.transpose();
        } else {
            const KalmanGain K = P_[m] * H_.transpose()
                * llt.solve(MeasCovMat::Identity());
            x_[m] = x_[m] + K * y;
            const StateMat I_KH = StateMat::Identity() - K * H_;
            P_[m] = I_KH * P_[m] * I_KH.transpose() + K * R_ * K.transpose();
        }

        // Trap 5: Enforce symmetry after Joseph update
        P_[m] = 0.5f * (P_[m] + P_[m].transpose());
    }
}

// ─────────────────────────────────────────────────────────────────
// IMM Step 6: Update model probabilities (Traps 1, 3, 4)
//
// Uses log-sum-exp to avoid underflow.
// Floor μ at kMinModelProb to prevent permanent model death.
// ─────────────────────────────────────────────────────────────────

void IMMFilter::update_model_probabilities() noexcept {
    // Incorporate mixing normaliser c̄ in log-domain
    float log_mu[kNumModels];
    for (int32_t m = 0; m < kNumModels; ++m) {
        log_mu[m] = log_likelihood_[m] + std::log(std::max(c_bar_[m], kLikelihoodEps));
    }

    // Log-sum-exp trick (Trap 1): shift by max to prevent underflow
    float max_log = log_mu[0];
    for (int32_t m = 1; m < kNumModels; ++m) {
        max_log = std::max(max_log, log_mu[m]);
    }

    float exp_sum = 0.0f;
    float exp_vals[kNumModels];
    for (int32_t m = 0; m < kNumModels; ++m) {
        exp_vals[m] = std::exp(log_mu[m] - max_log);
        exp_sum += exp_vals[m];
    }

    // Trap 3: Total likelihood zero guard
    if (exp_sum < kLikelihoodEps) {
        const float uniform = 1.0f / static_cast<float>(kNumModels);
        for (int32_t m = 0; m < kNumModels; ++m) {
            mu_(m) = uniform;
        }
    } else {
        for (int32_t m = 0; m < kNumModels; ++m) {
            mu_(m) = exp_vals[m] / exp_sum;
        }
    }

    // Trap 4: Floor to prevent permanent model death, then renormalise
    float mu_sum = 0.0f;
    for (int32_t m = 0; m < kNumModels; ++m) {
        mu_(m) = std::max(mu_(m), kMinModelProb);
        mu_sum += mu_(m);
    }
    for (int32_t m = 0; m < kNumModels; ++m) {
        mu_(m) /= mu_sum;
    }
}

// ─────────────────────────────────────────────────────────────────
// IMM Step 7: Combine estimates (weighted mean + spread-of-means)
//
//   x_c = Σ_j  μ_j · x_j
//   P_c = Σ_j  μ_j · [ P_j + (x_j − x_c)(x_j − x_c)^T ]
// ─────────────────────────────────────────────────────────────────

void IMMFilter::combine_estimates() noexcept {
    x_combined_ = StateVec::Zero();
    for (int32_t m = 0; m < kNumModels; ++m) {
        x_combined_ += mu_(m) * x_[m];
    }

    P_combined_ = StateMat::Zero();
    for (int32_t m = 0; m < kNumModels; ++m) {
        const StateVec dx = x_[m] - x_combined_;
        P_combined_ += mu_(m) * (P_[m] + dx * dx.transpose());
    }

    // Trap 5: Enforce symmetry on combined covariance
    P_combined_ = 0.5f * (P_combined_ + P_combined_.transpose());
}

// ─────────────────────────────────────────────────────────────────
// Public: predict (mix → predict → combine)
// ─────────────────────────────────────────────────────────────────

const StateVec& IMMFilter::predict() noexcept {
    if (initialized_) {
        compute_mixing_probabilities();
        mix_states();
        predict_all();
        combine_estimates();
        predicted_ = true;
    }
    return x_combined_;
}

// ─────────────────────────────────────────────────────────────────
// Public: update (full IMM cycle)
// ─────────────────────────────────────────────────────────────────

const StateVec& IMMFilter::update(const MeasVec& z) noexcept {
    if (meas_has_nan_or_inf(z)) { return x_combined_; }

    if (!initialized_) {
        init(z);
    } else {
        // Steps 1-3: Mixing + predict — skip if predict() was already called.
        if (!predicted_) {
            compute_mixing_probabilities();
            mix_states();
            predict_all();
        }
        predicted_ = false;

        // Step 4: Compute log-likelihoods
        compute_likelihoods(z);

        // Step 5: Per-model Kalman update (Joseph form)
        update_all(z);

        // Step 6: Update model probabilities (log-sum-exp)
        update_model_probabilities();

        // Step 7: Combine estimates
        combine_estimates();
    }
    return x_combined_;
}

// ─────────────────────────────────────────────────────────────────
// Public: reset
// ─────────────────────────────────────────────────────────────────

void IMMFilter::reset() noexcept {
    initialized_ = false;
    predicted_ = false;
    build_matrices();
}

// ─────────────────────────────────────────────────────────────────
// Public: setters
// ─────────────────────────────────────────────────────────────────

void IMMFilter::set_transition_matrix(const TransMat& pi) noexcept {
    pi_ = pi;
}

void IMMFilter::set_model_process_noise(int32_t idx, const StateMat& Q) noexcept {
    if (idx >= 0 && idx < kNumModels) {
        Q_[idx] = Q;
    }
}

void IMMFilter::set_measurement_noise(const MeasCovMat& R) noexcept {
    R_ = R;
}

// ─────────────────────────────────────────────────────────────────
// GMC / adaptive-R extensions
// ─────────────────────────────────────────────────────────────────

namespace {

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

void warp_state_inplace(StateVec& x, const HomMat& H) noexcept {
    const float w_half = 0.5F * x(2);
    const float h_half = 0.5F * x(3);
    const float cx = x(0) + w_half;
    const float cy = x(1) + h_half;

    const float pz = H(2, 0) * cx + H(2, 1) * cy + H(2, 2);
    if (std::abs(pz) < 1e-9F) { return; }
    const float cx_n = (H(0, 0) * cx + H(0, 1) * cy + H(0, 2)) / pz;
    const float cy_n = (H(1, 0) * cx + H(1, 1) * cy + H(1, 2)) / pz;

    const float qx = cx + x(4);
    const float qy = cy + x(5);
    const float qz = H(2, 0) * qx + H(2, 1) * qy + H(2, 2);
    if (std::abs(qz) < 1e-9F) { return; }
    const float qxn = (H(0, 0) * qx + H(0, 1) * qy + H(0, 2)) / qz;
    const float qyn = (H(1, 0) * qx + H(1, 1) * qy + H(1, 2)) / qz;

    x(0) = cx_n - w_half;
    x(1) = cy_n - h_half;
    x(4) = qxn - cx_n;
    x(5) = qyn - cy_n;
}

}  // namespace

const StateVec& IMMFilter::update(const MeasVec& z, float confidence) noexcept {
    if (!initialized_) {
        return update(z);
    }

    const float eff_conf = std::max(confidence, adaptive_r_floor_);

    const MeasCovMat R_saved = R_;
    R_ = R_saved * (adaptive_r_floor_ / eff_conf);
    static_cast<void>(update(z));
    R_ = R_saved;

    return x_combined_;
}

void IMMFilter::apply_gmc(const HomMat& H) noexcept {
    if (!initialized_) { return; }
    if (!hom_is_safe(H)) { return; }

    for (int32_t m = 0; m < kNumModels; ++m) {
        warp_state_inplace(x_[m], H);
    }
    warp_state_inplace(x_combined_, H);
}

void IMMFilter::set_gmc_failed(bool failed) noexcept {
    gmc_failed_ = failed;
}

void IMMFilter::set_gmc_q_boost(float boost) noexcept {
    if (boost > 0.0F) { gmc_q_boost_ = boost; }
}

void IMMFilter::set_adaptive_r_floor(float floor) noexcept {
    if (floor > 0.0F && floor <= 1.0F) { adaptive_r_floor_ = floor; }
}

// ─────────────────────────────────────────────────────────────────
// Singer model: physically derived F and Q
//
// Maneuver model:  a(k+1) = β·a(k) + w,  β = exp(-α·Δt)
//                  w ~ N(0, 2ασ²_a)
//
// State mapping (per spatial axis):
//   x  → pos index (0=x, 1=y)
//   vx → vel index (4=vx, 5=vy)
//   ax → acc index (8=ax, 9=ay)
// ─────────────────────────────────────────────────────────────────

void IMMFilter::build_singer_fq() noexcept {
    constexpr float kDt = 1.0F;  // 1 frame = 1 time unit (AV Rule 151)

    // Guard against near-zero alpha (would degenerate to CA numerically)
    const float alpha = std::max(singer_alpha_, 1e-4F);
    const float s2a   = std::max(singer_sigma2_, 1e-6F);
    const float beta  = std::exp(-alpha * kDt);

    // Pre-computed powers (AV Rule 151 — named, not inlined)
    const float a1 = alpha;
    const float a2 = a1 * a1;
    const float a3 = a2 * a1;
    const float a4 = a3 * a1;
    const float a5 = a4 * a1;
    const float b2 = beta * beta;

    // ── F_[kModelSinger]: rebuild from scratch ─────────────────
    F_[kModelSinger] = StateMat::Identity();
    // Position += velocity·Δt
    F_[kModelSinger](0, 4) = kDt;
    F_[kModelSinger](1, 5) = kDt;
    // w/h coupled to vw/vh (size change rate)
    F_[kModelSinger](2, 6) = kDt;
    F_[kModelSinger](3, 7) = kDt;
    // Position += integral of exp-correlated acceleration: (β-1+αΔt)/α²
    const float f_pa = (beta - 1.0F + alpha * kDt) / a2;
    F_[kModelSinger](0, 8) = f_pa;
    F_[kModelSinger](1, 9) = f_pa;
    // Velocity += (1-β)/α · acceleration
    const float f_va = (1.0F - beta) / a1;
    F_[kModelSinger](4, 8) = f_va;
    F_[kModelSinger](5, 9) = f_va;
    // Acceleration decays: a(k+1) = β·a(k)
    F_[kModelSinger](8, 8) = beta;
    F_[kModelSinger](9, 9) = beta;

    // ── Q_[kModelSinger]: physically derived cross-covariance ─────
    Q_[kModelSinger] = StateMat::Zero();
    const float q_pp = s2a * (2.0F*a3*kDt - 3.0F + 4.0F*beta - b2) / (2.0F*a5);
    const float q_pv = s2a * (1.0F - 2.0F*alpha*kDt*beta - b2)      / (2.0F*a4);
    const float q_vv = s2a * (1.0F - b2)                             / (2.0F*a3);
    const float q_aa = s2a * (1.0F - b2)                             / a1;

    // x-axis spatial cross terms: pos=0, vel=4, acc=8
    Q_[kModelSinger](0, 0) = q_pp;
    Q_[kModelSinger](0, 4) = q_pv;
    Q_[kModelSinger](4, 0) = q_pv;
    Q_[kModelSinger](4, 4) = q_vv;
    Q_[kModelSinger](8, 8) = q_aa;

    // y-axis spatial cross terms: pos=1, vel=5, acc=9
    Q_[kModelSinger](1, 1) = q_pp;
    Q_[kModelSinger](1, 5) = q_pv;
    Q_[kModelSinger](5, 1) = q_pv;
    Q_[kModelSinger](5, 5) = q_vv;
    Q_[kModelSinger](9, 9) = q_aa;

    // Size axes: simple diagonal (no physical maneuver model for w/h)
    Q_[kModelSinger](2, 2) = 1.0F;
    Q_[kModelSinger](3, 3) = 1.0F;
    Q_[kModelSinger](6, 6) = 0.1F;
    Q_[kModelSinger](7, 7) = 0.1F;
}

void IMMFilter::set_singer_params(float alpha, float sigma2_a) noexcept {
    singer_alpha_  = std::max(alpha,    1e-4F);  // guard: no div-by-zero in a2..a5
    singer_sigma2_ = std::max(sigma2_a, 1e-6F);  // guard: no negative variance
    build_singer_fq();
}

// ─────────────────────────────────────────────────────────────────
// Mahalanobis distance² against combined state
//
// d² = (z - H·x_c)ᵀ · S⁻¹ · (z - H·x_c)
// S   = H·P_combined·Hᵀ + R
// ─────────────────────────────────────────────────────────────────

float IMMFilter::mahalanobis_sq(const MeasVec& z) const noexcept {
    if (!initialized_) {
        return kMaxChi2;  // uninitialised → reject all
    }
    if (meas_has_nan_or_inf(z)) {
        return kMaxChi2;  // degenerate input → reject
    }

    const MeasVec innov = z - H_ * x_combined_;

    // Innovation covariance from combined estimate
    MeasCovMat S = H_ * P_combined_ * H_.transpose() + R_;
    S = 0.5F * (S + S.transpose());  // symmetry guard (Trap 5)

    // Trap 6+7: LLT for PD guarantee; fall back to kMaxChi2 on singular S
    const Eigen::LLT<MeasCovMat> llt(S);
    if (llt.info() != Eigen::Success) {
        return kMaxChi2;
    }
    const MeasVec S_inv_innov = llt.solve(innov);
    const float d2 = innov.dot(S_inv_innov);
    // Clamp to [0, kMaxChi2] — dot product is theoretically ≥0 but guard anyway
    return std::min(std::max(d2, 0.0F), kMaxChi2);
}

}  // namespace tracker
