// ---------------------------------------------------------------------------
// Project : Tracker
// File    : imm_filter.h
// Purpose : Interacting Multiple Model (IMM) estimator — 3 linear models
//           CV (Constant Velocity), CA (Constant Acceleration), Singer
//           All 10D state, differ only in Q.  No heap, no throw.
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#ifndef TRACKER_IMM_FILTER_H
#define TRACKER_IMM_FILTER_H

#include "kalman_types.h"
#include <cmath>

namespace tracker {

/// Model index constants (AV Rule 151 — named constants over literals)
constexpr int32_t kModelCV     = 0;
constexpr int32_t kModelCA     = 1;
constexpr int32_t kModelSinger = 2;

class IMMFilter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // Eigen fixed-size alignment for pybind11

    IMMFilter() noexcept;
    ~IMMFilter()                           noexcept = default;
    IMMFilter(const IMMFilter&)            noexcept = default;
    IMMFilter& operator=(const IMMFilter&) noexcept = default;
    IMMFilter(IMMFilter&&)                 noexcept = default;
    IMMFilter& operator=(IMMFilter&&)      noexcept = default;

    /// Initialise all models from first measurement @p z0 = [x, y, w, h].
    void init(const MeasVec& z0) noexcept;

    /// IMM predict cycle (mix → per-model predict).
    /// @return Reference to combined predicted state vector.
    [[nodiscard]] const StateVec& predict() noexcept;

    /// Full IMM cycle: mix → predict → likelihood → update → combine.
    /// If not yet initialised, calls init(z) instead.
    /// @return Reference to combined corrected state vector.
    [[nodiscard]] const StateVec& update(const MeasVec& z) noexcept;

    /// Adaptive-R IMM update: R scaled by 1 / max(conf, adaptive_r_floor_).
    [[nodiscard]] const StateVec& update(const MeasVec& z, float confidence) noexcept;

    /// Warp all per-model state means by a 3×3 homography (prev → curr).
    void apply_gmc(const HomMat& H) noexcept;

    /// Flag that GMC failed this frame. Next predict() uses Q*gmc_q_boost across
    /// all models; flag auto-clears after predict.
    void set_gmc_failed(bool failed) noexcept;

    /// Q multiplier used when GMC failed (default 4.0).
    void set_gmc_q_boost(float boost) noexcept;

    /// Confidence floor for adaptive R (default 0.4).
    void set_adaptive_r_floor(float floor) noexcept;

    /// Maximum R multiplier vs baseline for adaptive R (default 10.0).
    /// Prevents extreme noise inflation at very low confidence.
    void set_adaptive_r_cap(float cap) noexcept;

    /// Configure Singer maneuver model physics.
    /// @param alpha   Maneuver correlation time inverse (1/τ). Range: (0, 20].
    ///                α=1 → τ=1 frame, moderate maneuver; α=10 → rapid
    ///                direction change; α→0 degenerates to CA model.
    /// @param sigma2_a Acceleration variance [px²/frame⁴]. Range: (0, 500].
    ///                σ²=25 → ±5 px/frame² std dev. Larger → more agile.
    /// Immediately rebuilds F[Singer] and Q[Singer] from physical equations.
    void set_singer_params(float alpha, float sigma2_a) noexcept;

    /// Enable dynamic Q scaling based on aspect-ratio rate of change.
    /// On each predict(), boost = min(1 + |ΔAR| * sensitivity, boost_cap).
    /// Q_[i] is scaled by boost for that one predict cycle, then restored.
    /// sensitivity=0 disables the feature (default).
    /// Typical values: sensitivity 5–20, boost_cap 2–5.
    void set_ar_q_sensitivity(float sensitivity, float boost_cap = 3.0F) noexcept;

    /// Mahalanobis distance squared for measurement z against combined state.
    ///
    /// d² = (z − H·x̂)ᵀ · S⁻¹ · (z − H·x̂)
    /// where S = H·P_combined·Hᵀ + R  (innovation covariance, 4×4).
    ///
    /// χ²(4 dof) thresholds: p=0.10→7.78, p=0.01→13.28, p=0.001→18.47
    /// Returns kMaxChi2 (1e4) when filter uninitialized or S is singular.
    [[nodiscard]] float mahalanobis_sq(const MeasVec& z) const noexcept;

    /// @return Combined state estimate [x,y,w,h,vx,vy,vw,vh,ax,ay].
    [[nodiscard]] const StateVec& get_state() const noexcept { return x_combined_; }

    /// @return Combined error-covariance matrix (10×10).
    [[nodiscard]] const StateMat& get_covariance() const noexcept { return P_combined_; }

    /// @return Model probabilities [μ_CV, μ_CA, μ_Singer].
    [[nodiscard]] const ModelProb& get_model_probabilities() const noexcept { return mu_; }

    /// @return True once init() or update() has been called at least once.
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// Reset filter to uninitialised state.
    void reset() noexcept;

    /// Restore filter to a checkpoint (combined state, P, mode probs).
    /// Replicates @p x and @p P to all per-model x_[i]/P_[i]; sets mu_ = mu.
    /// initialized_=true, predicted_=false. F/Q/π/R untouched.
    /// Used by ORU re-update to rewind the filter to a saved snapshot.
    void restore_from(const StateVec& x,
                      const StateMat& P,
                      const ModelProb& mu) noexcept;

    /// Override Markov transition matrix (3×3, rows must sum to 1).
    void set_transition_matrix(const TransMat& pi) noexcept;

    /// Override process-noise Q for model @p idx (0=CV, 1=CA, 2=Singer).
    void set_model_process_noise(int32_t idx, const StateMat& Q) noexcept;

    /// Override measurement-noise R (shared by all models).
    void set_measurement_noise(const MeasCovMat& R) noexcept;

private:
    void build_matrices() noexcept;

    /// Rebuild F_[kModelSinger] and Q_[kModelSinger] from singer_alpha_ / singer_sigma2_.
    void build_singer_fq() noexcept;

    /// IMM sub-steps
    void compute_mixing_probabilities() noexcept;
    void mix_states() noexcept;
    void predict_all() noexcept;
    void compute_likelihoods(const MeasVec& z) noexcept;
    void update_all(const MeasVec& z) noexcept;
    void update_model_probabilities() noexcept;
    void combine_estimates() noexcept;

    // ── Per-model filter state ──────────────────────────────────
    StateVec   x_[kNumModels];      ///< model state estimates
    StateMat   P_[kNumModels];      ///< model error covariances
    StateMat   F_[kNumModels];      ///< state transition matrices
    StateMat   Q_[kNumModels];      ///< process noise (PRIMARY differentiator)

    // ── Shared measurement model ────────────────────────────────
    MeasMat    H_;                   ///< 4×10, same for all models
    MeasCovMat R_;                   ///< 4×4, same for all models

    // ── IMM bookkeeping ─────────────────────────────────────────
    ModelProb  mu_;                  ///< model probabilities [3×1]
    TransMat   pi_;                  ///< Markov transition matrix [3×3]
    float      log_likelihood_[kNumModels];  ///< log-likelihoods (for numerical stability)

    // ── Mixing workspace (avoids recomputing) ───────────────────
    float      mu_mix_[kNumModels][kNumModels];  ///< μ_{i|j} mixing weights
    float      c_bar_[kNumModels];               ///< normalising constants
    StateVec   x_mixed_[kNumModels];             ///< mixed initial states
    StateMat   P_mixed_[kNumModels];             ///< mixed initial covariances

    // ── Combined output ─────────────────────────────────────────
    StateVec   x_combined_;
    StateMat   P_combined_;

    bool       initialized_;
    bool       predicted_        = false;  ///< true after predict(), cleared by update()

    // ── GMC / adaptive-R ────────────────────────────────────────
    bool  gmc_failed_       = false;
    float gmc_q_boost_      = 4.0F;
    float adaptive_r_floor_ = 0.4F;
    float adaptive_r_cap_   = 10.0F;  ///< max R multiplier (D2A)

    // ── Singer model physics ─────────────────────────────────────
    float singer_alpha_  = 1.0F;   ///< maneuver time constant inverse (1/τ), default τ=1 frame
    float singer_sigma2_ = 25.0F;  ///< acceleration variance [px²/frame⁴], default ±5 px/frame²

    // ── Dynamic Q from aspect-ratio rate (disabled by default) ───
    float ar_q_sensitivity_ = 0.0F;  ///< px/frame-ratio scale; 0 = disabled
    float ar_q_boost_cap_   = 3.0F;  ///< max Q multiplier per frame
    float prev_meas_ar_     = 0.0F;  ///< AR from last accepted measurement (0 = unset)
};

}  // namespace tracker

#endif // TRACKER_IMM_FILTER_H
