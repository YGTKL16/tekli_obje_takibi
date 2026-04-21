// ---------------------------------------------------------------------------
// Project : Tracker
// File    : test_gmc.cpp
// Purpose : Unit tests for GMC warp, GMC-fail Q-boost, and adaptive-R update
//           across KalmanFilter and IMMFilter.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "imm_filter.h"
#include "kalman_filter.h"

using namespace tracker;

namespace {

MeasVec make_meas(float x, float y, float w, float h) {
    MeasVec z;
    z << x, y, w, h;
    return z;
}

HomMat identity_hom() {
    return HomMat::Identity();
}

HomMat translation_hom(float tx, float ty) {
    HomMat H = HomMat::Identity();
    H(0, 2) = tx;
    H(1, 2) = ty;
    return H;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// KalmanFilter: GMC warp
// ─────────────────────────────────────────────────────────────────

TEST(KalmanFilterGMC, ApplyIdentityLeavesStateUnchanged) {
    KalmanFilter kf;
    kf.init(make_meas(100.0F, 200.0F, 50.0F, 80.0F));
    const StateVec before = kf.get_state();
    kf.apply_gmc(identity_hom());
    const StateVec after = kf.get_state();
    for (int32_t i = 0; i < kStateDim; ++i) {
        EXPECT_NEAR(after(i), before(i), 1e-4F) << "idx=" << i;
    }
}

TEST(KalmanFilterGMC, ApplyTranslationShiftsCentre) {
    KalmanFilter kf;
    // top-left (100, 200), w=50, h=80 → centre (125, 240)
    kf.init(make_meas(100.0F, 200.0F, 50.0F, 80.0F));
    kf.apply_gmc(translation_hom(10.0F, -5.0F));
    const StateVec s = kf.get_state();
    // Centre shifted by (+10, -5); top-left therefore also shifts by (+10, -5).
    EXPECT_NEAR(s(0), 110.0F, 1e-3F);
    EXPECT_NEAR(s(1), 195.0F, 1e-3F);
    // Size unchanged.
    EXPECT_NEAR(s(2), 50.0F, 1e-4F);
    EXPECT_NEAR(s(3), 80.0F, 1e-4F);
    // Velocities unchanged under pure translation (difference of warps).
    EXPECT_NEAR(s(4), 0.0F, 1e-4F);
    EXPECT_NEAR(s(5), 0.0F, 1e-4F);
}

TEST(KalmanFilterGMC, ApplyNaNHomographyIsNoop) {
    KalmanFilter kf;
    kf.init(make_meas(100.0F, 200.0F, 50.0F, 80.0F));
    const StateVec before = kf.get_state();
    HomMat H = HomMat::Identity();
    H(0, 2) = std::numeric_limits<float>::quiet_NaN();
    kf.apply_gmc(H);
    const StateVec after = kf.get_state();
    for (int32_t i = 0; i < kStateDim; ++i) {
        EXPECT_FLOAT_EQ(after(i), before(i));
    }
}

TEST(KalmanFilterGMC, ApplyToUninitialisedIsNoop) {
    KalmanFilter kf;
    // Not initialised — apply_gmc must not touch anything.
    kf.apply_gmc(translation_hom(50.0F, 50.0F));
    EXPECT_FALSE(kf.is_initialized());
}

// ─────────────────────────────────────────────────────────────────
// KalmanFilter: GMC-fail Q-boost
// ─────────────────────────────────────────────────────────────────

TEST(KalmanFilterGMC, FailFlagInflatesPredictCovariance) {
    KalmanFilter kf_baseline;
    KalmanFilter kf_failed;
    const auto z0 = make_meas(100.0F, 100.0F, 50.0F, 50.0F);
    kf_baseline.init(z0);
    kf_failed.init(z0);

    kf_failed.set_gmc_q_boost(4.0F);
    kf_failed.set_gmc_failed(true);

    static_cast<void>(kf_baseline.predict());
    static_cast<void>(kf_failed.predict());

    const StateMat P_base = kf_baseline.get_covariance();
    const StateMat P_boost = kf_failed.get_covariance();

    // Position variance should be strictly larger on the boosted branch.
    EXPECT_GT(P_boost(0, 0), P_base(0, 0));
    EXPECT_GT(P_boost(1, 1), P_base(1, 1));
}

TEST(KalmanFilterGMC, FailFlagAutoClearsAfterPredict) {
    KalmanFilter kf_once;
    KalmanFilter kf_persistent;
    const auto z0 = make_meas(100.0F, 100.0F, 50.0F, 50.0F);
    kf_once.init(z0);
    kf_persistent.init(z0);

    kf_once.set_gmc_failed(true);       // one-shot
    kf_persistent.set_gmc_failed(true); // also one-shot; we re-arm below

    // First predict: both filters boosted.
    static_cast<void>(kf_once.predict());
    static_cast<void>(kf_persistent.predict());

    // Second predict: kf_once should NOT boost (flag auto-cleared);
    // kf_persistent is manually re-armed and should boost again.
    kf_persistent.set_gmc_failed(true);
    static_cast<void>(kf_once.predict());
    static_cast<void>(kf_persistent.predict());

    const StateMat P_once = kf_once.get_covariance();
    const StateMat P_persistent = kf_persistent.get_covariance();

    EXPECT_GT(P_persistent(0, 0), P_once(0, 0));
    EXPECT_GT(P_persistent(1, 1), P_once(1, 1));
}

// ─────────────────────────────────────────────────────────────────
// KalmanFilter: adaptive-R update
// ─────────────────────────────────────────────────────────────────

TEST(KalmanFilterAdaptiveR, HighConfidenceTightensR) {
    // Two filters initialised identically; feed the same jumped measurement.
    // The filter called with high confidence must move further toward z.
    KalmanFilter kf_hi;
    KalmanFilter kf_lo;
    const auto z0 = make_meas(100.0F, 100.0F, 50.0F, 50.0F);
    kf_hi.init(z0);
    kf_lo.init(z0);

    const auto z = make_meas(200.0F, 100.0F, 50.0F, 50.0F);  // big jump in x

    static_cast<void>(kf_hi.update(z, 0.95F));  // tight R → trust z more
    static_cast<void>(kf_lo.update(z, 0.40F));  // wide R  → trust prediction more

    const StateVec s_hi = kf_hi.get_state();
    const StateVec s_lo = kf_lo.get_state();

    EXPECT_GT(s_hi(0), s_lo(0));   // kf_hi closer to 200
    EXPECT_LT(s_hi(0) - 200.0F, s_lo(0) - 100.0F);  // kf_hi residual-to-z smaller
}

TEST(KalmanFilterAdaptiveR, RestoresBaseRAfterCall) {
    KalmanFilter kf;
    kf.init(make_meas(100.0F, 100.0F, 50.0F, 50.0F));
    static_cast<void>(kf.update(make_meas(101.0F, 101.0F, 50.0F, 50.0F), 0.9F));

    // After adaptive call, a follow-up plain update should behave as if R was
    // untouched (i.e., filter still nominally tracking).
    const auto z = make_meas(102.0F, 102.0F, 50.0F, 50.0F);
    const StateVec s = kf.update(z);
    EXPECT_NEAR(s(0), 102.0F, 5.0F);
    EXPECT_NEAR(s(1), 102.0F, 5.0F);
}

// ─────────────────────────────────────────────────────────────────
// IMMFilter: GMC + adaptive-R mirrors of the above
// ─────────────────────────────────────────────────────────────────

TEST(IMMFilterGMC, ApplyTranslationShiftsCombinedCentre) {
    IMMFilter imm;
    imm.init(make_meas(100.0F, 200.0F, 50.0F, 80.0F));
    imm.apply_gmc(translation_hom(15.0F, -7.0F));
    const StateVec s = imm.get_state();
    EXPECT_NEAR(s(0), 115.0F, 1e-3F);
    EXPECT_NEAR(s(1), 193.0F, 1e-3F);
    EXPECT_NEAR(s(2), 50.0F, 1e-4F);
    EXPECT_NEAR(s(3), 80.0F, 1e-4F);
}

TEST(IMMFilterGMC, FailFlagInflatesPredictCovarianceAllModels) {
    IMMFilter imm_base;
    IMMFilter imm_fail;
    const auto z0 = make_meas(100.0F, 100.0F, 50.0F, 50.0F);
    imm_base.init(z0);
    imm_fail.init(z0);

    imm_fail.set_gmc_q_boost(4.0F);
    imm_fail.set_gmc_failed(true);

    static_cast<void>(imm_base.predict());
    static_cast<void>(imm_fail.predict());

    const StateMat P_base = imm_base.get_covariance();
    const StateMat P_fail = imm_fail.get_covariance();

    EXPECT_GT(P_fail(0, 0), P_base(0, 0));
    EXPECT_GT(P_fail(1, 1), P_base(1, 1));
}

TEST(IMMFilterAdaptiveR, HighConfidenceTightensR) {
    IMMFilter imm_hi;
    IMMFilter imm_lo;
    const auto z0 = make_meas(100.0F, 100.0F, 50.0F, 50.0F);
    imm_hi.init(z0);
    imm_lo.init(z0);

    const auto z = make_meas(200.0F, 100.0F, 50.0F, 50.0F);

    static_cast<void>(imm_hi.update(z, 0.95F));
    static_cast<void>(imm_lo.update(z, 0.40F));

    const StateVec s_hi = imm_hi.get_state();
    const StateVec s_lo = imm_lo.get_state();

    EXPECT_GT(s_hi(0), s_lo(0));
}
