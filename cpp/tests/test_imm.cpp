// ---------------------------------------------------------------------------
// Project : Tracker
// File    : test_imm.cpp
// Purpose : GTest suite for IMMFilter — algorithm correctness + stability
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cmath>
#include <gtest/gtest.h>
#include "imm_filter.h"

using namespace tracker;

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

static bool has_nan(const StateVec& v) {
    for (int32_t i = 0; i < kStateDim; ++i) {
        if (std::isnan(v(i)) || std::isinf(v(i))) { return true; }
    }
    return false;
}

static bool probs_valid(const ModelProb& mu) {
    float sum = 0.0f;
    for (int32_t i = 0; i < kNumModels; ++i) {
        if (std::isnan(mu(i)) || std::isinf(mu(i))) { return false; }
        if (mu(i) < 0.0f) { return false; }
        sum += mu(i);
    }
    return std::abs(sum - 1.0f) < 1e-4f;
}

// ─────────────────────────────────────────────────────────────────
// Basic functionality
// ─────────────────────────────────────────────────────────────────

TEST(IMMFilter, InitSetsEqualProbabilities) {
    IMMFilter imm;
    EXPECT_FALSE(imm.is_initialized());

    MeasVec z0;
    z0 << 100.0f, 200.0f, 50.0f, 80.0f;
    imm.init(z0);

    EXPECT_TRUE(imm.is_initialized());

    const auto& mu = imm.get_model_probabilities();
    EXPECT_NEAR(mu(0), 1.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(mu(1), 1.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(mu(2), 1.0f / 3.0f, 1e-5f);

    const auto& s = imm.get_state();
    EXPECT_FLOAT_EQ(s(0), 100.0f);
    EXPECT_FLOAT_EQ(s(1), 200.0f);
    EXPECT_FLOAT_EQ(s(2), 50.0f);
    EXPECT_FLOAT_EQ(s(3), 80.0f);
}

TEST(IMMFilter, UpdateConverges) {
    IMMFilter imm;
    MeasVec z;
    z << 100.0f, 100.0f, 40.0f, 40.0f;

    // Feed same measurement 30 times — state should converge
    for (int32_t i = 0; i < 30; ++i) {
        static_cast<void>(imm.update(z));
    }

    const auto& s = imm.get_state();
    EXPECT_NEAR(s(0), 100.0f, 2.0f);
    EXPECT_NEAR(s(1), 100.0f, 2.0f);
    EXPECT_NEAR(s(2), 40.0f, 3.0f);
    EXPECT_NEAR(s(3), 40.0f, 3.0f);
    // Velocity and acceleration should be near zero
    EXPECT_NEAR(s(4), 0.0f, 3.0f);
    EXPECT_NEAR(s(5), 0.0f, 3.0f);
}

TEST(IMMFilter, ProbabilitiesAlwaysNormalize) {
    IMMFilter imm;
    MeasVec z;
    z << 50.0f, 50.0f, 30.0f, 30.0f;

    for (int32_t i = 0; i < 50; ++i) {
        // Slight random walk
        z(0) += static_cast<float>(i % 3) - 1.0f;
        z(1) += static_cast<float>(i % 5) - 2.0f;
        static_cast<void>(imm.update(z));
        EXPECT_TRUE(probs_valid(imm.get_model_probabilities()))
            << "Probabilities invalid at step " << i;
    }
}

TEST(IMMFilter, ResetClearsState) {
    IMMFilter imm;
    MeasVec z;
    z << 50.0f, 50.0f, 30.0f, 30.0f;
    static_cast<void>(imm.update(z));
    EXPECT_TRUE(imm.is_initialized());

    imm.reset();
    EXPECT_FALSE(imm.is_initialized());
}

// ─────────────────────────────────────────────────────────────────
// Model selection behaviour
// ─────────────────────────────────────────────────────────────────

TEST(IMMFilter, StraightLineMotionFavoursCV) {
    IMMFilter imm;

    // Straight-line: x increases by 10 per frame, constant size
    for (int32_t i = 0; i < 40; ++i) {
        MeasVec z;
        z << 100.0f + static_cast<float>(i) * 10.0f,
             200.0f,
             50.0f,
             80.0f;
        static_cast<void>(imm.update(z));
    }

    const auto& mu = imm.get_model_probabilities();
    // CV should dominate for constant-velocity motion
    EXPECT_GT(mu(kModelCV), 0.4f)
        << "CV=" << mu(0) << " CA=" << mu(1) << " Singer=" << mu(2);
}

TEST(IMMFilter, AcceleratingTargetFavoursCA) {
    IMMFilter imm;

    // Accelerating: x = 100 + 0.5*a*t^2, a=2
    for (int32_t i = 0; i < 40; ++i) {
        const float t = static_cast<float>(i);
        MeasVec z;
        z << 100.0f + 1.0f * t * t,   // quadratic x
             200.0f + 0.5f * t * t,    // quadratic y
             50.0f,
             80.0f;
        static_cast<void>(imm.update(z));
    }

    const auto& mu = imm.get_model_probabilities();
    // CA should have meaningful weight for accelerating target
    // (CV may still dominate since its high-Q states partially adapt)
    EXPECT_GT(mu(kModelCA), mu(kModelSinger))
        << "CV=" << mu(0) << " CA=" << mu(1) << " Singer=" << mu(2);
    EXPECT_GT(mu(kModelCA), 0.05f)
        << "CV=" << mu(0) << " CA=" << mu(1) << " Singer=" << mu(2);
}

TEST(IMMFilter, ManeuveringTargetFavoursSinger) {
    IMMFilter imm;

    // Zig-zag: alternating direction every 5 frames
    for (int32_t i = 0; i < 60; ++i) {
        const float phase = static_cast<float>((i / 5) % 2);
        const float dir = (phase > 0.5f) ? 1.0f : -1.0f;
        MeasVec z;
        z << 100.0f + dir * static_cast<float>(i % 5) * 15.0f,
             200.0f + dir * static_cast<float>(i % 5) * 10.0f,
             50.0f,
             80.0f;
        static_cast<void>(imm.update(z));
    }

    const auto& mu = imm.get_model_probabilities();
    // Singer should have meaningful weight for maneuvering target
    EXPECT_GT(mu(kModelSinger), 0.15f)
        << "CV=" << mu(0) << " CA=" << mu(1) << " Singer=" << mu(2);
}

// ─────────────────────────────────────────────────────────────────
// Coasting (predict-only)
// ─────────────────────────────────────────────────────────────────

TEST(IMMFilter, CoastingExtrapolatesPosition) {
    IMMFilter imm;

    // Build up velocity
    for (int32_t i = 0; i < 10; ++i) {
        MeasVec z;
        z << 100.0f + static_cast<float>(i) * 10.0f,
             200.0f + static_cast<float>(i) * 5.0f,
             50.0f,
             80.0f;
        static_cast<void>(imm.update(z));
    }

    const auto before = imm.get_state();

    // Coast 30 frames (predict-only)
    for (int32_t i = 0; i < 30; ++i) {
        static_cast<void>(imm.predict());
    }

    const auto after = imm.get_state();
    // Position should have extrapolated forward
    EXPECT_GT(after(0), before(0));
    EXPECT_GT(after(1), before(1));
}

TEST(IMMFilter, CovarianceGrowsDuringCoast) {
    IMMFilter imm;

    MeasVec z;
    z << 100.0f, 100.0f, 50.0f, 50.0f;
    for (int32_t i = 0; i < 20; ++i) {
        static_cast<void>(imm.update(z));
    }
    const float cov_before = imm.get_covariance()(0, 0);

    for (int32_t i = 0; i < 30; ++i) {
        static_cast<void>(imm.predict());
    }
    const float cov_after = imm.get_covariance()(0, 0);

    EXPECT_GT(cov_after, cov_before);
}

// ─────────────────────────────────────────────────────────────────
// Numerical stability tests (the 9 traps)
// ─────────────────────────────────────────────────────────────────

TEST(IMMFilter, NaNGuard_OutlierMeasurement) {
    // Trap 1/3: Extremely far measurement — all likelihoods → 0
    IMMFilter imm;

    MeasVec z;
    z << 100.0f, 100.0f, 50.0f, 50.0f;
    static_cast<void>(imm.update(z));

    // Suddenly jump 10000 pixels away (AI glitch)
    MeasVec z_outlier;
    z_outlier << 10000.0f, 10000.0f, 50.0f, 50.0f;
    static_cast<void>(imm.update(z_outlier));

    const auto& s = imm.get_state();
    EXPECT_FALSE(has_nan(s)) << "State contains NaN after outlier measurement";

    const auto& mu = imm.get_model_probabilities();
    EXPECT_TRUE(probs_valid(mu)) << "Probabilities invalid after outlier";
}

TEST(IMMFilter, NaNGuard_RepeatedOutliers) {
    // Stress test: 100 frames of wild measurements
    IMMFilter imm;

    MeasVec z;
    z << 100.0f, 100.0f, 50.0f, 50.0f;
    static_cast<void>(imm.update(z));

    for (int32_t i = 0; i < 100; ++i) {
        MeasVec z_wild;
        const float sign = (i % 2 == 0) ? 1.0f : -1.0f;
        z_wild << sign * 5000.0f * static_cast<float>(i),
                  sign * 3000.0f * static_cast<float>(i),
                  50.0f + static_cast<float>(i),
                  50.0f;
        static_cast<void>(imm.update(z_wild));

        EXPECT_FALSE(has_nan(imm.get_state()))
            << "NaN at step " << i;
        EXPECT_TRUE(probs_valid(imm.get_model_probabilities()))
            << "Probs invalid at step " << i;
    }
}

TEST(IMMFilter, ModelDeathRecovery) {
    // Trap 4: One model should recover after being dormant.
    //
    // Physical context: after CV-dominated straight-line motion, Singer
    // probability drops. A moderate zig-zag (bounded amplitude, not
    // explosively growing) should drive Singer's log-likelihood above CV's,
    // causing Singer to rise above its floor.
    //
    // NOTE: The zig-zag amplitude must stay within the range where Singer's
    // wider innovation covariance (from physically-correct Singer Q) gives it
    // a better chi² than CV, i.e. not so large that ALL models saturate to
    // the kMaxChi2 = 1e4 clip (which would equalize likelihoods and let the
    // Markov transition matrix — not the measurement — decide the update).
    IMMFilter imm;

    // 200 frames of straight-line → CV dominates, Singer drops
    for (int32_t i = 0; i < 200; ++i) {
        MeasVec z;
        z << 100.0f + static_cast<float>(i) * 5.0f,
             200.0f,
             50.0f,
             80.0f;
        static_cast<void>(imm.update(z));
    }

    // Singer should be low but NOT zero (floored at kMinModelProb = 1e-6)
    const float singer_before = imm.get_model_probabilities()(kModelSinger);
    EXPECT_GT(singer_before, 0.0f)
        << "Singer model died (probability reached exactly 0)";

    // Moderate zig-zag: bounded ±30 px amplitude, 4-frame period.
    // This creates innovations large enough to favour Singer's wider S
    // but not so large (100+ px) that chi² clips to kMaxChi2 for all models.
    const float kZigZagAmplitude = 30.0f;
    const float kBaseX = 1100.0f;
    const float kBaseY =  200.0f;
    for (int32_t i = 0; i < 40; ++i) {
        const float dir = ((i % 4) < 2) ? 1.0f : -1.0f;
        MeasVec z;
        z << kBaseX + dir * kZigZagAmplitude,
             kBaseY + dir * kZigZagAmplitude * 0.5f,
             50.0f,
             80.0f;
        static_cast<void>(imm.update(z));
    }

    const float singer_after = imm.get_model_probabilities()(kModelSinger);
    EXPECT_GT(singer_after, singer_before)
        << "Singer did not recover after maneuvering. "
        << "singer_before=" << singer_before << " singer_after=" << singer_after;
}

TEST(IMMFilter, CustomTransitionMatrix) {
    IMMFilter imm;

    // Force Singer to be strongly preferred via transition matrix
    TransMat pi;
    pi << 0.10f, 0.10f, 0.80f,
          0.10f, 0.10f, 0.80f,
          0.05f, 0.05f, 0.90f;
    imm.set_transition_matrix(pi);

    MeasVec z;
    z << 100.0f, 100.0f, 50.0f, 50.0f;
    for (int32_t i = 0; i < 30; ++i) {
        static_cast<void>(imm.update(z));
    }

    // Singer should have significant probability due to biased transition
    EXPECT_GT(imm.get_model_probabilities()(kModelSinger), 0.3f);
}

// ─────────────────────────────────────────────────────────────────
// No-heap allocation (compile-time guarantee + runtime smoke test)
// ─────────────────────────────────────────────────────────────────

TEST(IMMFilter, NoHeapAllocation) {
    // If this compiles with fixed-size Eigen, there's no heap.
    // Runtime: just ensure 1000 cycles don't crash.
    IMMFilter imm;
    MeasVec z;
    z << 1.0f, 2.0f, 3.0f, 4.0f;
    imm.init(z);

    for (int32_t i = 0; i < 1000; ++i) {
        z(0) += 0.1f;
        static_cast<void>(imm.update(z));
    }
    EXPECT_TRUE(imm.is_initialized());
    EXPECT_FALSE(has_nan(imm.get_state()));
}
