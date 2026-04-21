#include <cstdint>
#include <gtest/gtest.h>
#include "kalman_filter.h"
#include "tracker_state.h"

using namespace tracker;

TEST(Coasting, TransitionsToCoasting) {
    TrackerState ts;
    ts.force_tracking();

    // High confidence → stay TRACKING
    EXPECT_EQ(ts.step(0.9f), TrackState::TRACKING);
    EXPECT_EQ(ts.step(0.8f), TrackState::TRACKING);

    // Low confidence → COASTING
    EXPECT_EQ(ts.step(0.1f), TrackState::COASTING);
    EXPECT_EQ(ts.coast_count(), 1);
}

TEST(Coasting, RecoverFromCoasting) {
    TrackerState ts;
    ts.force_tracking();

    // Enter coasting
    static_cast<void>(ts.step(0.1f));
    EXPECT_EQ(ts.state(), TrackState::COASTING);

    // Recover with high confidence
    static_cast<void>(ts.step(0.8f));
    EXPECT_EQ(ts.state(), TrackState::TRACKING);
    EXPECT_EQ(ts.coast_count(), 0);
}

TEST(Coasting, LostAfterMaxFrames) {
    TrackerState ts;
    ts.set_max_coast_frames(10);
    ts.force_tracking();

    // Enter coasting
    static_cast<void>(ts.step(0.1f));
    EXPECT_EQ(ts.state(), TrackState::COASTING);

    // Coast for max frames
    for (int32_t i = 0; i < 10; ++i) {
        static_cast<void>(ts.step(0.1f));
    }
    EXPECT_EQ(ts.state(), TrackState::LOST);
}

TEST(Coasting, KalmanPredictOnlyDuringCoast) {
    KalmanFilter kf;
    TrackerState ts;
    ts.force_tracking();

    // Init KF with measurement
    MeasVec z;
    z << 100.0f, 100.0f, 50.0f, 50.0f;
    kf.init(z);

    // Simulate moving target
    for (int32_t i = 1; i <= 5; ++i) {
        MeasVec m;
        m << 100.0f + static_cast<float>(i) * 10.0f, 100.0f + static_cast<float>(i) * 5.0f, 50.0f, 50.0f;
        static_cast<void>(kf.update(m));
        static_cast<void>(ts.step(0.9f));
    }
    EXPECT_EQ(ts.state(), TrackState::TRACKING);

    const auto state_before_coast = kf.get_state();

    // Now target disappears — 30 frames of coasting
    for (int32_t i = 0; i < 30; ++i) {
        static_cast<void>(ts.step(0.05f));
        if (ts.state() == TrackState::COASTING || ts.state() == TrackState::TRACKING) {
            static_cast<void>(kf.predict());
        }
    }
    EXPECT_EQ(ts.state(), TrackState::COASTING);

    const auto state_after_coast = kf.get_state();

    // KF should have extrapolated position forward
    EXPECT_GT(state_after_coast(0), state_before_coast(0));
    EXPECT_GT(state_after_coast(1), state_before_coast(1));
}

TEST(Coasting, CovarianceGrowsDuringCoast) {
    KalmanFilter kf;
    MeasVec z;
    z << 100.0f, 100.0f, 50.0f, 50.0f;

    // Converge
    for (int32_t i = 0; i < 20; ++i) {
        static_cast<void>(kf.update(z));
    }
    const float cov_before = kf.get_covariance()(0, 0);

    // Predict without update (coasting)
    for (int32_t i = 0; i < 30; ++i) {
        static_cast<void>(kf.predict());
    }
    const float cov_after = kf.get_covariance()(0, 0);

    // Uncertainty should grow during coasting
    EXPECT_GT(cov_after, cov_before);
}
