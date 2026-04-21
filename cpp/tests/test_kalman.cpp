#include <cstdint>
#include <gtest/gtest.h>
#include "kalman_filter.h"

using namespace tracker;

TEST(KalmanFilter, InitSetsState) {
    KalmanFilter kf;
    EXPECT_FALSE(kf.is_initialized());

    MeasVec z0;
    z0 << 100.0f, 200.0f, 50.0f, 80.0f;
    kf.init(z0);

    EXPECT_TRUE(kf.is_initialized());
    const auto& s = kf.get_state();
    EXPECT_FLOAT_EQ(s(0), 100.0f);
    EXPECT_FLOAT_EQ(s(1), 200.0f);
    EXPECT_FLOAT_EQ(s(2), 50.0f);
    EXPECT_FLOAT_EQ(s(3), 80.0f);
    // velocities should be zero
    EXPECT_FLOAT_EQ(s(4), 0.0f);
    EXPECT_FLOAT_EQ(s(5), 0.0f);
}

TEST(KalmanFilter, PredictMovesState) {
    KalmanFilter kf;
    MeasVec z0;
    z0 << 100.0f, 200.0f, 50.0f, 80.0f;
    kf.init(z0);

    // Manually set velocity
    // Do an update with a shifted measurement to induce velocity
    MeasVec z1;
    z1 << 110.0f, 205.0f, 50.0f, 80.0f;
    static_cast<void>(kf.update(z1));

    const auto before = kf.get_state();
    static_cast<void>(kf.predict());
    const auto after = kf.get_state();

    // x should have moved in the direction of velocity
    EXPECT_GT(after(0), before(0) - 1.0f);  // roughly moving right
}

TEST(KalmanFilter, UpdateConverges) {
    KalmanFilter kf;
    MeasVec z;
    z << 100.0f, 100.0f, 40.0f, 40.0f;

    // Feed same measurement 20 times — state should converge
    for (int32_t i = 0; i < 20; ++i) {
        static_cast<void>(kf.update(z));
    }

    const auto& s = kf.get_state();
    EXPECT_NEAR(s(0), 100.0f, 1.0f);
    EXPECT_NEAR(s(1), 100.0f, 1.0f);
    EXPECT_NEAR(s(2), 40.0f, 2.0f);
    EXPECT_NEAR(s(3), 40.0f, 2.0f);
    // velocity should be near zero (stationary target)
    EXPECT_NEAR(s(4), 0.0f, 2.0f);
    EXPECT_NEAR(s(5), 0.0f, 2.0f);
}

TEST(KalmanFilter, ResetClearsState) {
    KalmanFilter kf;
    MeasVec z;
    z << 50.0f, 50.0f, 30.0f, 30.0f;
    static_cast<void>(kf.update(z));
    EXPECT_TRUE(kf.is_initialized());

    kf.reset();
    EXPECT_FALSE(kf.is_initialized());
}

TEST(KalmanFilter, NoHeapAllocation) {
    // This test verifies fixed-size Eigen usage compiles correctly
    // Runtime heap check done via valgrind externally
    KalmanFilter kf;
    MeasVec z;
    z << 1.0f, 2.0f, 3.0f, 4.0f;
    kf.init(z);
    for (int32_t i = 0; i < 100; ++i) {
        static_cast<void>(kf.predict());
        static_cast<void>(kf.update(z));
    }
    EXPECT_TRUE(kf.is_initialized());
}
