// ---------------------------------------------------------------------------
// Project : Tracker
// File    : test_safety_bridge.cpp
// Purpose : Unit tests for the Rust safety bridge (bbox clamping via FFI)
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "tracker_safety_ffi.h"
#include "tracker_safety.h"

// ── FFI-level tests (C linkage) ─────────────────────────────────

TEST(SafetyBridgeFFI, IdentityWhenInBounds) {
    TrackerSafetyBBox bbox{10.0F, 20.0F, 30.0F, 40.0F};
    TrackerSafetyBBox result = tracker_safety_clamp_bbox(bbox, 640.0F, 480.0F, 0.0F);
    EXPECT_FLOAT_EQ(result.x, 10.0F);
    EXPECT_FLOAT_EQ(result.y, 20.0F);
    EXPECT_FLOAT_EQ(result.w, 30.0F);
    EXPECT_FLOAT_EQ(result.h, 40.0F);
}

TEST(SafetyBridgeFFI, ClampsNegativeOrigin) {
    TrackerSafetyBBox bbox{-5.0F, -10.0F, 20.0F, 15.0F};
    TrackerSafetyBBox result = tracker_safety_clamp_bbox(bbox, 100.0F, 100.0F, 0.0F);
    EXPECT_FLOAT_EQ(result.x, 0.0F);
    EXPECT_FLOAT_EQ(result.y, 0.0F);
    EXPECT_GE(result.w, 0.0F);
    EXPECT_GE(result.h, 0.0F);
}

TEST(SafetyBridgeFFI, ClampsOverflow) {
    TrackerSafetyBBox bbox{90.0F, 90.0F, 50.0F, 50.0F};
    TrackerSafetyBBox result = tracker_safety_clamp_bbox(bbox, 100.0F, 100.0F, 0.0F);
    EXPECT_FLOAT_EQ(result.x, 90.0F);
    EXPECT_FLOAT_EQ(result.y, 90.0F);
    EXPECT_LE(result.x + result.w, 100.0F);
    EXPECT_LE(result.y + result.h, 100.0F);
}

TEST(SafetyBridgeFFI, HonorsMargin) {
    TrackerSafetyBBox bbox{0.0F, 0.0F, 100.0F, 100.0F};
    TrackerSafetyBBox result = tracker_safety_clamp_bbox(bbox, 100.0F, 100.0F, 5.0F);
    EXPECT_GE(result.x, 5.0F);
    EXPECT_GE(result.y, 5.0F);
}

TEST(SafetyBridgeFFI, ComponentInterface) {
    TrackerSafetyBBox result = tracker_safety_clamp_bbox_components(
        10.0F, 20.0F, 30.0F, 40.0F, 640.0F, 480.0F, 0.0F);
    EXPECT_FLOAT_EQ(result.x, 10.0F);
    EXPECT_FLOAT_EQ(result.y, 20.0F);
    EXPECT_FLOAT_EQ(result.w, 30.0F);
    EXPECT_FLOAT_EQ(result.h, 40.0F);
}

// ── C++ wrapper tests (MeasVec interface) ───────────────────────

TEST(SafetyBridgeCpp, ClampMeasurementInBounds) {
    tracker::MeasVec z;
    z << 100.0F, 200.0F, 50.0F, 60.0F;
    auto result = tracker::safety::clamp_measurement(z, 640.0F, 480.0F, 0.0F);
    EXPECT_FLOAT_EQ(result(0), 100.0F);
    EXPECT_FLOAT_EQ(result(1), 200.0F);
    EXPECT_FLOAT_EQ(result(2), 50.0F);
    EXPECT_FLOAT_EQ(result(3), 60.0F);
}

TEST(SafetyBridgeCpp, ClampMeasurementOutOfBounds) {
    tracker::MeasVec z;
    z << -10.0F, -20.0F, 800.0F, 600.0F;
    auto result = tracker::safety::clamp_measurement(z, 640.0F, 480.0F, 0.0F);
    EXPECT_GE(result(0), 0.0F);
    EXPECT_GE(result(1), 0.0F);
    EXPECT_LE(static_cast<float>(result(0) + result(2)), 640.0F);
    EXPECT_LE(static_cast<float>(result(1) + result(3)), 480.0F);
}
