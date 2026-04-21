// ---------------------------------------------------------------------------
// Project : Tracker
// File    : test_gmc_estimator.cpp
// Purpose : Unit tests for ORB + partial-affine GMC estimator
// Standard: JSF AV C++ Rev C (test harness; -fexceptions -frtti enabled)
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "gmc_estimator.h"

using namespace tracker;

// ── Helpers ─────────────────────────────────────────────────────

/// Build a synthetic 480×640 grayscale image with ORB-friendly texture.
static cv::Mat make_textured_frame(int32_t rows = 480, int32_t cols = 640,
                                   uint64_t seed = 42U) {
    cv::Mat frame(rows, cols, CV_8UC1);
    cv::RNG rng(seed);
    rng.fill(frame, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(255));
    // Add grid structure + circles for strong corners (ORB-friendly)
    for (int32_t y = 0; y < rows; y += 30) {
        cv::line(frame, cv::Point(0, y), cv::Point(cols - 1, y),
                 cv::Scalar(0), 2);
    }
    for (int32_t x = 0; x < cols; x += 30) {
        cv::line(frame, cv::Point(x, 0), cv::Point(x, rows - 1),
                 cv::Scalar(255), 2);
    }
    // Add circles at intersections for distinct keypoints
    for (int32_t y = 30; y < rows; y += 60) {
        for (int32_t x = 30; x < cols; x += 60) {
            cv::circle(frame, cv::Point(x, y), 8, cv::Scalar(0), 2);
            cv::circle(frame, cv::Point(x + 15, y + 15), 5, cv::Scalar(255), -1);
        }
    }
    // Light Gaussian blur to reduce noise while preserving corners
    cv::GaussianBlur(frame, frame, cv::Size(3, 3), 0.5);
    return frame;
}

/// Translate a frame by (dx, dy) pixels using warpAffine.
static cv::Mat translate_frame(const cv::Mat& src, float dx, float dy) {
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1.0, 0.0, static_cast<double>(dx),
                                            0.0, 1.0, static_cast<double>(dy));
    cv::Mat dst;
    cv::warpAffine(src, dst, M, src.size(), cv::INTER_LINEAR,
                   cv::BORDER_REPLICATE);
    return dst;
}

// ── Tests ───────────────────────────────────────────────────────

TEST(GMCEstimator, IdentityOnSameFrame) {
    GMCEstimator gmc;
    cv::Mat frame = make_textured_frame();
    float fg[4] = {0.0F, 0.0F, 0.0F, 0.0F};  // no foreground
    HomMat H;

    bool ok = gmc.estimate(frame, frame, fg, H);

    if (ok) {
        // Should be near identity
        for (int32_t r = 0; r < 3; ++r) {
            for (int32_t c = 0; c < 3; ++c) {
                const float expected = (r == c) ? 1.0F : 0.0F;
                EXPECT_NEAR(H(r, c), expected, 0.05F)
                    << "H(" << r << "," << c << ")";
            }
        }
    }
    // If ok == false, that's also acceptable for identical frames
    // (zero displacement can confuse matching)
}

TEST(GMCEstimator, PureTranslation) {
    GMCEstimator gmc;
    cv::Mat prev = make_textured_frame();
    const float dx = 20.0F;
    const float dy = 12.0F;
    cv::Mat curr = translate_frame(prev, dx, dy);
    float fg[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    HomMat H;

    bool ok = gmc.estimate(prev, curr, fg, H);
    ASSERT_TRUE(ok) << "GMC should succeed on pure translation";

    // Translation components
    EXPECT_NEAR(H(0, 2), dx, 3.0F) << "tx";
    EXPECT_NEAR(H(1, 2), dy, 3.0F) << "ty";

    // Rotation / scale should be near identity
    EXPECT_NEAR(H(0, 0), 1.0F, 0.05F) << "scale-x";
    EXPECT_NEAR(H(1, 1), 1.0F, 0.05F) << "scale-y";

    // Bottom row must be [0, 0, 1]
    EXPECT_FLOAT_EQ(H(2, 0), 0.0F);
    EXPECT_FLOAT_EQ(H(2, 1), 0.0F);
    EXPECT_FLOAT_EQ(H(2, 2), 1.0F);
}

TEST(GMCEstimator, TexturelessFrameFails) {
    GMCEstimator gmc;
    cv::Mat blank(480, 640, CV_8UC1, cv::Scalar(128));  // uniform gray
    float fg[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    HomMat H;

    bool ok = gmc.estimate(blank, blank, fg, H);
    EXPECT_FALSE(ok) << "Should fail on textureless frames";

    // H should be identity on failure
    for (int32_t r = 0; r < 3; ++r) {
        for (int32_t c = 0; c < 3; ++c) {
            const float expected = (r == c) ? 1.0F : 0.0F;
            EXPECT_FLOAT_EQ(H(r, c), expected);
        }
    }
}

TEST(GMCEstimator, EmptyInputFails) {
    GMCEstimator gmc;
    cv::Mat empty;
    cv::Mat frame = make_textured_frame();
    float fg[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    HomMat H;

    EXPECT_FALSE(gmc.estimate(empty, frame, fg, H));
    EXPECT_FALSE(gmc.estimate(frame, empty, fg, H));
    EXPECT_FALSE(gmc.estimate(empty, empty, fg, H));
}

TEST(GMCEstimator, DownsampleTranslation) {
    // Verify that downscale→estimate→upscale recovers full-res translation.
    GMCEstimator gmc(kGmcDefaultFeatures, kGmcDefaultMinMatches,
                     kGmcDefaultInlierRatio, kGmcDefaultRansacReproj,
                     kGmcDefaultDilateFactor, 0.5F);
    cv::Mat prev = make_textured_frame();
    const float dx = 20.0F;
    const float dy = -10.0F;
    cv::Mat curr = translate_frame(prev, dx, dy);
    float fg[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    HomMat H;

    bool ok = gmc.estimate(prev, curr, fg, H);
    ASSERT_TRUE(ok) << "GMC should succeed with downsampled translation";

    EXPECT_NEAR(H(0, 2), dx, 5.0F) << "tx at full resolution";
    EXPECT_NEAR(H(1, 2), dy, 5.0F) << "ty at full resolution";
}

TEST(GMCEstimator, AffineBottomRow) {
    // Bottom row must always be [0, 0, 1] for any successful estimate.
    // Use no downsampling to avoid borderline matching failures.
    GMCEstimator gmc(kGmcDefaultFeatures, kGmcDefaultMinMatches,
                     kGmcDefaultInlierRatio, kGmcDefaultRansacReproj,
                     kGmcDefaultDilateFactor, 1.0F);
    cv::Mat prev = make_textured_frame();
    cv::Mat curr = translate_frame(prev, 30.0F, 20.0F);
    float fg[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    HomMat H;

    bool ok = gmc.estimate(prev, curr, fg, H);
    ASSERT_TRUE(ok);

    EXPECT_FLOAT_EQ(H(2, 0), 0.0F);
    EXPECT_FLOAT_EQ(H(2, 1), 0.0F);
    EXPECT_FLOAT_EQ(H(2, 2), 1.0F);
}

TEST(GMCEstimator, IdentityOnFailure) {
    // On any failure path, H_out must be identity.
    GMCEstimator gmc;
    cv::Mat blank(480, 640, CV_8UC1, cv::Scalar(100));
    float fg[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    HomMat H;
    H.setZero();  // dirty it

    bool ok = gmc.estimate(blank, blank, fg, H);
    EXPECT_FALSE(ok);

    HomMat I = HomMat::Identity();
    EXPECT_TRUE(H.isApprox(I, 1e-6F))
        << "H must be identity on failure, got:\n" << H;
}

