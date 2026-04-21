// ---------------------------------------------------------------------------
// Project : Tracker
// File    : gmc_estimator.h
// Purpose : JSF-compliant Global Motion Compensation via ORB + Affine
//           (no exceptions, pre-allocated buffers, zero per-frame heap alloc)
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#ifndef TRACKER_GMC_ESTIMATOR_H
#define TRACKER_GMC_ESTIMATOR_H

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "kalman_types.h"

namespace tracker {

// ── Compile-time defaults ───────────────────────────────────────
constexpr int32_t kGmcDefaultFeatures      = 200;
constexpr int32_t kGmcDefaultMinMatches     = 6;
constexpr float   kGmcDefaultInlierRatio    = 0.3F;
constexpr float   kGmcDefaultRansacReproj   = 3.0F;
constexpr float   kGmcDefaultDilateFactor   = 1.4F;
constexpr float   kGmcDefaultDownsample     = 0.5F;

/// @brief ORB + partial-affine Global Motion Compensation.
///
/// Design constraints (JSF AV):
///   - No dynamic memory allocation after construction
///   - No exceptions  (-fno-exceptions in tracker_core; -fexceptions in
///     the boundary library tracker_gmc because OpenCV may throw)
///   - No RTTI        (-fno-rtti)
///   - Pre-allocated vectors reserved in ctor; clear() per frame (no dealloc)
///   - Fixed-width types (int32_t, float)
class GMCEstimator {
public:
    /// Construct with ORB detector and matcher; pre-allocates buffers.
    explicit GMCEstimator(
        int32_t n_features           = kGmcDefaultFeatures,
        int32_t min_matches          = kGmcDefaultMinMatches,
        float   inlier_ratio_thresh  = kGmcDefaultInlierRatio,
        float   ransac_reproj_thresh = kGmcDefaultRansacReproj,
        float   dilate_factor        = kGmcDefaultDilateFactor,
        float   downsample           = kGmcDefaultDownsample
    );

    ~GMCEstimator()                                = default;
    GMCEstimator(const GMCEstimator&)              = delete;
    GMCEstimator& operator=(const GMCEstimator&)   = delete;
    GMCEstimator(GMCEstimator&&)                   = default;
    GMCEstimator& operator=(GMCEstimator&&)        = default;

    /// Estimate 3×3 affine-embedded homography (prev → curr).
    ///
    /// @param prev_gray  Previous frame, single-channel uint8.
    /// @param curr_gray  Current  frame, single-channel uint8.
    /// @param fg_xywh    Foreground bounding box [x, y, w, h] to mask out.
    /// @param[out] H_out 3×3 result; identity on failure.
    /// @return true on success, false when the caller must inflate Q.
    [[nodiscard]] bool estimate(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const float    fg_xywh[4],
        HomMat&        H_out
    );

private:
    void build_mask(const cv::Size& size, const float fg[4], cv::Mat& mask);

    // ── Parameters ──────────────────────────────────────────────
    int32_t n_features_;
    int32_t min_matches_;
    float   inlier_ratio_thresh_;
    float   ransac_reproj_thresh_;
    float   dilate_factor_;
    float   downsample_;

    // ── Pre-allocated OpenCV objects ────────────────────────────
    cv::Ptr<cv::ORB>      orb_;
    cv::Ptr<cv::BFMatcher> matcher_;

    // ── Pre-allocated buffers (reserve in ctor, clear per frame) ─
    std::vector<cv::KeyPoint> kp_prev_;
    std::vector<cv::KeyPoint> kp_curr_;
    cv::Mat                   desc_prev_;
    cv::Mat                   desc_curr_;
    std::vector<cv::DMatch>   matches_;
    std::vector<cv::Point2f>  pts_prev_;
    std::vector<cv::Point2f>  pts_curr_;
    cv::Mat                   inlier_mask_;

    // ── Scratch mats for downsampled frames & mask ──────────────
    cv::Mat prev_small_;
    cv::Mat curr_small_;
    cv::Mat mask_;
};

}  // namespace tracker

#endif  // TRACKER_GMC_ESTIMATOR_H
