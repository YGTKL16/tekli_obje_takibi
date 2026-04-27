// ---------------------------------------------------------------------------
// Project : Tracker
// File    : gmc_estimator.cpp
// Purpose : ORB + estimateAffinePartial2D implementation (4 DOF)
// Standard: JSF AV C++ Rev C
// ---------------------------------------------------------------------------
#include "gmc_estimator.h"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace tracker {

// ── Construction ────────────────────────────────────────────────
GMCEstimator::GMCEstimator(
    int32_t n_features,
    int32_t min_matches,
    float   inlier_ratio_thresh,
    float   ransac_reproj_thresh,
    float   dilate_factor,
    float   downsample
)
    : n_features_{n_features}
    , min_matches_{min_matches}
    , inlier_ratio_thresh_{inlier_ratio_thresh}
    , ransac_reproj_thresh_{ransac_reproj_thresh}
    , dilate_factor_{dilate_factor}
    , downsample_{downsample}
    , orb_{cv::ORB::create(n_features)}
    , matcher_{cv::BFMatcher::create(cv::NORM_HAMMING, true)}
{
    // Pre-allocate buffers to avoid per-frame heap allocation.
    const int32_t cap = n_features;
    kp_prev_.reserve(static_cast<std::size_t>(cap));
    kp_curr_.reserve(static_cast<std::size_t>(cap));
    matches_.reserve(static_cast<std::size_t>(cap));
    pts_prev_.reserve(static_cast<std::size_t>(cap));
    pts_curr_.reserve(static_cast<std::size_t>(cap));
}

// ── Private: foreground mask ────────────────────────────────────
void GMCEstimator::build_mask(
    const cv::Size& size,
    const float     fg[4],
    cv::Mat&        mask
) {
    mask.create(size, CV_8UC1);
    mask.setTo(cv::Scalar(255));

    const float fx = fg[0];
    const float fy = fg[1];
    const float fw = fg[2];
    const float fh = fg[3];

    if (fw <= 0.0F || fh <= 0.0F) {
        return;  // no foreground to mask
    }

    const float cx = fx + fw * 0.5F;
    const float cy = fy + fh * 0.5F;
    const float dw = fw * dilate_factor_;
    const float dh = fh * dilate_factor_;

    const int32_t x0 = std::max(0, static_cast<int32_t>(cx - dw * 0.5F));
    const int32_t y0 = std::max(0, static_cast<int32_t>(cy - dh * 0.5F));
    const int32_t x1 = std::min(size.width,  static_cast<int32_t>(cx + dw * 0.5F));
    const int32_t y1 = std::min(size.height, static_cast<int32_t>(cy + dh * 0.5F));

    if (x1 > x0 && y1 > y0) {
        mask(cv::Rect(x0, y0, x1 - x0, y1 - y0)).setTo(cv::Scalar(0));
    }
}

// ── Public: estimate ────────────────────────────────────────────
bool GMCEstimator::estimate(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const float    fg_xywh[4],
    HomMat&        H_out
) {
    GMCEstimateStats stats{};
    return estimate_with_stats(prev_gray, curr_gray, fg_xywh, H_out, stats);
}

bool GMCEstimator::estimate_with_stats(
    const cv::Mat&    prev_gray,
    const cv::Mat&    curr_gray,
    const float       fg_xywh[4],
    HomMat&           H_out,
    GMCEstimateStats& stats_out
) {
    // Default to identity — caller checks return value for Q-boost decision.
    H_out.setIdentity();
    stats_out = GMCEstimateStats{};

    if (prev_gray.empty() || curr_gray.empty()) {
        return false;
    }

    // ── Downsample ──────────────────────────────────────────────
    float scale = downsample_;
    float fg_scaled[4] = {
        fg_xywh[0] * scale,
        fg_xywh[1] * scale,
        fg_xywh[2] * scale,
        fg_xywh[3] * scale
    };

    if (scale < 1.0F) {
        cv::resize(prev_gray, prev_small_, cv::Size(), scale, scale,
                   cv::INTER_NEAREST);
        cv::resize(curr_gray, curr_small_, cv::Size(), scale, scale,
                   cv::INTER_NEAREST);
    } else {
        prev_small_ = prev_gray;
        curr_small_ = curr_gray;
        scale = 1.0F;
    }

    // ── Build foreground mask ───────────────────────────────────
    build_mask(prev_small_.size(), fg_scaled, mask_);

    // ── ORB detect + compute ────────────────────────────────────
    kp_prev_.clear();
    kp_curr_.clear();
    orb_->detectAndCompute(prev_small_, mask_, kp_prev_, desc_prev_);
    orb_->detectAndCompute(curr_small_, mask_, kp_curr_, desc_curr_);

    if (desc_prev_.empty() || desc_curr_.empty()) {
        return false;
    }
    if (static_cast<int32_t>(kp_prev_.size()) < min_matches_ ||
        static_cast<int32_t>(kp_curr_.size()) < min_matches_) {
        return false;
    }

    // ── Match ───────────────────────────────────────────────────
    matches_.clear();
    matcher_->match(desc_prev_, desc_curr_, matches_);
    stats_out.match_count = static_cast<int32_t>(matches_.size());

    if (stats_out.match_count < min_matches_) {
        return false;
    }

    // ── Extract matched point pairs ─────────────────────────────
    pts_prev_.clear();
    pts_curr_.clear();
    for (const auto& m : matches_) {
        pts_prev_.push_back(kp_prev_[static_cast<std::size_t>(m.queryIdx)].pt);
        pts_curr_.push_back(kp_curr_[static_cast<std::size_t>(m.trainIdx)].pt);
    }

    // ── Partial affine (4 DOF: rotation + scale + translation) ──
    inlier_mask_ = cv::Mat();
    cv::Mat affine = cv::estimateAffinePartial2D(
        pts_prev_, pts_curr_, inlier_mask_,
        cv::RANSAC, ransac_reproj_thresh_
    );

    if (affine.empty() || inlier_mask_.empty()) {
        return false;
    }

    stats_out.has_affine = true;
    stats_out.inlier_count = cv::countNonZero(inlier_mask_);
    if (stats_out.match_count > 0) {
        stats_out.inlier_ratio =
            static_cast<float>(stats_out.inlier_count) / static_cast<float>(stats_out.match_count);
    }

    // ── Build 3×3 from 2×3 affine, then undo downscale ─────────
    // affine is CV_64F (2×3). Embed into 3×3 with bottom row [0, 0, 1].
    HomMat H_s;
    H_s.setIdentity();
    for (int32_t r = 0; r < 2; ++r) {
        for (int32_t c = 0; c < 3; ++c) {
            H_s(r, c) = static_cast<float>(affine.at<double>(r, c));
        }
    }

    if (scale < 1.0F) {
        // S * H_s * S^{-1}  where S = diag(scale, scale, 1)
        const float inv_scale = 1.0F / scale;
        // Scale the translation column
        H_s(0, 2) *= inv_scale;
        H_s(1, 2) *= inv_scale;
        // Off-diagonal rotation/scale elements stay the same because
        // S * [a b; c d] * S^{-1} = [a b; c d] when S is uniform.
    }

    H_out = H_s;
    if (stats_out.inlier_count < min_matches_) {
        return false;
    }
    if (stats_out.inlier_ratio < inlier_ratio_thresh_) {
        return false;
    }
    return true;
}

}  // namespace tracker
