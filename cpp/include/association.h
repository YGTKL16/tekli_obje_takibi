// ---------------------------------------------------------------------------
// Project : Tracker
// File    : association.h
// Purpose : No-heap single-track association for top-left [x, y, w, h] boxes
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#ifndef TRACKER_ASSOCIATION_H
#define TRACKER_ASSOCIATION_H

#include "kalman_types.h"

#include <array>
#include <cstdint>

namespace tracker {

constexpr int32_t kMaxAssociationCandidates = 32;

struct AssociationResult {
    int32_t matched_index = -1;
    float   matched_iou = 0.0F;
    float   matched_score = 0.0F;
    float   matched_cost = 1.0F;
    bool    has_match = false;
};

using CandidateBoxArray = std::array<MeasVec, kMaxAssociationCandidates>;
using CandidateScoreArray = std::array<float, kMaxAssociationCandidates>;

[[nodiscard]] float bbox_iou_top_left(
    const MeasVec& box_a,
    const MeasVec& box_b
) noexcept;

[[nodiscard]] AssociationResult associate_single_track(
    const MeasVec& tracker_bbox,
    const CandidateBoxArray& detection_bboxes,
    const CandidateScoreArray& detection_scores,
    int32_t num_detections,
    float iou_threshold,
    float score_weight
) noexcept;

}  // namespace tracker

#endif  // TRACKER_ASSOCIATION_H
