// ---------------------------------------------------------------------------
// Project : Tracker
// File    : association.cpp
// Purpose : No-heap single-track association for top-left [x, y, w, h] boxes
// ---------------------------------------------------------------------------
#include "association.h"

#include <cmath>

namespace tracker {

float bbox_iou_top_left(const MeasVec& box_a, const MeasVec& box_b) noexcept {
    const float aw = (box_a(2) > 0.0F) ? box_a(2) : 0.0F;
    const float ah = (box_a(3) > 0.0F) ? box_a(3) : 0.0F;
    const float bw = (box_b(2) > 0.0F) ? box_b(2) : 0.0F;
    const float bh = (box_b(3) > 0.0F) ? box_b(3) : 0.0F;

    if ((aw <= 0.0F) || (ah <= 0.0F) || (bw <= 0.0F) || (bh <= 0.0F)) {
        return 0.0F;
    }

    const float a_x2 = box_a(0) + aw;
    const float a_y2 = box_a(1) + ah;
    const float b_x2 = box_b(0) + bw;
    const float b_y2 = box_b(1) + bh;

    const float inter_x1 = (box_a(0) > box_b(0)) ? box_a(0) : box_b(0);
    const float inter_y1 = (box_a(1) > box_b(1)) ? box_a(1) : box_b(1);
    const float inter_x2 = (a_x2 < b_x2) ? a_x2 : b_x2;
    const float inter_y2 = (a_y2 < b_y2) ? a_y2 : b_y2;

    const float inter_w = ((inter_x2 - inter_x1) > 0.0F) ? (inter_x2 - inter_x1) : 0.0F;
    const float inter_h = ((inter_y2 - inter_y1) > 0.0F) ? (inter_y2 - inter_y1) : 0.0F;
    const float inter = inter_w * inter_h;
    const float union_area = (aw * ah) + (bw * bh) - inter;

    if (union_area <= 0.0F) {
        return 0.0F;
    }
    return inter / union_area;
}


AssociationResult associate_single_track(
    const MeasVec& tracker_bbox,
    const CandidateBoxArray& detection_bboxes,
    const CandidateScoreArray& detection_scores,
    int32_t num_detections,
    float iou_threshold,
    float score_weight
) noexcept {
    AssociationResult result{};

    if ((num_detections <= 0) || (num_detections > kMaxAssociationCandidates)) {
        return result;
    }

    for (int32_t idx = 0; idx < num_detections; ++idx) {
        const std::size_t array_idx = static_cast<std::size_t>(idx);
        const float iou = bbox_iou_top_left(tracker_bbox, detection_bboxes[array_idx]);
        if (iou < iou_threshold) {
            continue;
        }

        const float score = detection_scores[array_idx];
        const float cost = (1.0F - iou) - (score_weight * score);
        const bool better_cost = (!result.has_match) || (cost < result.matched_cost);
        const bool tie_break_score =
            result.has_match &&
            (std::fabs(cost - result.matched_cost) <= 1e-6F) &&
            (score > result.matched_score);

        if (better_cost || tie_break_score) {
            result.has_match = true;
            result.matched_index = idx;
            result.matched_iou = iou;
            result.matched_score = score;
            result.matched_cost = cost;
        }
    }

    return result;
}

}  // namespace tracker
