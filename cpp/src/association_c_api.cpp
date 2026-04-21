// ---------------------------------------------------------------------------
// Project : Tracker
// File    : association_c_api.cpp
// Purpose : C ABI bridge for no-heap single-track association
// ---------------------------------------------------------------------------
#include "association.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct TrackerAssociationResultC {
    int32_t matched_index;
    float   matched_iou;
    float   matched_score;
    float   matched_cost;
    int32_t has_match;
};

TrackerAssociationResultC tracker_associate_single_track(
    const float* tracker_bbox,
    const float* detection_bboxes,
    const float* detection_scores,
    int32_t num_detections,
    float iou_threshold,
    float score_weight
) {
    tracker::MeasVec tracker_box = tracker::MeasVec::Zero();
    for (int32_t idx = 0; idx < tracker::kMeasDim; ++idx) {
        tracker_box(idx) = tracker_bbox[idx];
    }

    tracker::CandidateBoxArray detection_boxes{};
    tracker::CandidateScoreArray scores{};

    if ((num_detections > 0) && (detection_bboxes != nullptr)) {
        for (int32_t det_idx = 0; det_idx < num_detections; ++det_idx) {
            const std::size_t box_offset =
                static_cast<std::size_t>(det_idx) * static_cast<std::size_t>(tracker::kMeasDim);
            detection_boxes[static_cast<std::size_t>(det_idx)] = tracker::MeasVec::Zero();
            for (int32_t coord_idx = 0; coord_idx < tracker::kMeasDim; ++coord_idx) {
                const std::size_t flat_idx = box_offset + static_cast<std::size_t>(coord_idx);
                detection_boxes[static_cast<std::size_t>(det_idx)](coord_idx) = detection_bboxes[flat_idx];
            }
            scores[static_cast<std::size_t>(det_idx)] =
                (detection_scores != nullptr) ? detection_scores[det_idx] : 0.0F;
        }
    }

    const tracker::AssociationResult result = tracker::associate_single_track(
        tracker_box,
        detection_boxes,
        scores,
        num_detections,
        iou_threshold,
        score_weight
    );

    return TrackerAssociationResultC{
        result.matched_index,
        result.matched_iou,
        result.matched_score,
        result.matched_cost,
        result.has_match ? 1 : 0,
    };
}

}  // extern "C"
