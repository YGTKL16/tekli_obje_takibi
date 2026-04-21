#include <gtest/gtest.h>

#include "association.h"

namespace {

tracker::MeasVec make_box(float x, float y, float w, float h) {
    tracker::MeasVec box = tracker::MeasVec::Zero();
    box << x, y, w, h;
    return box;
}

}  // namespace

TEST(Association, PicksHighestIoUCandidate) {
    const tracker::MeasVec tracker_box = make_box(0.0F, 0.0F, 10.0F, 10.0F);
    tracker::CandidateBoxArray detections{};
    tracker::CandidateScoreArray scores{};

    detections[0] = make_box(20.0F, 20.0F, 10.0F, 10.0F);
    detections[1] = make_box(1.0F, 1.0F, 10.0F, 10.0F);
    detections[2] = make_box(0.0F, 0.0F, 8.0F, 8.0F);

    const tracker::AssociationResult result = tracker::associate_single_track(
        tracker_box, detections, scores, 3, 0.0F, 0.0F
    );

    EXPECT_TRUE(result.has_match);
    EXPECT_EQ(result.matched_index, 1);
}

TEST(Association, RejectsLowIoUCandidates) {
    const tracker::MeasVec tracker_box = make_box(0.0F, 0.0F, 10.0F, 10.0F);
    tracker::CandidateBoxArray detections{};
    tracker::CandidateScoreArray scores{};
    detections[0] = make_box(50.0F, 50.0F, 10.0F, 10.0F);

    const tracker::AssociationResult result = tracker::associate_single_track(
        tracker_box, detections, scores, 1, 0.3F, 0.0F
    );

    EXPECT_FALSE(result.has_match);
    EXPECT_EQ(result.matched_index, -1);
}

TEST(Association, ScoreWeightBreaksIoUTies) {
    const tracker::MeasVec tracker_box = make_box(0.0F, 0.0F, 10.0F, 10.0F);
    tracker::CandidateBoxArray detections{};
    tracker::CandidateScoreArray scores{};

    detections[0] = make_box(0.0F, 0.0F, 10.0F, 10.0F);
    detections[1] = make_box(0.0F, 0.0F, 10.0F, 10.0F);
    scores[0] = 0.1F;
    scores[1] = 0.9F;

    const tracker::AssociationResult result = tracker::associate_single_track(
        tracker_box, detections, scores, 2, 0.0F, 0.5F
    );

    EXPECT_TRUE(result.has_match);
    EXPECT_EQ(result.matched_index, 1);
}
