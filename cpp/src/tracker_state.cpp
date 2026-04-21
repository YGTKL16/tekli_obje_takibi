// ---------------------------------------------------------------------------
// Project : Tracker
// File    : tracker_state.cpp
// Purpose : TrackerState implementation – confidence-driven state machine
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#include "tracker_state.h"

namespace tracker {

TrackerState::TrackerState() noexcept
    : state_(TrackState::LOST)
    , coast_count_(0)
    , conf_thresh_(0.3f)
    , max_coast_(60)
{}

TrackState TrackerState::step(float confidence) noexcept {
    switch (state_) {
        case TrackState::TRACKING:
            if (confidence < conf_thresh_) {
                state_ = TrackState::COASTING;
                coast_count_ = 1;
            }
            break;

        case TrackState::COASTING:
            if (confidence >= conf_thresh_) {
                state_ = TrackState::TRACKING;
                coast_count_ = 0;
            } else {
                ++coast_count_;
                if (coast_count_ > max_coast_) {
                    state_ = TrackState::LOST;
                }
            }
            break;

        case TrackState::LOST:
            if (confidence >= conf_thresh_) {
                state_ = TrackState::TRACKING;
                coast_count_ = 0;
            }
            break;

        default:  // AV Rule 193 – defensive
            break;
    }
    return state_;
}

void TrackerState::force_tracking() noexcept {
    state_ = TrackState::TRACKING;
    coast_count_ = 0;
}

}  // namespace tracker
