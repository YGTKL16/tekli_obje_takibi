// ---------------------------------------------------------------------------
// Project : Tracker
// File    : tracker_state.h
// Purpose : Three-state machine (TRACKING / COASTING / LOST) for track mgmt
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#ifndef TRACKER_TRACKER_STATE_H
#define TRACKER_TRACKER_STATE_H

#include <cstdint>

// AV Rule 185 – lightweight debug assertion (no exceptions)
#ifdef NDEBUG
#define TRACKER_ASSERT(cond) ((void)0)
#else
#define TRACKER_ASSERT(cond)                      \
    do {                                          \
        if (!(cond)) { __builtin_trap(); }        \
    } while (false)
#endif

namespace tracker {

/// Track lifecycle states (AV Rule 151 – scoped enum, fixed backing type).
enum class TrackState : uint8_t {
    TRACKING = 0,   ///< AI detection reliable – KF predict + update
    COASTING = 1,   ///< AI lost target – KF predict-only
    LOST     = 2    ///< Coasting exceeded max frames – target lost
};

/// @brief Confidence-driven state machine for a single tracked target.
///
/// Transition diagram:
///   LOST ──(conf ≥ thresh)──▸ TRACKING
///   TRACKING ──(conf < thresh)──▸ COASTING
///   COASTING ──(conf ≥ thresh)──▸ TRACKING
///   COASTING ──(coast_count > max)──▸ LOST
class TrackerState {
public:
    TrackerState() noexcept;
    ~TrackerState()                              noexcept = default;
    TrackerState(const TrackerState&)            noexcept = default;
    TrackerState& operator=(const TrackerState&) noexcept = default;
    TrackerState(TrackerState&&)                 noexcept = default;
    TrackerState& operator=(TrackerState&&)      noexcept = default;

    /// Advance the state machine with the latest AI confidence score.
    /// @param confidence  Detection confidence in [0, 1].
    /// @return Current state after transition.
    [[nodiscard]] TrackState step(float confidence) noexcept;

    /// Force re-init (e.g., operator selected a new target).
    void force_tracking() noexcept;

    /// @return Current lifecycle state.
    [[nodiscard]] TrackState state() const noexcept { return state_; }

    /// @return Number of consecutive predict-only (coast) frames.
    [[nodiscard]] int32_t coast_count() const noexcept { return coast_count_; }

    /// Set the confidence threshold for TRACKING ↔ COASTING transitions.
    /// @pre @p t must be in [0, 1].
    void set_confidence_threshold(float t) noexcept {
        TRACKER_ASSERT(t >= 0.0f && t <= 1.0f);
        conf_thresh_ = t;
    }

    /// Set the maximum number of coast frames before declaring LOST.
    /// @pre @p n must be > 0.
    void set_max_coast_frames(int32_t n) noexcept {
        TRACKER_ASSERT(n > 0);
        max_coast_ = n;
    }

private:
    TrackState state_;
    int32_t    coast_count_;
    float      conf_thresh_;    ///< default 0.3
    int32_t    max_coast_;      ///< default 60
};

}  // namespace tracker

#endif  // TRACKER_TRACKER_STATE_H
