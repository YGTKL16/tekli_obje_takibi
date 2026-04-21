// ---------------------------------------------------------------------------
// Project : Tracker
// File    : tracker_safety.h
// Purpose : C++ wrapper around the Rust tracker_safety FFI
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#ifndef TRACKER_SAFETY_H
#define TRACKER_SAFETY_H

#include "tracker_safety_ffi.h"
#include "kalman_types.h"

namespace tracker {
namespace safety {

/// @brief Clamp a Kalman measurement vector [x, y, w, h] to frame bounds.
///
/// Delegates to the Rust tracker_safety static library via C FFI.
/// @param z       Measurement vector [x, y, w, h]
/// @param fw      Frame width  in pixels
/// @param fh      Frame height in pixels
/// @param margin  Inset margin (pixels); 0 for no margin
/// @return Clamped measurement vector
[[nodiscard]] inline MeasVec clamp_measurement(
    const MeasVec& z,
    float fw,
    float fh,
    float margin) noexcept
{
    const TrackerSafetyBBox result = tracker_safety_clamp_bbox_components(
        static_cast<float>(z(0)),
        static_cast<float>(z(1)),
        static_cast<float>(z(2)),
        static_cast<float>(z(3)),
        fw, fh, margin);

    MeasVec out;
    out << result.x, result.y, result.w, result.h;
    return out;
}

} // namespace safety
} // namespace tracker

#endif // TRACKER_SAFETY_H
