// ---------------------------------------------------------------------------
// Project : Tracker
// File    : tracker_safety_ffi.h
// Purpose : C-linkage declarations for the Rust tracker_safety static library
// Standard: JSF AV C++ Rev C, MISRA C++ 2023
// ---------------------------------------------------------------------------
#ifndef TRACKER_SAFETY_FFI_H
#define TRACKER_SAFETY_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// ABI-compatible mirror of the Rust `#[repr(C)] BBox` struct.
typedef struct TrackerSafetyBBox {
    float x;
    float y;
    float w;
    float h;
} TrackerSafetyBBox;

/// @brief Clamp a bounding box so it lies within [margin, frame-margin].
TrackerSafetyBBox tracker_safety_clamp_bbox(
    TrackerSafetyBBox bbox,
    float frame_width,
    float frame_height,
    float margin);

/// @brief Clamp individual bbox components; returns a clamped BBox.
TrackerSafetyBBox tracker_safety_clamp_bbox_components(
    float x, float y, float w, float h,
    float frame_width, float frame_height,
    float margin);

#ifdef __cplusplus
}
#endif

#endif // TRACKER_SAFETY_FFI_H
