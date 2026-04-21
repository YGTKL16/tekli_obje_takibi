// deny(unsafe_code) — same as forbid but permits targeted #[allow] at FFI boundary.
#![deny(unsafe_code)]

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
}

fn clamp_scalar(value: f32, lower: f32, upper: f32) -> f32 {
    if value < lower {
        lower
    } else if value > upper {
        upper
    } else {
        value
    }
}

pub fn clamp_bbox(bbox: BBox, frame_width: f32, frame_height: f32, margin: f32) -> BBox {
    let safe_margin = if margin.is_finite() && margin > 0.0 {
        margin
    } else {
        0.0
    };

    let max_x = (frame_width - safe_margin).max(0.0);
    let max_y = (frame_height - safe_margin).max(0.0);
    let x = clamp_scalar(bbox.x, safe_margin, max_x);
    let y = clamp_scalar(bbox.y, safe_margin, max_y);
    let w = bbox.w.max(0.0).min((frame_width - x).max(0.0));
    let h = bbox.h.max(0.0).min((frame_height - y).max(0.0));

    BBox::new(x, y, w, h)
}

pub fn clamp_bbox_components(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    frame_width: f32,
    frame_height: f32,
    margin: f32,
) -> BBox {
    clamp_bbox(BBox::new(x, y, w, h), frame_width, frame_height, margin)
}

// ── C FFI entry points ──────────────────────────────────────────
// These are callable from C/C++ via the static library.
// The BBox struct is #[repr(C)] so layout is ABI-compatible.
// Safety rationale: no_mangle + extern "C" is the only unsafe surface;
// function bodies remain safe Rust.

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn tracker_safety_clamp_bbox(
    bbox: BBox,
    frame_width: f32,
    frame_height: f32,
    margin: f32,
) -> BBox {
    clamp_bbox(bbox, frame_width, frame_height, margin)
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn tracker_safety_clamp_bbox_components(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    frame_width: f32,
    frame_height: f32,
    margin: f32,
) -> BBox {
    clamp_bbox_components(x, y, w, h, frame_width, frame_height, margin)
}

#[cfg(test)]
mod tests {
    use super::{clamp_bbox, BBox};

    #[test]
    fn clamps_negative_origin_and_size() {
        let bbox = clamp_bbox(BBox::new(-4.0, -8.0, 20.0, 10.0), 100.0, 50.0, 0.0);
        assert_eq!(bbox, BBox::new(0.0, 0.0, 20.0, 10.0));
    }

    #[test]
    fn honors_margin_and_image_bounds() {
        let bbox = clamp_bbox(BBox::new(95.0, 48.0, 20.0, 10.0), 100.0, 50.0, 2.0);
        assert_eq!(bbox, BBox::new(95.0, 48.0, 5.0, 2.0));
    }

    #[test]
    fn sanitizes_invalid_margin() {
        let bbox = clamp_bbox(BBox::new(5.0, 5.0, 10.0, 10.0), 20.0, 20.0, f32::NAN);
        assert_eq!(bbox, BBox::new(5.0, 5.0, 10.0, 10.0));
    }
}
