"""End-to-end: TRT + IMM + DecisionMaker + measurement hygiene, writes annotated mp4."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import numpy as np
import cv2
from tracker.config import load_runtime_config
from tracker.trt_wrapper import TRTTrackWrapper
from tracker.decision import DecisionMaker
import tracker_cpp

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FILTER_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "filter_tuned.yaml")


def _load_filter_params() -> dict:
    """Load normalized runtime params with legacy-schema fallbacks."""
    return load_runtime_config(_FILTER_CONFIG)


fp = _load_filter_params()

SEQ_DIR = "data/contest_release/dataset1/basketball"
OUT_PATH = "/tmp/basketball_tracked.mp4"

video_path = os.path.join(SEQ_DIR, "basketball.mp4")
ann_path = os.path.join(SEQ_DIR, "annotation.txt")

with open(ann_path) as f:
    init_bbox = [float(x) for x in f.readline().strip().split(",")]

tracker = TRTTrackWrapper()
kf = tracker_cpp.IMMFilter()
sm = tracker_cpp.TrackerState()
sm.set_confidence_threshold(fp["sm_conf_threshold"])
sm.set_max_coast_frames(int(fp["sm_max_coast"]))
dec = DecisionMaker(
    conf_threshold=fp["conf_threshold"],
    iou_threshold=fp["iou_threshold"],
    coast_threshold=fp["coast_threshold"],
    max_coast_frames=int(fp["max_coast_frames"]),
    max_area_frac=fp["max_area_frac"],
    aspect_ratio_range=(fp["aspect_ratio_lo"], fp["aspect_ratio_hi"]),
    max_center_jump_frac=fp["max_center_jump_frac"],
)

cap = cv2.VideoCapture(video_path)
fps_v = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h))

frame_idx = 0
prev_bbox = init_bbox
last_good_bbox = init_bbox
reject_streak = 0
REINIT_AFTER = int(fp["reinit_after"])
reject_count = 0
coast_count = 0
update_count = 0
reinit_count = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    action = "INIT"
    conf = 1.0
    ai_bbox_dbg = None

    if frame_idx == 0:
        init_arr = np.array(init_bbox, dtype=np.float32)
        tracker.init(frame_rgb, init_arr)
        kf.init(init_arr)
        sm.force_tracking()
        out_bbox = init_bbox
        state_name = "INIT"
        coast_cnt = 0
    else:
        ai_bbox, conf = tracker.track(frame_rgb)
        ai_list = ai_bbox.tolist() if isinstance(ai_bbox, np.ndarray) else list(ai_bbox)
        ai_bbox_dbg = ai_list
        ts = sm.step(conf)
        state_name = ts.name
        coast_cnt = sm.coast_count()

        sane = dec.is_measurement_sane(ai_list, conf, w, h, prev_bbox=prev_bbox)
        strong = conf >= dec.conf_threshold

        if not sane or not strong:
            reject_streak += 1
            if reject_streak >= REINIT_AFTER:
                kf.init(np.array(last_good_bbox, dtype=np.float32))
                reinit_count += 1
                reject_streak = 0
                state = np.array(kf.get_state()).flatten()
                action = "REINIT"
            else:
                state = np.array(kf.predict()).flatten()
                action = "REJECT" if not sane else "COAST"
                if action == "REJECT":
                    reject_count += 1
                else:
                    coast_count += 1
            out_bbox = state[:4].tolist()
        else:
            state = np.array(kf.update(np.array(ai_list, dtype=np.float32))).flatten()
            _alo = fp["alpha_conf_lo"]
            _ahi = fp["alpha_conf_hi"]
            alpha = min(1.0, max(0.0, (conf - _alo) / max(_ahi - _alo, 1e-6)))
            out_bbox = [
                alpha * ai_list[0] + (1.0 - alpha) * float(state[0]),
                alpha * ai_list[1] + (1.0 - alpha) * float(state[1]),
                float(state[2]),
                float(state[3]),
            ]
            last_good_bbox = out_bbox
            reject_streak = 0
            action = "UPDATE"
            update_count += 1
            # Closed-loop: steer AI search window toward filtered bbox
            tracker.set_state(out_bbox)

    prev_bbox = out_bbox

    vis = frame_bgr.copy()
    x, y, bw, bh = [int(v) for v in out_bbox]
    if action == "UPDATE":
        color = (0, 255, 0)
    elif action == "COAST":
        color = (0, 255, 255)
    elif action == "REJECT":
        color = (0, 0, 255)
    elif action == "REINIT":
        color = (255, 0, 255)
    else:
        color = (255, 255, 255)
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

    if ai_bbox_dbg is not None:
        ax, ay, aw, ah = [int(v) for v in ai_bbox_dbg]
        cv2.rectangle(vis, (ax, ay), (ax + aw, ay + ah), (128, 128, 128), 1)

    label = f"f{frame_idx} conf={conf:.2f} {state_name} {action}"
    if state_name == "COASTING":
        label += f" c={coast_cnt}"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    writer.write(vis)

    if frame_idx % 50 == 0 or action == "REJECT":
        print(f"Frame {frame_idx:3d}: out=[{x:4d},{y:4d},{bw:3d},{bh:3d}] "
              f"conf={conf:.3f} {state_name} {action}")
    frame_idx += 1

cap.release()
writer.release()
print(f"\n{frame_idx} frames | UPDATE={update_count} COAST={coast_count} "
      f"REJECT={reject_count} REINIT={reinit_count}")
print(f"Wrote: {OUT_PATH}")
