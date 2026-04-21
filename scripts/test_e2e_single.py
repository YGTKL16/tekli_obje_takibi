"""End-to-end test: TRT wrapper on a single competition sequence."""
import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.tracker.trt_wrapper import TRTTrackWrapper

SEQ_DIR = "data/contest_release/dataset1/basketball"

# Load video
video_path = os.path.join(SEQ_DIR, "basketball.mp4")
ann_path = os.path.join(SEQ_DIR, "annotation.txt")

with open(ann_path) as f:
    parts = f.readline().strip().split(",")
    init_bbox = [float(x) for x in parts]

print(f"Init bbox: {init_bbox}")
print(f"Video: {video_path}")

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

# Init tracker
tracker = TRTTrackWrapper()

frame_idx = 0
times = []
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if frame_idx == 0:
        tracker.init(frame_rgb, init_bbox)
        print(f"Frame {frame_idx}: initialized")
    else:
        t0 = time.perf_counter()
        bbox, conf = tracker.track(frame_rgb)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}: bbox={bbox}, conf={conf:.4f}, time={times[-1]*1000:.1f}ms")

    frame_idx += 1

cap.release()

times = np.array(times)
print("\n--- Results ---")
print(f"Frames tracked: {len(times)}")
print(f"Avg latency: {times.mean()*1000:.1f} ms")
print(f"Median latency: {np.median(times)*1000:.1f} ms")
print(f"FPS: {1/times.mean():.0f}")
print(f"Min/Max: {times.min()*1000:.1f} / {times.max()*1000:.1f} ms")
