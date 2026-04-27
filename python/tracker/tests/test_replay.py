"""Regression tests for cached IMM replay policy."""

from pathlib import Path

import numpy as np
import pytest

from tracker.replay import DEFAULT_PARAMS, replay_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "cache" / "ai_outputs"
AB_TEST_SUBSET = [
    "dataset1/plane",
    "dataset1/surfer",
    "dataset1/volleyball",
    "dataset2/Girl2",
    "dataset2/Gull1",
    "dataset2/Kiting",
    "dataset2/ManRunning2",
    "dataset2/RcCar3",
    "dataset2/Surfing12",
    "dataset2/Wakeboarding2",
    "dataset3/air_conditioning_box2",
    "dataset3/basketball_player4-n",
    "dataset3/duck1_1",
    "dataset3/truck_night",
    "dataset4/car6",
    "dataset5/bike3",
    "dataset5/building2",
    "dataset5/car1_3",
    "dataset5/car1_s",
    "dataset5/person2_2",
]
TARGETED_SMOKE_SEQS = [
    "dataset1/surfer",
    "dataset3/truck_night",
    "dataset5/car1_s",
    "dataset5/car1_3",
]


def _write_cache(path: Path, ai_bboxes, confs, init_bbox, *, frame_w=640, frame_h=480):
    gt = np.asarray(ai_bboxes, dtype=np.float32)
    np.savez(
        path,
        ai_bboxes=np.asarray(ai_bboxes, dtype=np.float32),
        confs=np.asarray(confs, dtype=np.float32),
        gt=gt,
        init_bbox=np.asarray(init_bbox, dtype=np.float32),
        frame_w=np.int32(frame_w),
        frame_h=np.int32(frame_h),
    )


def _cache_path(seq_id: str) -> Path:
    return CACHE_DIR / f"{seq_id.replace('/', '__')}.npz"


def _require_cached_subset(seq_ids: list[str]) -> list[Path]:
    paths = [_cache_path(seq_id) for seq_id in seq_ids]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"missing cached replay inputs: {missing}")
    return paths


def test_replay_sequence_returns_ai_bbox_on_accepted_update(tmp_path):
    init_bbox = [100.0, 100.0, 50.0, 40.0]
    ai_bboxes = [
        init_bbox,
        [101.0, 100.0, 50.0, 40.0],
    ]
    confs = [1.0, 0.9]
    cache_path = tmp_path / "accepted_update.npz"
    _write_cache(cache_path, ai_bboxes, confs, init_bbox)

    pred_bboxes = replay_sequence(str(cache_path), DEFAULT_PARAMS)

    # With conf_bypass_threshold=0.15, conf=0.9 triggers bypass:
    # output IS the AI bbox directly (while KF is still updated internally).
    assert pred_bboxes[1] == pytest.approx(ai_bboxes[1])


def test_replay_sequence_returns_kf_prediction_while_coasting(tmp_path):
    init_bbox = [100.0, 100.0, 50.0, 40.0]
    ai_bboxes = [
        init_bbox,
        [140.0, 130.0, 50.0, 40.0],
    ]
    confs = [1.0, 0.01]
    cache_path = tmp_path / "coast_only.npz"
    _write_cache(cache_path, ai_bboxes, confs, init_bbox)

    pred_bboxes = replay_sequence(str(cache_path), DEFAULT_PARAMS)

    # AI-fallback: during coast (should_coast=True) the detection location is
    # preferred over KF extrapolation for F5 closed-loop feedback.
    assert pred_bboxes[1] == pytest.approx(ai_bboxes[1])
    assert pred_bboxes[1] != pytest.approx(init_bbox)


def test_replay_sequence_rejects_physically_implausible_measurement(tmp_path):
    init_bbox = [100.0, 100.0, 50.0, 40.0]
    ai_bboxes = [
        init_bbox,
        [500.0, 400.0, 50.0, 40.0],
    ]
    confs = [1.0, 0.95]
    cache_path = tmp_path / "judge_outlier.npz"
    _write_cache(cache_path, ai_bboxes, confs, init_bbox)

    params = {**DEFAULT_PARAMS, "conf_threshold": 0.1, "coast_threshold": 0.05}
    pred_bboxes = replay_sequence(str(cache_path), params)

    assert pred_bboxes[1] == pytest.approx(init_bbox)


@pytest.mark.parametrize("seq_id", TARGETED_SMOKE_SEQS)
def test_replay_runs_on_targeted_cached_sequences(seq_id):
    cache_path = _cache_path(seq_id)
    if not cache_path.exists():
        pytest.skip(f"missing cached replay input: {cache_path}")

    data = np.load(cache_path)
    gt = data["gt"]
    pred_bboxes = replay_sequence(str(cache_path), DEFAULT_PARAMS)

    assert len(pred_bboxes) == len(gt)
    assert all(len(bbox) == 4 for bbox in pred_bboxes)
    assert all(np.isfinite(bbox).all() for bbox in np.asarray(pred_bboxes, dtype=np.float32))


def test_replay_runs_on_cached_subset_without_invalid_boxes():
    cache_paths = _require_cached_subset(AB_TEST_SUBSET)

    for cache_path in cache_paths:
        data = np.load(cache_path)
        gt = data["gt"]
        pred_bboxes = replay_sequence(str(cache_path), DEFAULT_PARAMS)
        pred_arr = np.asarray(pred_bboxes, dtype=np.float32)
        assert pred_arr.shape[0] == gt.shape[0]
        assert pred_arr.shape[1] == 4
        assert np.isfinite(pred_arr).all()
