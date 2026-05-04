"""Direct MixFormerV2 wrapper checks."""

import os

import numpy as np
import pytest


def test_mixformerv2_default_paths_are_project_relative():
    from tracker.mixformerv2_wrapper import (
        DEFAULT_CHECKPOINT_PATH,
        DEFAULT_CONFIG_PATH,
        DEFAULT_REPO_ROOT,
        MixFormerV2Wrapper,
    )

    wrapper = MixFormerV2Wrapper()

    assert wrapper.repo_root == DEFAULT_REPO_ROOT
    assert wrapper.config_path == DEFAULT_CONFIG_PATH
    assert wrapper.checkpoint_path == DEFAULT_CHECKPOINT_PATH


def test_mixformerv2_init_defers_model_loading():
    from tracker.mixformerv2_wrapper import MixFormerV2Wrapper

    wrapper = MixFormerV2Wrapper()

    assert wrapper.network is None
    assert wrapper.initialized is False


def test_mixformerv2_track_before_init_returns_zero(sample_frame):
    from tracker.mixformerv2_wrapper import MixFormerV2Wrapper

    wrapper = MixFormerV2Wrapper()
    bbox, confidence = wrapper.track(sample_frame)

    assert np.allclose(bbox, 0.0)
    assert confidence == 0.0


def test_mixformerv2_direct_cpu_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("easydict")
    pytest.importorskip("einops")

    from tracker.mixformerv2_wrapper import (
        DEFAULT_CHECKPOINT_PATH,
        DEFAULT_CONFIG_PATH,
        DEFAULT_REPO_ROOT,
        MixFormerV2Wrapper,
    )

    missing = [
        path
        for path in (DEFAULT_REPO_ROOT, DEFAULT_CONFIG_PATH, DEFAULT_CHECKPOINT_PATH)
        if not os.path.exists(path)
    ]
    if missing:
        pytest.skip(f"MixFormerV2 assets unavailable: {missing}")

    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[92:132, 96:136] = np.array([220, 220, 220], dtype=np.uint8)
    init_bbox = np.array([96.0, 92.0, 40.0, 40.0], dtype=np.float32)

    wrapper = MixFormerV2Wrapper(device="cpu")
    wrapper.init(frame, init_bbox)
    bbox, confidence = wrapper.track(frame)

    assert bbox.shape == (4,)
    assert bbox.dtype == np.float32
    assert np.all(np.isfinite(bbox))
    assert bbox[2] > 0.0
    assert bbox[3] > 0.0
    assert 0.0 <= confidence <= 1.0
