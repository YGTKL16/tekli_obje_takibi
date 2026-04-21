"""Shared pytest fixtures for tracker tests."""

import os
import sys

import numpy as np
import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Ensure the primary build directory is always first in sys.path so that
# the freshest tracker_cpp .so is loaded (build-verify / build_debug may
# contain stale builds without recent binding changes).
_primary_build = os.path.join(PROJECT_ROOT, "build")
if os.path.isdir(_primary_build):
    if _primary_build in sys.path:
        sys.path.remove(_primary_build)
    sys.path.insert(0, _primary_build)


@pytest.fixture
def sample_bbox():
    """Standard [x, y, w, h] bounding box."""
    return np.array([100.0, 200.0, 50.0, 60.0], dtype=np.float32)


@pytest.fixture
def sample_frame():
    """480x640 BGR frame (zeros)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_kf_state():
    """8-element Kalman state [x, y, w, h, vx, vy, vw, vh]."""
    return np.array([100.0, 200.0, 50.0, 60.0, 1.0, 0.5, 0.0, 0.0], dtype=np.float32)
