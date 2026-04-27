"""Runtime regime detector for sequence-adaptive parameter selection.

4 mutually exclusive regimes, determined from per-frame signals:

    SMALL_TARGET   — fixed at frame 0, never changes (init_area < 500 px²)
    FAST_MANEUVER  — rolling 30-frame Singer μ spike OR high-fps sequence
    OCCLUSION      — sustained coasting + low confidence
    DEFAULT        — none of the above (golden standard i12 params)

Usage
-----
    detector = RegimeDetector(init_bbox, frame_hw=(fh, fw), native_fps=30)
    regime = detector.update(conf, mu_singer, coast_count, kf_vel_norm=0.0)
    overrides = detector.get_param_overrides()

The returned ``overrides`` dict should be applied to the decision / tracker
parameters that differ from the base config (i12_rescue_area_gate.yaml).
"""
from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Dict, Any

import numpy as np


class Regime(Enum):
    DEFAULT = "default"
    SMALL_TARGET = "small_target"
    FAST_MANEUVER = "fast_maneuver"
    OCCLUSION = "occlusion"


class RegimeDetector:
    """Lightweight, stateful regime classifier.

    All thresholds are class-level constants so they can be overridden in
    tests without constructing subclasses.
    """

    # --- detection thresholds -------------------------------------------------
    SMALL_AREA_THR: float = 500.0          # px²: bike3=170 < 500 → SMALL
    HIGH_FPS_THR: float = 60.0             # fps: 60+ → likely FAST_MANEUVER
    SINGER_MANEUVER_THR: float = 0.40      # rolling Singer μ median (0-1 scale)
    VEL_FAST_THR: float = 4.0             # rolling ||[vx,vy]|| median px/frame → FAST_MANEUVER
    OCCLUSION_COAST_THR: int = 8           # consecutive coast frames
    OCCLUSION_CONF_THR: float = 0.25       # conf below this while coasting
    HYSTERESIS_WINDOW: int = 30            # frames for rolling Singer median

    # --- parameter override tables -------------------------------------------
    _OVERRIDES: Dict[Regime, Dict[str, Any]] = {
        Regime.SMALL_TARGET: {
            # Tiny targets: block Great Rescue (template poison), block FazD,
            # allow long coasting (target may briefly disappear behind thin objects)
            "rescue_min_area": 1e9,          # block ALL rescues (no target is this large)
            "f5_feedback": True,             # always feed KF bbox back to AI
            "max_coast_frames": 60,          # patient coasting for tiny fast targets
            "refresh_min_bbox_area": 1e9,    # block FazD entirely (oversize threshold)
        },
        Regime.FAST_MANEUVER: {
            # Fast vehicles / UAVs: AI is more trustworthy than stale KF prediction;
            # disable closed-loop feedback that causes lag on high-speed targets.
            "f5_feedback": False,
            "maneuver_pi_enabled": True,     # dynamic pi injection on Mahal spike
            "max_coast_frames": 20,          # fail fast — don't coast on lost fast target
        },
        Regime.OCCLUSION: {
            # Slow/occluded targets: extend patience, trust AI more on re-entry
            "max_coast_frames": 80,
            "coast_threshold": 0.08,         # re-enter at low confidence (recovering)
            "f5_feedback": True,
        },
        Regime.DEFAULT: {},                  # nothing — i12 golden standard applies
    }

    def __init__(
        self,
        init_bbox: "np.ndarray | list[float]",
        frame_hw: "tuple[int, int] | None" = None,
        native_fps: float = 30.0,
    ) -> None:
        init_bbox = np.asarray(init_bbox, dtype=np.float32).reshape(4)
        init_area = float(init_bbox[2]) * float(init_bbox[3])

        # SMALL_TARGET is set once at init and is irrevocable
        self._is_small: bool = init_area < self.SMALL_AREA_THR
        self._is_high_fps: bool = native_fps >= self.HIGH_FPS_THR

        self._singer_history: deque[float] = deque(maxlen=self.HYSTERESIS_WINDOW)
        self._vel_history: deque[float] = deque(maxlen=self.HYSTERESIS_WINDOW)
        self._coast_count: int = 0
        self._conf: float = 1.0

        # Initialise regime immediately from static signals
        if self._is_small:
            self._regime = Regime.SMALL_TARGET
        elif self._is_high_fps:
            self._regime = Regime.FAST_MANEUVER
        else:
            self._regime = Regime.DEFAULT

    # ------------------------------------------------------------------
    def update(
        self,
        conf: float,
        mu_singer: float,
        coast_count: int,
        kf_vel_norm: float = 0.0,
    ) -> Regime:
        """Update regime from current-frame signals.

        Parameters
        ----------
        conf        : AI confidence for current frame (0-1)
        mu_singer   : IMM Singer model probability (0-1)
        coast_count : consecutive coast frames (StateManager / sm.coast_count())
        kf_vel_norm : ||[vx, vy]|| in px/frame (optional, for future extension)

        Returns
        -------
        Regime  — the active regime after this update
        """
        self._singer_history.append(float(mu_singer))
        self._vel_history.append(float(kf_vel_norm))
        self._coast_count = int(coast_count)
        self._conf = float(conf)

        # SMALL_TARGET: set at init and immutable
        if self._is_small:
            self._regime = Regime.SMALL_TARGET
            return self._regime

        rolling_singer = (
            float(np.median(self._singer_history))
            if self._singer_history
            else 0.0
        )
        rolling_vel = (
            float(np.median(self._vel_history))
            if self._vel_history
            else 0.0
        )

        if (self._is_high_fps
                or rolling_singer > self.SINGER_MANEUVER_THR
                or rolling_vel > self.VEL_FAST_THR):
            self._regime = Regime.FAST_MANEUVER
        elif (
            self._coast_count > self.OCCLUSION_COAST_THR
            and self._conf < self.OCCLUSION_CONF_THR
        ):
            self._regime = Regime.OCCLUSION
        else:
            self._regime = Regime.DEFAULT

        return self._regime

    # ------------------------------------------------------------------
    @property
    def regime(self) -> Regime:
        return self._regime

    def get_param_overrides(self) -> Dict[str, Any]:
        """Return flat parameter overrides for the active regime.

        Caller applies these on top of the base config (e.g. i12).
        An empty dict means "use base config unchanged" (DEFAULT regime).
        """
        return dict(self._OVERRIDES.get(self._regime, {}))

    def __repr__(self) -> str:
        return (
            f"RegimeDetector(regime={self._regime.value!r}, "
            f"is_small={self._is_small}, is_high_fps={self._is_high_fps}, "
            f"coast={self._coast_count}, conf={self._conf:.2f})"
        )
