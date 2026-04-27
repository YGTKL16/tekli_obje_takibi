"""Observation-Centric Re-Update (ORU) backfill for the IMM filter.

Implements the OC-SORT (Cao et al., CVPR 2023) ORU strategy adapted to our
3-model IMM stack. On a COASTING -> TRACKING transition the controller:

  1. Generates a virtual trajectory between the last accepted measurement
     before coasting (z_t1) and the re-acquired measurement (z_t2) via
     linear interpolation (constant-speed assumption).
  2. Restores the IMM to the snapshot taken at t1 (state, P, mu).
  3. Re-runs predict+update along the virtual trajectory using a synthetic
     low-confidence so the posterior does not over-commit to virtual data.

A multi-condition gate prevents firing on aggressive maneuvers, scale
collapse, low-confidence re-entries, or short coasts where backfill is
either unsafe or pointless.

Hybrid V1 architecture: orchestration in Python, IMM math stays in C++ via
the existing IMMFilter.update(z, conf) and IMMFilter.restore_from(...) bindings.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ImmSnapshot:
    """Frozen capture of the IMM filter state at one accepted-measurement frame."""

    state: np.ndarray  # (10,) combined state
    P: np.ndarray      # (10, 10) combined covariance
    mu: np.ndarray     # (3,) mode probabilities [CV, CA, Singer]
    z: np.ndarray      # (4,) accepted measurement bbox [x, y, w, h]
    frame_idx: int
    singer_mu: float   # convenience: mu[Singer] for fast gate checks


@dataclass
class OruConfig:
    """Tunable knobs for the ORU controller (mirrors `oru:` block in YAML)."""

    enabled: bool = True
    n_min: int = 5                  # min coast length to trigger
    n_max: int = 30                 # max coast length (above -> skip; traj unreliable)
    conf_reentry_min: float = 0.50  # re-entry conf must clear this
    velocity_cv_max: float = 0.5    # std(|v|)/mean(|v|) during coast must be <= this
    virtual_conf: float = 0.4       # synthetic conf for virtual updates (low -> R inflates)
    skip_if_singer_dom: bool = True
    singer_dom_thr: float = 0.7     # mu[Singer] above this -> skip (aggressive maneuver)
    ring_buffer_size: int = 35      # max snapshots retained
    min_velocity_samples: int = 3   # below this many coast samples, skip CV gate

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "OruConfig":
        if not d:
            return cls()
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**kwargs)


@dataclass
class _CoastWindow:
    """Per-coast scratchpad: anchor snapshot, length, velocity history."""

    anchor: Optional[ImmSnapshot] = None
    coast_start_frame: int = -1
    velocities: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.anchor = None
        self.coast_start_frame = -1
        self.velocities.clear()


def gen_virtual_trajectory(
    z_t1: np.ndarray,
    z_t2: np.ndarray,
    n: int,
) -> np.ndarray:
    """Linear interpolation between anchors over the missing frames.

    Returns shape (n-1, 4) — the virtual measurements for frames
    t1+1, t1+2, ..., t2-1 (z_t2 itself is the real re-entry measurement,
    handled by the caller's normal update path).
    """
    if n <= 1:
        return np.empty((0, 4), dtype=np.float32)
    z1 = np.asarray(z_t1, dtype=np.float32).reshape(4)
    z2 = np.asarray(z_t2, dtype=np.float32).reshape(4)
    # alphas for t = t1+1 .. t2-1 (exclude both anchors)
    alphas = np.arange(1, n, dtype=np.float32) / float(n)
    return (z1[None, :] + alphas[:, None] * (z2 - z1)[None, :]).astype(np.float32)


def velocity_cv(samples: list[float]) -> float:
    """Coefficient of variation: std/mean. Returns +inf if mean ~ 0."""
    if len(samples) < 2:
        return 0.0
    arr = np.asarray(samples, dtype=np.float64)
    mean = float(arr.mean())
    if mean < 1e-6:
        return float("inf")
    return float(arr.std() / mean)


class OruController:
    """Drives ORU snapshot capture, gating, and re-update.

    Lifecycle (called from the main pipeline loop):
        - on every accepted TRACKING update -> push_snapshot(...)
        - on TRACKING -> COASTING transition -> on_coast_start(frame_idx)
        - on every COASTING frame              -> record_coast_velocity(v)
        - on COASTING -> TRACKING transition  -> maybe_run(imm, z_t2, conf, frame_idx)
        - on LOST                              -> reset() (anchor invalidates)
    """

    def __init__(self, config: Optional[OruConfig] = None) -> None:
        self.cfg = config or OruConfig()
        self._snaps: collections.deque[ImmSnapshot] = collections.deque(
            maxlen=self.cfg.ring_buffer_size
        )
        self._coast = _CoastWindow()
        # Telemetry
        self.fired_count: int = 0
        self.gate_skips: dict[str, int] = collections.defaultdict(int)

    # ── Snapshot capture ───────────────────────────────────────────

    def push_snapshot(
        self,
        state: np.ndarray,
        P: np.ndarray,
        mu: np.ndarray,
        z: np.ndarray,
        frame_idx: int,
    ) -> None:
        """Record the post-update IMM state at an accepted TRACKING frame."""
        if not self.cfg.enabled:
            return
        snap = ImmSnapshot(
            state=np.asarray(state, dtype=np.float32).copy().reshape(10),
            P=np.asarray(P, dtype=np.float32).copy().reshape(10, 10),
            mu=np.asarray(mu, dtype=np.float32).copy().reshape(3),
            z=np.asarray(z, dtype=np.float32).copy().reshape(4),
            frame_idx=int(frame_idx),
            singer_mu=float(np.asarray(mu).reshape(3)[2]),
        )
        self._snaps.append(snap)

    # ── State transition hooks ─────────────────────────────────────

    def on_coast_start(self, frame_idx: int) -> None:
        """Latch the most recent snapshot as the t1 anchor."""
        if not self.cfg.enabled:
            return
        if not self._snaps:
            self._coast.reset()
            return
        self._coast.anchor = self._snaps[-1]
        self._coast.coast_start_frame = int(frame_idx)
        self._coast.velocities.clear()

    def record_coast_velocity(self, v_norm: float) -> None:
        """Append per-coast-frame |v| (used by the velocity-CV gate)."""
        if not self.cfg.enabled or self._coast.anchor is None:
            return
        self._coast.velocities.append(float(v_norm))

    def reset(self) -> None:
        """Drop the active coast (e.g. on LOST or hard reinit)."""
        self._coast.reset()

    # ── Re-update entry ────────────────────────────────────────────

    def maybe_run(
        self,
        imm,
        z_t2: np.ndarray,
        conf_t2: float,
        frame_idx: int,
    ) -> bool:
        """Run ORU backfill if the gate passes; otherwise no-op.

        Returns True iff the IMM was rewound and re-updated along virtual data.
        Caller is responsible for the final real-z update at frame t2 and for
        clearing any per-frame feedback (F5) during this call.
        """
        if not self.cfg.enabled:
            return False
        if self._coast.anchor is None:
            self._record_skip("no_anchor")
            return False

        anchor = self._coast.anchor
        n = int(frame_idx) - anchor.frame_idx  # gap in frames (>=2 for real backfill)

        if not self._gate_pass(anchor, conf_t2, n):
            return False

        virtual = gen_virtual_trajectory(anchor.z, z_t2, n)
        if virtual.shape[0] == 0:
            self._record_skip("empty_virtual")
            return False

        imm.restore_from(anchor.state, anchor.P, anchor.mu)
        for k in range(virtual.shape[0]):
            imm.predict()  # inflate P via Q before each update (required for correct Kalman gain)
            # Use no-conf overload → baseline R (no adaptive-R scaling).
            # virtual_conf was calibrated for floor=0.4 but floor is now 0.01,
            # so passing virtual_conf=0.4 would shrink R 40× and over-commit to
            # the linear interpolation. Neutral R is the correct prior here.
            imm.update(virtual[k].astype(np.float32))

        self.fired_count += 1
        # Coast window invalidated: re-acquired track owns its own future.
        self._coast.reset()
        return True

    # ── Gate ───────────────────────────────────────────────────────

    def _gate_pass(self, anchor: ImmSnapshot, conf_t2: float, n: int) -> bool:
        if n < self.cfg.n_min:
            self._record_skip("too_short")
            return False
        if n > self.cfg.n_max:
            self._record_skip("too_long")
            return False
        if conf_t2 < self.cfg.conf_reentry_min:
            self._record_skip("low_reentry_conf")
            return False
        if (
            self.cfg.skip_if_singer_dom
            and anchor.singer_mu > self.cfg.singer_dom_thr
        ):
            self._record_skip("singer_dom")
            return False
        if len(self._coast.velocities) >= self.cfg.min_velocity_samples:
            cv = velocity_cv(self._coast.velocities)
            if cv > self.cfg.velocity_cv_max:
                self._record_skip("velocity_cv")
                return False
        return True

    def _record_skip(self, reason: str) -> None:
        self.gate_skips[reason] += 1

    # ── Introspection ──────────────────────────────────────────────

    def telemetry(self) -> dict:
        return {
            "fired_count": int(self.fired_count),
            "gate_skips": dict(self.gate_skips),
            "snapshots_buffered": len(self._snaps),
            "coast_active": self._coast.anchor is not None,
        }
