"""Particle filter for single-object tracking during occlusion / coasting.

Manages pre-allocated particles in [cx, cy, vx, vy] state space.
Designed to complement the IMM Kalman filter: activated when the tracker
enters COASTING/LOST state, blended back to KF when the AI re-acquires.

Usage::

    pf = ParticleFilter(n_particles=500)

    # On TRACKING → COASTING transition:
    pf.init(last_good_bbox, velocity=np.array([vx, vy]))

    # Each coasting frame:
    pf.predict(process_noise_std=2.0)

    # If AI returns a detection while coasting:
    pf.update(obs_bbox)
    pf.resample()

    # Get bbox estimate:
    est_bbox = pf.estimate()  # [x, y, w, h]
"""
from __future__ import annotations

import numpy as np


class ParticleFilter:
    """Particle filter in [cx, cy, vx, vy] state space.

    Width/height are stored from the last :meth:`init` or :meth:`update`
    call and returned by :meth:`estimate`.

    Parameters
    ----------
    n_particles : int
        Number of particles (default 500). Pre-allocated at construction.
    """

    def __init__(self, n_particles: int = 500) -> None:
        self.n = int(n_particles)
        # Pre-allocated — no dynamic allocation in hot path
        self._particles = np.zeros((self.n, 4), dtype=np.float32)  # [cx, cy, vx, vy]
        self._weights = np.full(self.n, 1.0 / self.n, dtype=np.float32)
        self._indices = np.empty(self.n, dtype=np.int32)
        self._last_w: float = 10.0
        self._last_h: float = 10.0
        self.initialized: bool = False

    # ------------------------------------------------------------------
    def init(
        self,
        bbox,
        velocity=None,
        pos_std: float = 5.0,
        vel_std: float = 2.0,
    ) -> None:
        """Scatter particles around *bbox* center with Gaussian noise.

        Parameters
        ----------
        bbox : array-like [x, y, w, h]
            Initial target bounding box (top-left, width, height).
        velocity : array-like [vx, vy] | None
            Initial velocity estimate from KF state (pixels/frame).
        pos_std : float
            Particle position spread in pixels (default 5.0).
        vel_std : float
            Particle velocity spread in pixels/frame (default 2.0).
        """
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        self._last_w = max(float(w), 1.0)
        self._last_h = max(float(h), 1.0)
        cx = x + w * 0.5
        cy = y + h * 0.5
        vx = float(velocity[0]) if velocity is not None else 0.0
        vy = float(velocity[1]) if velocity is not None else 0.0

        rng = np.random
        self._particles[:, 0] = (cx + rng.normal(0.0, pos_std, self.n)).astype(np.float32)
        self._particles[:, 1] = (cy + rng.normal(0.0, pos_std, self.n)).astype(np.float32)
        self._particles[:, 2] = (vx + rng.normal(0.0, vel_std, self.n)).astype(np.float32)
        self._particles[:, 3] = (vy + rng.normal(0.0, vel_std, self.n)).astype(np.float32)
        self._weights[:] = 1.0 / self.n
        self.initialized = True

    # ------------------------------------------------------------------
    def predict(
        self,
        dt: float = 1.0,
        process_noise_std: float = 2.0,
        vel_noise_std: float = 0.6,
    ) -> None:
        """Propagate particles: position += velocity * dt + noise."""
        rng = np.random
        self._particles[:, 0] += (
            self._particles[:, 2] * dt
            + rng.normal(0.0, process_noise_std, self.n).astype(np.float32)
        )
        self._particles[:, 1] += (
            self._particles[:, 3] * dt
            + rng.normal(0.0, process_noise_std, self.n).astype(np.float32)
        )
        self._particles[:, 2] += rng.normal(0.0, vel_noise_std, self.n).astype(np.float32)
        self._particles[:, 3] += rng.normal(0.0, vel_noise_std, self.n).astype(np.float32)

    # ------------------------------------------------------------------
    def update(self, obs_bbox, obs_noise_std: float = 15.0) -> None:
        """Update particle weights from Gaussian likelihood of *obs_bbox*.

        Also updates stored w/h for the next :meth:`estimate` call.
        """
        ox, oy, ow, oh = (
            float(obs_bbox[0]),
            float(obs_bbox[1]),
            float(obs_bbox[2]),
            float(obs_bbox[3]),
        )
        self._last_w = max(ow, 1.0)
        self._last_h = max(oh, 1.0)
        ocx = ox + ow * 0.5
        ocy = oy + oh * 0.5

        dx = self._particles[:, 0] - ocx
        dy = self._particles[:, 1] - ocy
        d2 = dx * dx + dy * dy
        sigma2 = float(obs_noise_std) ** 2

        log_w = -0.5 * d2 / sigma2
        log_w -= log_w.max()  # numerical stability
        w = np.exp(log_w)
        w_sum = float(w.sum())
        if w_sum > 1e-10:
            self._weights[:] = (w / w_sum).astype(np.float32)
        else:
            self._weights[:] = 1.0 / self.n

    # ------------------------------------------------------------------
    def resample(self) -> None:
        """Low-variance (systematic) resampling — O(N), unbiased."""
        cumsum = np.cumsum(self._weights)
        cumsum[-1] = 1.0  # guard floating-point round-off
        u0 = float(np.random.uniform(0.0, 1.0 / self.n))
        positions = (u0 + np.arange(self.n, dtype=np.float32) / self.n).astype(np.float32)
        indices = np.searchsorted(cumsum, positions)
        np.copyto(self._indices, indices)
        self._particles[:] = self._particles[self._indices]
        self._weights[:] = 1.0 / self.n

    # ------------------------------------------------------------------
    def estimate(self) -> np.ndarray:
        """Weighted mean → [x, y, w, h] bbox (float32)."""
        cx = float(np.dot(self._weights, self._particles[:, 0]))
        cy = float(np.dot(self._weights, self._particles[:, 1]))
        return np.array(
            [cx - self._last_w * 0.5, cy - self._last_h * 0.5, self._last_w, self._last_h],
            dtype=np.float32,
        )
