"""Pre-computed Singer F/Q matrix lookup table — one-shot at import, O(1) at runtime.

Discretizes (α, σ²_a) space at module load time (~1 ms, 64 entries).
At runtime: O(1) nearest-grid-point snap, no floating-point loops.

The physics implemented here MUST match IMMFilter::build_singer_fq() in C++ exactly.
Any change to the formulas must be made in both places.

Grid coverage:
    ALPHA_GRID  : α ∈ [0.1, 10.0] — 8 points
    SIGMA2_GRID : σ²_a ∈ [1.0, 200.0] — 8 points
    Total       : 64 entries × 2 × (10×10) float32 ≈ 50 KB
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Discretization grids (named constants — AV Rule 151 equivalent in Python)
ALPHA_GRID  = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0], dtype=np.float32)
SIGMA2_GRID = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 50.0, 100.0, 200.0], dtype=np.float32)
_DT         = 1.0  # 1 frame = 1 time unit


@dataclass(frozen=True, slots=True)
class SingerEntry:
    """Singer model matrices for one (α, σ²_a) grid point."""
    alpha:   float
    sigma2:  float
    F: np.ndarray   # (10, 10) float32 — state transition
    Q: np.ndarray   # (10, 10) float32 — process noise


def _build_entry(alpha: float, sigma2_a: float) -> SingerEntry:
    """Physics-exact Singer F and Q — mirrors IMMFilter::build_singer_fq() (C++).

    References:
        Singer, R.A. (1970). "Estimating Optimal Tracking Filter Performance
        for Manned Maneuvering Targets." IEEE Trans. Aerospace Electronic Systems.
    """
    dt   = _DT
    a1   = max(float(alpha), 1e-4)   # guard: no div-by-zero
    s2a  = max(float(sigma2_a), 1e-6)
    beta = float(np.exp(-a1 * dt))

    a2, a3, a4, a5 = a1**2, a1**3, a1**4, a1**5
    b2 = beta**2

    # Transition coupling terms
    f_pa = (beta - 1.0 + a1 * dt) / a2   # pos ← acc integral
    f_va = (1.0 - beta) / a1              # vel ← acc step

    # Process noise cross-covariance entries
    q_pp = s2a * (2*a3*dt - 3 + 4*beta - b2) / (2*a5)
    q_pv = s2a * (1 - 2*a1*dt*beta - b2)     / (2*a4)
    q_vv = s2a * (1 - b2)                    / (2*a3)
    q_aa = s2a * (1 - b2)                    / a1

    # ── F matrix (10×10) ──────────────────────────────────────────
    F = np.eye(10, dtype=np.float32)
    # Position += velocity·dt  (all 4 spatial dims)
    F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = float(dt)
    # Position += f_pa · acceleration  (x and y only)
    F[0, 8] = F[1, 9] = float(f_pa)
    # Velocity += f_va · acceleration  (x and y only)
    F[4, 8] = F[5, 9] = float(f_va)
    # Acceleration decays: a(k+1) = β·a(k)
    F[8, 8] = F[9, 9] = float(beta)

    # ── Q matrix (10×10) ──────────────────────────────────────────
    Q = np.zeros((10, 10), dtype=np.float32)
    # x-axis: pos=0, vel=4, acc=8
    # y-axis: pos=1, vel=5, acc=9
    for pi, vi, ai in [(0, 4, 8), (1, 5, 9)]:
        Q[pi, pi] = float(q_pp)
        Q[pi, vi] = Q[vi, pi] = float(q_pv)
        Q[vi, vi] = float(q_vv)
        Q[ai, ai] = float(q_aa)
    # Size axes: simple diagonal — no physical Singer model for w/h
    Q[2, 2] = Q[3, 3] = 1.0
    Q[6, 6] = Q[7, 7] = 0.1

    return SingerEntry(alpha=float(alpha), sigma2=float(sigma2_a), F=F, Q=Q)


# ── Build LUT at import time (one-shot, ~1 ms) ───────────────────
_LUT: dict[tuple[float, float], SingerEntry] = {
    (float(a), float(s)): _build_entry(float(a), float(s))
    for a in ALPHA_GRID
    for s in SIGMA2_GRID
}


def lookup(alpha: float, sigma2_a: float) -> SingerEntry:
    """O(1) LUT lookup — snaps to nearest grid point.

    Args:
        alpha:    Maneuver correlation time inverse (1/τ). Valid range: (0, 20].
        sigma2_a: Acceleration variance [px²/frame⁴]. Valid range: (0, 500].

    Returns:
        SingerEntry with F and Q matrices for nearest grid point.

    Notes:
        For exact physics (arbitrary α, σ²_a), use IMMFilter.set_singer_params()
        which calls the C++ build_singer_fq() at runtime. The LUT is provided
        for offline inspection and Python-side validation.
    """
    a_idx   = int(np.argmin(np.abs(ALPHA_GRID  - float(alpha))))
    s_idx   = int(np.argmin(np.abs(SIGMA2_GRID - float(sigma2_a))))
    a_snap  = float(ALPHA_GRID[a_idx])
    s_snap  = float(SIGMA2_GRID[s_idx])
    return _LUT[(a_snap, s_snap)]
