"""ChaosDetector — Confidence Volatility / Chaos Trigger.

Tracks rolling std-dev of SGLATrack confidence over a sliding window.
High volatility  →  "chaotic environment"  →  reduced effective confidence
                 →  C++ R multiplier = floor/eff_conf > 1
                 →  R inflated  →  IMM trusts prediction over measurement.

Usage (pipeline.py):
    detector = ChaosDetector(ChaosConfig.from_dict(cfg.get("chaos_config")))
    eff_conf = detector.step(obs.confidence)   # call before step_guided_imm
    ...
    detector.reset()                            # call between sequences
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChaosConfig:
    """Tunable knobs for the ChaosDetector."""

    enabled: bool = True
    window: int = 10               # rolling window size (frames)
    chaos_thr: float = 0.08        # std-dev below this → no penalty
    chaos_max: float = 0.20        # std-dev at this level → max penalty
    min_chaos_conf_factor: float = 0.15  # conf reduced to this fraction at peak chaos
    min_samples: int = 5           # don't trigger until buffer has ≥ N samples

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ChaosConfig":
        if not d:
            return cls()
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**kwargs)


class ChaosDetector:
    """Sliding-window confidence volatility detector.

    Each call to ``step(conf)`` appends the raw AI confidence, computes the
    rolling std-dev, and returns a (possibly) deflated effective confidence.

    The deflation formula is a linear ramp:

        t          = clamp((σ − σ_thr) / (σ_max − σ_thr),  0, 1)
        eff_conf   = conf × (1 − t × (1 − factor_min))
        eff_conf   = max(eff_conf, 1e-3)          # avoid division-by-zero in C++

    At t=0 (calm):   eff_conf == conf  (no change)
    At t=1 (chaos):  eff_conf == conf × factor_min  (e.g. 0.15 × conf)
    With adaptive-R floor=0.4 and factor_min=0.15:
        conf=0.6 → eff=0.09 → multiplier = 0.4/0.09 ≈ 4.4× R inflation
    """

    def __init__(self, cfg: ChaosConfig) -> None:
        self.cfg = cfg
        self._buf: collections.deque[float] = collections.deque(maxlen=cfg.window)
        self._trigger_count: int = 0
        self._last_chaos_level: float = 0.0
        self._last_deflation: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, conf: float) -> float:
        """Append *conf*, return (deflated) effective confidence."""
        self._buf.append(float(conf))

        if not self.cfg.enabled:
            return conf

        if len(self._buf) < self.cfg.min_samples:
            return conf

        sigma = _std(self._buf)
        self._last_chaos_level = sigma

        thr = self.cfg.chaos_thr
        mx = self.cfg.chaos_max
        factor = self.cfg.min_chaos_conf_factor

        if sigma <= thr or mx <= thr:
            # Calm — no penalty
            self._last_deflation = 0.0
            return conf

        t = min((sigma - thr) / (mx - thr), 1.0)
        eff = conf * (1.0 - t * (1.0 - factor))
        eff = max(eff, 1e-3)

        if t > 0.0:
            self._trigger_count += 1

        self._last_deflation = conf - eff
        return eff

    def reset(self) -> None:
        """Clear window and counters (call between sequences)."""
        self._buf.clear()
        self._trigger_count = 0
        self._last_chaos_level = 0.0
        self._last_deflation = 0.0

    def telemetry(self) -> dict:
        return {
            "chaos_level": self._last_chaos_level,
            "trigger_count": self._trigger_count,
            "last_conf_deflation": self._last_deflation,
            "buffer_size": len(self._buf),
        }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _std(buf: collections.deque) -> float:
    """Population std-dev of a deque of floats."""
    n = len(buf)
    if n < 2:
        return 0.0
    mean = sum(buf) / n
    variance = sum((x - mean) ** 2 for x in buf) / n
    return math.sqrt(variance)
