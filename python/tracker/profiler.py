"""Frame-level microsecond profiler with per-stage budget enforcement.

Uses time.perf_counter_ns() (monotonic, OS-provided) for sub-microsecond
resolution. One FrameProfiler instance per pipeline run; context-manager
API keeps overhead minimal (no per-frame object creation).

Performance budgets are sourced from the project specifications:
    KF predict/update  < 0.5 ms p99
    IMM full cycle     < 1.0 ms p99
    AI inference       < 15 ms (→ 4-5 ms with TRT + IMM-guided skip)
    GMC                < 2.0 ms p99
    Total frame        ≤ 30 ms
"""
from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generator

import numpy as np


# Performance budgets per pipeline stage [ms, p99 target]
_BUDGETS_MS: dict[str, float] = {
    "read":     3.0,
    "gmc":      2.0,
    "ai":      15.0,   # drops to ~4-5 ms with TRT engine + IMM-guided skip
    "decision": 0.1,
    "kf":       1.0,
    "viz":      2.0,
    "total":   30.0,
}


@dataclass
class _StageStats:
    samples: list[float] = field(default_factory=list)

    def p50(self) -> float:
        return float(np.percentile(self.samples, 50)) if self.samples else 0.0

    def p99(self) -> float:
        return float(np.percentile(self.samples, 99)) if self.samples else 0.0

    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    def count(self) -> int:
        return len(self.samples)


class FrameProfiler:
    """Context-manager profiler for tracking pipeline stages.

    Usage::

        profiler = FrameProfiler()
        # Inside the frame loop:
        profiler.begin_frame()
        with profiler.stage("gmc"):
            ...gmc code...
        with profiler.stage("ai"):
            ...ai code...
        with profiler.stage("kf"):
            ...kf code...
        profiler.end_frame()
        # Print report every 100 frames:
        if frame_count % 100 == 0:
            print(profiler.report())
    """

    def __init__(self) -> None:
        self._stats: dict[str, _StageStats] = defaultdict(_StageStats)
        self._frame_start: int = 0
        self._violations: dict[str, int] = defaultdict(int)

    def begin_frame(self) -> None:
        """Record frame start timestamp."""
        self._frame_start = time.perf_counter_ns()

    @contextlib.contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager that times one pipeline stage.

        Records elapsed ms and increments violation counter if budget exceeded.
        """
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            self._stats[name].samples.append(dt_ms)
            budget = _BUDGETS_MS.get(name)
            if budget is not None and dt_ms > budget:
                self._violations[name] += 1

    def end_frame(self) -> None:
        """Record total frame time and check against 30 ms budget."""
        total_ms = (time.perf_counter_ns() - self._frame_start) / 1_000_000.0
        self._stats["total"].samples.append(total_ms)
        if total_ms > _BUDGETS_MS["total"]:
            self._violations["total"] += 1

    def report(self) -> str:
        """Return formatted table: stage, mean, p50, p99, budget, violations."""
        header = (
            f"\n{'Stage':<12} {'Mean ms':>9} {'p50 ms':>8} {'p99 ms':>8} "
            f"{'Budget':>8} {'Status':>8} {'Viols':>6}"
        )
        sep = "-" * 65
        rows: list[str] = [header, sep]
        for stage, budget in _BUDGETS_MS.items():
            if stage not in self._stats:
                continue
            s = self._stats[stage]
            v = self._violations.get(stage, 0)
            flag = "OVER" if v > 0 else "OK"
            rows.append(
                f"{stage:<12} {s.mean():>9.3f} {s.p50():>8.3f} {s.p99():>8.3f} "
                f"{budget:>8.1f} {flag:>8} {v:>6}"
            )
        total_frames = self._stats.get("total", _StageStats()).count()
        rows.append(f"\nFrames profiled: {total_frames}")
        return "\n".join(rows)

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._stats.clear()
        self._violations.clear()

    def get_stage_p99(self, name: str) -> float:
        """Return p99 latency [ms] for a named stage, or 0.0 if not recorded."""
        if name not in self._stats:
            return 0.0
        return self._stats[name].p99()
