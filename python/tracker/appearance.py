"""D4: Lightweight appearance-based Re-ID via patch colour histogram.

ReIDMemory stores the last `maxlen` accepted frames' (frame_idx, bbox, hist)
tuples.  During rescue (LOST state), `similarity()` computes cosine distance
between the query histogram and every stored entry, returning the best match.

Design constraints (JSF-compatible):
- No dynamic allocation after construction: deque fixed at maxlen.
- All arithmetic is NumPy vectorised (no Python loops on hot paths).
- No network inference — 32×32 BGR→HSV 8-bin histogram (~192 floats).
"""

from __future__ import annotations

import collections
from typing import NamedTuple

import numpy as np


class ReidMatch(NamedTuple):
    frame_idx: int
    bbox: np.ndarray   # [x, y, w, h] float32
    similarity: float  # cosine similarity in [0, 1]


def _extract_histogram(patch: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Compute an L2-normalised joint H+S+V histogram from a BGR patch.

    Returns a float32 vector of length ``3 * n_bins``.
    """
    import cv2  # lazy import — only needed at runtime
    if patch.size == 0:
        return np.zeros(3 * n_bins, dtype=np.float32)
    resized = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # Compute per-channel histogram in [0, 256)
    hists = []
    ranges = [(0, 180), (0, 256), (0, 256)]
    for ch, (lo, hi) in enumerate(ranges):
        h, _ = np.histogram(hsv[:, :, ch].flatten(), bins=n_bins, range=(lo, hi))
        hists.append(h.astype(np.float32))
    feat = np.concatenate(hists)
    norm = float(np.linalg.norm(feat))
    if norm > 1e-9:
        feat /= norm
    return feat


class ReIDMemory:
    """Fixed-capacity circular buffer of appearance descriptors.

    Parameters
    ----------
    maxlen:
        Maximum number of entries retained (FIFO eviction).
    n_bins:
        Number of histogram bins per HSV channel (default 8 → 24-dim vector).
    """

    def __init__(self, maxlen: int = 50, n_bins: int = 8) -> None:
        self._maxlen = maxlen
        self._n_bins = n_bins
        self._buf: collections.deque[tuple[int, np.ndarray, np.ndarray]] = (
            collections.deque(maxlen=maxlen)
        )
        self._frozen = False

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        bbox: np.ndarray,
    ) -> None:
        """Extract and store a descriptor for the current accepted frame.

        No-op when frozen (LOST state) or when bbox is degenerate.
        """
        if self._frozen:
            return
        bbox = np.asarray(bbox, dtype=np.float32).flatten()
        if bbox.shape[0] < 4 or bbox[2] < 2 or bbox[3] < 2:
            return
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        ih, iw = frame_bgr.shape[:2]
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, iw)
        y2 = min(y + h, ih)
        if x2 <= x1 or y2 <= y1:
            return
        patch = frame_bgr[y1:y2, x1:x2]
        hist = _extract_histogram(patch, self._n_bins)
        self._buf.append((frame_idx, bbox.copy(), hist))

    def freeze(self) -> None:
        """Stop accepting new entries (call on transition to LOST)."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Resume accepting entries (call on re-acquisition)."""
        self._frozen = False

    def similarity(
        self,
        frame_bgr: np.ndarray,
        query_bbox: np.ndarray,
    ) -> ReidMatch | None:
        """Return the best-matching stored entry for ``query_bbox``.

        Returns ``None`` when the buffer is empty or the query patch is bad.
        """
        if not self._buf:
            return None
        query_bbox = np.asarray(query_bbox, dtype=np.float32).flatten()
        if query_bbox.shape[0] < 4 or query_bbox[2] < 2 or query_bbox[3] < 2:
            return None
        x, y, w, h = (int(query_bbox[0]), int(query_bbox[1]),
                       int(query_bbox[2]), int(query_bbox[3]))
        ih, iw = frame_bgr.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, iw), min(y + h, ih)
        if x2 <= x1 or y2 <= y1:
            return None
        query_hist = _extract_histogram(frame_bgr[y1:y2, x1:x2], self._n_bins)

        best_idx = -1
        best_sim = -1.0
        best_bbox = None
        for i, (fidx, bbox, hist) in enumerate(self._buf):
            sim = float(np.dot(query_hist, hist))
            if sim > best_sim:
                best_sim = sim
                best_idx = fidx
                best_bbox = bbox

        if best_idx < 0 or best_bbox is None:
            return None
        return ReidMatch(frame_idx=best_idx, bbox=best_bbox, similarity=best_sim)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        """Reset buffer and unfreeze."""
        self._buf.clear()
        self._frozen = False
