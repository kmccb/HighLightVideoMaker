"""Find the live-video rectangle inside a screen recording.

App chrome (toolbars, letterbox bars) barely changes over time while the embedded video
does, so per-pixel temporal std separates them. Row/column activity profiles give a
rectangle that tolerates small overlays (play button, scrubber) inside the video.
"""
from __future__ import annotations

import numpy as np


def detect_crop(frames: np.ndarray, src_w: int, src_h: int,
                std_thresh: float = 8.0, line_frac: float = 0.25,
                min_area_frac: float = 0.15, max_area_frac: float = 0.95) -> dict | None:
    """frames: (N, h, w) gray samples in display orientation. Returns crop in source px or None."""
    if len(frames) < 4:
        return None
    std = frames.astype(np.float32).std(axis=0)
    active = std > std_thresh
    h, w = active.shape
    rows = np.where(active.mean(axis=1) > line_frac)[0]
    cols = np.where(active.mean(axis=0) > line_frac)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    y0, y1 = _longest_run(rows)
    x0, x1 = _longest_run(cols)
    area = (y1 - y0 + 1) * (x1 - x0 + 1) / (h * w)
    if not (min_area_frac <= area <= max_area_frac):
        return None
    sx, sy = src_w / w, src_h / h
    x, y = int(x0 * sx), int(y0 * sy)
    cw, ch = int((x1 - x0 + 1) * sx), int((y1 - y0 + 1) * sy)
    even = lambda v: max(2, v - v % 2)  # noqa: E731
    return {"x": even(x) if x else 0, "y": even(y) if y else 0, "w": even(cw), "h": even(ch)}


def _longest_run(idx: np.ndarray, max_gap: int = 3) -> tuple[int, int]:
    """Longest run of indices allowing small gaps (overlay text etc.)."""
    best = (idx[0], idx[0])
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > max_gap:
            if prev - start > best[1] - best[0]:
                best = (start, prev)
            start = i
        prev = i
    if prev - start > best[1] - best[0]:
        best = (start, prev)
    return int(best[0]), int(best[1])
