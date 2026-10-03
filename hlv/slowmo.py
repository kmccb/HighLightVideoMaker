"""Slow-motion ramp into the hit, and the frame interpolation that makes it smooth.

Speed curve (in source time, around the hit H):
    full speed -> ramp down over RAMP_IN -> hold `speed` from H-HOLD_BEFORE to H+HOLD_AFTER
    -> ramp back up over RAMP_OUT -> full speed.
Output frames land at fractional source times; real frames are used when the source has
them (60/120 fps clips), otherwise in-betweens are synthesised with optical flow.
"""
from __future__ import annotations

import cv2
import numpy as np

RAMP_IN, HOLD_BEFORE, HOLD_AFTER, RAMP_OUT = 0.5, 0.4, 0.25, 0.35


def window(hit: float, lo: float, hi: float) -> tuple[float, float, float, float] | None:
    """(ramp-in start, hold start, hold end, ramp-out end) in source seconds, clipped to
    [lo, hi]; None if there isn't room for a ramp."""
    a, b = hit - HOLD_BEFORE - RAMP_IN, hit - HOLD_BEFORE
    c, d = hit + HOLD_AFTER, hit + HOLD_AFTER + RAMP_OUT
    if b - lo < 0.15 or hi - c < 0.15:
        return None
    return max(a, lo), b, c, min(d, hi)


def speed_at(t: float, win, speed: float) -> float:
    if win is None:
        return 1.0
    a, b, c, d = win
    if t <= a or t >= d:
        return 1.0
    if b <= t <= c:
        return speed
    u = (t - a) / max(b - a, 1e-6) if t < b else 1 - (t - c) / max(d - c, 1e-6)
    u = u * u * (3 - 2 * u)                      # smoothstep: no jolt entering/leaving
    return 1.0 + (speed - 1.0) * u


def timeline(t_start: float, t_end: float, fps: float, win, speed: float) -> list[float]:
    """Source time of every output frame from t_start to t_end."""
    out, t = [], t_start
    while t < t_end - 1e-6:
        out.append(t)
        t += speed_at(t, win, speed) / fps
    return out


def atempo(f: float) -> str:
    """ffmpeg atempo chain for any factor (each stage must be within 0.5..2)."""
    parts = []
    while f < 0.5:
        parts.append("atempo=0.5")
        f /= 0.5
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    parts.append(f"atempo={f:.5f}")
    return ",".join(parts)


class Interpolator:
    """Optical-flow in-betweens between two frames (computed at half size for speed)."""

    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._key = None
        self._flows = None

    def _flows_for(self, a: np.ndarray, b: np.ndarray, key):
        if key != self._key:
            h, w = a.shape[:2]
            ga = cv2.cvtColor(cv2.resize(a, (w // 2, h // 2)), cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(cv2.resize(b, (w // 2, h // 2)), cv2.COLOR_BGR2GRAY)
            fab = cv2.resize(self.dis.calc(ga, gb, None), (w, h)) * 2
            fba = cv2.resize(self.dis.calc(gb, ga, None), (w, h)) * 2
            self._key, self._flows = key, (fab, fba)
        return self._flows

    def between(self, a: np.ndarray, b: np.ndarray, alpha: float, key=None) -> np.ndarray:
        if alpha <= 0.02:
            return a
        if alpha >= 0.98:
            return b
        fab, fba = self._flows_for(a, b, key if key is not None else (id(a), id(b)))
        h, w = a.shape[:2]
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        wa = cv2.remap(a, gx - alpha * fab[..., 0], gy - alpha * fab[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(b, gx - (1 - alpha) * fba[..., 0], gy - (1 - alpha) * fba[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(wa, 1 - alpha, wb, alpha, 0)
