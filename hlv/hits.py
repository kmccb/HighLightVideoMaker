"""The 'hit' moment of a play (tackle, catch, big block): what gets synced to a beat.

You set it with K on the review page (review.hit). Until then we suggest one:
  - with a spotlight: the moment your player stops hardest (biggest deceleration after
    he's been moving fast) - that's the collision;
  - otherwise: the peak of on-field motion during the play.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import camera, plays
from .config import DetectConfig
from .pipeline import CACHE_DIRNAME

FPS = 30.0


def _times(clip: dict) -> tuple[float, float]:
    r, d = clip.get("review") or {}, clip["detection"]
    snap = r.get("snap") if r.get("snap") is not None else d["snap"]
    end = r.get("end") if r.get("end") is not None else d["end"]
    return snap, end


def suggest(root: Path, clip: dict) -> dict:
    snap, end = _times(clip)
    lo, hi = snap + 0.5, max(snap + 0.6, end)
    cache = root / CACHE_DIRNAME
    if camera.anchors_of(clip):
        try:
            pl = camera.player_track(cache, clip)
        except Exception:  # noqa: BLE001 - fall back to motion
            pl = None
        if pl is not None:
            x, y, conf = pl[:, 0], pl[:, 1], np.nan_to_num(pl[:, 4])
            ok = ~np.isnan(x)
            if ok.sum() > 20:
                idx = np.arange(len(x))
                xs = np.interp(idx, idx[ok], x[ok])
                ys = np.interp(idx, idx[ok], y[ok]) * (9 / 16)   # roughly equal units
                v = gaussian_filter1d(np.hypot(np.gradient(xs), np.gradient(ys)), 3) * FPS
                dec = -np.gradient(gaussian_filter1d(v, 3))
                # A collision: fast before, sudden stop, while we still trust the track.
                peak_before = np.maximum.accumulate(v)
                score = dec * np.sqrt(np.maximum(peak_before, 0)) * (conf > 0.25)
                i0, i1 = int(lo * FPS), min(len(v) - 1, int(hi * FPS))
                if i1 > i0 and score[i0:i1].max() > 0:
                    i = i0 + int(np.argmax(score[i0:i1]))
                    # Contact is where the stop *begins*, not its sharpest point (that's
                    # when they hit the ground).
                    while i > i0 and dec[i - 1] > 0.35 * dec[i]:
                        i -= 1
                    return {"t": round(i / FPS, 3), "method": "player stops"}
    npz = cache / f"{clip['id']}.npz"
    if npz.exists():
        with np.load(npz) as z:
            sig = {k: z[k] for k in z.files}
        act = plays.activity(sig, FPS, DetectConfig.load(root / "hlv_config.json"))
        i0, i1 = int(lo * FPS), min(len(act) - 1, int(hi * FPS))
        if i1 > i0:
            return {"t": round((i0 + int(np.argmax(act[i0:i1]))) / FPS, 3), "method": "motion peak"}
    return {"t": round(snap + 0.6 * (end - snap), 3), "method": "estimate"}


def hit_of(root: Path, clip: dict) -> tuple[float | None, bool]:
    """(hit time, set_by_you). Hit time is None when you marked the play as having no hit."""
    r = clip.get("review") or {}
    if r.get("no_hit"):
        return None, True
    if r.get("hit") is not None:
        return float(r["hit"]), True
    return suggest(root, clip)["t"], False
