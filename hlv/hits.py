"""The 'hit' moment of a play (tackle, catch, big block): what gets synced to a beat.

You set it with K on the review page or on the Mark plays page (review.hit). Until then we
suggest one: a mix of signals learned from your marked hits (hitfx: someone right on him,
a crowd around him, a jump in loudness, mid-play, not while he's still accelerating).
Fallbacks if that can't run: the moment the player stops hardest, else the motion peak.
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


# Learned from Tom's 23 marked hits (2026-10-05; leave-one-out: 48% within 0.4 s, median
# 0.5 s off, vs ~17% for the old "player stops" rule). Signals: see hitfx.
WEIGHTS = {"accel": -1.0, "near": 0.5, "crowd": 0.5, "onset": 1.0, "u": 2.0, "_u0": 0.55}


def learned(root: Path, clip: dict) -> dict | None:
    """The learned suggestion, from the player track (if clicked), saved person detections
    (if the warm-up has run) and the audio. None if there's nothing to go on."""
    from . import detect, hitfx
    snap, end = _times(clip)
    cache = root / CACHE_DIRNAME
    player = None
    if camera.anchors_of(clip):
        try:
            player = camera.player_track(cache, clip)
        except Exception:  # noqa: BLE001
            player = None
    n = int(round(clip["probe"]["duration"] * FPS)) + 1
    if player is None:
        player = np.full((n, 5), np.nan, np.float32)
    camf = cache / "track" / f"{clip['id']}-cam.npy"
    cam = np.load(camf) if camf.exists() else None
    dets = detect.cached(cache, clip)
    f = hitfx.features(root, clip, player, cam, dets, snap, end)
    lo, hi = int((snap + 0.4) * FPS), min(len(player), int((end + 0.3) * FPS))
    if hi <= lo + 2:
        return None
    s = hitfx.score(f, WEIGHTS, lo, hi)
    how = "learned" + ("" if dets else " (sound + timing; people not detected yet)")
    return {"t": round((lo + int(np.argmax(s))) / FPS, 3), "method": how}


def suggest(root: Path, clip: dict) -> dict:
    try:
        r = learned(root, clip)
        if r is not None:
            return r
    except Exception:  # noqa: BLE001 - fall back to the old rules below
        pass
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
