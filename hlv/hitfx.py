"""Signals for finding a play's hit, frame by frame (30 fps over the clip):

  speed    the player's speed over the field (camera pan/zoom removed), in his heights / s
  decel    how hard he slows down (contact, tackle)
  accel    how hard he speeds up (a burst, a big move)
  near     how close the nearest other person is (1 = boxes touching, 0 = far)
  crowd    how many people are within about one body height of him
  loud     crowd / impact loudness (audio, smoothed), and `onset` its sudden rises
  u        how far through the play (0 = snap, 1 = end)

Built from the player track (your clicks), the person detections and the clip's audio.
hits.suggest() combines them with weights learned from the hits you've marked.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

FPS = 30.0
NAMES = ("speed", "decel", "accel", "near", "crowd", "loud", "onset", "u")


def _audio_db(src: Path, n: int) -> np.ndarray:
    from .audio import read_audio
    try:
        x = read_audio(src, 16000)
    except Exception:  # noqa: BLE001 - no audio track
        return np.zeros(n)
    hop = int(16000 / FPS)
    k = min(n, len(x) // hop)
    rms = np.sqrt(np.mean(x[:k * hop].reshape(k, hop) ** 2, axis=1) + 1e-10)
    out = np.full(n, float(20 * np.log10(rms.min() + 1e-10)) if k else -80.0)
    out[:k] = 20 * np.log10(rms + 1e-10)
    return out


def features(root: Path, clip: dict, player: np.ndarray, cam: np.ndarray | None,
             dets: dict[int, np.ndarray] | None, snap: float, end: float) -> dict[str, np.ndarray]:
    """player: smoothed track (N, 5) normalised; cam: (N, 2, 3) in 960-px analysis units."""
    n = len(player)
    x, y, h = player[:, 0], player[:, 1], player[:, 3]
    ok = ~np.isnan(x)
    idx = np.arange(n)
    if ok.sum() < 5:
        return {k: np.zeros(n) for k in NAMES}
    xs, ys = np.interp(idx, idx[ok], x[ok]), np.interp(idx, idx[ok], y[ok])
    hs = np.maximum(np.interp(idx, idx[ok], h[ok]), 0.01)
    # field motion = image motion minus what the camera alone would have done
    W, Hh = 960.0, 540.0
    vx, vy = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        px, py = xs[i - 1] * W, ys[i - 1] * Hh
        if cam is not None and i < len(cam):
            m = cam[i]
            px, py = m[0, 0] * px + m[0, 1] * py + m[0, 2], m[1, 0] * px + m[1, 1] * py + m[1, 2]
        vx[i], vy[i] = xs[i] * W - px, ys[i] * Hh - py
    speed = gaussian_filter1d(np.hypot(vx, vy) / (hs * Hh) * FPS, 3)
    dv = np.gradient(gaussian_filter1d(speed, 2)) * FPS
    near, crowd = np.zeros(n), np.zeros(n)
    if dets:
        ks = sorted(dets)
        nv, cv = [], []
        for i in ks:
            b = dets[i]
            if i >= n or not len(b):
                nv.append(0.0); cv.append(0.0)
                continue
            d = np.hypot((b[:, 0] - xs[i]) * 16 / 9, b[:, 1] - ys[i]) / hs[i]
            me = int(np.argmin(d))
            other = np.delete(d, me) if d[me] < 0.6 else d
            nv.append(float(np.clip(1.2 - other.min(), 0, 1.2)) if len(other) else 0.0)
            cv.append(float((other < 1.2).sum()))
        near = np.interp(idx, ks, nv)
        crowd = np.interp(idx, ks, cv)
    loud = _audio_db(Path(root) / clip["source"], n)
    loud_s = gaussian_filter1d(loud, 4)
    onset = np.maximum(np.gradient(gaussian_filter1d(loud, 2)) * FPS, 0)
    u = (idx / FPS - snap) / max(0.5, end - snap)
    return {"speed": speed, "decel": np.maximum(-dv, 0), "accel": np.maximum(dv, 0),
            "near": gaussian_filter1d(near, 2), "crowd": gaussian_filter1d(crowd, 3),
            "loud": loud_s, "onset": gaussian_filter1d(onset, 2), "u": u}


def score(feats: dict[str, np.ndarray], w: dict[str, float], lo: int, hi: int) -> np.ndarray:
    """Weighted sum of the signals, each standardised over the play window [lo, hi)."""
    s = np.zeros(hi - lo)
    for k, wk in w.items():
        if not wk or k not in feats:
            continue
        v = feats[k][lo:hi]
        if k == "u":                                   # prefer the middle of the play
            s += wk * -((v - w.get("_u0", 0.5)) ** 2) * 4
            continue
        sd = v.std()
        s += wk * ((v - v.mean()) / sd if sd > 1e-9 else 0)
    return s
