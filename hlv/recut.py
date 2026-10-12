"""Re-cutting the song to the reel: on every big play's drop the song comes back on one of its
biggest moments instead of just carrying on.

The drop (ghost.py) already cuts all sound for a third of a second before the hit, so the song can
return at any point of itself without an audible splice. `landmarks` finds the points worth
returning to - beats that carry one of the song's heavy hits and are followed by a loud stretch,
best of all coming out of a quieter one (a chorus or a drop starting). reel.plan_timing asks `pick`
for the next one at each drop and re-anchors its beat grid there, so the plays after it are timed
to the song as it will actually be heard; reel._assemble lays the song down in those pieces.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import beats

VERSION = 1
AFTER, BEFORE = 8.0, 4.0     # seconds of loudness looked at after / before a candidate
TAIL = 12.0                  # nothing this close to the song's end (it would run out under the play)


def landmarks(music: Path, cache_dir: Path) -> list[list[float]]:
    """[[song time s, score], ...], best moments scoring highest (about 0..3). Cached."""
    st = music.stat()
    f = cache_dir / "beats" / f"{music.stem}-{st.st_size}-{int(st.st_mtime)}-marks{VERSION}.json"
    if f.exists():
        return json.loads(f.read_text())
    grid = beats.cached(music, cache_dir)
    heavy = beats.heavy_hits(music, cache_dir)
    x = beats.read_audio(music)
    hop = beats.SR // 10                                         # loudness in tenths of a second
    n = len(x) // hop
    rms = np.sqrt((x[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1))
    loud = max(1e-6, float(np.percentile(rms, 90)))
    dur = n / 10
    ht = np.array([t for t, _ in heavy]) if heavy else np.zeros(0)
    hs = np.array([s for _, s in heavy]) if heavy else np.zeros(0)
    out = []
    for t in grid["beats"]:
        if t > dur - TAIL or not len(ht):
            continue
        near = np.abs(ht - t) <= 0.08
        if not near.any():
            continue
        hit = float(hs[near].max())                              # the heavy hit this beat carries
        if hit < 0.5:
            continue
        k = int(t * 10)
        after = float(rms[k:k + int(AFTER * 10)].mean()) / loud
        before = float(rms[max(0, k - int(BEFORE * 10)):k].mean()) / loud if k > 0 else after
        out.append([round(float(t), 3), round(hit + after + 0.8 * max(0.0, after - before), 3)])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def pick(marks: list, pos: float, used: list[float], ahead: float = 4.0) -> float | None:
    """Where the song comes back at a drop, given it's now at `pos`: the best moment further on
    (never one it has already used, or right next to one), slightly preferring nearer ones so the
    song still moves forward in order. When nothing is left ahead, the best unused one anywhere."""
    fresh = [(t, s) for t, s in marks if all(abs(t - u) > 6.0 for u in used)]
    cands = [(t, s) for t, s in fresh if t > pos + ahead] or [(t, s) for t, s in fresh if abs(t - pos) > 15.0]
    if not cands:
        return None
    return float(max(cands, key=lambda c: c[1] - 0.004 * abs(c[0] - pos))[0])


def cuts(music_start: float, timing: list[dict]) -> list[tuple[float, float]]:
    """(reel time, song time) of every point the song (re)starts from: the reel's start, then each drop's jump."""
    out = [(0.0, float(music_start))]
    for tm in timing:
        if tm.get("jump") is not None:
            out.append((float(tm["hit_reel"]), float(tm["jump"])))
    return out


def song_time(cut_list: list[tuple[float, float]], t: float) -> float:
    """The song time heard at reel time t."""
    k = max(i for i, (rt, _) in enumerate(cut_list) if rt <= t + 1e-9) if cut_list else None
    return t if k is None else cut_list[k][1] + (t - cut_list[k][0])
