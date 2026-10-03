"""Shared camera planning for the review page and renders.

Works on the 720p review proxy (fast to decode, same timeline as the source) and returns
crop rects normalised to the full display frame, so they apply to any resolution.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from . import autozoom, frames, motion, proxy, track
from .probe import probe

FPS = 30.0
TRACK_W = 960
_frames_cache: OrderedDict = OrderedDict()
_lock = threading.Lock()


def _proxy_frames(cache: Path, cid: str) -> np.ndarray:
    with _lock:
        if cid in _frames_cache:
            _frames_cache.move_to_end(cid)
            return _frames_cache[cid]
    p = proxy.proxy_path(cache, cid)
    info = probe(p)
    f = frames.read_frames(p, info.width, info.height, fps=FPS, width=TRACK_W, color=True)
    with _lock:
        _frames_cache[cid] = f
        while len(_frames_cache) > 3:  # a few clips' worth of 960px frames in memory
            _frames_cache.popitem(last=False)
    return f


def _proxy_size(cache: Path, cid: str) -> tuple[int, int]:
    """Size of the analysis frames (proxy scaled to TRACK_W) without decoding them."""
    info = probe(proxy.proxy_path(cache, cid))
    return TRACK_W, int(round(info.height * TRACK_W / info.width / 2)) * 2


def content_rect(clip: dict, W: int, H: int) -> tuple:
    """Real video area in analysis px (screen recordings: inside the app chrome)."""
    c = clip.get("crop")
    if not c:
        return (0.0, 0.0, float(W), float(H))
    sx, sy = W / clip["probe"]["width"], H / clip["probe"]["height"]
    return (c["x"] * sx, c["y"] * sy, c["w"] * sx, c["h"] * sy)


def anchors_of(clip: dict) -> list[dict]:
    sp = (clip.get("review") or {}).get("spotlight")
    return [a for a in sp if isinstance(a, dict)] if isinstance(sp, list) else []


def player_track(cache: Path, clip: dict) -> np.ndarray | None:
    anchors = anchors_of(clip)
    if not anchors:
        return None
    key = hashlib.sha1(("v11" + json.dumps(anchors, sort_keys=True)).encode()).hexdigest()[:12]
    f = cache / "track" / f"{clip['id']}-{key}.npy"
    if f.exists():
        res = np.load(f)
    else:
        fr = _proxy_frames(cache, clip["id"])
        res = track.track_player(fr, FPS, anchors)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, res)   # raw track cached; smoothing is cheap and tweakable
    return track.smooth_track(res, FPS)


def _camera_motion(cache: Path, cid: str, fr: np.ndarray) -> np.ndarray:
    f = cache / "track" / f"{cid}-cam.npy"
    if f.exists():
        m = np.load(f)
        if len(m) == len(fr):
            return m
    m = track.camera_motion(fr)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, m)
    return m


def _motion_box(cache: Path, clip: dict) -> np.ndarray:
    f = cache / "track" / f"{clip['id']}-motion.npy"
    if f.exists():
        return np.load(f)
    fr = _proxy_frames(cache, clip["id"])
    H, W = fr.shape[1:3]
    x0, y0, cw, ch = (int(round(v)) for v in content_rect(clip, W, H))
    fr = fr[:, y0:y0 + ch, x0:x0 + cw]
    small = np.stack([cv2.resize(x, (640, int(640 * ch / cw) // 2 * 2)) for x in fr])
    gray = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in small])
    box = motion.motion_signals(gray, small, track=True)["box"]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, box)
    return box


def plan(cache: Path, clip: dict) -> dict:
    """Everything the review page needs to preview the camera for one clip."""
    r = clip.get("review") or {}
    d = clip["detection"]
    snap = r.get("snap") if r.get("snap") is not None else d["snap"]
    end = r.get("end") if r.get("end") is not None else d["end"]
    level = float(r.get("zoom") or 0)
    W, H = _proxy_size(cache, clip["id"])
    content = content_rect(clip, W, H)
    if level <= 1 and not anchors_of(clip):
        # Nothing to plan: don't touch the frames at all (they're big).
        return {"fps": FPS, "mode": "off", "level": level, "rects": None, "player": None}
    player = player_track(cache, clip)
    rects, mode = None, "off"
    if level > 1:
        if player is not None:
            rects, mode = autozoom.plan_follow(player, W, H, content, snap, end, level), "player"
        else:
            # No spotlight: frame the busiest part of the play, gently.
            x0, y0, cw, ch = content
            zc = autozoom.ZoomConfig(max_zoom=min(level, 1.4), min_src_px=0,
                                     out_w=16, out_h=9)
            rr = autozoom.plan(_motion_box(cache, clip), cw, ch, snap, end, zc)
            rects, mode = rr + np.array([x0, y0, 0, 0]), "play"
    norm = np.array([W, H, W, H], np.float64)
    return {
        "fps": FPS, "mode": mode, "level": level,
        "rects": None if rects is None else np.round(rects / norm, 4).tolist(),
        "player": None if player is None else np.round(np.nan_to_num(player, nan=-1), 4).tolist(),
    }
