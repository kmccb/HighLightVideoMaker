"""Slow step: decode a clip once, extract all signals, cache to .npz.

Everything downstream (plays.py, score sweeps) runs off this cache.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from . import audio, crop as cropmod, frames, motion
from .probe import ClipInfo

CACHE_VERSION = 2


def clip_id(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return re.sub(r"[^a-z0-9]+", "-", "/".join(rel.parts).lower()).strip("-")


def classify_kind(info: ClipInfo, name: str, cuts: list[float], crop: dict | None,
                  min_cuts: int) -> str:
    if len(cuts) >= min_cuts or (cuts and info.duration > 40):
        return "pre_edited"
    if name.lower().startswith("screenrecording") or crop is not None:
        return "screen_recording"
    return "iphone"


def extract(path: Path, info: ClipInfo, cache_dir: Path, cid: str, fps: float = 30.0,
            width: int = 640, force: bool = False) -> dict:
    """Return dict with arrays + meta; cached on disk keyed by clip id + file size/mtime."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz, meta_path = cache_dir / f"{cid}.npz", cache_dir / f"{cid}.json"
    stamp = {"v": CACHE_VERSION, "size": path.stat().st_size, "mtime": path.stat().st_mtime,
             "fps": fps, "width": width}
    if not force and npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("stamp") == stamp:
            with np.load(npz) as z:
                return {"meta": meta, **{k: z[k] for k in z.files}}

    # Crop: only worth checking for screen-recording-like sources.
    crop = None
    looks_screen = (path.name.lower().startswith("screenrecording")
                    or info.height > info.width or info.rotation != 0)
    if looks_screen:
        samples = frames.sample_frames(path, info.width, info.height, 40, info.duration)
        crop = cropmod.detect_crop(samples, info.width, info.height)

    # One decode (the expensive part for 4K); gray derived for motion, colour kept for
    # the field mask and scene cuts.
    color = frames.read_frames(path, info.width, info.height, fps=fps, width=width, crop=crop, color=True)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in color]) if len(color) else \
        np.zeros((0, 1, 1), np.uint8)
    n = len(color)

    sig = motion.motion_signals(gray, color)
    sig["frame_diff"] = motion.frame_diff(gray)

    wt, ws = np.zeros(0), np.zeros(0)
    if info.has_audio:
        try:
            wt, ws = audio.whistle_score(audio.read_audio(path))
        except Exception:  # noqa: BLE001 - audio is optional evidence
            pass

    meta = {"stamp": stamp, "crop": crop, "n_frames": n, "fps": fps}
    np.savez_compressed(npz, whistle_t=wt, whistle_s=ws, **sig)
    meta_path.write_text(json.dumps(meta))
    return {"meta": meta, "whistle_t": wt, "whistle_s": ws, **sig}
