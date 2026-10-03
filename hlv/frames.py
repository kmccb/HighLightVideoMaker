"""Decode clips to small constant-frame-rate grayscale/colour arrays via an ffmpeg pipe.

The `fps` filter resamples the (possibly VFR) source onto a constant grid, so frame i
sits at i / fps seconds on the source timeline. ffmpeg auto-applies rotation metadata,
so frames (and crop boxes) are in display orientation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def _filters(fps: float, width: int, crop: dict | None, pix: str) -> str:
    vf = []
    if crop:
        vf.append(f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}")
    vf.append(f"fps={fps}")
    vf.append(f"scale={width}:-2:flags=area")
    vf.append(f"format={pix}")
    return ",".join(vf)


def _out_size(src_w: int, src_h: int, width: int, crop: dict | None) -> tuple[int, int]:
    if crop:
        src_w, src_h = crop["w"], crop["h"]
    h = int(round(src_h * width / src_w / 2)) * 2
    return width, h


def read_frames(path: Path, src_w: int, src_h: int, fps: float = 30.0, width: int = 640,
                crop: dict | None = None, color: bool = False) -> np.ndarray:
    """Return frames as uint8 array (N, H, W) gray or (N, H, W, 3) BGR."""
    pix = "bgr24" if color else "gray"
    w, h = _out_size(src_w, src_h, width, crop)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an",
           "-vf", _filters(fps, width, crop, pix), "-f", "rawvideo", "pipe:1"]
    ch = 3 if color else 1
    frame_bytes = w * h * ch
    # Decode straight into one pre-sized array: no raw-bytes copy, no concatenation.
    # (A long 960px colour clip is >1 GB; holding two copies per worker ran the machine out.)
    shape = (h, w, 3) if color else (h, w)
    dur = _duration(path)
    cap = int(np.ceil(dur * fps)) + 8 if dur else 256
    arr = np.empty((cap,) + shape, np.uint8)
    n = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        while True:
            if n == len(arr):   # longer than expected: grow (rare)
                arr = np.concatenate([arr, np.empty((max(64, n // 4),) + shape, np.uint8)])
            got = proc.stdout.readinto(memoryview(arr[n]).cast("B"))
            if not got or got < frame_bytes:
                break
            n += 1
    if proc.returncode not in (0, None):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return arr[:n]


def _duration(path: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(path)], capture_output=True, text=True, check=True).stdout
        return float(out.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def sample_frames(path: Path, src_w: int, src_h: int, n: int, duration: float,
                  width: int = 320) -> np.ndarray:
    """~n evenly spaced gray frames at low res (for crop detection)."""
    fps = max(n / max(duration, 0.1), 0.2)
    return read_frames(path, src_w, src_h, fps=round(fps, 3), width=width)
