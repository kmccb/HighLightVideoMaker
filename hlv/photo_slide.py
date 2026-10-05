"""Turn a photo into a reel slide: a slow pan-and-zoom ("Ken Burns") over a blurred fill.

The whole photo stays visible in the middle (tall iPhone photos too); a blurred, darkened
copy fills the rest of the 16:9 frame. The photo grows (or shrinks) 10% over the slide,
about a point a little above its middle; alternate slides zoom in and out.
Slides are cached in .hlv_cache/photos by a hash of everything that changes them.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import graphics
from .photos import open_image
from .pipeline import CACHE_DIRNAME

PHOTO_VERSION = 1
FPS = 30                 # same as segment.FPS (not imported: segment pulls in the tracker)
W, H = graphics.W, graphics.H
MAX_ZOOM = 1.10
FOCUS_Y = 0.4            # zoom centre, as a fraction of the photo height
THUMB = 160


def slide_length(photo_s: float) -> float:
    """Rendered length: enough for the longest beat-fitted length (1.25x) plus a little."""
    return round(photo_s * 1.25 + 0.2, 3)


def _cache(root: Path) -> Path:
    d = Path(root) / CACHE_DIRNAME / "photos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _src_key(root: Path, photo: dict, *extra) -> str:
    st = (Path(root) / photo["source"]).stat()
    return hashlib.sha1(json.dumps([st.st_size, st.st_mtime, *extra]).encode()).hexdigest()[:12]


def _background(img: np.ndarray, W: int = W, H: int = H) -> np.ndarray:
    h, w = img.shape[:2]
    k = max(W / w, H / h)
    big = cv2.resize(img, (int(np.ceil(w * k)), int(np.ceil(h * k))), interpolation=cv2.INTER_AREA)
    y0, x0 = (big.shape[0] - H) // 2, (big.shape[1] - W) // 2
    bg = cv2.GaussianBlur(big[y0:y0 + H, x0:x0 + W], (0, 0), 30)
    return (bg.astype(np.float32) * 0.45).astype(np.uint8)


def render(root: Path, photo: dict, seconds: float, zoom_out: bool = False,
           size: tuple[int, int] | None = None) -> Path:
    """Render (or reuse) the slide for this photo, at size (default 1920x1080)."""
    root = Path(root)
    W, H = size or (graphics.W, graphics.H)
    key = _src_key(root, photo, seconds, W, H, zoom_out, PHOTO_VERSION)
    out = _cache(root) / f"{photo['id']}-{key}.mp4"
    if out.exists():
        return out
    img = cv2.cvtColor(np.asarray(open_image(root / photo["source"])), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    fit = min(W / w, H / h)
    fg = cv2.resize(img, (max(1, round(w * fit * MAX_ZOOM)), max(1, round(h * fit * MAX_ZOOM))),
                    interpolation=cv2.INTER_AREA if fit * MAX_ZOOM < 1 else cv2.INTER_CUBIC)
    bg = _background(img, W, H)
    off = np.array([(W - w * fit) / 2, (H - h * fit) / 2])
    # Zoom about a point a little above the photo's middle: where faces and numbers
    # usually are in sports photos, so they stay in frame.
    focus = off + np.array([w * fit / 2, h * fit * FOCUS_Y])

    n = max(2, int(round(seconds * FPS)))
    tmp = out.with_suffix(".tmp.mp4")
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "pipe:0", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{n / FPS:.4f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(tmp)],
        stdin=subprocess.PIPE)
    try:
        for k in range(n):
            u = k / (n - 1)
            z = 1 + (MAX_ZOOM - 1) * (1 - u if zoom_out else u)      # steady motion, no easing
            s = z / MAX_ZOOM
            tx, ty = focus + z * (off - focus)
            m = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float64)
            frame = cv2.warpAffine(fg, m, (W, H), dst=bg.copy(), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_TRANSPARENT)
            enc.stdin.write(frame.tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed rendering {photo['source']}")
    tmp.replace(out)
    return out


def thumb(root: Path, photo: dict) -> Path:
    """Small JPEG for the Build panel list."""
    root = Path(root)
    d = _cache(root) / "thumbs"
    d.mkdir(exist_ok=True)
    out = d / f"{photo['id']}-{_src_key(root, photo, THUMB)}.jpg"
    if not out.exists():
        im = open_image(root / photo["source"])
        im.thumbnail((THUMB, THUMB), Image.LANCZOS)
        im.save(out, "JPEG", quality=85)
    return out
