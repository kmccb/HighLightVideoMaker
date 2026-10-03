"""Auto-zoom: a virtual camera operator that frames the action inside the source frame.

1. Track where players are moving (camera-compensated, on-field) every frame.
2. Plan a crop path offline. Because we know the whole clip, smoothing is zero-phase,
   so the virtual camera anticipates rather than lags.
3. Wide (full frame) before the snap so you see the formation, ease in once the play
   develops, ease back out after the play.
4. Render from the full-resolution source: 4K -> 1080p gives up to 2x zoom with no
   loss of sharpness; lower-res sources get proportionally less zoom.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import frames, motion
from .probe import ClipInfo

FPS = 30.0


@dataclass
class ZoomConfig:
    out_w: int = 1920
    out_h: int = 1080
    max_zoom: float = 2.0         # never crop tighter than full / max_zoom
    min_src_px: int = 960         # don't crop narrower than this many source pixels (sharpness)
    margin_x: float = 1.6         # crop width = mover spread * margin
    margin_y: float = 1.9
    center_sigma_s: float = 0.6   # smoothing of where the camera points
    zoom_sigma_s: float = 1.2     # smoothing of how far in it is (slower = calmer)
    ease_in_s: float = 1.0        # wide -> tracked after the snap
    ease_out_s: float = 1.0       # tracked -> wide after the end
    strength: float = 1.0         # 0 = no zoom, 1 = full planned zoom
    min_mass_frac: float = 0.35   # ignore motion clusters weaker than this x the play's typical


def track(path: Path, info: ClipInfo, crop: dict | None) -> np.ndarray:
    """Per-frame mover box at 30 fps on the (cropped) display frame, normalised."""
    color = frames.read_frames(path, info.width, info.height, fps=FPS, width=640, crop=crop, color=True)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in color])
    return motion.motion_signals(gray, color, track=True)["box"]


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _fill_nan(v: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(v)
    if not ok.any():
        return v
    idx = np.arange(len(v))
    return np.interp(idx, idx[ok], v[ok])


def plan(box: np.ndarray, W: int, H: int, snap: float, end: float, zc: ZoomConfig) -> np.ndarray:
    """Crop rect per frame (x, y, w, h) in source px, fixed output aspect, inside the frame."""
    n = len(box)
    aspect = zc.out_w / zc.out_h
    full_w = min(W, H * aspect)                      # widest window with output aspect
    min_w = max(full_w / zc.max_zoom, min(zc.min_src_px, full_w))

    # Weak clusters (play winding down, only the sideline fidgeting) are not worth
    # following: drop them and let the camera hold its last good framing.
    box = box.copy()
    t_idx = np.arange(n) / FPS
    in_play = (t_idx >= snap) & (t_idx <= end) & ~np.isnan(box[:, 6])
    if in_play.any():
        weak = box[:, 6] < zc.min_mass_frac * np.median(box[in_play, 6])
        box[weak, :] = np.nan
    cx, cy = _fill_nan(box[:, 0]) * W, _fill_nan(box[:, 1]) * H
    spread_w = (_fill_nan(box[:, 3]) - _fill_nan(box[:, 2])) * W * zc.margin_x
    spread_h = (_fill_nan(box[:, 5]) - _fill_nan(box[:, 4])) * H * zc.margin_y * aspect
    want_w = np.clip(np.maximum(spread_w, spread_h), min_w, full_w)
    if np.isnan(cx).all():                           # nothing moved: stay wide
        cx, cy, want_w = np.full(n, W / 2), np.full(n, H / 2), np.full(n, full_w)

    cx = gaussian_filter1d(cx, zc.center_sigma_s * FPS, mode="nearest")
    cy = gaussian_filter1d(cy, zc.center_sigma_s * FPS, mode="nearest")
    # Zoom on log scale so zooming in and out feel equally paced.
    lw = gaussian_filter1d(np.log(want_w), zc.zoom_sigma_s * FPS, mode="nearest")

    # Gate: wide before the snap, tracked during the play, wide again after.
    t = np.arange(n) / FPS
    g = _smoothstep((t - snap) / zc.ease_in_s) * (1 - _smoothstep((t - end) / zc.ease_out_s))
    g *= zc.strength
    lw = np.log(full_w) + g * (lw - np.log(full_w))
    cx = W / 2 + g * (cx - W / 2)
    cy = H / 2 + g * (cy - H / 2)

    w = np.exp(lw)
    h = w / aspect
    x = np.clip(cx - w / 2, 0, W - w)
    y = np.clip(cy - h / 2, 0, H - h)
    return np.stack([x, y, w, h], axis=1)


def render(path: Path, info: ClipInfo, crop: dict | None, rects: np.ndarray, t0: float, t1: float,
           out_zoom: Path, out_compare: Path | None, zc: ZoomConfig,
           player: np.ndarray | None = None, style: str = "off", color: str = "#ffd23f") -> None:
    """Render [t0, t1] with the planned crop. Optional side-by-side comparison video.
    player: (N, 5) normalised full-frame track for the highlight (-1/NaN = unknown)."""
    W = crop["w"] if crop else info.width
    H = crop["h"] if crop else info.height
    ow, oh = zc.out_w, zc.out_h
    if H < oh:  # low-res source: don't upscale past the source
        oh = H - H % 2
        ow = int(round(oh * zc.out_w / zc.out_h / 2)) * 2

    vf = ([f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}"] if crop else []) + [f"fps={FPS}", "format=bgr24"]
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(path), "-t", f"{t1 - t0:.3f}",
                            "-an", "-vf", ",".join(vf), "-f", "rawvideo", "pipe:1"],
                           stdout=subprocess.PIPE)

    def encoder(out: Path, w: int, h: int):
        out.parent.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(FPS), "-i", "pipe:0", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(path),
             "-map", "0:v", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
             str(out)], stdin=subprocess.PIPE)

    enc = encoder(out_zoom, ow, oh)
    cw, ch = 960, 540
    cmp_ = encoder(out_compare, cw * 2, ch) if out_compare else None
    fbytes = W * H * 3
    i0 = int(round(t0 * FPS))
    k = 0
    try:
        while True:
            buf = dec.stdout.read(fbytes)
            if len(buf) < fbytes:
                break
            f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            x, y, w, h = rects[min(i0 + k, len(rects) - 1)]
            xi, yi = int(x), int(y)
            wi, hi = min(int(np.ceil(w)), W - xi), min(int(np.ceil(h)), H - yi)
            zoomed = cv2.resize(f[yi:yi + hi, xi:xi + wi], (ow, oh), interpolation=cv2.INTER_AREA)
            if player is not None and style != "off":
                p = player[min(i0 + k, len(player) - 1)]
                if p[0] >= 0 and not np.isnan(p[0]):
                    s = ow / w   # source px -> output px
                    ox, oy = crop["x"] if crop else 0, crop["y"] if crop else 0
                    pcx = (p[0] * info.width - ox - x) * s
                    pcy = (p[1] * info.height - oy - y) * s
                    a = float(np.clip((p[4] - 0.2) / 0.3, 0, 1))
                    draw_highlight(zoomed, style, pcx, pcy, p[2] * info.width * s, p[3] * info.height * s,
                                   a, color, (i0 + k) / FPS)
            enc.stdin.write(zoomed.tobytes())
            if cmp_:
                left = cv2.resize(f, (cw, ch), interpolation=cv2.INTER_AREA)
                s = cw / W
                cv2.rectangle(left, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)),
                              (60, 220, 255), 2)
                cv2.putText(left, "original + auto-zoom frame", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                right = cv2.resize(zoomed, (cw, ch), interpolation=cv2.INTER_AREA)
                cv2.putText(right, f"auto-zoom {W / w:.2f}x", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cmp_.stdin.write(np.hstack([left, right]).tobytes())
            k += 1
    finally:
        dec.stdout.close()
        dec.wait()
        for p in (enc, cmp_):
            if p:
                p.stdin.close()
                p.wait()


def plan_follow(player: np.ndarray, W: int, H: int, content: tuple, snap: float, end: float,
                level: float, fps: float = FPS) -> np.ndarray:
    """Camera that follows one tracked player (from spotlight clicks).

    player: (N, 5) normalised cx, cy, w, h, conf. content: (x0, y0, cw, ch) px - the real
    video area (screen recordings have app chrome around it). Zoom eases in after the snap
    and out after the end, and backs off to wide wherever tracking confidence drops, so a
    lost player means a wider shot rather than a shot of the wrong thing.
    Returns (N, 4) crop rects x, y, w, h in px.
    """
    n = len(player)
    x0, y0, cw, ch = content
    aspect = 16 / 9
    fw = min(cw, ch * aspect)
    t = np.arange(n) / fps

    conf = np.nan_to_num(player[:, 4], nan=0.0)
    trust = np.clip((gaussian_filter1d(conf, 0.5 * fps, mode="nearest") - 0.2) / 0.3, 0, 1)
    px = _fill_nan(player[:, 0] * W)
    py = _fill_nan(player[:, 1] * H)
    if np.isnan(px).all():
        px, py = np.full(n, x0 + cw / 2), np.full(n, y0 + ch / 2)
    # Frame the player a little above centre-of-box (players are tall; keep the field ahead).
    px = gaussian_filter1d(px, 0.3 * fps, mode="nearest")
    py = gaussian_filter1d(py, 0.3 * fps, mode="nearest")

    # Start partly zoomed on the player and push in over the second before the snap,
    # hold through the play, ease back to the partial level after it.
    base = 0.4
    ramp = _smoothstep((t - (snap - 1.0)) / 1.0) * (1 - _smoothstep((t - end) / 1.0))
    g = base + (1 - base) * ramp
    lz = gaussian_filter1d(np.log(1 + (level - 1) * g * trust), 0.4 * fps, mode="nearest")
    w = fw / np.exp(lz)
    h = w / aspect
    x = np.clip(px - w / 2, x0, x0 + cw - w)
    y = np.clip(py - h / 2, y0, y0 + ch - h)
    return np.stack([x, y, w, h], axis=1)


def _hex_bgr(c: str) -> tuple:
    c = c.lstrip("#")
    return (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))


def draw_highlight(img: np.ndarray, style: str, cx: float, cy: float, bw: float, bh: float,
                   alpha: float, color: str, t: float) -> None:
    """Draw the player highlight in place (output px). Mirrors the review page's styles."""
    if style == "off" or alpha <= 0:
        return
    H, W = img.shape[:2]
    k = W / 960                                   # minimum sizes scale with output size
    col = np.array(_hex_bgr(color), np.float32)
    feet = cy + bh / 2
    if style == "shadow":
        # Dim everything except a soft elliptical pool of light around the player.
        R = max(bh * 1.0, 28 * k)
        yy, xx = np.ogrid[:H, :W]
        d = np.sqrt(((xx - cx) / (0.8 * R)) ** 2 + ((yy - cy) / R) ** 2)
        lit = np.clip((1.0 - d) / 0.45, 0, 1)      # 1 inside, soft edge, 0 outside
        dim = 1.0 - 0.5 * alpha * (1.0 - lit)
        img[:] = (img.astype(np.float32) * dim[..., None]).astype(np.uint8)
    if style in ("ring", "shadow"):
        # Sized from height (arms make width jumpy); feet sit in the back half of the ring.
        rx = max(bh * 0.38, 12 * k)
        ry = max(rx * 0.3, 4 * k)
        feet -= ry * 0.25
        layer = np.zeros_like(img)
        thick = max(2, int(round((2 if style == "shadow" else 3) * k)))
        cv2.ellipse(layer, (int(cx), int(feet)), (int(rx), int(ry)), 0, 0, 360,
                    tuple(int(v) for v in col), thick, cv2.LINE_AA)
        glow = cv2.GaussianBlur(layer, (0, 0), 4 * k)
        fill = np.zeros_like(img)
        cv2.ellipse(fill, (int(cx), int(feet)), (int(rx), int(ry)), 0, 0, 360,
                    tuple(int(v) for v in col), -1, cv2.LINE_AA)
        out = img.astype(np.float32)
        out += alpha * 0.18 * fill.astype(np.float32)
        out += alpha * 0.9 * glow.astype(np.float32)
        m = layer.any(axis=2)
        out[m] = (1 - alpha) * out[m] + alpha * layer[m].astype(np.float32)
        img[:] = np.clip(out, 0, 255).astype(np.uint8)
    elif style == "arrow":
        s = max(bh * 0.28, 11 * k)
        tip_y = cy - bh / 2 - 6 * k + np.sin(t * 5) * 2.5 * k
        pts = np.array([[cx, tip_y], [cx - s / 2, tip_y - s * 0.9], [cx, tip_y - s * 0.62],
                        [cx + s / 2, tip_y - s * 0.9]], np.float32)
        layer = img.copy()
        shadow = img.copy()
        cv2.fillPoly(shadow, [(pts + [1.5 * k, 2 * k]).astype(np.int32)], (0, 0, 0), cv2.LINE_AA)
        cv2.fillPoly(layer, [pts.astype(np.int32)], tuple(int(v) for v in col), cv2.LINE_AA)
        cv2.polylines(layer, [pts.astype(np.int32)], True, (20, 20, 20), max(1, int(k)), cv2.LINE_AA)
        img[:] = cv2.addWeighted(img, 1 - 0.5 * alpha, shadow, 0.5 * alpha, 0)
        m = np.any(layer != img, axis=2)
        img[m] = ((1 - alpha) * img[m] + alpha * layer[m]).astype(np.uint8)
