"""Animated intro: the player's name card, slammed in on the beat.

Timeline (each slam lands on a strong beat of the song when there's one):
  T1  giant jersey number smashes in (motion blur, shake, flash, shockwave, sparks), then
      grows into a faint outline behind everything
  T2  first name flies in from the left, last name from the right; second impact with a
      quick colour-split glitch
  T3  team-colour bars wipe in, details (position / class / size) and school slide up
  T4  logo pops with a bounce and glow; a light sheen sweeps across the name
  end white flash into the first play

Background: a blurred, darkened action frame of the player (first spotlighted play) slowly
pushing in, with embers, light streaks, grain and a vignette. Impact sounds (whoosh + boom)
are synthesised and returned with the video.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import wave
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import encoder, graphics
from .graphics import _detail_line, _rgb, font

W, H = graphics.W, graphics.H     # frame size; frame_size() switches it (portrait intros)
FPS = 30
SR = 48000
INTRO_VERSION = 2
HOLD_EXTRA = 1.5        # seconds the finished name card stays up before the flash into the first play
_size_lock = threading.RLock()


@contextmanager
def frame_size(w: int, h: int):
    """Render the intros at w x h (e.g. 1080x1920 for the vertical reel). The styles read the
    module-level W/H (and the quarter-size QW/QH), so set them in every intro module for the
    duration, one render at a time."""
    from . import intro_fx, intro_fx2
    mods = [sys.modules[__name__], intro_fx, intro_fx2]
    new = {"W": w, "H": h, "QW": w // 4, "QH": h // 4}
    with _size_lock:
        old = [(m, {k: getattr(m, k) for k in new if hasattr(m, k)}) for m in mods]
        for m, keys in old:
            for k in keys:
                setattr(m, k, new[k])
        try:
            yield
        finally:
            for m, keys in old:
                for k, v in keys.items():
                    setattr(m, k, v)


def portrait() -> bool:
    return H > W


def layout(has_logo: bool, last_size: int) -> dict:
    """Where the logo, number and name block go. Wide: logo left, name block right (centred
    when there's no logo). Portrait: logo on top, number and name block stacked under it."""
    if portrait():
        tx = W // 2
        logo_c = (W // 2, int(H * 0.22))
        y_first = int(H * 0.46) - (0 if has_logo else int(H * 0.08))
        num_y = y_first + 120
    else:
        tx = 1180 if has_logo else W // 2
        logo_c = (470, 520)
        y_first = 300
        num_y = 500
    y_last = y_first + 75 + last_size // 2 + 5
    y_bar = y_last + last_size // 2 + 30
    return {"tx": tx, "logo_c": logo_c, "y_first": y_first, "y_last": y_last, "y_bar": y_bar,
            "y_det": y_bar + 70, "y_school": y_bar + 140, "num_c": (tx, num_y)}


def fit(layer: np.ndarray | None, max_w: float | None = None) -> np.ndarray | None:
    """Shrink a text layer that's wider than the frame allows (long names in portrait)."""
    if layer is None:
        return None
    max_w = max_w or W - 90
    w = layer.shape[1]
    if w <= max_w:
        return layer
    k = max_w / w
    return cv2.resize(layer, (int(max_w), max(1, int(layer.shape[0] * k))), interpolation=cv2.INTER_AREA)


def _cover(img: np.ndarray) -> np.ndarray:
    """Crop the middle of an image to the frame's shape, then scale to the frame (no squashing)."""
    h, w = img.shape[:2]
    if w / h > W / H:
        cw = int(round(h * W / H))
        img = img[:, (w - cw) // 2:(w - cw) // 2 + cw]
    else:
        ch = int(round(w * H / W))
        img = img[(h - ch) // 2:(h - ch) // 2 + ch]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- timing
BURST_PAD = 0.35        # with a burst cut, render this far past the hit so the style's own flash-out falls after it


def beat_times(grid: dict | None, music_start: float) -> tuple[list[float], float]:
    """Slam times T1..T4 and total length to render. With grid["hits"] (the song's heavy hits, see
    beats.heavy_hits) the intro can follow the big hits (grid["sync"] == "hits") and end on one;
    with grid["burst"] it ends on a hit that blows apart into the first play, and renders
    BURST_PAD past it (cut_point() is where the reel cuts)."""
    T, end = _times(grid, music_start)
    pad = BURST_PAD if grid and grid.get("burst") else 0.0
    return T, round((end + pad) * FPS) / FPS


def cut_point(grid: dict | None, music_start: float) -> float:
    """Where the reel cuts from the intro to the first play (the end, less any burst pad)."""
    return round(_times(grid, music_start)[1] * FPS) / FPS


def _times(grid: dict | None, music_start: float) -> tuple[list[float], float]:
    T, end = _grid_times(grid, music_start)
    hits = [(t - music_start, s) for t, s in (grid or {}).get("hits") or []]
    if not hits or not (grid.get("sync") == "hits" or grid.get("burst")):
        return T, end
    beats = np.array(grid.get("beats") or []) - music_start
    bs = grid.get("strength") or [0.5] * len(beats)
    pool = [h for h in hits + [(float(t), 0.55 * float(s)) for t, s in zip(beats, bs)] if 0.45 <= h[0] <= 9.5]
    if grid.get("sync") != "hits":
        # steady-beat slams; only the end moves onto the biggest hit a little after the logo lands
        cand = [h for h in pool if T[3] + 0.9 <= h[0] <= T[3] + 3.5]
        if cand:
            end = max(cand, key=lambda h: h[1] - 0.05 * abs(h[0] - (T[3] + 1.6)))[0]
        return T, round(end * FPS) / FPS
    endc = [h for h in pool if 4.2 <= h[0] <= 9.0]
    if not endc:
        return T, end
    end = max(endc, key=lambda h: h[1] - 0.02 * h[0])[0]
    early = [h for h in pool if h[0] <= 2.3]
    t1 = max(early, key=lambda h: h[1])[0] if early else min(T[0], end - 3.0)
    picks = [t1]
    for t, _ in sorted((h for h in pool if t1 + 0.5 <= h[0] <= end - 0.9), key=lambda h: -h[1]):
        if all(abs(t - p) >= 0.5 for p in picks):
            picks.append(t)
        if len(picks) == 4:
            break
    picks.sort()
    while len(picks) < 4:                                   # not enough hits: spread the rest evenly
        gaps = np.diff(picks + [end - 0.9])
        i = int(np.argmax(gaps))
        picks.insert(i + 1, picks[i] + gaps[i] / 2)
    return [round(t * FPS) / FPS for t in picks], round(end * FPS) / FPS


def _grid_times(grid: dict | None, music_start: float) -> tuple[list[float], float]:
    """Slam times on the song's steady beat grid (or fixed ones without a song)."""
    if not grid or not grid.get("beats"):
        return [0.5, 1.3, 2.1, 2.6], 4.2 + HOLD_EXTRA
    b = np.array(grid["beats"]) - music_start
    s = np.array(grid.get("strength") or [1.0] * len(b))
    ok = b >= 0.45
    if not ok.any():
        return [0.5, 1.3, 2.1, 2.6], 4.2 + HOLD_EXTRA
    period = float(np.median(np.diff(b))) if len(b) > 1 else 0.5
    i0 = int(np.argmax(ok))
    # first reasonably strong beat within ~1 s, else the first beat
    for i in range(i0, min(len(b), i0 + max(2, int(1.0 / max(period, 0.2))))):
        if s[i] >= 0.5:
            i0 = i
            break
    g = max(1, int(round(1.0 / max(period, 0.2))))      # beats per ~1 s
    idx = [i0, i0 + g, i0 + 2 * g, i0 + 2 * g + max(1, g // 2)]
    idx = [min(i, len(b) - 1) for i in idx]
    ts = [float(b[i]) for i in idx]
    for k in range(1, 4):                               # guard short / odd grids
        if ts[k] <= ts[k - 1] + 0.3:
            ts[k] = ts[k - 1] + 0.6
    end = ts[3] + max(1.3, 2.5 * period) + HOLD_EXTRA
    return [round(t * FPS) / FPS for t in ts], round(min(end, 8.5) * FPS) / FPS


# ---------------------------------------------------------------- easing
def _clamp(u):
    return float(np.clip(u, 0.0, 1.0))


def ease_out_back(u, k=1.70158):
    u = _clamp(u) - 1
    return 1 + (k + 1) * u ** 3 + k * u ** 2


def ease_in(u):
    u = _clamp(u)
    return u * u * u


def ease_out(u):
    u = _clamp(u)
    return 1 - (1 - u) ** 3


def smooth(u):
    u = _clamp(u)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- layers
def _text_layer(text: str, size: int, fill, stroke: int = 0, stroke_fill=(0, 0, 0), style="Bold Condensed",
                spacing: int = 0) -> np.ndarray:
    """Tightly cropped RGBA (uint8) image of a text line."""
    f = font(size, style)
    pad = stroke + 8
    if spacing:
        widths = [f.getlength(ch) for ch in text]
        tw = int(sum(widths) + spacing * max(0, len(text) - 1))
    else:
        tw = int(f.getlength(text))
    asc, desc = f.getmetrics()
    img = Image.new("RGBA", (tw + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    if spacing:
        x = pad
        for ch, wch in zip(text, widths):
            dr.text((x, pad), ch, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
            x += wch + spacing
    else:
        dr.text((pad, pad), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    bbox = img.getbbox() or (0, 0, 1, 1)
    return np.array(img.crop(bbox))[:, :, [2, 1, 0, 3]].copy()   # RGBA -> BGRA


def _outline(layer: np.ndarray, color, width: int = 6) -> np.ndarray:
    """Outline-only version of a layer (for the ghost number)."""
    a = layer[:, :, 3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    ring = cv2.subtract(cv2.dilate(a, k), a)
    out = np.zeros_like(layer)
    out[:, :, :3] = color
    out[:, :, 3] = ring
    return out


def _glow(layer: np.ndarray, color, sigma: float) -> np.ndarray:
    pad = int(sigma * 3)
    a = cv2.copyMakeBorder(layer[:, :, 3], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    a = cv2.GaussianBlur(a, (0, 0), sigma)
    out = np.zeros(a.shape + (4,), np.uint8)
    out[:, :, :3] = color
    out[:, :, 3] = a
    return out


def _fx_color(pl: dict) -> tuple:
    """Effect colour (BGR): the accent, unless it's too pale to read (then a bright team colour)."""
    acc = np.array(_rgb(pl.get("accent"), (255, 210, 63)), np.float32)
    if 0.299 * acc[0] + 0.587 * acc[1] + 0.114 * acc[2] < 200:
        return tuple(int(v) for v in acc[::-1])
    team = np.array(_rgb(pl.get("color")), np.float32)
    bright = team / max(team.max(), 1) * 255            # full-brightness team hue
    return tuple(int(v) for v in (0.75 * bright + 0.25 * 255)[::-1])


def _shadow(layer: np.ndarray, sigma: float = 9) -> np.ndarray:
    return _glow(layer, (0, 0, 0), sigma)


def _paste(dst: np.ndarray, rgba: np.ndarray, cx: float, cy: float, scale: float = 1.0, alpha: float = 1.0,
           add: bool = False) -> None:
    """Alpha-blend (or add) a BGRA uint8 layer centred at (cx, cy) onto a float32 BGR frame."""
    if alpha <= 0.003 or scale <= 0.01:
        return
    h, w = rgba.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = rgba if (nw, nh) == (w, h) else cv2.resize(
        rgba, (nw, nh), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    dx0, dy0, dx1, dy1 = max(0, x0), max(0, y0), min(W, x0 + nw), min(H, y0 + nh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    src = img[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0].astype(np.float32)
    a = src[:, :, 3:4] * (alpha / 255.0)
    roi = dst[dy0:dy1, dx0:dx1]
    if add:
        roi += src[:, :, :3] * a
    else:
        roi *= (1 - a)
        roi += src[:, :, :3] * a


def _paste_blur(dst, rgba, p0, p1, s0, s1, alpha, n=5):
    """Motion blur: average n positions/scales along the move."""
    acc = np.zeros_like(dst)
    for k in range(n):
        u = k / (n - 1)
        tmp = dst.copy()
        _paste(tmp, rgba, p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u, s0 + (s1 - s0) * u, alpha)
        acc += tmp
    dst[:] = acc / n


# ---------------------------------------------------------------- effects
class Sparks:
    """Burst of sparks from a point: streaks with gravity and drag, white fading to accent."""

    def __init__(self, rng, x, y, t0, color, n=80, spread=(500, 1700), cone=None):
        self.t0, self.color = t0, np.array(color, np.float32)
        ang = rng.uniform(0, 2 * np.pi, n) if cone is None else rng.uniform(cone[0], cone[1], n)
        spd = rng.uniform(*spread, n)
        self.p0 = np.array([x, y], np.float32) + rng.normal(0, 18, (n, 2))
        self.v = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1)
        self.life = rng.uniform(0.35, 0.8, n)

    def draw(self, layer, t):
        dt = t - self.t0
        if dt < 0 or dt > 0.9:
            return
        drag = np.exp(-2.6 * dt)
        pos = self.p0 + self.v * (1 - drag) / 2.6 + np.array([0, 520]) * dt * dt
        prev_dt = max(0.0, dt - 0.035)
        dragp = np.exp(-2.6 * prev_dt)
        prv = self.p0 + self.v * (1 - dragp) / 2.6 + np.array([0, 520]) * prev_dt * prev_dt
        for i in range(len(pos)):
            u = dt / self.life[i]
            if u >= 1:
                continue
            c = (1 - u) * np.array([255, 255, 255]) + u * self.color
            cv2.line(layer, tuple(int(v) for v in prv[i]), tuple(int(v) for v in pos[i]),
                     tuple(float(v) * (1 - u) for v in c), 3 if u < 0.4 else 2, cv2.LINE_AA)


def _shake(t, hits, amp=26.0):
    """Damped camera shake (dx, dy, zoom) after each impact."""
    dx = dy = 0.0
    for t0, a in hits:
        d = t - t0
        if 0 <= d < 0.45:
            e = np.exp(-d / 0.11) * amp * a
            dx += e * np.sin(2 * np.pi * 23 * d)
            dy += e * np.cos(2 * np.pi * 19 * d + 1.3)
    return dx, dy


def _vignette() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(1.15 - 0.55 * r ** 1.6, 0.25, 1.0)[..., None]


# ---------------------------------------------------------------- background
def background_frame(src_bgr: np.ndarray | None, pl: dict) -> np.ndarray:
    """Dark, blurred, team-tinted base (float32 BGR, a bit larger than the frame for push-in)."""
    team = np.array(_rgb(pl.get("color"))[::-1], np.float32)
    if src_bgr is None:
        base = _cover(np.array(graphics._background(pl).convert("RGB"))[:, :, ::-1]).astype(np.float32)
    else:
        f = _cover(src_bgr).astype(np.float32)
        f = cv2.GaussianBlur(f, (0, 0), 7)
        gray = f.mean(axis=2, keepdims=True)
        f = 0.55 * f + 0.45 * gray                     # desaturate
        f = f * 0.30 + team * 0.10                     # darken + team tint
        base = f
    # team-colour light from the lower left, deep shadow top right
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.clip(1.0 - (xx * 0.75 + (H - yy) * 0.6) / (W * 0.9), 0, 1)[..., None] ** 1.5
    return base * (0.75 + 0.25 * (1 - d)) + team * 0.55 * d


# ---------------------------------------------------------------- audio
def _impact_sfx(t_hits: list[tuple[float, float]], duration: float, rng) -> np.ndarray:
    """Whoosh into each slam + boom on impact. Returns (n, 2) float32 at 48 kHz."""
    from scipy.signal import butter, lfilter
    n = int(duration * SR) + 1
    out = np.zeros(n, np.float32)
    b_hp, a_hp = butter(2, 1500 / (SR / 2), "high")
    b_bp, a_bp = butter(2, [300 / (SR / 2), 5000 / (SR / 2)], "band")
    for t0, strength in t_hits:
        i0 = int(t0 * SR)
        # whoosh: band-passed noise swelling over 0.35 s, cut off at the impact
        L = int(0.35 * SR)
        s = max(0, i0 - L)
        if i0 > s:
            env = np.linspace(0, 1, i0 - s) ** 2.2
            out[s:i0] += 0.35 * strength * lfilter(b_bp, a_bp, rng.standard_normal(i0 - s)).astype(np.float32) * env
        # boom: pitch-dropping sine + sub, long decay
        L = int(1.1 * SR)
        e = min(n, i0 + L)
        tt = np.arange(e - i0) / SR
        f = 35 + 60 * np.exp(-tt / 0.07)
        boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.32)
        sub = np.sin(2 * np.pi * 42 * tt) * np.exp(-tt / 0.5) * 0.5
        # crack: very short bright noise burst
        cr = np.zeros_like(tt)
        m = int(0.025 * SR)
        cr[:m] = lfilter(b_hp, a_hp, rng.standard_normal(m)) * np.exp(-np.arange(m) / (0.006 * SR))
        # tail: decaying filtered noise (a cheap room)
        tail = lfilter(b_bp, a_bp, rng.standard_normal(len(tt))) * np.exp(-tt / 0.28) * 0.12
        out[i0:e] += strength * (0.95 * boom + sub + 0.6 * cr + tail).astype(np.float32)
    peak = float(np.abs(out).max()) or 1.0
    out = np.tanh(out / peak * 1.6) * 0.89           # soft-clip for punch
    return np.stack([out, out], 1)


# ---------------------------------------------------------------- render
def render(out: Path, lib_root: Path, pl: dict, grid: dict | None, music_start: float,
           bg_src: np.ndarray | None, sfx: bool = True, seed: int = 7) -> tuple[Path, float]:
    """Render the animated intro to out (mp4 with impact sounds). Returns (path, duration)."""
    rng = np.random.default_rng(seed)
    T, dur = beat_times(grid, music_start)
    T1, T2, T3, T4 = T
    n_frames = int(round(dur * FPS))
    accent = _rgb(pl.get("accent"), (255, 210, 63))[::-1]          # BGR
    team = _rgb(pl.get("color"))[::-1]
    fxc = _fx_color(pl)
    white = (255, 255, 255)

    # ---- layers
    num_txt = f"#{pl['number']}" if pl.get("number") else ""
    parts = (pl.get("name") or "").upper().split()
    first, last = (" ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else ("", parts[0] if parts else "")
    L_num = fit(_text_layer(num_txt, 600, accent[::-1] + (255,), stroke=16, stroke_fill=team[::-1]), W - 60) if num_txt else None
    L_num_sh = _shadow(L_num, 12) if L_num is not None else None
    L_num_ghost = _outline(L_num, fxc, 5) if L_num is not None else None
    L_num_glow = _glow(L_num, fxc, 30) if L_num is not None else None
    last_size = 300 if len(last) <= 8 else 230
    L_first = fit(_text_layer(first, 130, white + (255,), spacing=16)) if first else None
    L_last = fit(_text_layer(last, last_size, white + (255,), stroke=4, stroke_fill=team[::-1])) if last else None
    L_first_sh = _shadow(L_first, 7) if L_first is not None else None
    L_last_sh = _shadow(L_last, 10) if L_last is not None else None
    L_last_glow = _glow(L_last, fxc, 22) if L_last is not None else None
    det = _detail_line(pl)
    L_det = fit(_text_layer(det, 54, (225, 230, 238, 255), style="SemiBold Condensed", spacing=2)) if det else None
    L_school = fit(_text_layer((pl.get("school") or "").upper(), 54, accent[::-1] + (255,),
                               style="SemiBold Condensed", spacing=3)) if pl.get("school") else None
    lg = graphics._logo(lib_root, pl, 380)
    L_logo = np.array(lg)[:, :, [2, 1, 0, 3]].copy() if lg is not None else None
    L_logo_glow = _glow(L_logo, fxc, 26) if L_logo is not None else None

    # ---- layout: logo left, text block right (centred when there's no logo); stacked in portrait
    lay = layout(L_logo is not None, last_size)
    tx, logo_c, num_c = lay["tx"], lay["logo_c"], lay["num_c"]
    y_first, y_last, y_bar, y_det, y_school = (lay[k] for k in ("y_first", "y_last", "y_bar", "y_det", "y_school"))

    base = background_frame(bg_src, pl)
    vig = _vignette()
    # light streaks: one blurred diagonal band, rendered once at half size and slid across
    hw, hh = W // 2, H // 2
    band = np.zeros((hh, hw + 700), np.float32)
    cv2.fillConvexPoly(band, np.array([[350, 0], [420, 0], [170, hh], [100, hh]], np.int32), 1.0)
    band = cv2.GaussianBlur(band, (0, 0), 12)
    grain = [rng.normal(0, 4.5, (H // 2, W // 2, 1)).astype(np.float32) for _ in range(6)]
    embers = np.stack([rng.uniform(0, W, 50), rng.uniform(0, H, 50), rng.uniform(20, 70, 50),
                       rng.uniform(1.5, 3.5, 50), rng.uniform(0, 6.28, 50)], 1)
    sparks = []
    if L_num is not None:
        sparks.append(Sparks(rng, num_c[0], num_c[1], T1, fxc, n=90))
    sparks.append(Sparks(rng, tx, y_last, T2, fxc, n=70, spread=(600, 1900), cone=(-0.5, 0.5)))
    sparks.append(Sparks(rng, tx, y_last, T2, fxc, n=70, spread=(600, 1900), cone=(np.pi - 0.5, np.pi + 0.5)))
    hits = [(T1, 1.0), (T2, 0.8), (T3, 0.25), (T4, 0.35)]

    # ---- encoder (+ audio)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav")
    audio = _impact_sfx([(T1 - 0.0, 1.0), (T2, 0.85), (T3, 0.35), (T4, 0.5)], dur, rng) if sfx \
        else np.zeros((int(dur * SR) + 1, 2), np.float32)
    with wave.open(str(wav), "wb") as wv:
        wv.setnchannels(2)
        wv.setsampwidth(2)
        wv.setframerate(SR)
        wv.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "pipe:0", "-i", str(wav), "-map", "0:v", "-map", "1:a", *encoder.h264(17),
         "-c:a", "aac", "-b:a", "192k", "-shortest",
         "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    try:
        for k in range(n_frames):
            t = k / FPS
            # background: fade up, slow push-in
            z = 1.0 + 0.10 * t / dur
            m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
            frame = cv2.warpAffine(base, m, (W, H), borderMode=cv2.BORDER_REFLECT)
            frame *= min(1.0, 0.25 + t / 0.4) if t < T1 else 1.0
            # light streaks + embers, composed at half size
            small = np.zeros((hh, hw), np.float32)
            for spd, off, gain in ((260, 0, 1.0), (190, 450, 0.6), (320, 850, 0.4)):
                xpos = int((off + spd * t) % (hw + 700))
                x0 = xpos - 700
                a0, a1 = max(0, x0), min(hw, x0 + hw + 700)
                if a1 > a0:
                    small[:, a0:a1] += band[:, a0 - x0:a1 - x0] * gain
            fxs = np.zeros((hh, hw, 3), np.float32)
            fxs += small[..., None] * 14
            for ex, ey, vs, sz, ph in embers:
                y = (ey - vs * t * 3) % H
                x = ex + 18 * np.sin(t * 1.3 + ph)
                a = 0.5 + 0.5 * np.sin(t * 4 + ph)
                cv2.circle(fxs, (int(x / 2), int(y / 2)), max(1, int(sz / 2)), tuple(float(c) * a for c in fxc), -1, cv2.LINE_AA)
            frame += cv2.resize(fxs, (W, H), interpolation=cv2.INTER_LINEAR)

            # ---- number: slam at T1, then grows into a faint outline behind the name
            if L_num is not None and t >= T1 - 0.22:
                if t < T1:
                    u = ease_in((t - (T1 - 0.22)) / 0.22)
                    s_prev = 3.4 - 2.4 * ease_in(((t - 1 / FPS) - (T1 - 0.22)) / 0.22)
                    _paste_blur(frame, L_num, num_c, num_c, max(s_prev, 3.4 - 2.4 * u), 3.4 - 2.4 * u, min(1, u * 2), n=9)
                else:
                    settle = ease_out_back((t - T1) / 0.18, 2.2)
                    s = 1.0 + 0.06 * (1 - settle)
                    g = smooth((t - (T2 - 0.12)) / 0.35)        # -> ghost
                    s_ghost = s * (1 + 0.45 * g) + 0.02 * (t - T1)
                    _paste(frame, L_num_glow, num_c[0], num_c[1], s_ghost, 0.55 * (1 - g) * (0.6 + 0.4 * np.exp(-(t - T1) / 0.3)), add=True)
                    _paste(frame, L_num_sh, num_c[0] + 14 * s_ghost, num_c[1] + 18 * s_ghost, s_ghost, 0.7 * (1 - g))
                    _paste(frame, L_num, num_c[0], num_c[1], s_ghost, 1 - g)
                    _paste(frame, L_num_ghost, num_c[0], num_c[1], s_ghost, 0.22 * g)

            # ---- slanted team-colour panel behind the name block (comes in with the names)
            if t >= T2 - 0.12:
                u = ease_out((t - (T2 - 0.12)) / 0.25)
                px0, px1 = tx - 600, tx + 600
                py0, py1 = y_first - 110, y_school + 70
                right = px0 + (px1 - px0) * u
                pts = np.array([[px0 + 90, py0], [right + 90, py0], [right - 90, py1], [px0 - 90, py1]], np.int32)
                pan = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(pan, pts, 255, cv2.LINE_AA)
                a = cv2.GaussianBlur(pan, (0, 0), 1.2).astype(np.float32)[..., None] / 255 * 0.55
                frame = frame * (1 - a) + np.array(team, np.float32) * 0.55 * a
                edge = np.array([[px0 + 90, py0], [px0 + 110, py0], [px0 - 70, py1], [px0 - 90, py1]], np.int32)
                if u > 0.05:
                    cv2.fillConvexPoly(frame, edge, tuple(float(c) for c in fxc), cv2.LINE_AA)

            # ---- bars wipe in behind the name (T3)
            if t >= T3 - 0.15:
                u = ease_out((t - (T3 - 0.15)) / 0.3)
                for j, (yy, bh_, col, a) in enumerate(((y_bar - 6, 14, fxc, 1.0), (y_bar + 16, 5, accent, 0.9))):
                    x0 = tx - 470
                    x1 = x0 + 940 * u
                    pts = np.array([[x0 + 20, yy], [x1 + 20, yy], [x1, yy + bh_], [x0, yy + bh_]], np.int32)
                    bar = np.zeros((H, W, 3), np.float32)
                    cv2.fillConvexPoly(bar, pts, tuple(float(c) for c in col), cv2.LINE_AA)
                    msk = (bar.sum(2, keepdims=True) > 0).astype(np.float32) * a
                    frame = frame * (1 - msk) + bar * msk

            # ---- names: fly in from both sides, collide at T2
            if t >= T2 - 0.2:
                if t < T2:
                    u = ease_in((t - (T2 - 0.2)) / 0.2)
                    up = ease_in(((t - 1 / FPS) - (T2 - 0.2)) / 0.2)
                    if L_first is not None:
                        _paste_blur(frame, L_first, (tx - 900 * (1 - up), y_first), (tx - 900 * (1 - u), y_first), 1, 1, u)
                    if L_last is not None:
                        _paste_blur(frame, L_last, (tx + 1100 * (1 - up), y_last), (tx + 1100 * (1 - u), y_last), 1, 1, u)
                else:
                    pop = 1 + 0.05 * (1 - ease_out_back((t - T2) / 0.2, 2.5))
                    if L_last_glow is not None:
                        _paste(frame, L_last_glow, tx, y_last, pop, 0.5 * np.exp(-(t - T2) / 0.35) + 0.12, add=True)
                    if L_first is not None:
                        _paste(frame, L_first_sh, tx + 8, y_first + 10, pop, 0.7)
                        _paste(frame, L_first, tx, y_first, pop)
                    if L_last is not None:
                        _paste(frame, L_last_sh, tx + 10, y_last + 14, pop, 0.8)
                        _paste(frame, L_last, tx, y_last, pop)
                        # light sheen across the last name after the logo lands
                        if T4 <= t < T4 + 0.6:
                            u = (t - T4) / 0.6
                            h, w = L_last.shape[:2]
                            xs = np.arange(w)[None, :] - (u * (w + 400) - 200) - (np.arange(h)[:, None] * 0.4)
                            sheen_band = np.exp(-(xs / 60.0) ** 2).astype(np.float32)
                            sheen = L_last.copy()
                            sheen[:, :, :3] = 255
                            sheen[:, :, 3] = (L_last[:, :, 3].astype(np.float32) * sheen_band * 0.85).astype(np.uint8)
                            _paste(frame, sheen, tx, y_last, pop, 1.0, add=True)

            # ---- details + school slide up (T3)
            if t >= T3 - 0.05:
                for L, y, delay in ((L_det, y_det, 0.0), (L_school, y_school, 0.12)):
                    if L is None:
                        continue
                    u = ease_out((t - (T3 - 0.05) - delay) / 0.35)
                    _paste(frame, L, tx, y + 40 * (1 - u), 1.0, u)

            # ---- logo pops (T4)
            if L_logo is not None and t >= T4 - 0.05:
                u = (t - (T4 - 0.05)) / 0.35
                s = 0.2 + 0.8 * ease_out_back(u, 2.6)
                _paste(frame, L_logo_glow, logo_c[0], logo_c[1], s, 0.7 * np.exp(-max(0, t - T4) / 0.6) + 0.15, add=True)
                _paste(frame, L_logo, logo_c[0], logo_c[1], s, min(1, u * 3))

            # ---- impact effects: sparks, shockwave, flash
            if any(0 <= t - t0 < 0.9 for t0 in (T1, T2)):
                fx = np.zeros((H, W, 3), np.float32)
                for sp in sparks:
                    sp.draw(fx, t)
                if L_num is not None and 0 <= t - T1 < 0.45:
                    u = (t - T1) / 0.45
                    cv2.circle(fx, num_c, int(120 + 1300 * ease_out(u)), tuple(float(c) * (1 - u) for c in fxc),
                               max(2, int(34 * (1 - u))), cv2.LINE_AA)
                glow = cv2.resize(cv2.GaussianBlur(cv2.resize(fx, (hw, hh), interpolation=cv2.INTER_AREA), (0, 0), 4),
                                  (W, H))
                frame += cv2.GaussianBlur(fx, (0, 0), 1.0) + glow * 0.9
            for t0, a in ((T1, 0.75), (T2, 0.55)):
                if 0 <= t - t0 < 0.14:
                    fl = a * (1 - (t - t0) / 0.14)
                    frame = frame * (1 - fl) + 255 * fl

            # ---- grade: vignette, grain, shake, glitch, end flash
            frame *= vig
            frame += cv2.resize(grain[k % len(grain)], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
            dx, dy = _shake(t, hits)
            if dx or dy:
                m = np.float32([[1.03, 0, dx - W * 0.015], [0, 1.03, dy - H * 0.015]])
                frame = cv2.warpAffine(frame, m, (W, H), borderMode=cv2.BORDER_REFLECT)
            if 0 <= t - T2 < 0.1:                       # colour-split glitch on the name slam
                sh = int(14 * (1 - (t - T2) / 0.1)) + 2
                frame[:, sh:, 2] = frame[:, :-sh, 2].copy()
                frame[:, :-sh, 0] = frame[:, sh:, 0].copy()
            tail = (t - (dur - 0.17)) / 0.17
            if tail > 0:                                 # flash out into the first play
                frame = frame * (1 - 0.85 * tail) + 255 * 0.85 * tail
            enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
        wav.unlink(missing_ok=True)
    if enc.returncode != 0:
        raise RuntimeError("intro render failed")
    return out, n_frames / FPS


STYLES = ("animated", "inferno", "storm", "shatter", "stadium",
          "glitch", "smoke", "neon", "warp", "gold")     # "animated" = Slam (this module)


def cache_key(pl: dict, logo: Path | None, grid: dict | None, music_start: float, bg_id: str, sfx: bool,
              style: str = "animated", size: tuple[int, int] | None = None) -> str:
    payload = {"v": INTRO_VERSION, **({"style": style} if style != "animated" else {}),
               **({"size": list(size)} if size and tuple(size) != (graphics.W, graphics.H) else {}), "pl": {k: pl.get(k) for k in ("name", "number", "position", "class_year", "tagline",
                                                                  "height_weight", "school", "color", "accent")},
               "logo": [str(logo), logo.stat().st_mtime] if logo else None,
               "beats": beat_times(grid, music_start), "bg": bg_id, "sfx": sfx,
               **({"cut": cut_point(grid, music_start)} if grid and grid.get("burst") else {})}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]



def pick_background(root: Path, clips: list[dict]) -> tuple[np.ndarray | None, str]:
    """A frame of the player for the backdrop: the first play with a spotlight (else the first
    play), taken from its review proxy just after the snap, cropped to the real picture."""
    from . import camera, proxy, segment
    if not clips:
        return None, ""
    c = next((c for c in clips if camera.anchors_of(c)), clips[0])
    _, _, snap = segment._window(c)
    src = proxy.proxy_path(root / ".hlv_cache", c["id"])
    if not src.exists():
        return None, ""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{snap + 0.3:.3f}", "-i", str(src), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    if img is None:
        return None, ""
    h, w = img.shape[:2]
    x0, y0, cw, ch = (int(round(v)) for v in camera.content_rect(c, w, h))
    img = img[y0:y0 + ch, x0:x0 + cw]
    return img, f"{c['id']}@{snap:.2f}"


def build(root: Path, lib_root: Path, pl: dict, clips: list[dict], grid: dict | None, music_start: float,
          sfx: bool = True, style: str = "animated", size: tuple[int, int] | None = None) -> tuple[Path, float]:
    """Render (or reuse) the animated intro for this project in one of STYLES. Returns (mp4, duration).
    size: (w, h) to render at, e.g. (1080, 1920) for the vertical reel; default 1920x1080."""
    from . import profiles
    bg, bg_id = pick_background(root, clips)
    logo = profiles.asset(lib_root, pl.get("logo", ""))
    style = style if style in STYLES else "animated"
    size = tuple(size) if size else (graphics.W, graphics.H)
    key = cache_key(pl, logo, grid, music_start, bg_id, sfx, style, size)
    out = root / ".hlv_cache" / "intro" / f"intro-{key}.mp4"
    cut = cut_point(grid, music_start)                 # the reel uses the intro up to here
    if not out.exists():
        with frame_size(*size):
            if style != "animated":
                from . import intro_fx
                intro_fx.render(style, out, lib_root, pl, grid, music_start, bg, sfx=sfx)
            else:
                render(out, lib_root, pl, grid, music_start, bg, sfx=sfx)
    return out, cut


def timing_grid(grid: dict | None, music: Path | None, cache: Path, settings: dict) -> dict | None:
    """The beat grid plus what the intro's timing needs from the settings: the song's heavy hits
    (Intro timing: Big hits) and whether it ends in a burst into the first play."""
    from . import beats, burst
    if not grid:
        return grid
    sync = settings.get("intro_sync", "beat")
    out = settings.get("intro_out", "fade")
    if sync != "hits" and out not in burst.STYLES:
        return grid
    g = {**grid, "sync": sync, "burst": out in burst.STYLES}
    if music:
        g["hits"] = beats.heavy_hits(music, cache)
    return g


def preview(root: Path, intro_mp4: Path, music: Path | None, music_start: float, settings: dict) -> Path:
    """The intro with the song underneath, as it will sound in the reel."""
    out = intro_mp4.with_name(intro_mp4.stem + "-preview.mp4")
    if not music:
        return intro_mp4
    mvol = float(settings.get("music_volume", 0.9))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(intro_mp4), "-ss", f"{music_start:.3f}", "-i", str(music),
                    "-filter_complex",
                    f"[1:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:d=0.5,volume={mvol}[m];"
                    "[0:a][m]amix=inputs=2:normalize=0:duration=first[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(out)], check=True)
    return out
