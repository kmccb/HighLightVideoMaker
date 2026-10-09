"""Four more animated intros, alongside intro.py's "Slam": Inferno, Thunderstruck, Shatter and
Stadium Lights. Same beats (T1 number, T2 name, T3 details, T4 logo), same layout (logo left,
name block right), same player data - each with its own world, effects and entrances.

  inferno  flames roar up on T1; the number burns in (dissolving glowing edge, white-hot then
           cooling); the name rises out of the fire; heat haze, embers and ash; fireball flash out
  storm    rolling storm clouds; forked lightning strikes on every beat with a white flash and
           shake; the number is a live wire (electric arcs crawl its edge); the name is revealed
           by the strike; a final massive strike whites out
  shatter  the number punches through a pane of glass toward the camera (slow-motion shards,
           chromatic split); the name slams in from both sides through the debris
  stadium  Friday-night lights bang on bank by bank; volumetric beams through haze; anamorphic
           lens flares; chrome number with a moving glint; the name on a broadcast bar

Everything is generated (no stock footage). Fire, clouds, beams and glows are computed at a
quarter or half size and scaled up, so a style renders in well under a minute.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from . import encoder, graphics
from .graphics import _detail_line, _rgb
from .intro import (FPS, H, SR, W, Sparks, _fx_color, _glow, _impact_sfx, _outline, _paste, _paste_blur,
                    _shadow, _shake, _text_layer, _vignette, background_frame, beat_times, ease_in, ease_out,
                    ease_out_back, fit, layout, smooth)

STYLES = ("inferno", "storm", "shatter", "stadium")
QW, QH = W // 4, H // 4


# ---------------------------------------------------------------- shared pieces
def _noise(rng, h: int, w: int, sigma: float) -> np.ndarray:
    """Smooth random field in 0..1 (float32)."""
    n = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    n -= n.min()
    return n / max(float(n.max()), 1e-6)


def _bloom(fx: np.ndarray, sigma: float = 5, gain: float = 1.0) -> np.ndarray:
    """fx (full-res float BGR) plus a soft glow of itself, computed at quarter size."""
    small = cv2.resize(fx, (QW, QH), interpolation=cv2.INTER_AREA)
    return fx + cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma), (W, H)) * gain


def _up(q: np.ndarray) -> np.ndarray:
    return cv2.resize(q, (W, H), interpolation=cv2.INTER_CUBIC)


def _tinted(layer: np.ndarray, bgr) -> np.ndarray:
    out = layer.copy()
    out[:, :, :3] = bgr
    return out


class Card:
    """Text layers and layout shared by the styles."""

    def __init__(self, lib_root: Path, pl: dict, num_fill=None, num_stroke=None):
        self.accent = _rgb(pl.get("accent"), (255, 210, 63))[::-1]          # BGR
        self.team = _rgb(pl.get("color"))[::-1]
        self.fxc = _fx_color(pl)
        white = (255, 255, 255)
        num_txt = f"#{pl['number']}" if pl.get("number") else ""
        parts = (pl.get("name") or "").upper().split()
        first, last = (" ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else ("", parts[0] if parts else "")
        fill = num_fill or self.accent
        stroke = num_stroke or self.team
        self.num = fit(_text_layer(num_txt, 600, tuple(fill[::-1]) + (255,), stroke=16,
                                   stroke_fill=tuple(stroke[::-1])), W - 60) if num_txt else None
        self.last_size = 300 if len(last) <= 8 else 230
        self.first = fit(_text_layer(first, 130, white + (255,), spacing=16)) if first else None
        self.last = fit(_text_layer(last, self.last_size, white + (255,), stroke=4,
                                    stroke_fill=tuple(self.team[::-1]))) if last else None
        det = _detail_line(pl)
        self.det = fit(_text_layer(det, 54, (225, 230, 238, 255), style="SemiBold Condensed", spacing=2)) if det else None
        self.school = fit(_text_layer((pl.get("school") or "").upper(), 54, tuple(self.accent[::-1]) + (255,),
                                      style="SemiBold Condensed", spacing=3)) if pl.get("school") else None
        lg = graphics._logo(lib_root, pl, 380)
        self.logo = np.array(lg)[:, :, [2, 1, 0, 3]].copy() if lg is not None else None
        # layout (as Slam): logo left, name block right; centred when there's no logo; stacked in portrait
        lay = layout(self.logo is not None, self.last_size)
        self.tx, self.logo_c, self.num_c = lay["tx"], lay["logo_c"], lay["num_c"]
        self.y_first, self.y_last, self.y_bar = lay["y_first"], lay["y_last"], lay["y_bar"]
        self.y_det, self.y_school = lay["y_det"], lay["y_school"]

    def details(self, frame, t, T3, T4, glow_col=None):
        """Details + school slide up at T3, logo pops at T4 (all styles)."""
        if t >= T3 - 0.05:
            for L, y, delay in ((self.det, self.y_det, 0.0), (self.school, self.y_school, 0.12)):
                if L is not None:
                    u = ease_out((t - (T3 - 0.05) - delay) / 0.35)
                    _paste(frame, L, self.tx, y + 40 * (1 - u), 1.0, u)
        if self.logo is not None and t >= T4 - 0.05:
            u = (t - (T4 - 0.05)) / 0.35
            s = 0.2 + 0.8 * ease_out_back(u, 2.6)
            if glow_col is not None:
                if not hasattr(self, "_logo_glow"):
                    self._logo_glow = _glow(self.logo, glow_col, 26)
                _paste(frame, self._logo_glow, *self.logo_c, s, 0.7 * np.exp(-max(0, t - T4) / 0.6) + 0.15, add=True)
            _paste(frame, self.logo, *self.logo_c, s, min(1, u * 3))

    def bars(self, frame, t, T3, cols):
        if t >= T3 - 0.15:
            u = ease_out((t - (T3 - 0.15)) / 0.3)
            for yy, bh_, col in ((self.y_bar - 6, 14, cols[0]), (self.y_bar + 16, 5, cols[1])):
                x0 = self.tx - 470
                x1 = x0 + 940 * u
                pts = np.array([[x0 + 20, yy], [x1 + 20, yy], [x1, yy + bh_], [x0, yy + bh_]], np.int32)
                cv2.fillConvexPoly(frame, pts, tuple(float(c) for c in col), cv2.LINE_AA)


def _encoder(out: Path, dur: float, hits, sfx: bool, rng):
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav")
    audio = _impact_sfx(hits, dur, rng) if sfx else np.zeros((int(dur * SR) + 1, 2), np.float32)
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
    return enc, wav


def _finish(frame, k, grain, vig) -> np.ndarray:
    frame = frame * vig
    frame += cv2.resize(grain[k % len(grain)], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    return np.clip(frame, 0, 255).astype(np.uint8)


def _apply_shake(frame, t, hits, amp=26.0):
    dx, dy = _shake(t, hits, amp)
    if dx or dy:
        m = np.float32([[1.03, 0, dx - W * 0.015], [0, 1.03, dy - H * 0.015]])
        frame = cv2.warpAffine(frame, m, (W, H), borderMode=cv2.BORDER_REFLECT)
    return frame


def _chroma(frame, px: int):
    if px > 0:
        frame[:, px:, 2] = frame[:, :-px, 2].copy()
        frame[:, :-px, 0] = frame[:, px:, 0].copy()
    return frame


# ================================================================ INFERNO
_FIRE_STOPS = [(0.0, (0, 0, 0)), (0.22, (8, 10, 90)), (0.45, (0, 45, 210)), (0.65, (20, 130, 255)),
               (0.85, (90, 210, 255)), (1.0, (220, 250, 255))]
_FIRE_LUT = np.zeros((256, 3), np.float32)
for _i in range(256):
    _v = _i / 255
    for (a, ca), (b, cb) in zip(_FIRE_STOPS, _FIRE_STOPS[1:]):
        if a <= _v <= b:
            _u = (_v - a) / (b - a)
            _FIRE_LUT[_i] = np.array(ca) * (1 - _u) + np.array(cb) * _u
            break


class FireSim:
    """Classic rising-heat fire: each step every cell becomes the average of the cells just
    below it (so heat climbs and spreads), minus a smooth "cooling map" that scrolls upward
    with the flames. The cooling map carves the heat into licking tongues that flicker and
    shed wisps at the top. Hot spots along the base sway slowly, so flames gather into
    columns. Runs at a third of the frame size; ~8 steps a frame."""

    def __init__(self, rng, w: int | None = None, h: int | None = None):
        w, h = w or W // 3, h or H // 3
        self.rng, self.w, self.h = rng, w, h
        self.heat = np.zeros((h, w), np.float32)
        self.cool = _noise(rng, h * 4, w, 3)                  # scrolls up one row per step
        self.sway = _noise(rng, h * 6, w, 9)                   # sideways wiggle field, rising
        self.mx, self.my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.src = _noise(rng, 6, w * 4, 7)[3]               # hot spots along the base
        self.n = 0

    def step(self, height: float, t: float, n: int) -> np.ndarray:
        k = 2.2 / max(height * self.h, 4)                       # cooling per row -> flame height
        sway = int(self.w * 0.6 + 30 * np.sin(t * 0.8) + 14 * np.sin(t * 2.1))
        rows = np.arange(self.h)
        for _ in range(n):
            spots = self.src[sway:sway + self.w]
            self.heat[-2:] = np.clip(0.6 + 1.4 * (spots - 0.45) + 0.3 * self.rng.random(self.w, dtype=np.float32), 0, 1.3)
            b1 = np.roll(self.heat, -1, 0)
            avg = (b1 + np.roll(b1, 1, 1) + np.roll(b1, -1, 1) + np.roll(self.heat, -2, 0)) * 0.25
            cool = self.cool[(rows + self.n) % self.cool.shape[0]]
            self.heat[:-2] = np.maximum(avg[:-2] - cool[:-2] * k, 0)
            self.n += 1
        return self.heat

    def look(self, t: float) -> np.ndarray:
        """The heat as it's drawn: each row nudged sideways by a slow rising wave, so the
        tongues sway and curl instead of standing like columns."""
        rows = (np.arange(self.h) + int(t * 90)) % len(self.sway)
        dx = (self.sway[rows] - 0.5) * 70 * (1 - 0.6 * np.arange(self.h)[:, None] / self.h)
        return cv2.remap(self.heat, self.mx + dx.astype(np.float32), self.my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)


def _inferno(k, t, ctx):
    c, T, dur, rng = ctx["card"], ctx["T"], ctx["dur"], ctx["rng"]
    T1, T2, T3, T4 = T
    # flame height: a low flicker, roaring up on T1 (and again a little on T2)
    h = 0.12 + 0.08 * smooth(t / max(T1, 0.1))
    for t0, peak in ((T1, 0.95), (T2, 0.7)):
        if t >= t0 - 0.06:
            h = max(h, 0.42 + (peak - 0.42) * np.exp(-max(0.0, t - t0) / 0.45))
    if t >= T1:
        h = max(h, 0.42)
    roar = any(-0.08 <= t - t0 <= 0.45 for t0 in (T1, T2))
    ctx["fire"].step(h, t, 200 if k == 0 else (26 if roar else 8))   # flames shoot up on the beat
    heat = ctx["fire"].look(t)
    fire = np.clip(heat, 0, 1) ** 1.25
    fire_rgb = _FIRE_LUT[(fire * 255).astype(np.uint8)]
    big = cv2.resize(fire_rgb, (W, H), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), 0.9)                   # soften the grid, keep the tongues
    # smoke: the heat field blurred, lifted and darkened, drifting above the flames
    sm = cv2.GaussianBlur(heat, (0, 0), 9)
    sm = np.roll(sm, -int(ctx["fire"].h * 0.22), axis=0)
    sm[-int(ctx["fire"].h * 0.22):] = 0
    smoke = np.clip(sm * 1.4 - heat * 0.8, 0, 1)
    smoke = cv2.resize(smoke, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    light = float(np.mean(heat[-int(ctx["fire"].h * 0.4):]))     # how bright the fire is now
    flick = 0.85 + 0.3 * light + 0.06 * np.sin(t * 23) * light
    # backdrop: near black, warm firelight from below that flickers with the flames
    frame = ctx["base"].copy()
    frame += ctx["warm"] * (0.25 + 0.9 * light) * flick
    frame = frame * (1 - 0.55 * smoke) + smoke * np.array([24, 26, 30], np.float32) * 0.6
    frame += cv2.resize(cv2.GaussianBlur(fire_rgb, (0, 0), 10), (W, H)) * 0.6
    frame += big * 1.1
    v = (np.arange(QH, dtype=np.float32) / QH)[:, None]
    base = (np.clip((v - (1 - h)) / max(h, 0.05), 0, 1) ** 1.15).astype(np.float32)   # for the heat haze
    # ---- number burns in (dissolving glowing edge), white-hot then cools to the accent colour
    if c.num is not None and t >= T1 - 0.1:
        r = smooth((t - (T1 - 0.1)) / 0.55) * 1.08
        nm = ctx["burn"]
        a = c.num[:, :, 3].astype(np.float32) / 255
        show = (nm < r).astype(np.float32) * a
        edge = ((nm >= r - 0.09) & (nm < r)).astype(np.float32) * a
        cool = smooth((t - T1) / 1.1)
        hot = np.array([200, 245, 255], np.float32)
        col = c.num[:, :, :3].astype(np.float32) * cool + hot * (1 - cool)
        ghost = smooth((t - (T2 - 0.1)) / 0.4)                        # recedes behind the name
        lay = np.zeros(c.num.shape, np.uint8)
        lay[:, :, :3] = np.clip(col, 0, 255)
        lay[:, :, 3] = np.clip(show * 255 * (1 - 0.78 * ghost), 0, 255)
        s = 1.0 + 0.35 * ghost
        _paste(frame, ctx["num_glow"], *c.num_c, s, (0.9 * (1 - cool) + 0.25) * (1 - 0.6 * ghost) * flick, add=True)
        _paste(frame, lay, *c.num_c, s)
        if r < 1.05:
            e = np.zeros(c.num.shape, np.uint8)
            e[:, :, :3] = (60, 190, 255)
            e[:, :, 3] = np.clip(edge * 255, 0, 255)
            _paste(frame, e, *c.num_c, s, 1.0, add=True)
            _paste(frame, _glow(e, (40, 150, 255), 10), *c.num_c, s, 0.9, add=True)
    # ---- name rises out of the fire at T2
    if t >= T2 - 0.25:
        u = ease_out((t - (T2 - 0.25)) / 0.45)
        for L, G, y, d in ((c.first, ctx["first_glow"], c.y_first, 0.0), (c.last, ctx["last_glow"], c.y_last, 0.06)):
            if L is None:
                continue
            uu = ease_out((t - (T2 - 0.25) - d) / 0.45)
            yy = y + 160 * (1 - uu)
            _paste(frame, G, c.tx, yy, 1.0, (0.7 * np.exp(-max(0, t - T2) / 0.6) + 0.3) * uu, add=True)
            _paste(frame, _shadow(L, 8) if "sh" not in ctx else ctx["sh"][id(L)], c.tx + 8, yy + 12, 1.0, 0.7 * uu)
            _paste(frame, L, c.tx, yy, 1.0, uu)
    c.bars(frame, t, T3, (ctx["orange"], c.accent))
    c.details(frame, t, T3, T4, glow_col=ctx["orange"])
    # ---- embers, sparks, ash
    fx = np.zeros((H, W, 3), np.float32)
    for sp in ctx["sparks"]:
        sp.draw(fx, t)
    em = ctx["embers"]
    for ex, ey, vs, sz, ph in em:
        y = (ey - vs * t) % H
        x = ex + 30 * np.sin(t * 1.1 + ph)
        a = 0.55 + 0.45 * np.sin(t * 6 + ph)
        cv2.circle(fx, (int(x), int(y)), max(1, int(sz)), (40 * a, 150 * a, 255 * a), -1, cv2.LINE_AA)
    frame = frame + _bloom(fx, 6, 1.2)
    for ex, ey, vs, sz, ph in ctx["ash"]:
        y = (ey - vs * t) % H
        x = ex + 40 * np.sin(t * 0.7 + ph)
        cv2.circle(frame, (int(x), int(y)), max(1, int(sz)), (95, 100, 105), -1, cv2.LINE_AA)
    # ---- heat haze over everything (stronger near the fire)
    hz = ctx["haze"]
    rr = (np.arange(QH) + int(t * 90)) % hz.shape[0]
    d = (hz[rr] - 0.5) * (2.5 + 7 * cv2.resize(base, (QW, QH)))
    dx = _up(d.astype(np.float32))
    frame = cv2.remap(frame, ctx["mx"] + dx, ctx["my"] + dx * 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frame = _apply_shake(frame, t, ctx["hits"], 22)
    # ---- fireball flash out
    tail = (t - (dur - 0.32)) / 0.32
    if tail > 0:
        r = np.sqrt(ctx["rr2"]) / (W * (0.15 + 0.9 * tail))
        ball = np.clip(1.2 - r, 0, 1)[..., None]
        col = np.array([60, 170, 255], np.float32) * (1 - tail) + np.array([230, 245, 255], np.float32) * tail
        frame = frame * (1 - ball * tail) + col * ball * tail * 1.1
    return frame


def _inferno_setup(ctx, rng):
    c = ctx["card"]
    T1, T2 = ctx["T"][0], ctx["T"][1]
    ctx["base"] = background_frame(ctx["bg"], ctx["pl"]) * 0.35
    ctx["fire"] = FireSim(rng)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    ctx["warm"] = (np.clip(yy - 0.25, 0, 1) ** 1.6) * np.array([10, 45, 120], np.float32)   # firelight from below
    ctx["haze"] = _noise(rng, QH * 3, QW, 4)
    ctx["orange"] = (30, 120, 255)
    if c.num is not None:
        ctx["burn"] = _noise(rng, c.num.shape[0], c.num.shape[1], 14)
        ctx["num_glow"] = _glow(c.num, (30, 140, 255), 34)
    ctx["first_glow"] = _glow(c.first, (30, 110, 255), 18) if c.first is not None else None
    ctx["last_glow"] = _glow(c.last, (30, 110, 255), 26) if c.last is not None else None
    ctx["sh"] = {id(L): _shadow(L, 8) for L in (c.first, c.last) if L is not None}
    up = (-np.pi * 0.85, -np.pi * 0.15)
    ctx["sparks"] = [Sparks(rng, c.num_c[0], c.num_c[1] + 120, T1, (40, 160, 255), n=120, spread=(500, 1700), cone=up),
                     Sparks(rng, c.tx, c.y_last + 80, T2, (40, 160, 255), n=90, spread=(400, 1400), cone=up)]
    n = 90
    ctx["embers"] = np.stack([rng.uniform(0, W, n), rng.uniform(0, H, n), rng.uniform(60, 220, n),
                              rng.uniform(1.2, 3.2, n), rng.uniform(0, 6.28, n)], 1)
    ctx["ash"] = np.stack([rng.uniform(0, W, 40), rng.uniform(0, H, 40), rng.uniform(25, 70, 40),
                           rng.uniform(1, 2.2, 40), rng.uniform(0, 6.28, 40)], 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ctx["mx"], ctx["my"] = xx, yy
    ctx["rr2"] = (xx - c.num_c[0]) ** 2 + (yy - c.num_c[1]) ** 2


# ================================================================ THUNDERSTRUCK
def _bolt(rng, x0, y0, x1, y1, disp, depth=6, branches=True):
    """Forked lightning: midpoint displacement, with side branches. Returns a list of polylines."""
    pts = [(x0, y0), (x1, y1)]
    d = disp
    for _ in range(depth):
        nxt = [pts[0]]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            mx, my = (ax + bx) / 2, (ay + by) / 2
            L = np.hypot(bx - ax, by - ay) or 1
            off = rng.normal(0, d)
            nxt += [(mx - (by - ay) / L * off, my + (bx - ax) / L * off), (bx, by)]
        pts = nxt
        d *= 0.55
    lines = [np.array(pts, np.float32)]
    if branches:
        for _ in range(rng.integers(2, 5)):
            i = int(rng.integers(len(pts) // 5, len(pts) * 4 // 5))
            bx, by = pts[i]
            ang = np.arctan2(y1 - y0, x1 - x0) + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
            ln = np.hypot(x1 - x0, y1 - y0) * rng.uniform(0.15, 0.35)
            lines += [b * 1 for b in _bolt(rng, bx, by, bx + np.cos(ang) * ln, by + np.sin(ang) * ln,
                                           disp * 0.4, depth - 2, False)]
    return lines


def _storm(k, t, ctx):
    c, T, dur, rng = ctx["card"], ctx["T"], ctx["dur"], ctx["rng"]
    T1, T2, T3, T4 = T
    blue = (255, 200, 140)
    # flash level: each strike double-pulses
    fl = 0.0
    for st in ctx["strikes"]:
        d = t - st["t"]
        if 0 <= d < 0.35:
            fl = max(fl, st["power"] * (np.exp(-d / 0.05) + 0.6 * np.exp(-abs(d - 0.09) / 0.03)))
    # clouds: two layers drifting, lit by the flashes
    cl = ctx["clouds"]
    x1 = int(t * 14) % (cl.shape[1] - QW)
    x2 = int(t * 26) % (cl.shape[1] - QW)
    cloud = 0.6 * cl[:QH, x1:x1 + QW] + 0.4 * cl[QH:2 * QH, x2:x2 + QW]
    cloud = np.clip(cloud * 1.5 - 0.35, 0, 1) * ctx["sky"]
    sky = cloud[..., None] * (np.array([70, 52, 40], np.float32) + np.array([255, 225, 200], np.float32) * fl * 0.9)
    frame = ctx["base"] + _up(sky)
    fx = np.zeros((H, W, 3), np.float32)
    # ---- bolts
    for st in ctx["strikes"]:
        d = t - st["t"]
        if 0 <= d < 0.3:
            on = 1.0 if (d < 0.06 or 0.08 < d < 0.14 or (d > 0.18 and rng.random() < 0.5)) else 0.25
            a = on * (1 - d / 0.3)
            for j, ln in enumerate(st["lines"]):
                thick = 5 if j == 0 else 2
                cv2.polylines(fx, [ln.astype(np.int32)], False, tuple(v * a for v in blue), thick + 6, cv2.LINE_AA)
                cv2.polylines(fx, [ln.astype(np.int32)], False, (255 * a, 255 * a, 255 * a), thick, cv2.LINE_AA)
    # ---- number: dark steel with an electric outline, revealed by the T1 strike
    if c.num is not None and t >= T1:
        ghost = smooth((t - (T2 - 0.1)) / 0.4)
        s = 1.0 + 0.35 * ghost
        flick = 1.0 if t - T1 > 0.35 else (1.0 if rng.random() < 0.7 else 0.2)
        _paste(frame, ctx["num_glow"], *c.num_c, s, (0.6 + 0.4 * fl) * (1 - 0.5 * ghost) * flick, add=True)
        _paste(frame, ctx["num_dark"], *c.num_c, s, (1 - 0.75 * ghost) * flick)
        _paste(frame, ctx["num_edge"], *c.num_c, s, (0.9 - 0.5 * ghost) * flick, add=True)
        # arcs crawling along the edge
        ep = ctx["edge_pts"]
        if len(ep):
            h, w = c.num.shape[:2]
            for _ in range(7 if t - T1 < 1.5 else 3):
                i = int(rng.integers(len(ep)))
                j = min(len(ep) - 1, i + int(rng.integers(15, 90)))
                (ya, xa), (yb, xb) = ep[i], ep[j]
                X0 = c.num_c[0] + (xa - w / 2) * s
                Y0 = c.num_c[1] + (ya - h / 2) * s
                X1 = c.num_c[0] + (xb - w / 2) * s
                Y1 = c.num_c[1] + (yb - h / 2) * s
                for ln in _bolt(rng, X0, Y0, X1, Y1, 14, 4, False):
                    cv2.polylines(fx, [ln.astype(np.int32)], False, (255, 230, 180), 2, cv2.LINE_AA)
    # ---- name: struck in at T2 (flash reveal, flicker), electric glow
    if t >= T2:
        d = t - T2
        flick = 1.0 if d > 0.3 else (1.0 if rng.random() < 0.75 else 0.15)
        for L, G, y in ((c.first, ctx["first_glow"], c.y_first), (c.last, ctx["last_glow"], c.y_last)):
            if L is None:
                continue
            _paste(frame, G, c.tx, y, 1.0, (0.9 * np.exp(-d / 0.4) + 0.25 + 0.3 * fl) * flick, add=True)
            _paste(frame, ctx["sh"][id(L)], c.tx + 8, y + 12, 1.0, 0.7 * flick)
            _paste(frame, L, c.tx, y, 1.0, flick)
    c.bars(frame, t, T3, (blue, c.accent))
    c.details(frame, t, T3, T4, glow_col=blue)
    frame = frame + _bloom(fx, 4, 1.6)
    frame = frame + 255 * fl * 0.55
    rain = ctx["rain"]
    for rx, ry, sp in rain:
        y = (ry + sp * t) % H
        x = (rx - 0.25 * sp * t) % W
        cv2.line(frame, (int(x), int(y)), (int(x - 6), int(y + 24)), (150, 140, 130), 1, cv2.LINE_AA)
    frame = _apply_shake(frame, t, ctx["hits"], 30)
    tail = (t - (dur - 0.3)) / 0.3
    if tail > 0:
        frame = frame * (1 - tail) + 255 * tail
    return frame


def _storm_setup(ctx, rng):
    c = ctx["card"]
    T1, T2, T3, T4 = ctx["T"]
    dur = ctx["dur"]
    ctx["base"] = background_frame(ctx["bg"], ctx["pl"]) * 0.22
    ctx["clouds"] = _noise(rng, QH * 2, QW * 3, 9) * 0.7 + _noise(rng, QH * 2, QW * 3, 3) * 0.3
    ctx["sky"] = np.clip(1.25 - np.arange(QH, dtype=np.float32)[:, None] / QH * 1.1, 0.15, 1)
    if c.num is not None:
        dark = c.num.copy()
        dark[:, :, :3] = (np.array(c.num[:, :, :3], np.float32) * 0.25 + np.array([40, 30, 25])).astype(np.uint8)
        ctx["num_dark"] = dark
        edge = _outline(c.num, (255, 225, 170), 5)
        ctx["num_edge"] = edge
        ctx["num_glow"] = _glow(edge, (255, 190, 120), 22)
        ys, xs = np.nonzero(edge[:, :, 3] > 128)
        order = np.argsort(np.arctan2(ys - ys.mean(), xs - xs.mean())) if len(ys) else []
        ctx["edge_pts"] = np.stack([ys, xs], 1)[order][::6] if len(ys) else []
    ctx["first_glow"] = _glow(c.first, (255, 190, 120), 16) if c.first is not None else None
    ctx["last_glow"] = _glow(c.last, (255, 190, 120), 24) if c.last is not None else None
    ctx["sh"] = {id(L): _shadow(L, 8) for L in (c.first, c.last) if L is not None}
    targets = [(T1, c.num_c, 1.0), (T2, (c.tx, c.y_last - 80), 0.85), (T3, (c.tx - 300, c.y_bar), 0.45),
               (T4, c.logo_c if c.logo is not None else (c.tx + 300, c.y_bar), 0.55), (dur - 0.3, (W / 2, H * 0.7), 1.2)]
    strikes = []
    for t0, (x, y), p in targets:
        x0 = x + rng.uniform(-380, 380)
        strikes.append({"t": t0, "power": p, "lines": _bolt(rng, x0, -20, x, y, 90)})
    for t0 in np.sort(rng.uniform(0.1, max(0.2, T1 - 0.2), 2)):          # distant flickers before T1
        x = rng.uniform(100, W - 100)
        strikes.append({"t": float(t0), "power": 0.25, "lines": _bolt(rng, x, -20, x + rng.uniform(-200, 200),
                                                                      rng.uniform(250, 450), 50)})
    ctx["strikes"] = strikes
    ctx["rain"] = np.stack([rng.uniform(0, W, 140), rng.uniform(0, H, 140), rng.uniform(900, 1500, 140)], 1)
    ctx["hits"] = [(T1, 1.0), (T2, 0.8), (T3, 0.3), (T4, 0.4)]


# ================================================================ SHATTER
def _shatter(k, t, ctx):
    c, T, dur = ctx["card"], ctx["T"], ctx["dur"]
    T1, T2, T3, T4 = T
    frame = ctx["base"].copy()
    fx = np.zeros((H, W, 3), np.float32)
    # ---- number: flies at the glass from far behind, punches through at T1
    if c.num is not None and t >= T1 - 0.35:
        if t < T1:
            u = ease_in((t - (T1 - 0.35)) / 0.35)
            up = ease_in(((t - 1 / FPS) - (T1 - 0.35)) / 0.35)
            _paste_blur(frame, c.num, c.num_c, c.num_c, 0.15 + 0.85 * up, 0.15 + 0.85 * u, min(1, u * 1.5), n=7)
        else:
            ghost = smooth((t - (T2 - 0.1)) / 0.4)
            s = 1.0 + 0.08 * np.exp(-(t - T1) / 0.15) + 0.35 * ghost
            _paste(frame, ctx["num_glow"], *c.num_c, s, 0.5 * np.exp(-(t - T1) / 0.4) * (1 - ghost), add=True)
            _paste(frame, ctx["num_sh"], c.num_c[0] + 16, c.num_c[1] + 20, s, 0.7 * (1 - ghost))
            _paste(frame, c.num, *c.num_c, s, 1 - 0.78 * ghost)
    # ---- the pane: intact until T1, then shards fly at the camera (slow motion for 0.4 s)
    d = t - T1
    if d < 0:
        for sh in ctx["shards"]:
            cv2.polylines(fx, [sh["pts"].astype(np.int32)], True, (40, 40, 40), 1, cv2.LINE_AA)
        x = int((t / max(T1, 0.1)) * (W + 600)) - 300                   # glint sliding across the glass
        cv2.line(fx, (x, 0), (x - 260, H), (90, 90, 90), 40)
    else:
        ts = 0.35 * d if d < 0.4 else 0.14 + (d - 0.4) * 1.1               # slow-mo, then real time
        fall = max(0.0, t - T3)
        A = np.zeros((H, W), np.float32)          # all shards in one pass: tint amount + brightness
        B = np.zeros((H, W), np.float32)
        for sh in ctx["shards"]:
            a = np.clip(1 - ts / sh["life"], 0, 1)
            if a <= 0:
                continue
            cen = sh["c"] + sh["v"] * ts + np.array([0, 900]) * fall * fall
            sc = 1 + sh["z"] * ts
            ang = sh["rot"] * ts
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
            p = ((sh["pts"] - sh["c"]) @ R.T) * sc + cen
            pi = p.astype(np.int32)
            glint = 0.5 + 0.5 * np.sin(ang * 3 + sh["ph"])
            cv2.fillConvexPoly(A, pi, float(0.25 * a), cv2.LINE_AA)
            cv2.fillConvexPoly(B, pi, float(a * (60 + 120 * glint)), cv2.LINE_AA)
            cv2.polylines(fx, [pi], True, tuple(v * a * (0.35 + 0.5 * glint) for v in (230, 240, 255)), 1, cv2.LINE_AA)
        frame = frame * (1 - A[..., None]) + B[..., None]
    # ---- names slam in from both sides through the debris at T2
    if t >= T2 - 0.2:
        if t < T2:
            u = ease_in((t - (T2 - 0.2)) / 0.2)
            up = ease_in(((t - 1 / FPS) - (T2 - 0.2)) / 0.2)
            if c.first is not None:
                _paste_blur(frame, c.first, (c.tx - 900 * (1 - up), c.y_first), (c.tx - 900 * (1 - u), c.y_first), 1, 1, u)
            if c.last is not None:
                _paste_blur(frame, c.last, (c.tx + 1100 * (1 - up), c.y_last), (c.tx + 1100 * (1 - u), c.y_last), 1, 1, u)
        else:
            pop = 1 + 0.05 * (1 - ease_out_back((t - T2) / 0.2, 2.5))
            for L, y in ((c.first, c.y_first), (c.last, c.y_last)):
                if L is not None:
                    _paste(frame, ctx["sh"][id(L)], c.tx + 10, y + 14, pop, 0.8)
                    _paste(frame, L, c.tx, y, pop)
            if c.last is not None:
                _paste(frame, ctx["last_glow"], c.tx, c.y_last, pop, 0.45 * np.exp(-(t - T2) / 0.35) + 0.1, add=True)
    c.bars(frame, t, T3, (c.fxc, c.accent))
    c.details(frame, t, T3, T4, glow_col=c.fxc)
    frame = frame + _bloom(fx, 3, 1.0)
    for t0, a in ((T1, 0.9), (T2, 0.5)):
        if 0 <= t - t0 < 0.12:
            f = a * (1 - (t - t0) / 0.12)
            frame = frame * (1 - f) + 255 * f
    frame = _apply_shake(frame, t, ctx["hits"], 30)
    for t0, mx in ((T1, 22), (T2, 14)):
        if 0 <= t - t0 < 0.25:
            frame = _chroma(frame, int(mx * (1 - (t - t0) / 0.25)) + 1)
    tail = (t - (dur - 0.17)) / 0.17
    if tail > 0:
        frame = frame * (1 - 0.85 * tail) + 255 * 0.85 * tail
    return frame


def _shatter_setup(ctx, rng):
    from scipy.spatial import Delaunay
    c = ctx["card"]
    T1, T2, T3, T4 = ctx["T"]
    ctx["base"] = background_frame(ctx["bg"], ctx["pl"]) * 0.8
    if c.num is not None:
        ctx["num_glow"] = _glow(c.num, c.fxc, 30)
        ctx["num_sh"] = _shadow(c.num, 12)
    ctx["sh"] = {id(L): _shadow(L, 9) for L in (c.first, c.last) if L is not None}
    ctx["last_glow"] = _glow(c.last, c.fxc, 22) if c.last is not None else None
    # shards: points dense around the impact, sparse at the edges, triangulated
    cx, cy = c.num_c
    r = np.abs(rng.normal(0, 380, 140)) + 10
    a = rng.uniform(0, 2 * np.pi, 140)
    pts = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    pts = np.vstack([pts, [[-50, -50], [W + 50, -50], [-50, H + 50], [W + 50, H + 50],
                           [W / 2, -50], [W / 2, H + 50], [-50, H / 2], [W + 50, H / 2]]])
    pts[:, 0] = np.clip(pts[:, 0], -50, W + 50)
    pts[:, 1] = np.clip(pts[:, 1], -50, H + 50)
    shards = []
    for tri in Delaunay(pts).simplices:
        p = pts[tri].astype(np.float32)
        cen = p.mean(0)
        v = cen - np.array([cx, cy], np.float32)
        dist = float(np.hypot(*v)) or 1.0
        speed = 2600 / (1 + dist / 300) + rng.uniform(100, 500)
        shards.append({"pts": p, "c": cen, "v": v / dist * speed, "z": 2.4 / (1 + dist / 400),
                       "rot": rng.normal(0, 3), "ph": rng.uniform(0, 6.28),
                       "life": rng.uniform(0.9, 1.6)})
    ctx["shards"] = shards
    ctx["hits"] = [(T1, 1.2), (T2, 0.8), (T3, 0.25), (T4, 0.35)]


# ================================================================ STADIUM LIGHTS
def _chrome(layer: np.ndarray) -> np.ndarray:
    """Metallic version of a text layer: banded vertical gradient plus a bevel highlight."""
    a = layer[:, :, 3].astype(np.float32) / 255
    h, w = a.shape
    v = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    band = 0.55 + 0.45 * np.cos(v * np.pi * 2.3 + 0.6)
    g = np.clip(band, 0, 1) * 210 + 40
    blur = cv2.GaussianBlur(a, (0, 0), 6)
    gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    light = np.clip(-(gx + gy) * 6, -1, 1)                              # lit from the top left
    val = np.clip(g + 90 * light, 0, 255)
    out = np.zeros((h, w, 4), np.uint8)
    out[:, :, 0] = np.clip(val * 1.05, 0, 255)                          # a touch cool
    out[:, :, 1] = val
    out[:, :, 2] = np.clip(val * 0.95, 0, 255)
    out[:, :, 3] = layer[:, :, 3]
    return out


def _stadium(k, t, ctx):
    c, T, dur = ctx["card"], ctx["T"], ctx["dur"]
    T1, T2, T3, T4 = T
    frame = ctx["base"].copy()
    hz = ctx["haze"]
    x0 = int(t * 22) % (hz.shape[1] - QW)
    haze = hz[:, x0:x0 + QW]
    beams = np.zeros((QH, QW), np.float32)
    lamps = np.zeros((H, W, 3), np.float32)
    flare = np.zeros((H, W, 3), np.float32)
    for b in ctx["banks"]:
        d = t - b["t"]
        if d < 0:
            continue
        on = min(1.0, d / 0.04) * (1.0 if d > 0.25 or int(d * 40) % 3 else 0.55)   # bang on with a flicker
        surge = 1 + 1.6 * np.exp(-d / 0.18)
        sway = 0.04 * np.sin(t * 0.8 + b["ph"])
        beams += b["mask"](sway) * on * surge
        for (lx, ly) in b["lamps"]:
            cv2.circle(lamps, (int(lx), int(ly)), 9, (255 * on, 250 * on, 235 * on), -1, cv2.LINE_AA)
        if d < 0.9:                                                          # anamorphic flare as it hits
            a = np.exp(-d / 0.3)
            cx, cy = b["c"]
            cv2.line(flare, (0, int(cy)), (W, int(cy)), (255 * a, 200 * a, 140 * a), 3)
            for f, rad, col in ((0.6, 40, (120, 200, 255)), (1.2, 70, (255, 160, 120)), (1.6, 25, (200, 255, 200))):
                gx, gy = cx + (W / 2 - cx) * f, cy + (H / 2 - cy) * f
                cv2.circle(flare, (int(gx), int(gy)), rad, tuple(v * a * 0.35 for v in col), -1, cv2.LINE_AA)
    beams *= 0.55 + 0.45 * haze
    frame += _up(cv2.GaussianBlur(beams, (0, 0), 3))[..., None] * np.array([150, 160, 165], np.float32)
    frame += _up(cv2.GaussianBlur(haze, (0, 0), 4) * (0.15 + 0.25 * min(1.0, max(0.0, t - T1))))[..., None] * 40
    # ---- chrome number drops in at T1 with a glint sweep, then recedes behind the bar
    if c.num is not None and t >= T1 - 0.05:
        u = ease_out((t - (T1 - 0.05)) / 0.35)
        ghost = smooth((t - (T2 - 0.1)) / 0.4)
        s = 1.25 - 0.25 * u + 0.3 * ghost
        _paste(frame, ctx["num_sh"], c.num_c[0] + 18, c.num_c[1] + 24, s, 0.75 * u * (1 - ghost))
        _paste(frame, ctx["num_chrome"], *c.num_c, s, u * (1 - 0.75 * ghost))
        g = (t - T1) / 0.7
        if 0 <= g < 1:
            L = ctx["num_chrome"]
            h, w = L.shape[:2]
            xs = np.arange(w)[None, :] - (g * (w + 500) - 250) - np.arange(h)[:, None] * 0.5
            sheen = L.copy()
            sheen[:, :, :3] = 255
            sheen[:, :, 3] = (L[:, :, 3].astype(np.float32) * np.exp(-(xs / 45.0) ** 2) * 0.9).astype(np.uint8)
            _paste(frame, sheen, *c.num_c, s, 1.0, add=True)
    # ---- broadcast bar slides in from the left at T2 with the name on it
    if t >= T2 - 0.15:
        u = ease_out((t - (T2 - 0.15)) / 0.35)
        bx0, bx1 = c.tx - 620, c.tx + 620
        right = bx0 + (bx1 - bx0) * u
        y0, y1 = c.y_first - 95, c.y_last + c.last_size // 2 + 10
        pts = np.array([[bx0, y0], [right + 60, y0], [right, y1], [bx0 - 60, y1]], np.int32)
        pan = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(pan, pts, 255, cv2.LINE_AA)
        a = pan.astype(np.float32)[..., None] / 255 * 0.82
        frame = frame * (1 - a) + np.array(c.team, np.float32) * 0.8 * a
        cv2.line(frame, (int(bx0 - 60), y1 + 6), (int(right), y1 + 6), tuple(float(v) for v in c.accent), 8, cv2.LINE_AA)
        if u > 0.6:
            uu = ease_out((t - (T2 + 0.05)) / 0.3)
            for L, y in ((c.first, c.y_first), (c.last, c.y_last)):
                if L is not None:
                    _paste(frame, ctx["sh"][id(L)], c.tx + 8 - 120 * (1 - uu), y + 10, 1.0, 0.6 * uu)
                    _paste(frame, L, c.tx - 120 * (1 - uu), y, 1.0, uu)
    c.bars(frame, t, T3, ((255, 255, 255), c.accent))
    c.details(frame, t, T3, T4, glow_col=(255, 240, 220))
    glow = _bloom(lamps + flare, 3, 2.2)
    frame += glow
    frame = _apply_shake(frame, t, ctx["hits"], 14)
    tail = (t - (dur - 0.22)) / 0.22
    if tail > 0:                                                          # flare whiteout
        frame = frame * (1 - 0.9 * tail) + 255 * 0.9 * tail
    return frame


def _stadium_setup(ctx, rng):
    c = ctx["card"]
    T1, T2, T3, T4 = ctx["T"]
    ctx["base"] = background_frame(ctx["bg"], ctx["pl"]) * 0.18
    ctx["haze"] = _noise(rng, QH, QW * 3, 10) * 0.8 + _noise(rng, QH, QW * 3, 3) * 0.2
    if c.num is not None:
        ctx["num_chrome"] = _chrome(c.num)
        ctx["num_sh"] = _shadow(c.num, 14)
    ctx["sh"] = {id(L): _shadow(L, 7) for L in (c.first, c.last) if L is not None}
    banks = []
    yy, xx = np.mgrid[0:QH, 0:QW].astype(np.float32)
    for (bx, t0, ph) in ((W * 0.18, T1 - 0.02, 0.0), (W * 0.82, T1 + (T2 - T1) * 0.5, 1.3), (W * 0.5, T2 - 0.02, 2.4)):
        by = 70.0
        lamps = [(bx + (i - 1.5) * 34, by + (j - 1) * 30) for i in range(4) for j in range(3)]

        def mask(sway, bx=bx, by=by):
            # a cone from the bank down onto the field, brightest near the source
            qx, qy = bx / 4, by / 4
            ang = np.arctan2(xx - qx, yy - qy + 1e-3) - sway * (1 if bx < W / 2 else -1)
            centre = np.arctan2((W / 2 - bx) * 0.6, H) if bx != W / 2 else 0.0
            width = 0.22
            cone = np.clip(1 - np.abs(ang - centre) / width, 0, 1) ** 1.5
            fall = np.clip(1 - (yy - qy) / QH, 0, 1) ** 0.8 * (yy > qy)
            return cone * fall * 1.4

        banks.append({"t": t0, "c": (bx, by), "lamps": lamps, "mask": mask, "ph": ph})
    ctx["banks"] = banks
    ctx["hits"] = [(T1, 0.6), (T2, 0.5), (T3, 0.15), (T4, 0.25)]


# ================================================================ render
_STYLE = {"inferno": (_inferno_setup, _inferno), "storm": (_storm_setup, _storm),
          "shatter": (_shatter_setup, _shatter), "stadium": (_stadium_setup, _stadium)}


def render(style: str, out: Path, lib_root: Path, pl: dict, grid: dict | None, music_start: float,
           bg_src: np.ndarray | None, sfx: bool = False, seed: int = 7) -> tuple[Path, float]:
    """Render one of the styles to out (mp4). Returns (path, duration)."""
    setup, draw = _STYLE[style]
    rng = np.random.default_rng(seed)
    T, dur = beat_times(grid, music_start)
    n_frames = int(round(dur * FPS))
    card = Card(lib_root, pl)
    ctx = {"card": card, "T": T, "dur": dur, "rng": rng, "bg": bg_src, "pl": pl,
           "hits": [(T[0], 1.0), (T[1], 0.8), (T[2], 0.25), (T[3], 0.35)]}
    setup(ctx, rng)
    vig = _vignette()
    grain = [rng.normal(0, 4.5, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
    enc, wav = _encoder(out, dur, [(T[0], 1.0), (T[1], 0.85), (T[2], 0.35), (T[3], 0.5)], sfx, rng)
    try:
        for k in range(n_frames):
            t = k / FPS
            frame = draw(k, t, ctx)
            frame *= min(1.0, 0.3 + t / 0.35) if t < T[0] else 1.0
            enc.stdin.write(_finish(frame, k, grain, vig).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
        wav.unlink(missing_ok=True)
    if enc.returncode != 0:
        raise RuntimeError(f"{style} intro render failed")
    return out, n_frames / FPS


# the second set of styles (glitch, smoke, neon, warp, gold) lives in intro_fx2
from . import intro_fx2 as _fx2  # noqa: E402

_STYLE.update(_fx2.STYLES)
STYLES = STYLES + tuple(_fx2.STYLES)
