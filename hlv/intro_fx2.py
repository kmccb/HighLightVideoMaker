"""Five more intros (same beats T1-T4 and layout as the others, see intro_fx.Card):

  glitch   digital corruption: scanlines, data blocks, RGB split; the number tears in, the
           name decodes out of scrambled characters
  smoke    team-colour smoke bombs billow in; the number punches a hole through the smoke,
           the name slams in as it swirls
  neon     a dark brick wall; the number and name are neon tubes that buzz and flicker on,
           light spilling onto the wall
  warp     hyperspace: light streaks rush out of the centre, the number blasts out of it
           with a zoom blur and a shockwave, the name follows
  gold     black velvet; gold metal lettering revealed by a sweeping light, sparkling
           glints and falling gold dust
"""
from __future__ import annotations

import string

import cv2
import numpy as np

from .intro import (FPS, H, W, Sparks, _glow, _outline, _paste, _paste_blur, _shadow, _text_layer,
                    background_frame, ease_in, ease_out, ease_out_back, smooth)
from .intro_fx import QH, QW, Card, _apply_shake, _bloom, _chroma, _noise, _up

GLYPHS = string.ascii_uppercase + string.digits + "#@%&$/<>"


def _ghost_num(frame, c, ctx, t, T1, T2, layer, glow=None, glow_a=0.5):
    """The number after it lands: grows slightly and fades back behind the name (all styles)."""
    g = smooth((t - (T2 - 0.1)) / 0.4)
    s = 1.0 + 0.35 * g
    if glow is not None:
        _paste(frame, glow, *c.num_c, s, glow_a * (1 - 0.6 * g), add=True)
    _paste(frame, ctx["num_sh"], c.num_c[0] + 14, c.num_c[1] + 18, s, 0.6 * (1 - g))
    _paste(frame, layer, *c.num_c, s, 1 - 0.78 * g)


def _names(frame, c, ctx, t, T2, how="slam"):
    """First + last name entrances: 'slam' (fly in from both sides), 'rise', 'zoom'."""
    if t < T2 - 0.2:
        return
    for L, y, d in ((c.first, c.y_first, 0.0), (c.last, c.y_last, 0.05)):
        if L is None:
            continue
        sh = ctx["sh"][id(L)]
        if how == "slam":
            side = -1 if L is c.first else 1
            dist = 900 if L is c.first else 1100
            if t < T2:
                u = ease_in((t - (T2 - 0.2)) / 0.2)
                up = ease_in(((t - 1 / FPS) - (T2 - 0.2)) / 0.2)
                _paste_blur(frame, L, (c.tx + side * dist * (1 - up), y), (c.tx + side * dist * (1 - u), y), 1, 1, u)
            else:
                pop = 1 + 0.05 * (1 - ease_out_back((t - T2) / 0.2, 2.5))
                _paste(frame, sh, c.tx + 9, y + 12, pop, 0.75)
                _paste(frame, L, c.tx, y, pop)
        elif how == "zoom":
            u = ease_out((t - (T2 - 0.12) - d) / 0.3)
            if u <= 0:
                continue
            s = 2.4 - 1.4 * u
            _paste(frame, sh, c.tx + 9, y + 12, s, 0.75 * u)
            _paste(frame, L, c.tx, y, s, min(1, u * 1.6))
        else:                                                      # rise
            u = ease_out((t - (T2 - 0.2) - d) / 0.45)
            if u <= 0:
                continue
            _paste(frame, sh, c.tx + 9, y + 12 + 90 * (1 - u), 1.0, 0.7 * u)
            _paste(frame, L, c.tx, y + 90 * (1 - u), 1.0, u)


def _common(ctx):
    c = ctx["card"]
    ctx["sh"] = {id(L): _shadow(L, 9) for L in (c.first, c.last) if L is not None}
    if c.num is not None:
        ctx["num_sh"] = _shadow(c.num, 12)


def _tail_flash(frame, t, dur, col=(255, 255, 255), length=0.18):
    u = (t - (dur - length)) / length
    if u > 0:
        frame = frame * (1 - 0.88 * u) + np.array(col, np.float32) * 0.88 * u
    return frame


# ================================================================ GLITCH
def _glitch_setup(ctx, rng):
    _common(ctx)
    c = ctx["card"]
    ctx["base"] = background_frame(ctx["bg"], ctx["pl"]) * 0.28
    lines = np.ones((H, 1, 1), np.float32)
    lines[::3] = 0.72
    ctx["scan"] = lines
    ctx["cyan"], ctx["mag"] = (255, 230, 40), (255, 40, 255)
    if c.num is not None:
        ctx["num_glow"] = _glow(c.num, ctx["cyan"], 26)
    # bursts of corruption: on the beats plus a few random ones
    T1, T2, T3, T4 = ctx["T"]
    ctx["bursts"] = sorted([T1, T2, T3, T4] + list(rng.uniform(T1 + 0.3, ctx["dur"] - 0.4, 3)))


def _glitch_amount(t, bursts):
    a = 0.0
    for b in bursts:
        if 0 <= t - b < 0.35:
            a = max(a, 1 - (t - b) / 0.35)
    return a


def _corrupt(frame, rng, amt):
    """Tear horizontal slices sideways and stamp data blocks; amt 0..1."""
    if amt <= 0.02:
        return frame
    out = frame.copy()
    for _ in range(int(3 + 14 * amt)):
        y0 = int(rng.integers(0, H - 10))
        h = int(rng.integers(4, 10 + int(70 * amt)))
        dx = int(rng.normal(0, 90 * amt))
        out[y0:y0 + h] = np.roll(frame[y0:y0 + h], dx, axis=1)
    for _ in range(int(2 + 10 * amt)):
        x, y = int(rng.integers(0, W - 60)), int(rng.integers(0, H - 20))
        w, h = int(rng.integers(20, 220)), int(rng.integers(4, 26))
        col = rng.choice([0, 1, 2])
        out[y:y + h, x:x + w, col] = np.clip(out[y:y + h, x:x + w, col] + 140 * amt, 0, 255)
    return _chroma(out, int(4 + 26 * amt))


def _glitch(k, t, ctx):
    c, T, dur, rng = ctx["card"], ctx["T"], ctx["dur"], ctx["rng"]
    T1, T2, T3, T4 = T
    frame = ctx["base"].copy()
    # digital noise snow before the number lands
    if t < T1:
        snow = rng.random((QH, QW)).astype(np.float32) * (40 * (1 - t / max(T1, 0.1)))
        frame += _up(snow)[..., None]
    if c.num is not None and t >= T1 - 0.05:
        if t < T1 + 0.35 and rng.random() < 0.35:                       # flickers in
            pass
        else:
            _ghost_num(frame, c, ctx, t, T1, T2, c.num, ctx["num_glow"], 0.55)
    # names decode: scrambled characters that lock left to right
    if t >= T2 - 0.05:
        u = np.clip((t - (T2 - 0.05)) / 0.55, 0, 1)
        for text, size, y, key in ((ctx["first_txt"], 130, c.y_first, "f"), (ctx["last_txt"], c.last_size, c.y_last, "l")):
            if not text:
                continue
            n_lock = int(len(text) * u + 0.0001)
            shown = "".join(ch if (i < n_lock or ch == " ") else rng.choice(list(GLYPHS)) for i, ch in enumerate(text))
            if u >= 1:
                L = c.first if key == "f" else c.last
            else:
                L = _text_layer(shown, size, (255, 255, 255, 255), stroke=4 if key == "l" else 0,
                                stroke_fill=tuple(c.team[::-1]), spacing=16 if key == "f" else 0)
            if L is not None:
                _paste(frame, _shadow(L, 8) if u < 1 else ctx["sh"][id(L)], c.tx + 9, y + 12, 1.0, 0.7)
                _paste(frame, L, c.tx, y, 1.0)
    c.bars(frame, t, T3, (ctx["cyan"], ctx["mag"]))
    c.details(frame, t, T3, T4, glow_col=ctx["cyan"])
    frame *= ctx["scan"]
    frame = _corrupt(frame, rng, max(_glitch_amount(t, ctx["bursts"]), 1.0 if t > dur - 0.25 else 0.0))
    frame = _apply_shake(frame, t, ctx["hits"], 16)
    u = (t - (dur - 0.2)) / 0.2
    if u > 0:                                                            # dissolve into white blocks
        m = (rng.random((H // 40 + 1, W // 40 + 1)) < u).astype(np.float32)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        frame = frame * (1 - m) + 255 * m
    return frame


def _glitch_setup_text(ctx, rng):
    _glitch_setup(ctx, rng)
    parts = (ctx["pl"].get("name") or "").upper().split()
    ctx["first_txt"], ctx["last_txt"] = (" ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else ("", parts[0] if parts else "")


# ================================================================ SMOKE
def _smoke_setup(ctx, rng):
    _common(ctx)
    c = ctx["card"]
    ctx["base"] = background_frame(ctx["bg"], ctx["pl"]) * 0.22
    ctx["s1"] = _noise(rng, QH * 2, QW * 2, 10)
    ctx["s2"] = _noise(rng, QH * 2, QW * 2, 5)
    team = np.array(c.team, np.float32)
    ctx["col_a"] = np.clip(team / max(team.max(), 1) * 230 + 25, 0, 255)          # bright team colour
    acc = np.array(c.accent, np.float32)
    ctx["col_b"] = acc if acc.sum() < 600 else np.array([200, 200, 210], np.float32)
    yy, xx = np.mgrid[0:QH, 0:QW].astype(np.float32)
    ctx["qxx"], ctx["qyy"] = xx, yy
    if c.num is not None:
        ctx["num_glow"] = _glow(c.num, tuple(int(v) for v in ctx["col_a"]), 30)
    ctx["sparks"] = [Sparks(rng, *c.num_c, ctx["T"][0], tuple(int(v) for v in ctx["col_a"]), n=80)]


def _smoke(k, t, ctx):
    c, T, dur = ctx["card"], ctx["T"], ctx["dur"]
    T1, T2, T3, T4 = T
    xx, yy = ctx["qxx"], ctx["qyy"]
    # two smoke bombs from the lower corners, billowing up and in
    grow = smooth(t / max(T1 + 0.4, 0.5))
    dens = np.zeros((QH, QW), np.float32)
    for (sx, col) in ((0.05, "col_a"), (0.95, "col_b")):
        cx, cy = sx * QW, QH * 1.05
        r = np.hypot((xx - cx) / (QW * (0.35 + 0.55 * grow)), (yy - cy) / (QH * (0.6 + 0.6 * grow)))
        plume = np.clip(1.15 - r, 0, 1) ** 1.3
        dens_c = plume
        ctx.setdefault("_d", {})[col] = dens_c
    ox, oy = int(t * 18) % QW, int(t * 30) % QH
    tex = 0.6 * ctx["s1"][oy:oy + QH, ox:ox + QW] + 0.4 * ctx["s2"][(oy * 2) % QH:(oy * 2) % QH + QH, ox:ox + QW]
    frame = ctx["base"].copy()
    smoke_rgb = np.zeros((QH, QW, 3), np.float32)
    total = np.zeros((QH, QW), np.float32)
    for col in ("col_a", "col_b"):
        d = np.clip(ctx["_d"][col] * (0.4 + 1.1 * tex) - 0.15, 0, 1)
        # the number punches a hole through the smoke at T1
        if t >= T1:
            rr = np.hypot(xx - c.num_c[0] / 4, (yy - c.num_c[1] / 4) * 1.3) / (QW * (0.05 + 0.3 * ease_out((t - T1) / 0.6)))
            d *= np.clip(rr, 0, 1) ** 0.6 * 0.6 + 0.4
        smoke_rgb += d[..., None] * ctx[col]
        total += d
    smoke_rgb = cv2.GaussianBlur(smoke_rgb, (0, 0), 2)
    total = cv2.GaussianBlur(np.clip(total, 0, 1), (0, 0), 2)
    frame = frame * (1 - 0.8 * _up(total)[..., None]) + _up(smoke_rgb) * 0.85
    fx = np.zeros((H, W, 3), np.float32)
    for sp in ctx["sparks"]:
        sp.draw(fx, t)
    if c.num is not None and t >= T1 - 0.18:
        if t < T1:
            u = ease_in((t - (T1 - 0.18)) / 0.18)
            _paste_blur(frame, c.num, c.num_c, c.num_c, 0.3 + 0.7 * u, 0.3 + 0.7 * u, u, n=6)
        else:
            _ghost_num(frame, c, ctx, t, T1, T2, c.num, ctx["num_glow"], 0.5 * np.exp(-(t - T1) / 0.5) + 0.15)
    _names(frame, c, ctx, t, T2, "slam")
    c.bars(frame, t, T3, (tuple(int(v) for v in ctx["col_a"]), c.accent))
    c.details(frame, t, T3, T4, glow_col=tuple(int(v) for v in ctx["col_a"]))
    frame = frame + _bloom(fx, 3, 1.0)
    for t0, a in ((T1, 0.6), (T2, 0.4)):
        if 0 <= t - t0 < 0.12:
            f = a * (1 - (t - t0) / 0.12)
            frame = frame * (1 - f) + 255 * f
    frame = _apply_shake(frame, t, ctx["hits"], 26)
    return _tail_flash(frame, t, dur)


# ================================================================ NEON
def _brick_wall(rng) -> np.ndarray:
    img = np.zeros((H, W, 3), np.float32)
    img[:] = (34, 34, 52)
    bw, bh = 120, 46
    for r, y in enumerate(range(0, H + bh, bh)):
        off = (r % 2) * bw // 2
        for x in range(-bw, W + bw, bw):
            shade = rng.uniform(0.75, 1.15)
            col = tuple(float(v * shade) for v in (40, 46, 78))
            cv2.rectangle(img, (x + off + 3, y + 3), (x + off + bw - 3, y + bh - 3), col, -1)
    img += rng.normal(0, 6, (H, W, 1)).astype(np.float32)
    return cv2.GaussianBlur(img, (0, 0), 1.0) * 0.55


def _neon_setup(ctx, rng):
    _common(ctx)
    c = ctx["card"]
    ctx["wall"] = _brick_wall(rng)
    team = np.array(c.team, np.float32)
    tube_a = tuple(int(v) for v in np.clip(team / max(team.max(), 1) * 255, 60, 255))     # team colour, full
    acc = np.array(c.accent, np.float32)
    tube_b = tuple(int(v) for v in acc) if acc.sum() < 650 else (255, 120, 255)        # pink if accent is pale
    ctx["tube_a"], ctx["tube_b"] = tube_a, tube_b

    def tube(L, col):
        core = _outline(L, (255, 255, 255), 3)
        rim = _outline(L, col, 7)
        return rim, core, _glow(rim, col, 16), _glow(rim, col, 60)
    ctx["num_t"] = tube(c.num, tube_a) if c.num is not None else None
    ctx["first_t"] = tube(c.first, tube_b) if c.first is not None else None
    ctx["last_t"] = tube(c.last, tube_b) if c.last is not None else None
    T1, T2 = ctx["T"][0], ctx["T"][1]
    # on/off pattern while each sign warms up (and a stutter later)
    ctx["flick"] = {key: (t0, rng.random(40) < 0.55) for key, t0 in (("num", T1), ("first", T2), ("last", T2 + 0.08))}
    ctx["stutter"] = float(rng.uniform(T2 + 1.0, ctx["dur"] - 0.6))


def _neon_on(ctx, key, t):
    t0, pat = ctx["flick"][key]
    d = t - t0
    if d < 0:
        return 0.0
    if d < 0.4:
        return 1.0 if pat[int(d * 60) % len(pat)] else 0.08
    if 0 <= t - ctx["stutter"] < 0.12 and key == "last":
        return 0.15
    return 0.92 + 0.08 * np.sin(t * 120)                                  # mains hum


def _neon(k, t, ctx):
    c, T, dur = ctx["card"], ctx["T"], ctx["dur"]
    T1, T2, T3, T4 = T
    frame = ctx["wall"].copy()
    spill = np.zeros((H, W, 3), np.float32)
    for key, tubes, pos, s in (("num", ctx["num_t"], c.num_c, 1.25), ("first", ctx["first_t"], (c.tx, c.y_first), 1.0),
                               ("last", ctx["last_t"], (c.tx, c.y_last), 1.0)):
        if tubes is None:
            continue
        on = _neon_on(ctx, key, t)
        if key == "num":                                   # the number dims once the name is lit
            on *= 1 - 0.72 * smooth((t - (T2 - 0.05)) / 0.35)
        if on <= 0:
            continue
        rim, core, g1, g2 = tubes
        _paste(spill, g2, *pos, s, 0.9 * on, add=True)                      # light on the wall
        _paste(frame, g1, *pos, s, 1.0 * on, add=True)
        _paste(frame, rim, *pos, s, on, add=True)
        _paste(frame, core, *pos, s, 0.85 * on, add=True)
    frame = frame + spill * 0.8
    c.bars(frame, t, T3, (ctx["tube_a"], ctx["tube_b"]))
    c.details(frame, t, T3, T4, glow_col=ctx["tube_a"])
    u = (t - (dur - 0.22)) / 0.22
    if u > 0:                                                              # power surge to white
        frame = frame * (1 + 2.5 * u) + 255 * u * 0.6
    return frame


# ================================================================ WARP
def _warp_setup(ctx, rng):
    _common(ctx)
    c = ctx["card"]
    ctx["base"] = background_frame(ctx["bg"], ctx["pl"]) * 0.15
    n = 420
    ctx["stars"] = np.stack([rng.uniform(0, 2 * np.pi, n), rng.uniform(0.02, 1.0, n), rng.uniform(0.6, 1.4, n)], 1)
    team = np.array(c.team, np.float32)
    ctx["tint"] = tuple(float(v) for v in np.clip(team / max(team.max(), 1) * 255 * 0.7 + 80, 0, 255))
    if c.num is not None:
        ctx["num_glow"] = _glow(c.num, tuple(int(v) for v in ctx["tint"]), 34)
    ctx["sparks"] = [Sparks(rng, *c.num_c, ctx["T"][0], tuple(int(v) for v in ctx["tint"]), n=110, spread=(800, 2400))]


def _warp(k, t, ctx):
    c, T, dur = ctx["card"], ctx["T"], ctx["dur"]
    T1, T2, T3, T4 = T
    frame = ctx["base"].copy()
    fx = np.zeros((H, W, 3), np.float32)
    # speed: builds up to the jump at T1, cruising after, surging again at the end
    v = 0.15 + 2.6 * ease_in(min(1, t / max(T1, 0.1))) if t < T1 else 0.9 + 1.6 * np.exp(-(t - T1) / 0.4)
    if t > dur - 0.5:
        v += 4 * (t - (dur - 0.5)) / 0.5
    cx, cy = W / 2, H / 2
    R = np.hypot(W, H) / 2
    for ang, r0, sp in ctx["stars"]:
        r = ((r0 + t * 0.35 * sp * (1 + v)) % 1.0) ** 1.6 * R
        ln = min(R, r * (0.04 + 0.22 * v))
        x0, y0 = cx + np.cos(ang) * r, cy + np.sin(ang) * r
        x1, y1 = cx + np.cos(ang) * (r - ln), cy + np.sin(ang) * (r - ln)
        b = min(1.0, r / R * 1.8)
        col = tuple(float(ch) * b for ch in ctx["tint"])
        cv2.line(fx, (int(x1), int(y1)), (int(x0), int(y0)), col, 2, cv2.LINE_AA)
    for sp in ctx["sparks"]:
        sp.draw(fx, t)
    if 0 <= t - T1 < 0.5:                                                   # shockwave
        u = (t - T1) / 0.5
        cv2.circle(fx, (int(c.num_c[0]), int(c.num_c[1])), int(80 + 1500 * ease_out(u)),
                   tuple(float(ch) * (1 - u) for ch in ctx["tint"]), max(2, int(30 * (1 - u))), cv2.LINE_AA)
    frame = frame + _bloom(fx, 4, 1.3)
    if c.num is not None and t >= T1 - 0.25:
        if t < T1:                                                          # blasts out of the centre
            u = ease_in((t - (T1 - 0.25)) / 0.25)
            p0 = (cx + (c.num_c[0] - cx) * 0, cy + (c.num_c[1] - cy) * 0)
            _paste_blur(frame, c.num, p0, c.num_c, 0.05, 0.05 + 0.95 * u, u, n=10)
        else:
            _ghost_num(frame, c, ctx, t, T1, T2, c.num, ctx["num_glow"], 0.6 * np.exp(-(t - T1) / 0.4) + 0.15)
    _names(frame, c, ctx, t, T2, "zoom")
    c.bars(frame, t, T3, (tuple(int(v) for v in ctx["tint"]), c.accent))
    c.details(frame, t, T3, T4, glow_col=tuple(int(v) for v in ctx["tint"]))
    for t0, a in ((T1, 0.85), (T2, 0.35)):
        if 0 <= t - t0 < 0.12:
            f = a * (1 - (t - t0) / 0.12)
            frame = frame * (1 - f) + 255 * f
    frame = _apply_shake(frame, t, ctx["hits"], 24)
    return _tail_flash(frame, t, dur, length=0.2)


# ================================================================ GOLD
def _gold(layer: np.ndarray) -> np.ndarray:
    """Gold metal: banded warm gradient (deep bronze -> gold -> pale highlight) + bevel light."""
    a = layer[:, :, 3].astype(np.float32) / 255
    h, w = a.shape
    v = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    band = 0.5 + 0.5 * np.cos(v * np.pi * 2.6 + 0.9)
    blur = cv2.GaussianBlur(a, (0, 0), 5)
    gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    light = np.clip(-(gx + gy) * 6, -1, 1)
    lum = np.clip(0.35 + 0.55 * band + 0.35 * light, 0, 1.2)
    dark, mid, hi = np.array([20, 70, 120]), np.array([40, 175, 235]), np.array([190, 245, 255])
    rgb = np.where(lum[..., None] < 0.6, dark + (mid - dark) * (lum[..., None] / 0.6),
                   mid + (hi - mid) * np.clip((lum[..., None] - 0.6) / 0.6, 0, 1))
    out = np.zeros((h, w, 4), np.uint8)
    out[:, :, :3] = np.clip(rgb, 0, 255)
    out[:, :, 3] = layer[:, :, 3]
    return out


def _star(size: int) -> np.ndarray:
    """A four-point sparkle (BGRA)."""
    s = size * 2 + 1
    img = np.zeros((s, s), np.float32)
    cv2.line(img, (0, size), (s - 1, size), 1.0, 2, cv2.LINE_AA)
    cv2.line(img, (size, 0), (size, s - 1), 1.0, 2, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + cv2.GaussianBlur(img, (0, 0), size / 4) * 0.6
    cv2.circle(img, (size, size), 3, 1.0, -1, cv2.LINE_AA)
    out = np.zeros((s, s, 4), np.uint8)
    out[:, :, :3] = (220, 245, 255)
    out[:, :, 3] = np.clip(img * 255, 0, 255)
    return out


def _gold_setup(ctx, rng):
    _common(ctx)
    c = ctx["card"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - W / 2) / W, (yy - H / 2) / H)
    ctx["base"] = (np.clip(0.9 - r * 1.4, 0.05, 1) ** 1.5)[..., None] * np.array([14, 16, 22], np.float32) \
        + background_frame(ctx["bg"], ctx["pl"]) * 0.08
    ctx["num_g"] = _gold(c.num) if c.num is not None else None
    ctx["first_g"] = _gold(c.first) if c.first is not None else None
    ctx["last_g"] = _gold(c.last) if c.last is not None else None
    ctx["gold_glow"] = (40, 160, 230)
    if ctx["num_g"] is not None:
        ctx["num_glow"] = _glow(ctx["num_g"], ctx["gold_glow"], 30)
    ctx["star"] = _star(46)
    n = 120
    ctx["dust"] = np.stack([rng.uniform(0, W, n), rng.uniform(-H, H, n), rng.uniform(40, 140, n),
                            rng.uniform(1.5, 5, n), rng.uniform(0, 6.28, n)], 1)
    # glints: (time, layer key, relative x, relative y)
    ctx["glints"] = [(float(rng.uniform(ctx["T"][0] + 0.2, ctx["dur"] - 0.3)), rng.choice(["num", "last", "first"]),
                      float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.6))) for _ in range(9)]


def _sweep(L, u, width=70.0):
    """A diagonal band of light crossing a layer, u 0..1."""
    h, w = L.shape[:2]
    xs = np.arange(w)[None, :] - (u * (w + 500) - 250) - np.arange(h)[:, None] * 0.5
    sh = L.copy()
    sh[:, :, :3] = (215, 245, 255)
    sh[:, :, 3] = (L[:, :, 3].astype(np.float32) * np.exp(-(xs / width) ** 2) * 0.95).astype(np.uint8)
    return sh


def _gold_fx(k, t, ctx):
    c, T, dur = ctx["card"], ctx["T"], ctx["dur"]
    T1, T2, T3, T4 = T
    frame = ctx["base"].copy()
    fx = np.zeros((H, W, 3), np.float32)
    for x, y0, vs, sz, ph in ctx["dust"]:                                   # falling gold dust (bokeh)
        y = (y0 + vs * t) % (H * 2) - H * 0.5
        a = 0.35 + 0.35 * np.sin(t * 3 + ph)
        cv2.circle(fx, (int(x + 20 * np.sin(t + ph)), int(y)), int(sz), (40 * a, 150 * a, 220 * a), -1, cv2.LINE_AA)
    frame = frame + cv2.GaussianBlur(fx, (0, 0), 2) * 0.9
    layers = {"num": (ctx["num_g"], c.num_c, T1), "first": (ctx["first_g"], (c.tx, c.y_first), T2),
              "last": (ctx["last_g"], (c.tx, c.y_last), T2 + 0.08)}
    g_num = smooth((t - (T2 - 0.1)) / 0.4)
    for key, (L, pos, t0) in layers.items():
        if L is None or t < t0 - 0.3:
            continue
        u = ease_out((t - (t0 - 0.3)) / 0.5)
        if key == "num":
            s = 1.0 + 0.35 * g_num + 0.15 * (1 - u)
            alpha = u * (1 - 0.75 * g_num)
            _paste(frame, ctx["num_glow"], *pos, s, 0.4 * alpha, add=True)
            _paste(frame, ctx["num_sh"], pos[0] + 14, pos[1] + 18, s, 0.6 * alpha)
        else:
            s, alpha = 1.0 + 0.08 * (1 - u), u
            _paste(frame, ctx["sh"][id(c.first if key == "first" else c.last)], pos[0] + 9, pos[1] + 12, s, 0.7 * alpha)
        _paste(frame, L, *pos, s, alpha)
        sw = (t - t0) / 0.8
        if 0 <= sw < 1:
            _paste(frame, _sweep(L, sw), *pos, s, 1.0, add=True)
        for gt, gk, rx, ry in ctx["glints"]:
            if gk == key and 0 <= t - gt < 0.35:
                gu = (t - gt) / 0.35
                h, w = L.shape[:2]
                _paste(frame, ctx["star"], pos[0] + (rx - 0.5) * w * s, pos[1] + (ry - 0.5) * h * s,
                       0.4 + 0.8 * np.sin(np.pi * gu), np.sin(np.pi * gu), add=True)
    if 0 <= t - T1 < 0.9:                                                    # anamorphic flare as the number lands
        a = np.exp(-(t - T1) / 0.3)
        cv2.line(frame, (0, int(c.num_c[1])), (W, int(c.num_c[1])), (60 * a, 170 * a, 255 * a), 3)
    c.bars(frame, t, T3, (ctx["gold_glow"], (190, 245, 255)))
    c.details(frame, t, T3, T4, glow_col=ctx["gold_glow"])
    frame = frame + _bloom(np.clip(frame - 200, 0, 255), 4, 0.6)
    return _tail_flash(frame, t, dur, col=(200, 240, 255), length=0.22)


STYLES = {"glitch": (_glitch_setup_text, _glitch), "smoke": (_smoke_setup, _smoke), "neon": (_neon_setup, _neon),
          "warp": (_warp_setup, _warp), "gold": (_gold_setup, _gold_fx)}
