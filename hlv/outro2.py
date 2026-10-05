"""Three more ScottForge endings, using the same pieces as outro.py's "forge" (the anvil mark,
Scott + italic Forge wordmark in Fraunces, label, website, invite):

  anvil    darkness; the anvil mark slams down, a hammer strike throws a huge shower of sparks
           and their light reveals the wordmark, which then glows on
  molten   glowing molten metal pours up into the letters (a bright, wavy fill line that cools
           behind it), dripping sparks, heat shimmer
  minimal  clean and quiet: a hairline draws across, the mark and wordmark fade up while the
           letter spacing settles, the website types itself out
"""
from __future__ import annotations

import cv2
import numpy as np

from .intro import Sparks, _clamp, _glow, _paste, ease_out, ease_out_back, smooth

SPARK_YEL = (87, 200, 255)


def make(style: str, v: dict):
    return {"anvil": _anvil, "molten": _molten, "minimal": _minimal}[style](v)


def _text_block(frame, v, t, T_LABEL, T_URL, T_INV, url_typed=False):
    W = v["W"]
    if v["L_label"] is not None and t > T_LABEL:
        u = (t - T_LABEL) / 0.5
        _paste(frame, v["L_label"], W / 2 + 34, v["y_label"], 1.0, smooth(u))
        lw = int(52 * ease_out(u))
        lx = int(W / 2 + 34 - v["L_label"].shape[1] / 2 - 20)
        cv2.line(frame, (lx - lw, v["y_label"]), (lx, v["y_label"]), tuple(float(c) * smooth(u) for c in v["ember"]), 2, cv2.LINE_AA)
    if v["L_url"] is not None and t > T_URL:
        u = (t - T_URL) / 0.5
        L = v["L_url"]
        if url_typed:                                           # reveal left to right, with a caret
            frac = _clamp((t - T_URL) / 0.9)
            w = L.shape[1]
            cut = int(w * frac)
            part = L.copy()
            part[:, cut:, 3] = 0
            _paste(frame, part, W / 2, v["y_url"], 1.0, 1.0)
            if frac < 1 or int(t * 3) % 2 == 0:
                x = int(W / 2 - w / 2 + cut + 8)
                cv2.line(frame, (x, v["y_url"] - 30), (x, v["y_url"] + 30), tuple(float(c) for c in v["ember"]), 4)
        else:
            _paste(frame, v["L_url_glow"], W / 2, v["y_url"], 1.0, 0.35 * smooth(u), add=True)
            _paste(frame, L, W / 2, v["y_url"] + 24 * (1 - ease_out(u)), 1.0, smooth(u))
    if v["L_inv"] is not None and t > T_INV:
        u = (t - T_INV) / 0.6
        _paste(frame, v["L_inv"], W / 2, v["y_inv"] + 20 * (1 - ease_out(u)), 1.0, smooth(u))


def _finish(frame, fx, v, k, t):
    W, H = v["W"], v["H"]
    glow = cv2.resize(cv2.GaussianBlur(cv2.resize(fx, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3), (W, H))
    frame = frame + fx + glow * 1.4
    frame += cv2.resize(v["grain"][k % len(v["grain"])], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    fade = smooth(t / 0.4) * (1 - smooth((t - (v["seconds"] - 0.7)) / 0.7))
    return np.clip(frame * fade, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- anvil strike
def _anvil(v):
    W, H, rng, ember = v["W"], v["H"], v["rng"], v["ember"]
    T_SLAM, T_HIT = 0.45, 0.95
    mc = v["mark_c"]
    shower = [Sparks(rng, mc[0], mc[1] - 30, T_HIT, SPARK_YEL, n=260, spread=(500, 2300), cone=(np.pi * 1.02, np.pi * 1.98)),
              Sparks(rng, mc[0], mc[1] - 30, T_HIT + 0.05, ember, n=160, spread=(300, 1500))]
    yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
    rr = np.hypot(xx - mc[0] / 4, (yy - mc[1] / 4) * 1.2)
    base = np.zeros((H, W, 3), np.float32) + np.array([6, 7, 9], np.float32)

    def frame_at(k):
        t = k / v["FPS"]
        frame = base.copy()
        fx = np.zeros((H, W, 3), np.float32)
        # light thrown by the strike: a warm pool that flares and settles to a glow
        light = 0.0 if t < T_HIT else 1.6 * np.exp(-(t - T_HIT) / 0.35) + 0.35
        pool = np.clip(1 - rr / (W / 4 * (0.25 + 0.5 * min(light, 1))), 0, 1) ** 1.6 * light
        frame += cv2.resize(pool, (W, H))[..., None] * np.array(ember, np.float32) * 0.35
        # mark slams down from above
        if t >= T_SLAM - 0.2:
            u = _clamp((t - (T_SLAM - 0.2)) / 0.2)
            y = mc[1] - 420 * (1 - u ** 2)
            sc = 1.0 + 0.12 * max(0.0, 1 - (t - T_SLAM) / 0.15) if t > T_SLAM else 1.0
            _paste(frame, v["L_anvil_glow"], mc[0], y, sc, 0.4 + light * 0.6, add=True)
            _paste(frame, v["L_anvil"], mc[0], y, sc, min(1, u * 2))
            if t > T_HIT:
                _paste(frame, v["L_dots_glow"], *mc, 1.0, 0.8 + 0.4 * np.sin(t * 9), add=True)
                _paste(frame, v["L_dots"], *mc, 1.0, 1.0)
        for s in shower:
            s.draw(fx, t)
        if 0 <= t - T_HIT < 0.07:                                # the strike flash: a quick hit, centred on the anvil
            f = 1 - (t - T_HIT) / 0.07
            frame = frame + cv2.resize(np.clip(1.2 - rr / (W / 4 * 0.45), 0, 1), (W, H))[..., None] * 255 * f
        # wordmark: lit by the sparks first, then glowing on its own
        reveal = _clamp((t - T_HIT - 0.05) / 0.6)
        for L, c in ((v["L_scott"], v["scott_c"]), (v["L_forge"], v["forge_c"])):
            if reveal > 0:
                _paste(frame, L, *c, 1.0, smooth(reveal))
        if reveal > 0:
            heat = 0.75 + 0.18 * np.sin(t * 5.3)
            _paste(frame, v["L_forge_glow"], *v["forge_c"], 1.0, smooth(reveal) * heat, add=True)
        if 0 <= t - T_SLAM < 0.25:                               # dust ring from the slam
            u = (t - T_SLAM) / 0.25
            cv2.ellipse(fx, (int(mc[0]), int(mc[1] + 80)), (int(60 + 260 * u), int(14 + 40 * u)), 0, 0, 360,
                        tuple(float(c) * (1 - u) * 0.6 for c in ember), 3, cv2.LINE_AA)
        _text_block(frame, v, t, 1.9, 2.3, 2.7)
        return _finish(frame, fx, v, k, t)
    return frame_at


# ---------------------------------------------------------------- molten pour
def _molten(v):
    W, H, rng, ember = v["W"], v["H"], v["rng"], v["ember"]
    T0, FILL = 0.5, 1.4
    hot = np.array([170, 235, 255], np.float32)                  # white-yellow at the fill line
    words = [(v["L_scott"], v["scott_c"], 0.0), (v["L_forge"], v["forge_c"], 0.25)]
    drips = [Sparks(rng, v["forge_c"][0], v["forge_c"][1] + 70, T0 + FILL + 0.25, SPARK_YEL, n=50, spread=(150, 600),
                     cone=(np.pi * 0.25, np.pi * 0.75))]
    base = np.zeros((H, W, 3), np.float32) + np.array([7, 8, 10], np.float32)

    def fill_layer(L, frac, t):
        h, w = L.shape[:2]
        a = L[:, :, 3].astype(np.float32) / 255
        xs = np.arange(w, dtype=np.float32)
        line = h * (1 - frac) + 7 * np.sin(xs / 37 + t * 9) + 4 * np.sin(xs / 13 - t * 14)   # wavy surface
        ys = np.arange(h, dtype=np.float32)[:, None]
        filled = (ys >= line[None, :]).astype(np.float32)
        band = np.exp(-((ys - line[None, :]) / 10) ** 2)                                 # bright surface
        cool = np.clip((ys - line[None, :]) / (h * 0.6), 0, 1)                            # cooler further below
        col = (hot * (1 - cool[..., None]) + np.array(L[:, :, :3], np.float32) * cool[..., None])
        out = np.zeros((h, w, 4), np.uint8)
        out[:, :, :3] = np.clip(col + band[..., None] * 120, 0, 255)
        out[:, :, 3] = np.clip(a * filled * 255, 0, 255)
        return out

    def frame_at(k):
        t = k / v["FPS"]
        frame = base.copy()
        fx = np.zeros((H, W, 3), np.float32)
        warm = sum(_clamp((t - T0 - d) / FILL) for _, _, d in words) / 2
        frame += np.array(ember, np.float32) * 0.06 * warm
        for L, c, d in words:                                     # empty outlines first, then filled
            _paste(frame, _outline_cached(L), *c, 1.0, 0.35 * smooth(t / 0.4))
            frac = _clamp((t - T0 - d) / FILL)
            if frac > 0:
                lay = fill_layer(L, ease_out(frac) if frac < 1 else 1.0, t)
                _paste(frame, _glow(lay, ember, 18), *c, 1.0, 0.7 * (1.2 - 0.6 * frac), add=True)
                _paste(frame, lay, *c, 1.0, 1.0)
        mark_u = _clamp((t - T0 - FILL - 0.1) / 0.4)
        if mark_u > 0:
            _paste(frame, v["L_anvil_glow"], *v["mark_c"], 1.0, 0.8 * mark_u, add=True)
            _paste(frame, v["L_anvil"], *v["mark_c"], 0.9 + 0.1 * ease_out_back(mark_u), mark_u)
            _paste(frame, v["L_dots"], *v["mark_c"], 1.0, mark_u)
        for s in drips:
            s.draw(fx, t)
        _text_block(frame, v, t, 2.2, 2.55, 2.95)
        return _finish(frame, fx, v, k, t)
    return frame_at


_outlines: dict[int, np.ndarray] = {}


def _outline_cached(L):
    from .intro import _outline
    if id(L) not in _outlines:
        _outlines[id(L)] = _outline(L, (90, 100, 110), 2)
    return _outlines[id(L)]


# ---------------------------------------------------------------- minimal
def _minimal(v):
    W, H, ember = v["W"], v["H"], v["ember"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = np.clip(1.05 - 0.5 * np.hypot((xx - W / 2) / W, (yy - H / 2) / H) * 1.6, 0.6, 1)[..., None]
    base = (np.zeros((H, W, 3), np.float32) + np.array([16, 17, 19], np.float32)) * vign

    def frame_at(k):
        t = k / v["FPS"]
        frame = base.copy()
        fx = np.zeros((H, W, 3), np.float32)
        u = ease_out(_clamp((t - 0.3) / 0.9))                     # hairline draws out from the centre
        y_line = (v["scott_c"][1] + v["y_url"]) // 2 + 20
        half = int(380 * u)
        cv2.line(frame, (W // 2 - half, y_line), (W // 2 + half, y_line), tuple(float(c) for c in ember), 2, cv2.LINE_AA)
        mu = smooth(_clamp((t - 0.7) / 0.7))
        if mu > 0:
            _paste(frame, v["L_anvil"], *v["mark_c"], 1.0, mu)
            _paste(frame, v["L_dots"], *v["mark_c"], 1.0, mu)
        for L, c, d in ((v["L_scott"], v["scott_c"], 0.9), (v["L_forge"], v["forge_c"], 1.05)):
            wu = _clamp((t - d) / 0.9)
            if wu > 0:
                # letters drift together as they fade up (spacing settles)
                dx = (c[0] - W / 2) * 0.06 * (1 - ease_out(wu))
                _paste(frame, L, c[0] + dx, c[1], 1.0, smooth(wu))
        _text_block(frame, v, t, 1.9, 2.3, 3.3, url_typed=True)
        return _finish(frame, fx, v, k, t)
    return frame_at
