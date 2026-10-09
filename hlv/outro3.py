"""ScottForge endings with their own sound (outro.py's other styles are silent):

  hammer   close-up in a dark smithy: a hammer swings in and strikes a glowing bar on the anvil
           three times (scale flies off as sparks, the bar dents and flares, the camera jolts);
           on the last, biggest blow the scene falls back into soft focus and the logo forges
           on, white-hot cooling to its colours. Sound: forge roar and crackle, swing whooshes,
           the thunk of hot steel and the anvil's ring in a stone room.
  quench   a glowing blade is plunged into a water trough: a violent hiss, boiling, steam
           billows up lit orange from below, the blade's glow fades from the waterline up,
           and the logo appears as the steam clears.

make(style, v) returns frame_at(k) like outro2's; sound(style, seconds, seed) returns the
(n, 2) float32 48 kHz track. Everything is drawn and synthesised here: free and offline.
"""
from __future__ import annotations

import cv2
import numpy as np

from .intro import Sparks, _clamp, _glow, _paste, ease_in, ease_out, ease_out_back, smooth
from .outro2 import _text_block

STYLES = ("hammer", "quench")
VERSION = 1
SR = 48000
SPARK_YEL = (87, 200, 255)
HOT = np.array([175, 238, 255], np.float32)          # white-hot (BGR)

# blackbody-ish ramp, 0 = black iron .. 1 = white-yellow
_T = [0.0, 0.2, 0.4, 0.55, 0.7, 0.85, 1.0]
_R = [0, 70, 170, 235, 255, 255, 255]
_G = [0, 5, 25, 85, 160, 220, 250]
_B = [0, 0, 0, 10, 40, 120, 210]


def heat(T: np.ndarray) -> np.ndarray:
    """BGR float colour of steel at heat T (0..1)."""
    T = np.clip(T, 0, 1)
    return np.stack([np.interp(T, _T, _B), np.interp(T, _T, _G), np.interp(T, _T, _R)], -1).astype(np.float32)


def _noise(rng, h, w, scale, octaves=3) -> np.ndarray:
    """Smooth value noise in 0..1."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(1, int(scale / 2 ** o))
        small = rng.random((h // s + 2, w // s + 2)).astype(np.float32)
        out += amp * cv2.resize(small, (w + 2 * s, h + 2 * s), interpolation=cv2.INTER_CUBIC)[s:s + h, s:s + w]
        tot += amp
        amp *= 0.5
    out /= tot
    return np.clip((out - out.min()) / max(1e-6, out.max() - out.min()), 0, 1)


def make(style: str, v: dict):
    return {"hammer": _hammer, "quench": _quench}[style](v)


def sound(style: str, seconds: float, seed: int = 11) -> np.ndarray:
    return {"hammer": _hammer_sound, "quench": _quench_sound}[style](seconds, np.random.default_rng(seed))


# ================================================================ shared
def _bloom(frame, fx, W, H, k=1.4):
    glow = cv2.resize(cv2.GaussianBlur(cv2.resize(fx, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3), (W, H))
    wide = cv2.resize(cv2.GaussianBlur(cv2.resize(fx, (W // 16, H // 16), interpolation=cv2.INTER_AREA), (0, 0), 2), (W, H))
    return frame + fx + glow * k + wide * 0.8


def _soft(frame, W, H, sigma=3.0):
    """Cheap heavy blur (depth of field) via a small copy."""
    return cv2.resize(cv2.GaussianBlur(cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), sigma),
                      (W, H), interpolation=cv2.INTER_LINEAR)


def _shift(frame, dx, dy, z=1.0, cy=None):
    if abs(dx) < 0.05 and abs(dy) < 0.05 and abs(z - 1) < 1e-4:
        return frame
    H, W = frame.shape[:2]
    cy = H / 2 if cy is None else cy
    m = np.float32([[z, 0, (1 - z) * W / 2 + dx], [0, z, (1 - z) * cy + dy]])
    return cv2.warpAffine(frame, m, (W, H), borderMode=cv2.BORDER_REFLECT)


def _shake(t, hits, amp):
    dx = dy = 0.0
    for t0, a in hits:
        d = t - t0
        if 0 <= d < 0.5:
            e = np.exp(-d / 0.09) * amp * a
            dx += e * np.sin(2 * np.pi * 17 * d + 0.4)
            dy += e * np.cos(2 * np.pi * 13 * d)
    return dx, dy


def _hot_copy(L):
    out = L.copy()
    out[:, :, :3] = HOT.astype(np.uint8)
    return out


def _forged_logo(frame, fx, v, t, t0, sparks_ok=True):
    """The mark and wordmark forge on: they appear white-hot and cool to their colours."""
    mc = v["mark_c"]
    u = (t - t0) / 0.35
    if u > 0:
        sc = 1.18 - 0.18 * ease_out_back(u, 2.0)
        a = _clamp(u * 2.5)
        cool = np.exp(-max(0.0, t - t0) / 0.55)
        _paste(frame, v["L_anvil_glow"], *mc, sc, a * (0.55 + 1.3 * cool), add=True)
        _paste(frame, v["L_anvil"], *mc, sc, a)
        _paste(frame, v["_hot_anvil"], *mc, sc, a * cool)
        d = _clamp((t - t0 - 0.25) / 0.2)
        _paste(frame, v["L_dots_glow"], *mc, sc, d * (0.8 + 0.4 * np.sin(t * 9)), add=True)
        _paste(frame, v["L_dots"], *mc, sc, d)
    for L, Lh, c, d0 in ((v["L_scott"], v["_hot_scott"], v["scott_c"], 0.12), (v["L_forge"], v["_hot_forge"], v["forge_c"], 0.24)):
        u = (t - t0 - d0) / 0.45
        if u > 0:
            cool = np.exp(-max(0.0, t - t0 - d0) / 0.6)
            y = c[1] + 18 * (1 - ease_out(u))
            _paste(frame, L, c[0], y, 1.0, smooth(u))
            _paste(frame, Lh, c[0], y, 1.0, smooth(u) * cool)
            _paste(frame, _glow_cached(Lh), c[0], y, 1.0, smooth(u) * cool * 0.9, add=True)
    u = (t - t0 - 0.24) / 0.45
    if u > 0:
        hv = 0.75 + 0.18 * np.sin(t * 5.3) + 0.08 * np.sin(t * 13.1)
        _paste(frame, v["L_forge_glow"], *v["forge_c"], 1.0, smooth(u) * hv, add=True)


_glows: dict[int, np.ndarray] = {}


def _glow_cached(L):
    if id(L) not in _glows:
        _glows[id(L)] = _glow(L, tuple(int(c) for c in HOT), 16)
    return _glows[id(L)]


def _prep_logo(v):
    v["_hot_anvil"] = _hot_copy(v["L_anvil"])
    v["_hot_scott"] = _hot_copy(v["L_scott"])
    v["_hot_forge"] = _hot_copy(v["L_forge"])


def _end(frame, v, k, t, fade_in=0.25):
    W, H = v["W"], v["H"]
    frame += cv2.resize(v["grain"][k % len(v["grain"])], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    fade = smooth(t / fade_in) * (1 - smooth((t - (v["seconds"] - 0.7)) / 0.7))
    return np.clip(frame * fade, 0, 255).astype(np.uint8)


class Debris:
    """Forge-scale sparks: streaks with drag and gravity that cool white -> yellow -> orange -> red,
    a few bouncing off the anvil face, plus slow tumbling flakes of scale."""

    def __init__(self, rng, x, y, t0, n, floor=None, power=1.0):
        self.t0, self.floor = t0, floor
        ang = np.where(rng.random(n) < 0.5, rng.uniform(np.pi * 1.0, np.pi * 1.42, n),
                       rng.uniform(np.pi * 1.58, np.pi * 2.0, n))          # low and sideways, both ways
        ang = np.where(rng.random(n) < 0.2, rng.uniform(np.pi * 1.25, np.pi * 1.75, n), ang)
        spd = rng.uniform(250, 1700, n) * power * rng.choice([1.0, 0.5], n, p=[0.7, 0.3])
        self.p0 = np.stack([x + rng.normal(0, 14, n), y + rng.normal(0, 4, n)], 1).astype(np.float32)
        self.v = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1).astype(np.float32)
        self.life = rng.uniform(0.35, 1.5, n)
        self.w = rng.choice([1, 2, 2, 3], n)
        self.flick = rng.uniform(0, 2 * np.pi, n)
        m = n // 5                                                     # scale flakes
        self.f0 = np.stack([x + rng.normal(0, 30, m), y + rng.normal(0, 6, m)], 1).astype(np.float32)
        self.fv = np.stack([rng.normal(0, 260, m), rng.uniform(-520, -120, m)], 1).astype(np.float32)
        self.fsize = rng.uniform(2.5, 6, m)
        self.flife = rng.uniform(0.8, 1.8, m)

    def _pos(self, dt):
        drag = np.exp(-1.8 * dt)
        p = self.p0 + self.v * (1 - drag) / 1.8 + np.array([0, 1500], np.float32) * dt * dt * 0.5
        if self.floor is not None:                                     # bounce off the anvil face
            fx0, fx1, fy = self.floor
            over = (p[:, 1] > fy) & (p[:, 0] > fx0) & (p[:, 0] < fx1) & (self.v[:, 1] > 0)
            p[over, 1] = fy - (p[over, 1] - fy) * 0.35
        return p

    def draw(self, fx, t):
        dt = t - self.t0
        if dt < 0 or dt > 2.0:
            return
        p1, p0 = self._pos(dt), self._pos(max(0.0, dt - 1 / 55))
        u = dt / self.life
        for i in np.nonzero(u < 1)[0]:
            T = 1.0 - 0.75 * u[i]
            c = heat(np.array(T)) * (1.6 - u[i]) * (0.75 + 0.25 * np.sin(dt * 40 + self.flick[i]))
            cv2.line(fx, (int(p0[i, 0]), int(p0[i, 1])), (int(p1[i, 0]), int(p1[i, 1])),
                     tuple(float(x) for x in c), int(self.w[i]), cv2.LINE_AA)
        fp = self.f0 + self.fv * dt + np.array([0, 900], np.float32) * dt * dt * 0.5
        fu = dt / self.flife
        for i in np.nonzero(fu < 1)[0]:
            c = heat(np.array(0.62 - 0.35 * fu[i])) * (1 - fu[i]) * (0.6 + 0.4 * abs(np.sin(dt * 14 + i)))
            ax = (self.fsize[i], self.fsize[i] * (0.3 + 0.7 * abs(np.sin(dt * 9 + i))))
            cv2.ellipse(fx, (int(fp[i, 0]), int(fp[i, 1])), (max(1, int(ax[0])), max(1, int(ax[1]))),
                        float(dt * 300 + i * 40), 0, 360, tuple(float(x) for x in c), -1, cv2.LINE_AA)


def _smithy_bg(v, rng):
    """Dark smithy: a forge glowing at the back left, soft out-of-focus embers, drifting haze.
    The website's forge clip (if any) is blended in, heavily blurred, for real flicker."""
    W, H = v["W"], v["H"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.zeros((H, W, 3), np.float32) + np.array([10, 12, 15], np.float32)
    base *= (0.7 + 0.3 * (1 - yy / H))[..., None]
    fcx, fcy = (0.12 * W, 0.42 * H) if W > H else (0.1 * W, 0.3 * H)
    forge = np.exp(-(((xx - fcx) / (0.33 * max(W, H) * 0.6)) ** 2 + ((yy - fcy) / (0.28 * H)) ** 2))[..., None]
    forge_c = np.array([20, 70, 210], np.float32)
    haze = [_noise(rng, H // 8, W // 8 + 60, 22, 3) for _ in range(2)]
    bokeh = [(rng.uniform(0, W), rng.uniform(0, H * 0.75), rng.uniform(10, 34), rng.uniform(0, 6.3), rng.uniform(8, 30))
             for _ in range(14)]
    clip = None
    vid = v.get("video")
    if vid:
        import subprocess
        n = int(v["seconds"] * v["FPS"]) + 2
        w4, h4 = W // 4, H // 4
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-stream_loop", "-1", "-ss", "1.0", "-i", str(vid), "-t", f"{n / v['FPS']:.3f}", "-an",
             "-vf", f"fps={v['FPS']},scale={w4}:{h4}:force_original_aspect_ratio=increase,crop={w4}:{h4},format=bgr24",
             "-f", "rawvideo", "pipe:1"], capture_output=True).stdout
        fb = w4 * h4 * 3
        clip = [np.frombuffer(raw[i * fb:(i + 1) * fb], np.uint8).reshape(h4, w4, 3) for i in range(len(raw) // fb)] or None

    def bg(t, k):
        fl = 0.8 + 0.12 * np.sin(t * 7.1) + 0.06 * np.sin(t * 17.3 + 1) + 0.05 * np.sin(t * 3.1)
        f = base + forge * forge_c * 0.55 * fl
        if clip is not None:
            c = cv2.GaussianBlur(clip[min(k, len(clip) - 1)].astype(np.float32), (0, 0), 6)
            f += cv2.resize(c, (W, H)) * 0.22
        sm = W // 8
        off = int(t * 9) % 60
        hz = cv2.resize(haze[0][:, off:off + sm] * 0.6 + haze[1][:, 60 - off:60 - off + sm] * 0.4, (W, H))
        f += hz[..., None] * np.array([14, 20, 30], np.float32) * (0.5 + forge * 1.5)
        bk = np.zeros((H // 4, W // 4, 3), np.float32)
        for (bx, by, r, ph, sp) in bokeh:
            y = (by - sp * t) % H
            a = 0.12 + 0.08 * np.sin(t * 2 + ph)
            cv2.circle(bk, (int((bx + 12 * np.sin(t + ph)) / 4), int(y / 4)), max(1, int(r / 4)), (30 * a, 90 * a, 200 * a), -1, cv2.LINE_AA)
        return f + cv2.resize(cv2.GaussianBlur(bk, (0, 0), 1.2), (W, H))
    return bg


# ================================================================ hammer
STRIKES = (0.55, 1.30, 2.05)


def _hammer_pose(t):
    """(angle in degrees: 0 = head down on the bar, + = raised; hand lift 0..1.3)."""
    keys = [(0.0, 82, 1.0, None), (0.28, 82, 1.0, None), (0.55, 0, 0, "in"), (0.70, 22, 0.15, "out"),
            (0.96, 76, 0.92, "smooth"), (1.05, 80, 1.0, "smooth"), (1.30, 0, 0, "in"), (1.45, 22, 0.15, "out"),
            (1.66, 84, 1.0, "smooth"), (1.80, 90, 1.08, "smooth"), (2.05, 0, 0, "in"), (2.24, 18, 0.12, "out"),
            (2.85, 100, 1.5, "smooth"), (99, 100, 1.5, None)]
    for (t0, a0, l0, _), (t1, a1, l1, kind) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            u = (t - t0) / (t1 - t0)
            e = ease_in(u) ** 0.75 if kind == "in" else ease_out(u) if kind == "out" else smooth(u)
            return a0 + (a1 - a0) * e, l0 + (l1 - l0) * e
    return keys[-1][1], keys[-1][2]


def _build_hammer(L_handle=880):
    """Hammer albedo (BGR 0..1) + alpha on a canvas; returns (img, alpha, hand_xy). Head vertical
    (face down), handle running right to the hand and on past it."""
    head_c = (90, 170)
    cw, ch = head_c[0] + L_handle + 400, 340
    alb = np.zeros((ch, cw, 3), np.float32)
    a = np.zeros((ch, cw), np.float32)
    hx, hy = head_c[0] + L_handle, head_c[1] + 6
    # handle: hickory, swelling towards the grip, grain along its length
    xs = np.arange(cw, dtype=np.float32)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    half = np.interp(xs, [head_c[0], head_c[0] + 300, hx, cw], [15, 16, 21, 22])
    across = (yy - (hy - 0.006 * (xx - head_c[0]))) / half[None, :]
    m = (np.abs(across) <= 1) & (xx >= head_c[0])
    rng = np.random.default_rng(3)
    grain = 0.85 + 0.15 * np.sin(xx / 9 + 6 * _noise(rng, ch, cw, 40, 2) + across * 2.5)
    shade = (0.55 + 0.45 * np.sqrt(np.clip(1 - across ** 2, 0, 1))) * (1 - 0.35 * np.clip(across, 0, 1))
    wood = np.array([0.20, 0.36, 0.55], np.float32)
    alb[m] = (wood * (grain * shade)[..., None])[m]
    a[m] = 1
    # head: drop-forged, flat polished face below, chamfers, wedge peen above
    rel = np.array([(-64, 130), (64, 130), (67, 112), (59, 96), (53, 30), (55, -20), (41, -88), (15, -127), (-15, -127),
                    (-41, -88), (-55, -20), (-53, 30), (-59, 96), (-67, 112)], np.float32)
    pts = (rel + np.array(head_c, np.float32)).astype(np.int32)
    hm = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(hm, [pts], 255, cv2.LINE_AA)
    hmf = hm.astype(np.float32) / 255
    rx = (xx - head_c[0]) / 62
    ry = yy - head_c[1]
    steel = 0.30 + 0.30 * np.clip(1 - rx ** 2, 0, 1) + 0.45 * np.exp(-((rx + 0.32) / 0.12) ** 2)
    steel += 0.25 * np.exp(-((rx - 0.55) / 0.08) ** 2)
    steel *= 0.9 + 0.1 * _noise(rng, ch, cw, 6, 2)
    steel = np.where(ry > 108, 0.95, steel)                         # polished striking face
    steel = np.where((ry > 94) & (ry <= 108), 0.75, steel)          # chamfer
    steel = np.where(ry < -112, 0.8, steel)                         # peen edge
    col = np.array([0.46, 0.47, 0.50], np.float32)[None, None] * steel[..., None]
    alb = alb * (1 - hmf[..., None]) + col * hmf[..., None]
    a = np.maximum(a, hmf)
    a = cv2.GaussianBlur(a, (3, 3), 0.7)
    return alb, a, (hx, hy), head_c


def _build_anvil(W, H, xh, fy):
    """Anvil albedo + alpha over the whole frame (side view, close up)."""
    rel = [(-1150, 30), (-1000, 10), (-850, 2), (-700, 0), (420, 0), (432, 10), (432, 105), (330, 118), (230, 160),
           (110, 250), (95, 330), (100, 900), (-380, 900), (-375, 330), (-360, 250), (-480, 160), (-620, 125),
           (-760, 110), (-900, 85), (-1040, 58)]
    pts = np.array([(xh + x, fy + y) for x, y in rel], np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
    a = m.astype(np.float32) / 255
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = np.random.default_rng(5)
    tex = 0.8 + 0.2 * _noise(rng, H, W, 10, 3)
    pits = _noise(rng, H, W, 3, 1)
    body = (0.30 + 0.10 * np.exp(-((yy - fy - 40) / 60) ** 2)) * tex * (0.9 + 0.1 * pits)
    body *= 1 - 0.45 * np.clip((yy - fy) / 500, 0, 1)
    face = np.clip(1 - (yy - fy) / 9, 0, 1)                           # polished top band
    alb = (np.array([0.40, 0.42, 0.45], np.float32)[None, None] * body[..., None] +
           np.array([0.85, 0.88, 0.92], np.float32)[None, None] * (face * 0.9)[..., None])
    # the edge where the face meets the side catches the light
    alb += np.array([0.3, 0.32, 0.35], np.float32) * np.exp(-((yy - fy - 11) / 2.5) ** 2)[..., None] * 0.6
    return alb, a


def _hammer(v):
    W, H, rng, FPS = v["W"], v["H"], v["rng"], v["FPS"]
    _prep_logo(v)
    tall = H > W
    xh = int(W / 2 + 150) if tall else 1100                    # where the hammer lands
    fy = int(H * 0.62) if tall else 720                        # anvil face (top)
    thick = 50
    bar_top0 = fy - thick
    bx0, bx1 = xh - 520, xh + 1100                              # bar from its hot end into the tongs
    anvil_alb, anvil_a = _build_anvil(W, H, xh, fy)
    h_alb, h_a, hand_c, head_c = _build_hammer()
    L_handle = hand_c[0] - head_c[0]
    h_pre = h_alb * h_a[..., None]
    hand0 = np.array([xh + L_handle, bar_top0 - 130 + (hand_c[1] - head_c[1])], np.float32)
    bg = _smithy_bg(v, rng)

    # tongs gripping the far end of the bar (off to the right)
    tong = np.zeros((H, W), np.float32)
    tx = xh + 700
    for poly in ([(tx, bar_top0 - 6), (tx + 160, bar_top0 - 14), (tx + 700, bar_top0 - 110), (tx + 700, bar_top0 - 88),
                  (tx + 150, bar_top0 + 4), (tx, bar_top0 + 6)],
                 [(tx, fy - 6), (tx + 150, fy - 4), (tx + 700, fy + 100), (tx + 700, fy + 122), (tx + 160, fy + 16), (tx, fy + 6)]):
        cv2.fillPoly(tong, [np.array(poly, np.int32)], 1.0, cv2.LINE_AA)
    tong_alb = np.zeros((H, W, 3), np.float32) + np.array([0.16, 0.17, 0.19], np.float32)

    # static scene (anvil + tongs), composited once; lit per frame
    st_a = np.maximum(anvil_a, tong)
    st_alb = anvil_alb * anvil_a[..., None] + tong_alb * (tong * (1 - anvil_a))[..., None]

    # the bar: oxide scale (dark patches), knocked off where it's hit
    bw, bh = bx1 - bx0, thick + 30
    scale_n = _noise(rng, bh, bw, 16, 3)
    surf_n = _noise(rng, bh, bw, 4, 2)
    bxs = np.arange(bx0, bx1, dtype=np.float32)
    byy = np.arange(fy - bh, fy, dtype=np.float32)[:, None]
    T_line = 0.97 - 0.62 * np.clip((bxs - (xh - 260)) / 1150, 0, 1) ** 0.9
    T_line *= np.clip((bxs - bx0) / 25, 0.6, 1)                    # the very end is a touch cooler

    # light probes along the bar (computed at 1/8 res)
    lw, lh = W // 8, H // 8
    ly, lx = np.mgrid[0:lh, 0:lw].astype(np.float32) * 8
    probes = np.linspace(bx0 + 40, min(bx1, W + 200), 9)

    debris = [Debris(rng, xh + rng.normal(0, 10), bar_top0, t0, n=int(170 * p), floor=(xh - 700, xh + 420, bar_top0 - 2),
                     power=p) for t0, p in zip(STRIKES, (0.9, 1.0, 1.35))]
    logo_sparks = Sparks(rng, v["mark_c"][0], v["mark_c"][1], 2.3, SPARK_YEL, n=70, spread=(300, 1100))
    T_LOGO = 2.28

    def bar_state(t):
        dent = np.zeros_like(bxs)
        Tb = T_line * (1 - 0.03 * t)
        clean = np.zeros_like(bxs)
        for i, t0 in enumerate(STRIKES):
            if t >= t0:
                d = 5.0 + 2.5 * i
                off = bxs - xh - (i - 1) * 26
                dent += d * np.exp(-(off / 75) ** 4) - 2.2 * np.exp(-((np.abs(off) - 105) / 30) ** 2)
                Tb = Tb + 0.32 * np.exp(-(t - t0) / 0.14) * np.exp(-(off / 90) ** 2)
                clean = np.maximum(clean, np.exp(-(off / 85) ** 2))
        return dent, Tb, clean

    def frame_at(k):
        t = k / FPS
        dent, Tb, clean = bar_state(t)
        # ---- light: the bar (inverse-square-ish), the strike flash, the forge at the back, a cool fill
        light = np.zeros((lh, lw, 3), np.float32) + np.array([0.24, 0.21, 0.19], np.float32)
        light += np.array([0.16, 0.14, 0.12], np.float32) * np.clip(1 - ly / H, 0, 1)[..., None]   # cool top light
        for px in probes:
            Ti = float(np.interp(px, bxs, Tb))
            col = heat(np.array(min(1.0, Ti + 0.1))) / 255
            d2 = ((lx - px) ** 2 + (ly - (bar_top0 + 15)) ** 2) / 210 ** 2
            light += (col * Ti ** 3 * 0.55)[None, None] / (1 + d2)[..., None]
        for t0 in STRIKES:
            if 0 <= t - t0 < 0.4:
                d2 = ((lx - xh) ** 2 + (ly - bar_top0) ** 2) / 330 ** 2
                light += np.array([0.75, 0.9, 1.0], np.float32) * (2.2 * np.exp(-(t - t0) / 0.06)) / (1 + d2)[..., None]
        light += np.array([0.02, 0.06, 0.18], np.float32) * np.clip(1 - lx / W, 0, 1)[..., None] ** 2
        Lmap = cv2.resize(light, (W, H), interpolation=cv2.INTER_LINEAR)

        frame = bg(t, k)
        frame += Lmap * 10                                         # a little of the light reaches the walls
        lit = st_alb * Lmap * 255
        frame = frame * (1 - st_a[..., None]) + lit

        # ---- the bar (emissive)
        top = bar_top0 + dent
        bar_y = byy
        inside = (bar_y >= top[None, :]) & (bar_y < fy)
        cap = bxs[None, :] < bx0 + 16
        r = 16.0
        cy = np.clip(bar_y, top[None, :] + r, fy - r)
        inside &= ~cap | (((bxs[None, :] - (bx0 + r)) ** 2 + (bar_y - cy) ** 2) <= r * r)
        thk = np.maximum(1, fy - top)[None, :]
        prof = 0.86 + 0.14 * np.sin(np.pi * np.clip((bar_y - top[None, :]) / thk, 0, 1))
        blot = np.clip((scale_n - 0.52) / 0.14, 0, 1) * (1 - clean[None, :])
        sc = 1 - 0.38 * blot
        Tpix = Tb[None, :] * prof * (0.9 + 0.1 * surf_n) * sc
        bcol = heat(Tpix)
        ia = inside.astype(np.float32)
        ia = cv2.GaussianBlur(ia, (3, 3), 0.6)
        x_a, x_b = max(0, bx0), min(W, bx1)
        sl = slice(x_a - bx0, x_b - bx0)
        roi = frame[fy - bh:fy, x_a:x_b]
        roi[:] = roi * (1 - ia[:, sl, None]) + bcol[:, sl] * ia[:, sl, None]
        fx = np.zeros((H, W, 3), np.float32)
        fx[fy - bh:fy, x_a:x_b] += bcol[:, sl] * ia[:, sl, None] * 0.35
        # tongs bite over the bar
        frame = frame * (1 - tong[..., None]) + (tong_alb * Lmap * 255) * tong[..., None]

        # ---- the hammer (motion-blurred over half a frame)
        acc_c = np.zeros((H, W, 3), np.float32)
        acc_a = np.zeros((H, W), np.float32)
        a0, a1 = _hammer_pose(t - 0.5 / FPS)[0], _hammer_pose(t)[0]
        subs = int(np.clip(abs(a1 - a0) / 0.6, 1, 64))
        for j in range(subs):
            ts = t - (1 / FPS) * 0.5 * (1 - j / max(1, subs - 1))
            ang, lift = _hammer_pose(ts)
            hand = hand0 + lift * np.array([70, -560], np.float32)
            M = cv2.getRotationMatrix2D((float(hand_c[0]), float(hand_c[1])), -ang, 1.0)
            M[:, 2] += hand - np.array(hand_c, np.float32)
            acc_c += cv2.warpAffine(h_pre, M, (W, H), flags=cv2.INTER_LINEAR)
            acc_a += cv2.warpAffine(h_a, M, (W, H), flags=cv2.INTER_LINEAR)
        acc_c /= subs
        acc_a /= subs
        frame = frame * (1 - acc_a[..., None]) + acc_c * Lmap * 255

        # ---- sparks + strike flash
        for d in debris:
            d.draw(fx, t)
        for t0 in STRIKES:
            dt = t - t0
            if 0 <= dt < 0.07:
                f = (1 - dt / 0.07) ** 2
                cv2.ellipse(fx, (xh, int(bar_top0 + 4)), (int(70 + 120 * dt / 0.07), int(14 + 20 * dt / 0.07)), 0, 0, 360,
                            tuple(float(c) * f * 1.4 for c in HOT), -1, cv2.LINE_AA)
        frame = _bloom(frame, fx, W, H)

        # ---- camera: slow push-in, a jolt on each blow
        dx, dy = _shake(t, [(s, a) for s, a in zip(STRIKES, (0.6, 0.7, 1.2))], 16)
        frame = _shift(frame, dx, dy, 1.0 + 0.035 * t / v["seconds"], cy=fy)

        # ---- the logo: the scene falls back into soft focus, the logo forges on
        if t > T_LOGO - 0.1:
            u = smooth((t - T_LOGO + 0.1) / 0.7)
            frame = frame * (1 - u) + _soft(frame, W, H, 2.5) * u
            frame *= 1 - 0.62 * u
            lfx = np.zeros((H, W, 3), np.float32)
            logo_sparks.draw(lfx, t)
            _forged_logo(frame, lfx, v, t, T_LOGO)
            _text_block(frame, v, t, 3.0, 3.35, 3.7)
            frame = frame + lfx + _soft(lfx, W, H, 2) * 1.2
        return _end(frame, v, k, t)
    return frame_at


# ---------------------------------------------------------------- hammer sound
def _filt(x, kind, f, order=2):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(order, f, kind, fs=SR, output="sos"), x)


def _room_ir(rng, d=1.3, bright=3500):
    n = int(d * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / (d / 6.5))
    ir = _filt(ir, "low", bright)
    ir[:int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))   # pre-delay
    for dly, g in ((0.011, 0.5), (0.019, 0.35), (0.027, 0.3), (0.041, 0.2)):   # early reflections off stone
        ir[int(dly * SR)] += g * 6
    return (ir / np.abs(ir).max()).astype(np.float32)


def _add(buf, x, t):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(x))
    buf[i:j] += x[:j - i]


def _strike(rng, power=1.0, ring=1.0):
    """Hammer on hot steel on the anvil: click, a heavy thunk, the anvil's inharmonic ring."""
    d = 2.6 * ring
    n = int(d * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    # click: the first millisecond of contact
    m = int(0.003 * SR)
    out[:m] += rng.standard_normal(m) * np.linspace(1, 0, m) * 0.9
    # thunk: hot steel is soft, so a lot of the blow is a dull, heavy body sound
    nb = _filt(rng.standard_normal(n), "low", 1800) * np.exp(-t / 0.018)
    f = 95 + 120 * np.exp(-t / 0.012)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.09)
    out += 0.9 * nb / (np.abs(nb).max() + 1e-9) + 0.75 * body
    # anvil ring: modes of a steel block (inharmonic), each a slightly detuned pair -> shimmer
    modes = [(1046, 1.5, 0.55), (1468, 1.1, 0.85), (2093, 0.9, 0.6), (2789, 0.7, 0.45), (3517, 0.55, 0.36),
             (4398, 0.4, 0.26), (5602, 0.28, 0.16), (7203, 0.16, 0.09), (619, 0.9, 0.22)]
    k = rng.uniform(0.985, 1.015)
    ringsum = np.zeros(n)
    for fr, dec, amp in modes:
        fr *= k
        for det in (-1.3, 1.3):
            ringsum += amp * np.sin(2 * np.pi * (fr + det) * t + rng.uniform(0, 6.28)) * np.exp(-t / (dec * ring))
    ringsum *= 1 - np.exp(-t / 0.0015)
    out += 0.42 * ring * ringsum
    # the hammer head's own short, bright ping
    for fr, dec in ((3870, 0.05), (6120, 0.03), (8350, 0.02)):
        out += 0.12 * np.sin(2 * np.pi * fr * t) * np.exp(-t / dec)
    return (out * power).astype(np.float32)


def _crackle(rng, d, rate, gain, lo=1500):
    """Sparse pops and ticks: burning coal, flying scale."""
    n = int(d * SR)
    out = np.zeros(n)
    for _ in range(rng.poisson(rate * d)):
        i = rng.integers(0, max(1, n - 2000))
        L = int(rng.uniform(0.002, 0.012) * SR)
        a = rng.uniform(0.2, 1) ** 2
        out[i:i + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * a
    return (_filt(out, "high", lo) * gain).astype(np.float32)


def _forge_bed(rng, d):
    """Forge roar: a low rumble and a breathy mid roar, breathing slowly, plus crackle."""
    n = int(d * SR)
    t = np.arange(n) / SR
    brown = np.cumsum(rng.standard_normal(n))
    brown = _filt(brown - np.convolve(brown, np.ones(4801) / 4801, "same"), "low", 180)
    brown /= np.abs(brown).max() + 1e-9
    roar = _filt(rng.standard_normal(n), "band", [180, 900])
    roar /= np.abs(roar).max() + 1e-9
    breathe = 0.75 + 0.15 * np.sin(2 * np.pi * 0.37 * t) + 0.1 * np.sin(2 * np.pi * 0.83 * t + 1)
    return (0.22 * brown + 0.07 * roar * breathe).astype(np.float32) + _crackle(rng, d, 9, 0.18)


def _whoosh(rng, d=0.28, gain=0.12):
    n = int(d * SR)
    x = _filt(rng.standard_normal(n), "band", [250, 1600])
    env = np.linspace(0, 1, n) ** 2.5
    return (x / (np.abs(x).max() + 1e-9) * env * gain).astype(np.float32)


def _master(L, R, seconds, fade_in=0.2):
    n = int(seconds * SR)
    out = np.stack([L[:n], R[:n]], 1)
    t = np.arange(n) / SR
    env = np.clip(t / fade_in, 0, 1) * np.clip((seconds - t) / 0.8, 0, 1)
    out *= env[:, None]
    peak = float(np.abs(out).max()) or 1.0
    out = np.tanh(out / peak * 1.25) / np.tanh(1.25) * 0.89        # gentle glue, peak about -1 dBFS
    return out.astype(np.float32)


def _hammer_sound(seconds, rng):
    from scipy.signal import fftconvolve
    n = int((seconds + 2) * SR)
    dry = np.zeros(n, np.float32)
    bed = _forge_bed(rng, seconds + 2)
    for i, t0 in enumerate(STRIKES):
        big = i == len(STRIKES) - 1
        _add(dry, _whoosh(rng, 0.26 if not big else 0.32, 0.1 if not big else 0.16), t0 - (0.26 if not big else 0.32))
        _add(dry, _strike(rng, 1.25 if big else 0.85 + 0.08 * i, 1.1 if big else 0.8), t0)
        _add(dry, _crackle(rng, 0.9, 70 if big else 45, 0.25), t0 + 0.01)       # scale and sparks skittering
    # the logo forming: a soft low swell of the forge, no music-like tones
    t1 = 2.2
    m = int(1.6 * SR)
    tt = np.arange(m) / SR
    swell = _filt(rng.standard_normal(m), "low", 300) * np.sin(np.pi * np.clip(tt / 1.6, 0, 1)) ** 2
    _add(dry, (swell / (np.abs(swell).max() + 1e-9) * 0.18).astype(np.float32), t1)
    irs = [_room_ir(np.random.default_rng(s)) for s in (21, 22)]
    chans = []
    for c, ir in enumerate(irs):
        wet = fftconvolve(dry, ir)[:n] * 0.11
        # a little spread: the sparks fly left on one channel, right on the other
        chans.append(dry * (1.0 if c == 0 else 0.96) + wet + bed[:n] * (1.0 if c == 0 else 0.92))
    return _master(chans[0], chans[1], seconds)


# ================================================================ quench
T_DIP = 0.75


class Steam:
    """Billowing steam: soft noise puffs that rise, swell and fade, drawn at quarter size."""

    def __init__(self, rng, W, H):
        self.rng, self.W, self.H = rng, W, H
        self.sprites = []
        for i in range(8):
            s = 128
            yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
            r = np.hypot(xx - s / 2, yy - s / 2) / (s / 2)
            nz = _noise(rng, s, s, 7, 4)
            self.sprites.append(np.exp(-(r / 0.5) ** 2) * np.clip(nz * 1.7 - 0.45, 0, 1) ** 1.5)
        self.p = []

    def emit(self, x, y, t, n, strength):
        r = self.rng
        for _ in range(n):
            self.p.append(dict(t0=t, x=x + r.normal(0, 45), y=y + r.normal(0, 8), vx=r.normal(0, 90),
                               vy=-r.uniform(140, 420) * strength, s0=r.uniform(70, 150), grow=r.uniform(160, 380),
                               life=r.uniform(1.6, 3.6), a=r.uniform(0.10, 0.24) * strength, spr=int(r.integers(0, 8))))

    def draw(self, t):
        """(density, warmth) maps at quarter size."""
        w, h = self.W // 4, self.H // 4
        dens = np.zeros((h, w), np.float32)
        for p in self.p:
            dt = t - p["t0"]
            if dt < 0 or dt > p["life"]:
                continue
            u = dt / p["life"]
            x = (p["x"] + p["vx"] * dt + 30 * np.sin(dt * 1.7 + p["s0"])) / 4
            y = (p["y"] + p["vy"] * (1 - np.exp(-dt * 0.9)) / 0.9) / 4
            sz = int((p["s0"] + p["grow"] * dt) / 4)
            if sz < 2:
                continue
            a = p["a"] * smooth(dt / 0.15) * (1 - u) ** 1.4
            spr = cv2.resize(self.sprites[p["spr"]], (sz, sz))
            x0, y0 = int(x - sz / 2), int(y - sz / 2)
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + sz), min(h, y0 + sz)
            if xb <= xa or yb <= ya:
                continue
            dens[ya:yb, xa:xb] += spr[ya - y0:yb - y0, xa - x0:xb - x0] * a
        return 1 - np.exp(-1.5 * dens)


def _quench(v):
    W, H, rng, FPS = v["W"], v["H"], v["rng"], v["FPS"]
    _prep_logo(v)
    tall = H > W
    wy = int(H * 0.66) if tall else 735                        # water line
    bxc = W // 2 + (0 if tall else 120)                        # blade centre x
    blade_len, blade_w = (820, 118) if not tall else (900, 120)
    bg = _smithy_bg(v, rng)
    steam = Steam(rng, W, H)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # blade layer (vertical, tip down): flat with a central ridge and a bevel, a tang going up
    bl_h = blade_len + 500
    bl = np.zeros((bl_h, blade_w + 40), np.float32)
    by, bx = np.mgrid[0:bl_h, 0:blade_w + 40].astype(np.float32)
    cx = (blade_w + 40) / 2
    tip = bl_h - 10
    half = np.where(by > tip - 160, (blade_w / 2) * np.sqrt(np.clip((tip - by) / 160, 0, 1)), blade_w / 2)
    half = np.where(by < 500, 14, half)                          # tang (into the tongs)
    blade_m = (np.abs(bx - cx) <= half) & (by <= tip)
    bl[blade_m] = 1
    bl = cv2.GaussianBlur(bl, (3, 3), 0.7)
    edge = np.clip(np.abs(bx - cx) / np.maximum(half, 1), 0, 1)            # 0 at the spine, 1 at the edge
    ridge = 0.72 + 0.28 * np.sqrt(np.clip(1 - edge, 0, 1)) + 0.08 * np.exp(-((bx - cx) / 5) ** 2)
    surf = 0.88 + 0.12 * _noise(rng, bl_h, blade_w + 40, 5, 2)
    tongs_w = 70

    # trough: the dark wooden/steel wall in front of the water, the water surface a thin band
    wall = (yy > wy + 14).astype(np.float32)
    stave = 150
    sx = (xx % stave) / stave
    grain = 0.75 + 0.25 * _noise(rng, H // 6, W, 4, 2).repeat(6, 0)[:H] if H % 6 == 0 else 0.85
    plank = (0.2 + 0.06 * np.sin((xx // stave) * 2.7)) * (0.75 + 0.25 * np.sin(np.pi * sx) ** 0.5) * grain
    plank *= np.where((sx < 0.025) | (sx > 0.975), 0.25, 1.0)
    wall_alb = plank[..., None] * np.array([0.42, 0.62, 0.85], np.float32)
    hoop = (np.abs(yy - (wy + 130)) < 15)
    hoop_sh = 0.3 + 0.35 * np.exp(-((yy - (wy + 124)) / 5) ** 2)
    wall_alb[hoop] = (hoop_sh[..., None] * np.array([0.55, 0.57, 0.6], np.float32))[hoop]
    wall_alb *= (1 - 0.5 * np.clip((yy - wy) / (H - wy + 1), 0, 1))[..., None]
    ripple_n = _noise(rng, 64, W // 4 + 200, 8, 2)
    bubbles = []

    def blade_y(t):
        """y of the tip."""
        if t < T_DIP:                                                  # swung in over the trough, a small lift
            return wy - 330 + 150 * ease_out(t / 0.5) - 25 * smooth((t - 0.5) / (T_DIP - 0.5))
        u = (t - T_DIP) / 0.3
        return wy - 205 + (blade_len * 0.62 + 205) * ease_out(u)                         # plunge

    emitted = [0.0]
    T_LOGO = 2.45
    logo_sparks = Sparks(rng, v["mark_c"][0], v["mark_c"][1], T_LOGO + 0.05, SPARK_YEL, n=50, spread=(250, 900))

    def frame_at(k):
        t = k / FPS
        # steam (emitted while the blade is hot under water); keep particle state frame-ordered
        while emitted[0] <= t:
            te = emitted[0]
            if te >= T_DIP:
                s = np.exp(-(te - T_DIP) / 0.9)
                burst = 3.0 if te - T_DIP < 0.1 else 1.0
                steam.emit(bxc, wy - 10, te, int(2 + 9 * s * burst), 0.7 + 0.9 * s)
                if s > 0.05:
                    for _ in range(int(6 * s * burst)):
                        bubbles.append((te, bxc + rng.normal(0, 55 * (1 + burst)), rng.uniform(0.15, 0.5), rng.uniform(3, 10)))
            emitted[0] += 1 / FPS
        tip = blade_y(t)
        depth = max(0.0, tip - wy)                                      # how far it's in
        cool_t = max(0.0, t - T_DIP)
        # ---- heat along the blade: hot at the tip; below the water it dies fast, above it cools upward
        rows = np.arange(bl_h, dtype=np.float32)
        y_rows = tip - (bl_h - 10) + rows                              # frame y of each blade row
        base_T = 0.95 - 0.55 * np.clip((tip - y_rows) / blade_len, 0, 1) ** 1.2
        base_T = np.where(rows < 500, 0.25 * np.clip((rows - 380) / 120, 0, 1), base_T)   # the tang is dull
        under = y_rows > wy
        front = wy - 260 * (1 - np.exp(-cool_t / 1.2))                # cooling creeps up from the waterline
        above_cool = np.where(y_rows > front, np.exp(-cool_t / 0.5), 1.0) if t >= T_DIP else 1.0
        base_T = base_T * np.where(under, np.exp(-cool_t / 0.25) * 0.6, above_cool) * (1 - 0.06 * t)
        Tm = base_T[:, None] * ridge * surf
        col = heat(Tm)
        dark = np.array([42, 44, 48], np.float32) * ridge[..., None]     # cold steel underneath the glow
        col = np.maximum(col, dark)

        frame = bg(t, k)
        # light from the blade on the scene
        glow_T = float(np.mean(base_T[~under & (rows > 500)])) if np.any(~under & (rows > 500)) else 0.0
        steam_light = float(np.exp(-cool_t / 0.35) * 0.9) if t >= T_DIP else 0.0
        lcx, lcy = bxc, min(wy - 40, tip - blade_len * 0.3)
        lmap = np.exp(-(((xx[::8, ::8] - lcx) / 500) ** 2 + ((yy[::8, ::8] - lcy) / 420) ** 2))
        lmap = cv2.resize(lmap, (W, H))[..., None] * (glow_T ** 2 * 0.9 + steam_light)
        warm = np.array([40, 110, 255], np.float32)
        frame += lmap * warm * 0.25

        # water surface band: dark, ripples catching the glow
        off = int(t * 60) % 200
        rp = cv2.resize(ripple_n[:, off:off + W // 4], (W, 30))
        band_y0 = wy - 16
        dist = np.abs(np.arange(W, dtype=np.float32) - bxc)
        ring = 0.0
        if t > T_DIP:
            rr = (t - T_DIP) * 520
            ring = np.exp(-((dist - rr) / 30) ** 2) * np.exp(-(t - T_DIP) / 0.9)
        sheen = (0.25 + 0.75 * rp ** 3) * (0.25 + np.exp(-(dist / 650) ** 2) * (glow_T + steam_light * 1.5))[None, :]
        sheen += ring[None, :] * 0.8 if t > T_DIP else 0
        sheen += (np.exp(-(dist / (blade_w * 0.6)) ** 2) * (glow_T * 2.2 + steam_light) * (0.5 + 0.5 * rp))
        frame[band_y0:band_y0 + 30] = (np.array([18, 16, 14], np.float32) +
                                       sheen[..., None] * np.array([40, 120, 255], np.float32) * 0.55)

        # blade above the water (the wall hides the rest; a glow shows through the water)
        bl_x0 = int(bxc - (blade_w + 40) / 2)
        y0 = int(round(tip - (bl_h - 10)))
        ya, yb = max(0, y0), min(wy - 2 if t >= T_DIP else H, y0 + bl_h)
        fx = np.zeros((H, W, 3), np.float32)
        if yb > ya:
            a = bl[ya - y0:yb - y0][..., None]
            c = col[ya - y0:yb - y0]
            roi = frame[ya:yb, bl_x0:bl_x0 + blade_w + 40]
            roi[:] = roi * (1 - a) + c * a
            fx[ya:yb, bl_x0:bl_x0 + blade_w + 40] += c * a * 0.35
            # the tongs' jaws gripping the tang
            jy = int(y0 + 470)
            if jy > 0:
                cv2.rectangle(frame, (int(bxc - tongs_w / 2), max(0, jy - 70)), (int(bxc + tongs_w / 2), jy + 10), (18, 19, 22), -1)
                cv2.rectangle(frame, (int(bxc - 9), 0), (int(bxc - 3), max(0, jy - 70)), (22, 23, 26), -1)
                cv2.rectangle(frame, (int(bxc + 3), 0), (int(bxc + 9), max(0, jy - 70)), (22, 23, 26), -1)
        # glow under the water, fading fast
        if t >= T_DIP and depth > 0:
            g = np.exp(-cool_t / 0.3)
            sub = np.exp(-(((xx[::4, ::4] - bxc) / 90) ** 2 + ((yy[::4, ::4] - (wy + depth * 0.5)) / (depth * 0.6 + 40)) ** 2))
            fx += cv2.resize(sub, (W, H))[..., None] * np.array([30, 110, 255], np.float32) * g * 0.6
        # trough wall in front
        frame = frame * (1 - wall[..., None]) + (wall_alb * (60 + lmap * 160)) * wall[..., None]
        cv2.line(frame, (0, wy + 14), (W, wy + 14), tuple(float(c) for c in np.array([60, 70, 85]) * (0.5 + glow_T)), 3, cv2.LINE_AA)
        # boiling at the surface
        for (tb, bxp, life, r) in bubbles:
            u = (t - tb) / life
            if 0 <= u < 1:
                c = np.array([90, 170, 255], np.float32) * (1 - u) * (0.4 + steam_light)
                cv2.circle(fx, (int(bxp), int(wy - 6 - 10 * u)), int(r * (0.5 + u)), tuple(float(x) for x in c), 2, cv2.LINE_AA)

        # steam: lit warm near the blade early on, cool grey later
        dens = steam.draw(t)
        d_full = cv2.resize(dens, (W, H))[..., None]
        scol = np.array([150, 158, 165], np.float32) + lmap * warm * 0.7 + np.array([10, 30, 70], np.float32) * steam_light
        frame = _bloom(frame, fx, W, H, 1.2)
        sa = np.clip(d_full * 0.8, 0, 0.85)
        frame = frame * (1 - sa) + scol * (0.55 + 0.6 * d_full) * sa
        dx, dy = _shake(t, [(T_DIP + 0.02, 0.5)], 10)
        frame = _shift(frame, dx, dy, 1.0 + 0.03 * t / v["seconds"])

        if t > T_LOGO - 0.2:
            u = smooth((t - T_LOGO + 0.2) / 0.8)
            frame = frame * (1 - u) + _soft(frame, W, H, 2.5) * u
            frame *= 1 - 0.55 * u
            lfx = np.zeros((H, W, 3), np.float32)
            logo_sparks.draw(lfx, t)
            _forged_logo(frame, lfx, v, t, T_LOGO)
            _text_block(frame, v, t, 3.2, 3.55, 3.9)
            frame = frame + lfx + _soft(lfx, W, H, 2) * 1.2
            # the last of the steam drifts across in front of the logo
            front_d = np.clip(d_full * 0.3 * (1 - smooth((t - T_LOGO) / 1.6)), 0, 0.4)
            frame = frame * (1 - front_d) + np.array([150, 156, 162], np.float32) * front_d
        return _end(frame, v, k, t)
    return frame_at


def _quench_sound(seconds, rng):
    from scipy.signal import fftconvolve
    n = int((seconds + 2) * SR)
    t = np.arange(n) / SR
    dry = np.zeros(n, np.float32)
    bed = _forge_bed(rng, seconds + 2) * 0.7
    # tongs clink as the blade is lifted over the trough
    m = int(0.4 * SR)
    tt = np.arange(m) / SR
    clink = sum(a * np.sin(2 * np.pi * f * tt) * np.exp(-tt / d) for f, d, a in ((2310, 0.09, 0.3), (3720, 0.06, 0.2), (5180, 0.04, 0.12)))
    _add(dry, clink.astype(np.float32) * 0.6, 0.18)
    # splash: the plunge
    m = int(0.35 * SR)
    sp = _filt(rng.standard_normal(m), "band", [400, 6000]) * np.exp(-np.arange(m) / (0.07 * SR))
    _add(dry, (sp / np.abs(sp).max() * 0.55).astype(np.float32), T_DIP)
    # the hiss: violent, roaring at first, sputtering as it dies; amplitude-modulated by boiling
    d = seconds - T_DIP + 0.5
    m = int(d * SR)
    tt = np.arange(m) / SR
    hiss = _filt(rng.standard_normal(m), "high", 2200)
    hiss = _filt(hiss, "low", 11000)
    body = _filt(rng.standard_normal(m), "band", [600, 3000])
    env = (1 - np.exp(-tt / 0.02)) * (0.75 * np.exp(-tt / 0.9) + 0.25 * np.exp(-tt / 2.4))
    sputter = np.clip(1 + 0.5 * _filt(rng.standard_normal(m), "low", 35) * 8, 0.2, 1.8)
    h = (hiss / np.abs(hiss).max() * 0.55 + body / np.abs(body).max() * 0.2) * env * sputter
    _add(dry, h.astype(np.float32), T_DIP + 0.01)
    # boiling: bursts of small bubbles (short upward chirps), thinning out
    for _ in range(int(220)):
        tb = T_DIP + rng.exponential(0.8)
        if tb > seconds - 0.5:
            continue
        L = int(rng.uniform(0.01, 0.03) * SR)
        tt = np.arange(L) / SR
        f0 = rng.uniform(250, 900)
        chirp = np.sin(2 * np.pi * np.cumsum(f0 * (1 + 3 * tt / tt[-1])) / SR) * np.exp(-tt / (L / SR / 3))
        _add(dry, (chirp * rng.uniform(0.05, 0.18) * np.exp(-(tb - T_DIP) / 1.2)).astype(np.float32), tb)
    # a low whump as the steam erupts
    m = int(0.8 * SR)
    tt = np.arange(m) / SR
    whump = np.sin(2 * np.pi * np.cumsum(60 + 40 * np.exp(-tt / 0.05)) / SR) * np.exp(-tt / 0.25)
    _add(dry, (whump * 0.45).astype(np.float32), T_DIP)
    irs = [_room_ir(np.random.default_rng(s), 1.1) for s in (31, 32)]
    chans = []
    for c, ir in enumerate(irs):
        wet = fftconvolve(dry, ir)[:n] * 0.08
        chans.append(dry * (1.0 if c == 0 else 0.95) + wet + bed[:n])
    return _master(chans[0], chans[1], seconds)
