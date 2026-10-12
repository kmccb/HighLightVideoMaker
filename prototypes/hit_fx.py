"""Prototype: seven effects for a big HIT (a collision, not a run), each as its own sample video.

hit_fx.py <project> <clip id> <out dir> [--only shockwave,impact,...] [--word "LIGHTS OUT"]
          [--opp X,Y --opp-t T]   the player he hits, in source-crop pixels at time T (collision course)

1 shockwave  2 impact frames  3 triple hit on the beat  4 instant replay  5 screen crack
6 collision course  7 stamp.  720p, the reel's song under each, a beat of the song on the hit.
"""
from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import signal

from hlv import beats, camera, hits, project, track

FPS, W, H = 30, 1280, 720
SR = 48000
GAP = 0.36                 # silence before the hit
LEAD, TAIL = 2.6, 2.6      # footage either side of the hit
VIEW_H = 856               # source rows shown (the reel's 1.5x zoom on this clip)
BLUE, GOLD, RED = (191, 79, 31), (63, 210, 255), (112, 84, 255)      # BGR
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


# ---------------------------------------------------------------- footage
def load(root: Path, c: dict, hit: float):
    """The play around the hit as 720p frames following the player, + where the contact is in each."""
    src, crop = root / c["source"], c.get("crop") or {"x": 0, "y": 0, "w": c["probe"]["width"], "h": c["probe"]["height"]}
    cw, ch = crop["w"], crop["h"]
    t0 = round(max(0.0, hit - LEAD) * FPS) / FPS
    n = int(round((min(c["probe"]["duration"] - 0.1, hit + TAIL) - t0) * FPS))
    tr = np.nan_to_num(track.smooth_track(np.load(camera._track_file(root / ".hlv_cache", c)), FPS), nan=-1)
    rows = np.array([tr[min(len(tr) - 1, int(round((t0 + i / FPS) * FPS)))] for i in range(n)], np.float64)
    pt = np.stack([(rows[:, 0] + 0.5 * rows[:, 2]) * cw, rows[:, 1] * ch], 1)          # contact: his leading edge
    vw, vh = VIEW_H * W / H, VIEW_H
    k = signal.windows.gaussian(61, 15)
    ctr = np.stack([np.convolve(np.pad(pt[:, j], 30, mode="edge"), k / k.sum(), "valid") for j in (0, 1)], 1)
    ox = np.clip(ctr[:, 0] - vw / 2, 0, cw - vw)
    oy = np.clip(ctr[:, 1] - vh * 0.55, 0, ch - vh)
    V = np.empty((n, H, W, 3), np.uint8)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", str(src), "-vf",
                          f"fps={FPS},crop={cw}:{ch}:{crop['x']}:{crop['y']}", "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    for i in range(n):
        buf = p.stdout.read(cw * ch * 3)
        if len(buf) < cw * ch * 3:
            n = i
            break
        f = np.frombuffer(buf, np.uint8).reshape(ch, cw, 3)
        M = np.float32([[W / vw, 0, -ox[i] * W / vw], [0, H / vh, -oy[i] * H / vh]])
        V[i] = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_AREA)
    p.stdout.close(), p.wait()
    s = W / vw
    view = lambda i, x, y: ((x - ox[i]) * s, (y - oy[i]) * s)      # noqa: E731  source-crop px -> view px
    P = np.stack([(pt[:n, 0] - ox[:n]) * s, (pt[:n, 1] - oy[:n]) * s], 1)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{n / FPS:.4f}", "-i", str(src), "-vn",
                          "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    game = np.zeros((int(n / FPS * SR) + SR, 2), np.float32)
    a = np.frombuffer(raw, np.float32).reshape(-1, 2)
    game[:len(a)] = a[:len(game)]
    return V[:n], P, game, t0, view


# ---------------------------------------------------------------- picture helpers
def ease(u):
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)


def move(f, c, zoom=1.0, dx=0.0, dy=0.0):
    M = np.float32([[zoom, 0, c[0] * (1 - zoom) + dx], [0, zoom, c[1] * (1 - zoom) + dy]])
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def flash(f, a, col=(255, 255, 255)):
    return cv2.addWeighted(f, 1 - a, np.full_like(f, col), a, 0) if a > 0 else f


_rng = np.random.default_rng(3)
_SH = _rng.uniform(-1, 1, (64, 2))


def punch(f, c, k, amp=20.0, bump=0.07, fl=0.7):
    """k frames after the hit: shake + zoom bump + flash, dying away."""
    if k < 0 or k > 22:
        return f
    d = _SH[k] * amp * np.exp(-k / 5.0)
    return flash(move(f, c, 1 + bump * np.exp(-k / 4.0), d[0], d[1]), fl * (1.0, 0.45, 0.15, 0)[min(k, 3)])


def tag(f, text):
    cv2.rectangle(f, (24, 24), (36 + 15 * len(text), 62), (12, 10, 10), -1)
    cv2.putText(f, text, (34, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)
    return f


def _remap(img, mx, my, *a, **k):
    return cv2.remap(img, np.float32(mx), np.float32(my), *a, **k)


def ripple(f, c, tt):
    d = np.hypot(XX - c[0], YY - c[1]) + 1e-3
    r0, wd, amp = 50 + 1500 * tt, 70.0, 34 * np.exp(-tt / 0.28)
    z = (d - r0) / wd
    g = np.exp(-z * z)
    ux, uy = (XX - c[0]) / d, (YY - c[1]) / d
    out = np.empty_like(f)
    for ch_, m in enumerate((1.3, 1.0, 0.75)):                    # colours split a little on the wave
        k = amp * m * g * z * 1.6
        out[..., ch_] = _remap(f[..., ch_], XX - ux * k, YY - uy * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out.astype(np.float32) + (g * amp * 2.2)[..., None], 0, 255).astype(np.uint8)


def speed_lines(c, seed, n=80):
    r = np.random.default_rng(seed)
    m = np.zeros((H, W), np.uint8)
    for a in r.uniform(0, 2 * np.pi, n):
        r_in, hw = r.uniform(170, 420), np.deg2rad(r.uniform(0.35, 1.3))
        pts = [(c[0] + r_in * np.cos(a), c[1] + r_in * np.sin(a)),
               (c[0] + 1600 * np.cos(a - hw), c[1] + 1600 * np.sin(a - hw)), (c[0] + 1600 * np.cos(a + hw), c[1] + 1600 * np.sin(a + hw))]
        cv2.fillPoly(m, [np.int32(pts)], 255, cv2.LINE_AA)
    return m


def xray(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.clip(np.abs(g - cv2.GaussianBlur(g, (0, 0), 28)) * 5.5, 0, 255).astype(np.uint8)


class Crack:
    """Glass cracked from one point: jagged spokes and two broken rings; every piece slides a little."""

    def __init__(self, c, seed=5, spokes=12):
        r = np.random.default_rng(seed)
        d = np.hypot(XX - c[0], YY - c[1])
        ang = np.arctan2(YY - c[1], XX - c[0])
        cuts = np.sort(r.uniform(-np.pi, np.pi, spokes))
        sec = np.zeros((H, W), np.int32)
        for a in cuts:
            ph, q = r.uniform(0, 1), r.uniform(48, 80)                  # a jagged line: straight runs with kinks, a few px either side
            zig = (np.abs(((d / q + ph) % 1) - 0.5) * 4 - 1) * r.uniform(5, 11) + (np.abs(((d / 19 + ph) % 1) - 0.5) * 4 - 1) * 2.5
            sec += ang > a + zig / np.maximum(d, 30)
        sec %= spokes
        band = (d > 80 + 14 * (sec % 3) - 9 * (sec % 2)).astype(np.int32) + (d > 215 + 38 * ((sec * 5) % 4) - 30)
        cell = sec * 3 + band
        mids = np.append(cuts, cuts[0] + 2 * np.pi)
        mids = (mids[:-1] + mids[1:]) / 2 + (cuts[1] - cuts[0]) * 0
        self.dx, self.dy, self.gain = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), np.ones((H, W), np.float32)
        for s in range(spokes):
            for b in range(3):
                m = cell == ((s + 1) % spokes) * 3 + b
                mag, tw = r.uniform(5, 17) * (0.5 + 0.5 * b), r.uniform(-5, 5)
                a = mids[s]
                self.dx[m], self.dy[m] = mag * np.cos(a) - tw * np.sin(a), mag * np.sin(a) + tw * np.cos(a)
                self.gain[m] = r.uniform(0.9, 1.12)
        e = (cell != np.roll(cell, 1, 0)) | (cell != np.roll(cell, 1, 1))
        e[0], e[:, 0] = False, False
        self.edge = cv2.GaussianBlur(cv2.dilate(e.astype(np.uint8) * 255, np.ones((2, 2), np.uint8)), (0, 0), 0.7).astype(np.float32) / 255
        self.shadow = np.roll(self.edge, (2, 3), (0, 1))

    def apply(self, f, s):
        out = _remap(f, XX - self.dx * s, YY - self.dy * s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        out *= (1 + (self.gain - 1) * s)[..., None]
        out *= (1 - 0.7 * s * self.shadow)[..., None]
        out += (255 - out) * (0.9 * min(1.0, s * 1.4) * self.edge)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


def stamp_layer(word, size=150):
    font = ImageFont.truetype(r"C:\Windows\Fonts\impact.ttf", size)
    x0, y0, x1, y1 = font.getbbox(word)
    tw, th, pad = x1 - x0, y1 - y0, 34
    im = Image.new("RGBA", (tw + 4 * pad, th + 3 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(pad, 0), (im.width, 0), (im.width - pad, im.height - 14), (0, im.height - 14)], fill=(BLUE[2], BLUE[1], BLUE[0], 235))
    d.polygon([(8, im.height - 10), (im.width - pad - 6, im.height - 10), (im.width - pad - 10, im.height), (4, im.height)], fill=(GOLD[2], GOLD[1], GOLD[0], 255))
    d.text((2 * pad - x0 + 5, pad - y0 + 5), word, font=font, fill=(0, 0, 0, 150))
    d.text((2 * pad - x0, pad - y0), word, font=font, fill=(255, 255, 255, 255))
    im = im.transform(im.size, Image.AFFINE, (1, 0.16, -0.16 * im.height / 2, 0, 1, 0), Image.BICUBIC).rotate(4, expand=True, resample=Image.BICUBIC)
    a = np.array(im)
    return a[..., [2, 1, 0, 3]]


def paste(f, layer, cx, cy, scale=1.0, alpha=1.0):
    lay = cv2.resize(layer, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    h, w = lay.shape[:2]
    x, y = int(cx - w / 2), int(cy - h / 2)
    a0, b0, a1, b1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if a1 <= a0 or b1 <= b0:
        return f
    cut = lay[b0 - y:b1 - y, a0 - x:a1 - x].astype(np.float32)
    al = cut[..., 3:] / 255 * alpha
    f[b0:b1, a0:a1] = (f[b0:b1, a0:a1] * (1 - al) + cut[..., :3] * al).astype(np.uint8)
    return f


_dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
_flows: dict[int, np.ndarray] = {}


def between(V, x):
    """Frame at fractional index x, by motion (optical flow), for the slow replay."""
    i, u = int(np.floor(x)), x - np.floor(x)
    if u < 0.02 or i + 1 >= len(V):
        return V[min(i, len(V) - 1)]
    if i not in _flows:
        _flows[i] = _dis.calc(cv2.cvtColor(V[i], cv2.COLOR_BGR2GRAY), cv2.cvtColor(V[i + 1], cv2.COLOR_BGR2GRAY), None)
    fl = _flows[i]
    a = _remap(V[i], XX - u * fl[..., 0], YY - u * fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b = _remap(V[i + 1], XX + (1 - u) * fl[..., 0], YY + (1 - u) * fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(a, 1 - u, b, u, 0)


# ---------------------------------------------------------------- the seven effects
# Each returns (frames, src index per frame or -1 where the game sound stops, hit time in the output,
#               sound events [(name, time, gain)], spans where the song is muffled, drop?)
def fx_shockwave(V, P, iH, per, a):
    out = []
    for i, f in enumerate(V):
        k = i - iH
        if 0 <= k < 22:
            f = punch(ripple(f, P[iH], k / FPS), P[iH], k, amp=26, bump=0.09)
        out.append(f)
    return out, list(range(len(V))), iH / FPS, [("boom", 0, 0.7), ("crack", 0, 0.5), ("air", 0.01, 0.25), ("ohh", 0.18, 0.2)], [], True


def fx_impact(V, P, iH, per, a):
    out, idx, c, hold = [], [], P[iH], 7
    x = xray(V[iH])
    for i in range(iH):
        out.append(V[i]), idx.append(i)
    for j in range(hold):
        ln = speed_lines(c, 10 + j // 2)
        g = np.maximum(x, ln)
        if j in (2, 3):
            fr = cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR)                                   # black on white
        elif j >= 4:
            fr = (g[..., None] / 255.0 * np.array([255, 190, 120]) + (1 - g[..., None] / 255.0) * np.array([40, 12, 4])).astype(np.uint8)
        else:
            fr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)                                         # white on black
        out.append(move(fr, c, 1.04 + 0.03 * j, *(_SH[j] * 10))), idx.append(-1)
    for i in range(iH, len(V)):
        out.append(punch(V[i], c, i - iH, amp=18, fl=0.9)), idx.append(i)
    th = iH / FPS
    return out, idx, th, [("boom", 0, 0.7), ("crack", 0, 0.6), ("crack", 2 / FPS, 0.3), ("crack", 4 / FPS, 0.3), ("boom", hold / FPS, 0.35), ("ohh", 0.3, 0.18)], [], True


def fx_triple(V, P, iH, per, a):
    out, idx, c, step, post = [], [], P[iH], int(round(per * FPS)), 5
    for i in range(iH + post):
        out.append(punch(V[i], c, i - iH, amp=10, bump=0.04, fl=0.35)), idx.append(i)
    for rep, (zoom, amp, fl) in enumerate(((1.16, 16, 0.55), (1.36, 30, 0.9))):
        last = rep == 1
        for j in range(step if not last else len(V) - iH + step - post):
            i = iH + post - step + j
            if i >= len(V):
                break
            z = zoom if not last else 1 + (zoom - 1) * (1 - ease((i - iH - 8) / 14))        # ease back out after the last one
            f = punch(move(V[i], c, z), c, i - iH, amp=amp, bump=0.05 + 0.04 * rep, fl=fl)
            out.append(flash(f, 0.5) if j == 0 else f), idx.append(i if last and i >= iH else -1)
    th = iH / FPS
    ev = [("boom", 0, 0.4), ("crack", 0, 0.25), ("boom", per, 0.55), ("crack", per, 0.4), ("boom", 2 * per, 0.85), ("crack", 2 * per, 0.6),
          ("air", 2 * per, 0.25), ("ohh", 2 * per + 0.2, 0.2)]
    return out, idx, th, ev, [], False


def fx_replay(V, P, iH, per, a):
    out, idx, c = [], [], P[iH]
    live_to, back, fwd, speed = min(len(V) - 1, iH + 24), 18, 12, 0.34
    for i in range(live_to):
        out.append(punch(V[i], c, i - iH, amp=12, bump=0.04, fl=0.4)), idx.append(i)
    t_rw0 = len(out) / FPS
    rw = 12
    for j in range(rw):                                              # the tape runs back
        i = int(round(live_to + (iH - back - live_to) * (j + 1) / rw))
        g = cv2.cvtColor(cv2.cvtColor(V[i], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        f = cv2.addWeighted(V[i], 0.35, g, 0.65, 0)
        for y in _rng.integers(0, H - 14, 7):
            f[y:y + 12] = np.roll(f[y:y + 12], int(_rng.integers(-60, 60)), 1)
        cv2.putText(f, "<<", (W - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
        out.append(f), idx.append(-1)
    nrep = int(round((back + fwd) / speed))
    vig = np.clip(1.15 - np.hypot((XX - W / 2) / (W * 0.62), (YY - H / 2) / (H * 0.62)) ** 2.2 * 0.75, 0.35, 1)[..., None]
    t_hit2 = None
    for j in range(nrep):
        x = iH - back + j * speed
        if t_hit2 is None and x >= iH:
            t_hit2 = len(out) / FPS
        cc = P[min(len(P) - 1, int(round(x)))]
        f = move(between(V, x), cc, 1 + 0.42 * ease(j / 26), (W / 2 - cc[0]) * 0.5 * ease(j / 26), (H * 0.52 - cc[1]) * 0.5 * ease(j / 26))
        f = punch(f, (W / 2, H / 2), int((x - iH) / speed / 2) if x >= iH else -1, amp=9, bump=0.03, fl=0.35)
        f = (f * vig).astype(np.uint8)
        cv2.circle(f, (W - 214, 56), 9, (60, 60, 255), -1, cv2.LINE_AA)
        cv2.putText(f, "REPLAY", (W - 194, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        out.append(flash(f, 0.6 * (1 - j / 3)) if j < 3 else f), idx.append(-1)
    t_back = len(out) / FPS
    for i in range(live_to, len(V)):
        out.append(flash(V[i], 0.6 * (1 - (i - live_to) / 4)) if i - live_to < 4 else V[i]), idx.append(i)
    th = iH / FPS
    ev = [("boom", 0, 0.6), ("crack", 0, 0.4), ("whoosh", t_rw0 - th - 0.05, 0.45), ("boom", t_hit2 - th, 0.75), ("air", t_hit2 - th, 0.3), ("ohh", t_hit2 - th + 0.3, 0.22)]
    return out, idx, th, ev, [(t_rw0, t_back)], True


def fx_crack(V, P, iH, per, a):
    out, cr = [], Crack(P[iH])
    for i, f in enumerate(V):
        k = i - iH
        if k >= 0:
            s = 1 - ease((k / FPS - 0.5) / 0.6)
            if s > 0:
                f = cr.apply(f, s)
            f = punch(f, P[iH], k, amp=16, bump=0.05, fl=0.8)
        out.append(f)
    ev = [("boom", 0, 0.65), ("crack", 0, 0.7), ("glass", 0.0, 0.5), ("glass", 0.12, 0.25), ("ohh", 0.25, 0.18)]
    return out, list(range(len(V))), iH / FPS, ev, [], True


def fx_course(V, P, iH, per, a):
    out, idx = [], []
    iz = int(np.clip(round((a.opp_t - a.t0) * FPS), 2, iH - 6))
    nfz = int(round(2 * per * FPS))
    me, opp = np.array(P[iz]) - (38, 0), np.array(a.view(iz, *a.opp))
    meet = me + (opp - me) * 0.5
    fz = V[iz]
    g = cv2.cvtColor(cv2.cvtColor(fz, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    dark = (g * 0.3).astype(np.float32)
    spot = np.zeros((H, W), np.float32)
    for c in (me, opp):
        spot = np.maximum(spot, np.clip(1.35 - np.hypot((XX - c[0]) / 105, (YY - c[1]) / 150), 0, 1))
    spot = cv2.GaussianBlur(np.clip(spot * 1.6, 0, 1), (0, 0), 12)[..., None]
    for i in range(iz):
        out.append(V[i]), idx.append(i)
    for j in range(nfz):
        u = j / nfz
        k = ease(j / 7) * (1 - ease((j - nfz + 5) / 5))
        f = fz.astype(np.float32) * (1 - k) + (dark * (1 - spot) + fz * spot) * k
        f = f.astype(np.uint8)
        grow = ease((u - 0.12) / 0.6)
        for c, col in ((me, GOLD), (opp, RED)):
            v = meet - c
            ln = np.hypot(*v)
            s0 = c + v / ln * 70
            e = s0 + (meet - v / ln * 22 - s0) * grow
            if grow > 0.02:
                segs = max(1, int(np.hypot(*(e - s0)) / 26))
                for q in range(segs):                                 # dashes
                    p0, p1 = s0 + (e - s0) * q / segs, s0 + (e - s0) * (q + 0.6) / segs
                    cv2.line(f, tuple(np.int32(p0)), tuple(np.int32(p1)), col, 5, cv2.LINE_AA)
                n_ = np.array([-v[1], v[0]]) / ln
                cv2.fillPoly(f, [np.int32([e + v / ln * 20, e + n_ * 13, e - n_ * 13])], col, cv2.LINE_AA)
            cv2.ellipse(f, (int(c[0]), int(c[1] + 118)), (62, 17), 0, 0, 360, col, 3, cv2.LINE_AA)
        if grow > 0.9:
            r = 16 + 10 * np.sin((u - 0.6) * 22)
            cv2.circle(f, tuple(np.int32(meet)), int(r), (255, 255, 255), 3, cv2.LINE_AA)
        out.append(flash(f, 0.5) if j == nfz - 1 else f), idx.append(-1)
    for i in range(iz, len(V)):
        out.append(punch(V[i], P[iH], i - iH, amp=24, bump=0.08, fl=0.8)), idx.append(i)
    th = (iH + nfz) / FPS
    tz = iz / FPS - th
    ev = [("thud", tz, 0.6), ("thud", tz + per, 0.6), ("boom", 0, 0.7), ("crack", 0, 0.5), ("air", 0.01, 0.25), ("ohh", 0.2, 0.2)]
    return out, idx, th, ev, [(iz / FPS, iz / FPS + nfz / FPS)], True


def fx_stamp(V, P, iH, per, a):
    out, lay = [], stamp_layer(a.word)
    k0, hold = 5, int(1.5 * FPS)
    for i, f in enumerate(V):
        k = i - iH
        f = punch(f, P[iH], k, amp=18, bump=0.06, fl=0.7)
        j = k - k0
        if 0 <= j < hold + 6:
            f = f.copy()
            if j < 5:
                f = paste(f, lay, W / 2, H - 150, 2.8 - 1.8 * ease((j + 1) / 5), 0.35 + 0.65 * (j + 1) / 5)
            elif j < hold:
                d = _SH[j] * 14 * np.exp(-(j - 5) / 3.0)
                f = paste(f, lay, W / 2 + d[0], H - 150 + d[1])
                f = flash(f, 0.35) if j == 5 else f
            else:
                u = (j - hold + 1) / 6
                f = paste(f, lay, W / 2 + 500 * u * u, H - 150, 1.0, 1 - u)
        out.append(f)
    ev = [("boom", 0, 0.65), ("crack", 0, 0.5), ("thud", (k0 + 5) / FPS, 0.9), ("crack", (k0 + 5) / FPS, 0.25), ("ohh", 0.2, 0.2)]
    return out, list(range(len(V))), iH / FPS, ev, [], True


EFFECTS = [("shockwave", "1  SHOCKWAVE", fx_shockwave), ("impact", "2  IMPACT FRAMES", fx_impact), ("triple", "3  TRIPLE HIT ON THE BEAT", fx_triple),
           ("replay", "4  INSTANT REPLAY", fx_replay), ("crack", "5  SCREEN CRACK", fx_crack), ("course", "6  COLLISION COURSE", fx_course),
           ("stamp", "7  STAMP", fx_stamp)]


# ---------------------------------------------------------------- sound
def bp(x, lo, hi):
    return signal.sosfilt(signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x, axis=0)


def bank():
    r = np.random.default_rng(7)
    norm = lambda x: x / max(1e-9, np.abs(x).max())    # noqa: E731
    t = lambda s: np.arange(int(s * SR)) / SR           # noqa: E731
    b = {}
    bt = t(1.6)
    b["boom"] = np.tanh(np.sin(2 * np.pi * np.cumsum(38 + 95 * np.exp(-bt / 0.07)) / SR) * np.exp(-bt / 0.42) * 2.2)
    ck = t(0.25)
    b["crack"] = norm(bp(r.standard_normal(len(ck)), 1200, 7000) * np.exp(-ck / 0.035))
    at = t(1.4)
    b["air"] = norm(bp(r.standard_normal(len(at)), 150, 900) * np.exp(-at / 0.5))
    tt = t(0.5)
    b["thud"] = np.tanh(np.sin(2 * np.pi * np.cumsum(45 + 60 * np.exp(-tt / 0.03)) / SR) * np.exp(-tt / 0.12) * 2.5)
    rn = t(1.3)
    ru = rn / rn[-1]
    ris = sum(bp(r.standard_normal(len(rn)), lo, hi) * np.clip((ru - w0) / (w1 - w0), 0, 1) ** 2
              for lo, hi, w0, w1 in ((400, 1500, 0.0, 0.5), (1500, 5000, 0.3, 0.85), (5000, 12000, 0.6, 1.0))) * ru ** 2.5
    ris[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))
    b["riser"] = norm(ris)
    wt = t(0.45)
    b["whoosh"] = norm(bp(r.standard_normal(len(wt)), 500, 6000) * np.sin(np.pi * wt / 0.45) ** 2 * (1 + 0.5 * np.sin(2 * np.pi * 30 * wt)))
    gt = t(0.9)
    gl = np.zeros(len(gt))
    for _ in range(26):                                              # glass: a scatter of tiny high rings
        f0, at0 = r.uniform(3200, 9500), r.uniform(0, 0.35)
        gl += np.sin(2 * np.pi * f0 * gt) * np.exp(-np.clip(gt - at0, 0, None) / r.uniform(0.02, 0.09)) * (gt >= at0) * r.uniform(0.3, 1)
    b["glass"] = norm(gl + 0.6 * bp(r.standard_normal(len(gt)), 4000, 12000) * np.exp(-gt / 0.05))
    ot = t(2.2)
    ohh = np.stack([bp(r.standard_normal(len(ot)), 300, 1100) + 0.5 * bp(r.standard_normal(len(ot)), 1100, 2600) for _ in range(2)], 1)
    b["ohh"] = norm(ohh * (np.clip(ot / 0.25, 0, 1) * np.exp(-np.clip(ot - 0.5, 0, None) / 0.6) * (1 + 0.2 * np.sin(2 * np.pi * 1.3 * ot)))[:, None])
    return b


def soundtrack(n_frames, idx, th, events, muffle, drop, game, song_at, music, B):
    N = int(round(n_frames / FPS * SR))
    tt = np.arange(N) / SR
    mix = np.zeros((N, 2), np.float32)
    spf = SR // FPS
    gs = np.zeros((N, 2), np.float32)
    for k, i in enumerate(idx):                                      # the play's own sound, wherever it runs live
        if i >= 0 and k > 0 and idx[k - 1] == i - 1:
            gs[k * spf:(k + 1) * spf] = game[i * spf:(i + 1) * spf][:max(0, N - k * spf)]
    genv = np.ones(N, np.float32)
    if drop:
        genv[(tt > th - GAP) & (tt < th)] = 0
    mix += signal.sosfilt(signal.butter(2, 60, btype="high", fs=SR, output="sos"), gs, axis=0).astype(np.float32) * genv[:, None] * 0.45
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{song_at - th:.4f}", "-t", f"{N / SR:.4f}", "-i", str(music), "-vn",
                          "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    song = np.zeros((N, 2), np.float32)
    a = np.frombuffer(raw, np.float32).reshape(-1, 2)
    song[:min(N, len(a))] = a[:N]
    if muffle:
        low = signal.sosfilt(signal.butter(4, 420, fs=SR, output="sos"), song, axis=0).astype(np.float32)
        m = np.zeros(N, np.float32)
        for a0, a1 in muffle:
            m = np.maximum(m, np.clip((tt - a0) / 0.08, 0, 1) * np.clip((a1 - tt) / 0.08, 0, 1))
        song = song * (1 - m[:, None]) + low * m[:, None] * 0.8
    menv = np.full(N, 0.75, np.float32) * np.clip(tt / 0.5, 0, 1)
    if drop:
        menv[(tt >= th - GAP) & (tt < th)] = 0
        menv[tt >= th] = 0.95
    menv *= np.clip((N / SR - tt) / 1.0, 0, 1)
    mix += song * menv[:, None]

    def place(x, at, gain):
        s = int(round(at * SR))
        x = np.stack([x, x], 1) if x.ndim == 1 else x
        a0, b0 = max(0, s), min(N, s + len(x))
        if b0 > a0:
            mix[a0:b0] += (gain * x[a0 - s:b0 - s]).astype(np.float32)
    if drop:
        place(B["riser"], th - GAP - 1.3, 0.45)
    for name, at, gain in events:
        place(B[name], th + at, gain)
    mix = np.tanh(mix * 1.1) / 1.1
    return mix


def write(out: Path, frames, mix):
    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(2), wf.setsampwidth(2), wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-i", str(wav), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-shortest", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for f in frames:
        enc.stdin.write(np.ascontiguousarray(f).tobytes())
    enc.stdin.close(), enc.wait()
    wav.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project"), ap.add_argument("clip"), ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--word", default="LIGHTS OUT")
    ap.add_argument("--opp", default=None, help="X,Y of the player he hits, source-crop px, at --opp-t")
    ap.add_argument("--opp-t", type=float, default=None)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    root = Path(a.project).resolve()
    proj = project.load(root)
    settings = project.settings(proj)
    c = next(c for c in proj["clips"] if c["id"] == a.clip)
    hit = hits.hit_of(root, c)[0]
    V, P, game, t0, view = load(root, c, hit)
    iH = int(round((hit - t0) * FPS))
    a.t0, a.view = t0, view
    a.opp = tuple(float(v) for v in a.opp.split(",")) if a.opp else None
    music = project.music_path(root, settings)
    bt = beats.cached(music, root / ".hlv_cache")
    bs, st = np.array(bt["beats"]), np.array(bt["strength"])
    per = float(np.median(np.diff(bs)))
    ok = np.where((bs > 40) & (bs < bt["duration"] - 30))[0]
    song_at = float(bs[ok[np.argmax(st[ok][:120])]])
    print(f"{len(V)} frames, hit at frame {iH}; beat every {per:.3f}s; song {song_at:.2f}s lands on the hit", flush=True)
    B = bank()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    only = [s for s in a.only.split(",") if s]
    made = []
    for key, label, fn in EFFECTS:
        if only and key not in only:
            continue
        if key == "course" and not a.opp:
            print("collision course needs --opp and --opp-t: skipped")
            continue
        frames, idx, th, ev, muffle, drop = fn(V, P, iH, per, a)
        n = len(frames)
        frames = [tag(f.copy(), label) for f in frames]
        for k in range(8):
            frames[k] = (frames[k] * (k / 8)).astype(np.uint8)
            frames[n - 1 - k] = (frames[n - 1 - k] * (k / 8)).astype(np.uint8)
        mix = soundtrack(n, idx, th, ev, muffle, drop, game, song_at, music, B)
        out = outdir / f"{a.name}{label.split()[0]} {label.split(None, 1)[1].title()}.mp4"
        write(out, frames, mix)
        made.append(out)
        print(f"{out.name}: {n / FPS:.1f}s, hit at {th:.2f}s", flush=True)
    if len(made) > 1:
        lst = outdir / "all.txt"
        lst.write_text("".join(f"file '{m.resolve().as_posix()}'\n" for m in made), encoding="utf-8")
        allv = outdir / f"{a.name}All 7 big-hit effects.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(allv)])
        lst.unlink()
        print(allv.name, flush=True)


if __name__ == "__main__":
    main()
