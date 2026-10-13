"""The ghost trail for a long run: a kickoff return, a breakaway, anything where the camera pans so
far that the run doesn't fit in the one frame the ordinary trail (ghost.render) is drawn in.

The camera's movement over the whole run is chained together, which gives every moment of the run
a place on one long panorama (drawn in the picture of the run's middle frame, so neither end is
stretched much). The empty field is built for that whole panorama (a median over time, tile by
tile) and he is cut out of his own frames at evenly spaced moments. The freeze is then a camera
move that retraces the real camera: it whips back through the pan to where the run began, then
runs forward through it again with the field empty, the ghosts landing one after another, until it
arrives back on him and settles on the reel's own framing. Because each picture is drawn as the
real camera saw that part of the field, nothing looks stretched however far the pan went. Same
length and same result files as ghost.render, so nothing else changes.
"""
from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path

import cv2
import numpy as np

from . import encoder, ghost

FPS = 30
LONG_MAX = 16.0           # seconds of run before the freeze it can show
SMALL_W = 960             # the camera's movement is followed at this width (every frame of the run is held)
GHOSTS = 14
T_WHIP = 0.7              # seconds for the camera to get back to the start of the run
PULL = 1.3                # the travelling view is this much wider than the reel's own framing
TILE = 768


def _tr(x: float, y: float) -> np.ndarray:
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1.0]])


def _quad(Hm: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    return cv2.perspectiveTransform(np.float64([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2), Hm).reshape(-1, 2)


def render(root: Path, clip: dict, pl: dict | None, tF: float, start: float, secs: float, period: float | None,
           size: tuple[int, int], mp4: Path, meta: Path, plan: dict, log=print) -> Path | None:
    def none(why: str):
        meta.write_text(json.dumps({"none": why}))
        log(f"  no ghost trail for {clip['source']}: {why}")
        return None

    OW, OH = size
    A = OW / OH
    src = Path(root) / clip["source"]
    info, crop = clip["probe"], clip.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    Wc, Hc = (crop["w"], crop["h"]) if crop else (info["width"], info["height"])
    s1 = min(1.0, SMALL_W / Wc)
    W1, H1 = int(round(Wc * s1 / 2)) * 2, int(round(Hc * s1 / 2)) * 2
    ta = round(max(start, tF - LONG_MAX) * FPS) / FPS
    n = int(round((tF - ta) * FPS)) + 1
    fr = ghost._read(src, ta, n, crop, (W1, H1))
    if len(fr) < n:
        return none("the video ends before the hit")
    mid = n // 2
    G1 = ghost._homographies(fr, mid)                           # every frame -> the middle frame's picture
    del fr
    k = min(Wc, ghost.WORK_W) / W1                              # small px -> working px
    Wp, Hp = int(round(W1 * k / 2)) * 2, int(round(H1 * k / 2)) * 2
    K, Ki = np.diag([k, k, 1.0]), np.diag([1 / k, 1 / k, 1.0])
    HP = [K @ g @ Ki for g in G1]                               # the same, at working size
    HP = [h / h[2, 2] for h in HP]
    player = np.array(plan["player"], np.float64)

    own, pos = {}, {}                                           # him in frame i: in its own picture / on the panorama (working px)
    for i in range(n):
        cx, cy, bw, bh, conf = player[min(len(player) - 1, int(round((ta + i / FPS) * FPS)))]
        if cx < 0 or np.isnan(cx) or conf < 0.3 or bw <= 0:
            continue
        own[i] = ((cx * info["width"] - ox) * s1 * k, (cy * info["height"] - oy) * s1 * k, bw * info["width"] * s1 * k, bh * info["height"] * s1 * k)
        pos[i] = (*ghost._pt(HP[i], own[i][0], own[i][1]), own[i][2], own[i][3])
    last = max((i for i in own if i >= n - 7), default=None)
    if last is None:
        return none("he isn't found at the hit")
    keys = sorted(i for i in own if i < last)
    if len(keys) < 12:
        return none("he isn't followed for enough of the run")
    pts = np.array([pos[i][:2] for i in keys[::5]])
    length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    fw = float(np.median([own[i][2] for i in keys]))
    chosen = ghost.pick({i: pos[i] for i in keys}, pos[last][:2], max(1.55 * fw, length / GHOSTS), GHOSTS + 2)   # spread over the whole run
    if len(chosen) < ghost.MIN_GHOSTS:
        return none("he doesn't cover enough ground before the hit")
    own[n - 1] = own[last]
    track_i = sorted(own)
    him = lambda r: tuple(np.interp(r, track_i, [own[i][j] for i in track_i]) for j in range(4))    # noqa: E731

    # ---- the camera: the real one, replayed. r = which frame of the run it stands in ----
    wide = list(ghost.reel_view(clip, plan, tF, size) * s1 * k)             # the reel's own framing of the freeze frame
    tvw = wide[2] * PULL
    tvh = tvw / A
    t_first = T_WHIP + 0.05
    room = max(0.8, secs - ghost.T_OUT - 0.9 - t_first)
    step = min(period / 2, room / len(chosen)) if period else room / len(chosen)
    T_G = [round(t_first + j * step, 4) for j in range(len(chosen))]
    t_arrive = T_G[-1] + 0.45
    t_hold_end = secs - ghost.T_OUT

    def cam(t: float):
        """(frame of the run the camera stands in, its window in that frame, how far back on the live picture)."""
        out_u = ghost._ease((t - t_hold_end) / ghost.T_OUT)
        if t < T_WHIP:                                          # whipping back to the start of the run
            u = ghost._ease(t / T_WHIP)
            r = (n - 1) + (chosen[0] - (n - 1)) * u
        else:
            u = 1.0
            r = float(np.interp(t, [T_WHIP, *T_G, t_arrive], [chosen[0], *chosen, n - 1]))
        hx, hy, _, hh = him(r)
        tg = [hx - tvw / 2, hy - 0.1 * hh - tvh / 2, tvw, tvh]
        z = u * (1 - out_u)
        return r, [wide[j] + (tg[j] - wide[j]) * z for j in range(4)], out_u

    def H_at(r: float) -> np.ndarray:
        i = int(np.clip(np.floor(r), 0, n - 1))
        u = float(r - i)
        return HP[i] if u < 1e-3 or i + 1 >= n else (1 - u) * HP[i] + u * HP[i + 1]

    def out_from_pano(r: float, rect) -> np.ndarray:
        """Panorama (middle frame's picture, working px) -> the output picture, for the camera standing in frame r."""
        x, y, w, h = rect
        V = np.array([[OW / w, 0, -x * OW / w], [0, OH / h, -y * OH / h], [0, 0, 1.0]])
        Hr = H_at(r)
        P = V @ np.linalg.inv(Hr)
        return P / float(P[2] @ (Hr @ np.array([x + w / 2, y + h / 2, 1.0])))      # depth 1 at the middle of the view

    # the part of the panorama any of those pictures shows
    quads = []
    for t in np.linspace(0, secs, 150):
        r, rect, _ = cam(t)
        quads.append(_quad(H_at(r), *rect))
    quads.append(_quad(HP[n - 1], 0, 0, Wp, Hp))
    qa = np.concatenate(quads)
    if not np.isfinite(qa).all():
        return none("the camera moves too far over this run to show it")
    cx0, cy0 = int(np.floor(qa[:, 0].min())) - 8, int(np.floor(qa[:, 1].min())) - 8
    CW, CH = int(np.ceil(qa[:, 0].max())) + 8 - cx0, int(np.ceil(qa[:, 1].max())) + 8 - cy0
    if CW > 9 * Wp or CH > 4 * Hp:
        return none("the camera moves too far over this run to show it")
    T0 = _tr(-cx0, -cy0)

    # ---- the empty field over that whole stretch: a median over time, tile by tile ----
    sample = sorted(set(range(0, n, max(2, n // 70))) | set(chosen) | {n - 1})
    frP = ghost._read(src, ta, n, crop, (Wp, Hp), keep=set(sample))
    if any(i not in frP for i in sample):
        return none("the video couldn't be read again")
    box = {i: _quad(T0 @ HP[i], 0, 0, Wp, Hp) for i in sample}
    inside = cv2.erode(np.ones((Hp, Wp), np.uint8), np.ones((5, 5), np.uint8), borderValue=0)
    plate = np.zeros((CH, CW, 3), np.uint8)
    cover = np.zeros((CH, CW), np.uint8)
    for x0 in range(0, CW, TILE):
        x1 = min(CW, x0 + TILE)
        use = [i for i in sample if box[i][:, 0].min() < x1 and box[i][:, 0].max() > x0]
        if not use:
            continue
        use = [use[j] for j in sorted(set(np.linspace(0, len(use) - 1, min(len(use), 13)).round().astype(int)))]
        Tt = _tr(-x0, 0) @ T0
        stack = np.full((len(use), CH, x1 - x0, 3), np.nan, np.float32)
        for j, i in enumerate(use):
            m = cv2.warpPerspective(inside, Tt @ HP[i], (x1 - x0, CH), flags=cv2.INTER_NEAREST) > 0
            stack[j][m] = cv2.warpPerspective(frP[i], Tt @ HP[i], (x1 - x0, CH), flags=cv2.INTER_LINEAR)[m]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(stack, axis=0)
        ok = ~np.isnan(med[..., 0])
        plate[:, x0:x1][ok] = med[ok].astype(np.uint8)
        cover[:, x0:x1] = ok
        del stack
    if not cover.all():                                         # what no frame saw: a soft fill from what's around it
        sm = (max(1, CW // 8), max(1, CH // 8))
        c8 = cv2.resize(cover.astype(np.float32), sm, interpolation=cv2.INTER_AREA)
        p8 = cv2.resize(plate.astype(np.float32) * cover[..., None], sm, interpolation=cv2.INTER_AREA)
        for _ in range(6):
            c8, p8 = cv2.GaussianBlur(c8, (0, 0), 6), cv2.GaussianBlur(p8, (0, 0), 6)
        fill = cv2.resize(p8 / np.maximum(c8[..., None], 1e-4), (CW, CH), interpolation=cv2.INTER_LINEAR)
        plate[cover == 0] = np.clip(fill[cover == 0], 0, 255).astype(np.uint8)

    # the freeze frame itself sits on the panorama where it belongs, its edges faded into it
    edge = np.minimum(np.minimum(np.arange(Wp)[None, :], Wp - 1 - np.arange(Wp)[None, :]),
                      np.minimum(np.arange(Hp)[:, None], Hp - 1 - np.arange(Hp)[:, None])).astype(np.float32)
    feather = np.clip(edge / 40, 0, 1)
    canvas = plate.copy()
    fa = cv2.warpPerspective(feather, T0 @ HP[n - 1], (CW, CH), flags=cv2.INTER_LINEAR)[..., None]
    ff = cv2.warpPerspective(frP[n - 1], T0 @ HP[n - 1], (CW, CH), flags=cv2.INTER_LINEAR)
    canvas = (canvas * (1 - fa) + ff * fa).astype(np.uint8)
    del fa, ff
    full = None                                                 # the start and end must be as sharp as the live picture
    if Wc > Wp:
        got = ghost._read(src, ta, n, crop, (Wc, Hc), keep={n - 1})
        if n - 1 in got:
            full = ghost._like(got[n - 1], frP[n - 1])
    Sfull = np.diag([Wp / Wc, Hp / Hc, 1.0])

    def cut(i: int, grow: float = 1.0):
        """Him in frame i, cut out by what differs from the empty field: (x, y in frame i, picture, soft mask)."""
        x, y, bw, bh = own[i]
        X0, Y0 = int(np.clip(x - 1.7 * bw, 0, Wp - 4)), int(np.clip(y - 1.2 * bh, 0, Hp - 4))
        w, h = int(min(3.4 * bw, Wp - X0)), int(min(2.4 * bh, Hp - Y0))
        img = frP[i][Y0:Y0 + h, X0:X0 + w]
        field = cv2.warpPerspective(plate, _tr(-X0, -Y0) @ np.linalg.inv(T0 @ HP[i]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lx, ly = x - X0, y - Y0
        diff = cv2.absdiff(img, field).max(axis=2)
        m = (cv2.GaussianBlur(diff, (0, 0), 1.2) > 26).astype(np.uint8)
        oval = np.zeros((h, w), np.uint8)
        cv2.ellipse(oval, (int(lx), int(ly)), (int(bw * 0.66 * grow), int(bh * 0.66 * grow)), 0, 0, 360, 1, -1)
        m &= oval
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        core = (slice(max(0, int(ly - bh * 0.5)), int(ly + bh * 0.5) + 1), slice(max(0, int(lx - bw * 0.35)), int(lx + bw * 0.35) + 1))
        for lb in range(1, nlab):
            if stats[lb, cv2.CC_STAT_AREA] >= 0.03 * bw * bh and (lab[core] == lb).any():
                keep[lab == lb] = 1
        if keep.sum() < 0.12 * bw * bh:                         # too little found: a soft oval on him instead
            keep[:] = 0
            cv2.ellipse(keep, (int(lx), int(ly)), (int(bw * 0.42), int(bh * 0.52)), 0, 0, 360, 1, -1)
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return X0, Y0, img.copy(), cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.3)

    tc = (pl or {}).get("color") or "#1f4fd1"
    team = np.array([int(tc[j:j + 2], 16) for j in (5, 3, 1)], np.float32)          # BGR
    ghosts = []
    for i in chosen:
        X0, Y0, img, gm = cut(i)
        rim = np.clip(cv2.GaussianBlur(gm, (0, 0), 3.5) * 1.5 - gm, 0, 1) * 0.55
        ghosts.append((HP[i] @ _tr(X0, Y0), img, gm, rim))       # (its place on the panorama, picture, mask, rim)
    lX, lY, _, l_m = cut(n - 1, 1.15)
    l_H = HP[n - 1] @ _tr(lX, lY)
    feet_i = [i for i in track_i if chosen[0] <= i <= last]
    feet = np.array([ghost._pt(HP[i], own[i][0], own[i][1] + own[i][3] * 0.5) for i in feet_i], np.float64)
    if len(feet) > 9:
        feet = cv2.GaussianBlur(feet.reshape(-1, 1, 2), (1, 9), 0).reshape(-1, 2)
    del frP
    S = min(OW, OH) / 1080.0
    white = np.full((OH, OW, 3), 255, np.uint8)

    def layer(P: np.ndarray, Hl: np.ndarray, *imgs):
        """Pieces that live at Hl on the panorama, as they land in the output: (rows, cols, pieces...) or None."""
        h, w = imgs[0].shape[:2]
        Mq = P @ Hl
        if Mq[2] @ np.array([w / 2, h / 2, 1.0]) < 0.2:                   # behind / far off to the side of this camera
            return None
        q = _quad(Mq, 0, 0, w, h)
        if not np.isfinite(q).all() or np.abs(q).max() > 20000:
            return None
        bx0, by0 = max(0, int(np.floor(q[:, 0].min()))), max(0, int(np.floor(q[:, 1].min())))
        bx1, by1 = min(OW, int(np.ceil(q[:, 0].max()))), min(OH, int(np.ceil(q[:, 1].max())))
        if bx1 - bx0 < 2 or by1 - by0 < 2:
            return None
        Mg = _tr(-bx0, -by0) @ Mq
        return (slice(by0, by1), slice(bx0, bx1),
                *[cv2.warpPerspective(im, Mg, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR) for im in imgs])

    def frame_at(t: float) -> np.ndarray:
        r, rect, out_u = cam(t)
        P = out_from_pano(r, rect)
        out = cv2.warpPerspective(canvas, P @ np.linalg.inv(T0), (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if full is not None and r > n - 1 - 45:                 # the freeze frame, from the full-size picture
            Mf = P @ HP[n - 1]
            af = cv2.warpPerspective(feather, Mf, (OW, OH), flags=cv2.INTER_LINEAR)[..., None]
            if af.max() > 0:
                sharp = cv2.warpPerspective(full, Mf @ Sfull, (OW, OH), flags=cv2.INTER_AREA if OW / rect[2] * Wp / Wc < 0.75 else cv2.INTER_CUBIC)
                out = (out * (1 - af) + sharp * af).astype(np.uint8)
        lit = out
        dim = 0.5 * ghost._ease(t / 0.4) * (1 - out_u)
        img = cv2.convertScaleAbs(out, alpha=1 - dim)
        shown = [j for j, tg in enumerate(T_G) if t >= tg]
        if shown and out_u < 1:                                 # the path, drawn up to the newest ghost
            upto = last if len(shown) == len(ghosts) and t >= T_G[-1] + 0.25 else chosen[shown[-1]]
            p = feet[[q for q, fi in enumerate(feet_i) if fi <= upto]]
            if len(p) >= 3:
                v = np.c_[p, np.ones(len(p))] @ P.T
                okp = v[:, 2] > 0.2
                p = (v[okp, :2] / v[okp, 2:3])
                p = p[(np.abs(p) < 8000).all(axis=1)]
            if len(p) >= 3:
                line = np.zeros((OH, OW), np.float32)
                cv2.polylines(line, [np.round(p * 8).astype(np.int32)], False, 1.0, max(2, int(round(3 * S))), cv2.LINE_AA, shift=3)
                glow = cv2.GaussianBlur(line, (0, 0), 7 * S) * 2.2
                img = np.clip(img + (glow[..., None] * team + line[..., None] * 150) * (0.9 * (1 - out_u)), 0, 255).astype(np.uint8)
        for j in shown:
            Hl, gimg, gm, rim = ghosts[j]
            L = layer(P, Hl, gimg, gm, rim)
            if L is None:
                continue
            rs, cs_, g, m, rm = L
            age = t - T_G[j]
            alpha = (0.55 + 0.4 * (j + 1) / len(ghosts)) * ghost._ease(age / 0.12) * (1 - out_u)
            popv = max(0.0, 1 - age / 0.22)                     # a white flash as it lands
            g = g.astype(np.float32)
            if popv > 0:
                g = g * (1 + 0.9 * popv) + 255 * 0.5 * popv
            m = m[..., None] * alpha
            patch = img[rs, cs_].astype(np.float32) + rm[..., None] * team * (1 - out_u)
            img[rs, cs_] = np.clip(patch * (1 - m) + g * m, 0, 255).astype(np.uint8)
        if dim > 0:                                             # he stays lit
            L = layer(P, l_H, l_m)
            if L is not None:
                rs, cs_, m = L
                m = m[..., None] * dim
                img[rs, cs_] = np.clip(img[rs, cs_] * (1 - m) + lit[rs, cs_] * m, 0, 255).astype(np.uint8)
        flash = max(0.0, 1 - t / 0.18) * 0.55                   # the hit itself
        return cv2.addWeighted(img, 1 - flash, white, flash, 0) if flash > 0 else img

    tmp = mp4.with_name(mp4.stem + ".tmp.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i", f"anullsrc=r={ghost.SR}:cl=stereo",
                            *encoder.h264(17, software=True), "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(tmp)],
                           stdin=subprocess.PIPE)
    try:
        for j in range(int(round(secs * FPS))):
            enc.stdin.write(np.ascontiguousarray(frame_at(j / FPS)).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ghost trail render failed for {clip['source']}")
    tmp.replace(mp4)
    meta.write_text(json.dumps({"pops": T_G, "seconds": secs, "ghosts": len(ghosts), "long": True}))
    log(f"  ghost trail {clip['source']}: {len(ghosts)} ghosts over the whole run ({(tF - ta):.1f} s, a picture {CW / Wp:.1f} frames wide)")
    return mp4
