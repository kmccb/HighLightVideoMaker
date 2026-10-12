"""Prototype: real yards and speed for a play, read off the field's own yard lines.

field_stats.py <project> <clip id> <debug.jpg>

The camera's movement is removed (every frame mapped into the hit frame's picture, as for the
ghost trail), a clean plate of the empty field is made, and the 5-yard lines are found on it:
straight white lines that all run to one vanishing point. Lines that are evenly spaced on the
ground are spaced by a projective rule in the picture, so three or more of them give a ruler
along the field; where the player's feet fall on that ruler, frame by frame, gives yards along the
field and from that his speed. Only distance along the field is measured (not side to side), so a
diagonal run reads a little low, never high.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from hlv import camera, ghost, hits, project, segment

FPS = 30
MPH = 3600 / 1760           # yards per second -> miles per hour


def _family(lines, rng, tol_deg=1.2):
    """The biggest set of lines that run to one point: (indices, vanishing point as a homogeneous vector)."""
    best, best_vp = [], None
    for _ in range(800):
        i, j = rng.choice(len(lines), 2, replace=False)
        vp = np.cross(lines[i][0], lines[j][0])
        if np.abs(vp).max() < 1e-12:
            continue
        inl = []
        for k, (ln, mid, d, L) in enumerate(lines):
            to = vp[:2] - mid * vp[2]                            # direction from the line's middle to the point (also right when it's at infinity)
            n = np.hypot(*to)
            if n > 1e-9 and abs(d[0] * to[1] - d[1] * to[0]) / n < np.sin(np.radians(tol_deg)):
                inl.append(k)
        if sum(lines[k][3] for k in inl) > sum(lines[k][3] for k in best):
            best, best_vp = inl, vp
    return best, best_vp


def yard_lines(plate: np.ndarray, at: tuple[float, float], debug: list | None = None):
    """The 5-yard lines in a picture of the empty field. Depending on where the camera is they can run
    up the picture (from the sideline) or nearly across it (from behind the end zone), so nothing is
    assumed about their direction: the straight white lines are sorted into families that each run to
    one vanishing point, and the yard lines are the family that makes an even ruler with the most
    lines. Returns (vanishing point, a point and direction of the ruler line through `at`, where each
    yard line crosses it), or None."""
    H, W = plate.shape[:2]
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    field = cv2.dilate(cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)), np.ones((9, 9), np.uint8))
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # a painted line is brighter than the turf on both sides of it: top-hat picks those out whatever the light
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    white = ((top > max(18.0, float(np.percentile(top[field > 0], 97)) * 0.6)) & (field > 0) & (hsv[..., 1] < 110)).astype(np.uint8) * 255
    if debug is not None:
        debug.append(white)
    segs = cv2.HoughLinesP(white, 1, np.pi / 720, threshold=60, minLineLength=int(0.14 * H), maxLineGap=int(0.04 * H))
    if segs is None:
        return None
    lines = []                                                   # homogeneous line, midpoint, direction, length
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(np.float64):
        ln = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        d = np.array([x2 - x1, y2 - y1])
        L = float(np.hypot(*d))
        lines.append((ln / np.hypot(ln[0], ln[1]), np.array([(x1 + x2) / 2, (y1 + y2) / 2]), d / L, L))
    rng = np.random.default_rng(0)
    left = list(range(len(lines)))
    best = None
    fams = []
    for _ in range(3):                                           # up to three families
        if len(left) < 3:
            break
        sub = [lines[k] for k in left]
        inl, vp = _family(sub, rng)
        if len(inl) < 3:
            break
        fam = [sub[k] for k in inl]
        fams.append((vp, sum(f[3] for f in fam)))
        left = [left[k] for k in range(len(sub)) if k not in set(inl)]
        dmean = np.mean([f[2] * np.sign(f[2] @ fam[0][2]) for f in fam], axis=0)
        tdir = np.array([-dmean[1], dmean[0]]) / np.hypot(*dmean)     # the ruler runs square across them, through his feet
        tline = np.cross([at[0], at[1], 1.0], [at[0] + tdir[0], at[1] + tdir[1], 1.0])
        xs = []
        for ln, mid, d, L in fam:
            pt = np.cross(ln, tline)
            if abs(pt[2]) > 1e-9:
                xs.append((float((pt[:2] / pt[2] - np.array(at)) @ tdir), L))
        xs.sort()
        gaps = np.diff([x for x, _ in xs])
        big = float(np.median(gaps[gaps > 0.004 * W])) if (gaps > 0.004 * W).any() else 0.0
        merged = []                                              # both edges of one painted line, or pieces of it: one line
        for x, L in xs:
            if merged and x - merged[-1][0] < max(0.004 * W, 0.3 * big):
                a, b = merged[-1]
                merged[-1] = ((a * b + x * L) / (b + L), b + L)
            else:
                merged.append((x, L))
        ss = [m[0] for m in merged]
        r = ruler(ss)
        if r and (best is None or len(ss) > len(best[3])):
            best = (vp, np.array(at, np.float64), tdir, ss, r)
    if best is None:
        return None
    return (*best, [f for f in fams if f[0] is not best[0]])


def ground(v1: np.ndarray, others: list, W: int, H: int):
    """A flat map of the field from the picture. The yard lines run to v1; lines along the field
    (sidelines, hash rows) run to another point, and since those two directions are square to each
    other on the ground, the pair gives the camera's focal length and with it the field as seen
    from above. Returns (function picture point -> (across, along) in unscaled ground units, focal
    length) for the best second family, or None when no believable one was found."""
    cx, cy = W / 2, H / 2
    for v2, _ in sorted(others, key=lambda f: -f[1]):
        w = v1[2] * v2[2]
        if abs(w) < 1e-12:
            continue
        f2 = -((v1[0] - cx * v1[2]) * (v2[0] - cx * v2[2]) + (v1[1] - cy * v1[2]) * (v2[1] - cy * v2[2])) / w
        if not (0.4 * W) ** 2 < f2 < (8 * W) ** 2:
            continue
        f = float(np.sqrt(f2))
        Ki = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1.0]])
        r1 = Ki @ v1
        r1 /= np.linalg.norm(r1)
        r2 = Ki @ v2
        r2 -= (r2 @ r1) * r1
        r2 /= np.linalg.norm(r2)
        nrm = np.cross(r1, r2)

        def to_ground(pt, Ki=Ki, r1=r1, r2=r2, nrm=nrm):
            q = Ki @ np.array([pt[0], pt[1], 1.0])
            return np.array([q @ r1, q @ r2]) / (q @ nrm)
        return to_ground, f
    return None


def ruler(xs: list[float]):
    """Evenly spaced lines on the ground cross a row of the picture at x = (a u + b) / (c u + 1) for
    u = 0, 1, 2, ... Fit that, allowing for lines that weren't found (a gap about twice its
    neighbours). Returns a function picture x -> yards, or None."""
    if len(xs) < 3:
        return None
    xs = np.array(xs, np.float64)
    best = None
    gaps = np.diff(xs)
    for skips in _skip_options(gaps):
        u = np.concatenate([[0], np.cumsum(skips)]).astype(np.float64)
        # x (c u + 1) = a u + b  ->  a u + b - c u x = x
        A = np.c_[u, np.ones_like(u), -u * xs]
        sol, *_ = np.linalg.lstsq(A, xs, rcond=None)
        a, b, c = sol
        fit = (a * u + b) / (c * u + 1)
        err = float(np.abs(fit - xs).max()) + 4.0 * (sum(skips) - len(skips))     # prefer fewer invented lines
        if best is None or err < best[0]:
            best = (err, a, b, c, u)
    err, a, b, c, u = best
    if err > max(6.0, 0.03 * float(np.median(gaps))) + 4.0 * (u[-1] - (len(xs) - 1)):
        return None
    return lambda x: 5.0 * (x - b) / (a - c * x), err, u


def _skip_options(gaps: np.ndarray):
    """Ways of counting the lines between neighbours: 1 each, or 2 where a gap is far bigger than those next to it."""
    base = [1] * len(gaps)
    yield base
    med = float(np.median(gaps))
    for k, g in enumerate(gaps):
        if g > 1.6 * med:
            alt = list(base)
            alt[k] = int(round(g / med))
            yield alt
    wide = [int(max(1, round(g / med))) for g in gaps]
    if wide != base:
        yield wide


def main():
    root = Path(sys.argv[1]).resolve()
    proj = project.load(root)
    c = next(c for c in proj["clips"] if c["id"] == sys.argv[2])
    cache = root / ".hlv_cache"
    info, crop = c["probe"], c.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    Wc, Hc = (crop["w"], crop["h"]) if crop else (info["width"], info["height"])
    ws = min(1.0, 1920 / Wc)
    W, H = int(round(Wc * ws / 2)) * 2, int(round(Hc * ws / 2)) * 2
    t_in, t_out, snap = segment._window(c)
    hit = hits.hit_of(root, c)[0]
    tF = round(float(hit) * FPS) / FPS
    ta = round(max(0.0, snap - 0.3, tF - 8.0) * FPS) / FPS
    tb = min(info["duration"] - 0.2, t_out, tF + 2.0)
    n = int(round((tb - ta) * FPS))
    iF = int(round((tF - ta) * FPS))
    frames = ghost._read(root / c["source"], ta, n, crop, (W, H))
    n = len(frames)
    G = ghost._homographies(frames, iF)
    idx = list(range(0, n, max(3, n // 45)))
    stack = np.stack([cv2.warpPerspective(frames[k], G[k], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for k in idx])
    plate = np.median(stack, axis=0).astype(np.uint8)
    del stack
    player = np.array(camera.plan(cache, c)["player"])
    feet = {}
    for i in range(n):
        p = player[min(len(player) - 1, int(round((ta + i / FPS) * FPS)))]
        if p[0] < 0 or np.isnan(p[0]) or p[4] < 0.3:
            continue
        x, y = (p[0] * info["width"] - ox) * ws, (p[1] * info["height"] - oy) * ws + p[3] * info["height"] * ws * 0.5
        feet[i] = ghost._pt(G[i], x, y)
    if len(feet) < 20:
        print("not tracked enough")
        return
    at = (float(np.median([f[0] for f in feet.values()])), float(np.median([f[1] for f in feet.values()])))
    dbg: list = []
    found = yard_lines(plate, at, dbg)
    if not found:
        print("no even set of yard lines found")
        cv2.imwrite(sys.argv[3], np.hstack([plate, cv2.cvtColor(dbg[0], cv2.COLOR_GRAY2BGR)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return
    vp, t0, tdir, xs, (yards, err, u), others = found
    tline = np.cross([t0[0], t0[1], 1.0], [t0[0] + tdir[0], t0[1] + tdir[1], 1.0])
    print(f"{c['source']}: {len(xs)} yard lines; ruler fits to {err:.1f} px; line numbers {u.astype(int).tolist()}")
    ks = sorted(feet)

    def along(pt):                                               # the yard line through his feet, followed to the ruler
        x = np.cross(np.cross(vp, [pt[0], pt[1], 1.0]), tline)
        return yards(float((x[:2] / x[2] - t0) @ tdir))

    yd = np.array([along(feet[k]) for k in ks])
    tt = np.array(ks) / FPS
    smooth = lambda z, sg: cv2.GaussianBlur(np.asarray(z, np.float64).reshape(-1, 1), (1, 0), sg, borderType=cv2.BORDER_REPLICATE).ravel()    # noqa: E731
    sm = smooth(yd, 7.5)
    v = smooth(np.abs(np.gradient(sm, tt)), 4.5)
    print(f"tracked {tt[0]:.1f}-{tt[-1]:.1f}s; ALONG THE FIELD ONLY: covered {sm.max() - sm.min():.1f} yd (net {abs(sm[-1] - sm[0]):.1f}), top {v.max() * MPH:.1f} mph")
    gm = ground(vp, others, W, H)
    if gm:
        to_ground, f = gm
        # the scale: each yard line is 5 yards further along than the last
        al = np.array([to_ground(t0 + tdir * x)[1] for x in xs])
        k_, c_ = np.polyfit(u * 5.0, al, 1)
        resid = float(np.abs(al - (k_ * u * 5.0 + c_)).max() / abs(k_))
        xy = np.array([to_ground(feet[k]) for k in ks]) / abs(k_)        # yards: (across, along)
        xs_, ys_ = smooth(xy[:, 0], 7.5), smooth(xy[:, 1], 7.5)
        sp = smooth(np.hypot(np.gradient(xs_, tt), np.gradient(ys_, tt)), 4.5)
        dist = float(np.hypot(np.diff(xs_), np.diff(ys_)).sum())
        print(f"FULL MAP (focal {f:.0f} px, yard lines land within {resid:.2f} yd of an even 5): ran {dist:.1f} yd "
              f"({abs(ys_[-1] - ys_[0]):.1f} along the field, {abs(xs_[-1] - xs_[0]):.1f} across), top speed {sp.max() * MPH:.1f} mph at {tt[int(sp.argmax())]:.1f}s, "
              f"average {dist / (tt[-1] - tt[0]) * MPH:.1f} mph")
    else:
        print("no second set of lines (sidelines / hash rows) found: only distance along the field can be measured")
    out = plate.copy()
    for k, x in enumerate(xs):
        on = t0 + tdir * x
        ln = np.cross(vp, [on[0], on[1], 1.0])
        d = np.array([-ln[1], ln[0]]) / np.hypot(ln[0], ln[1])
        cv2.line(out, tuple(np.round(on - d * 2500).astype(int)), tuple(np.round(on + d * 2500).astype(int)), (0, 0, 255), 2)
        cv2.putText(out, f"{int(u[k]) * 5}", (int(on[0]) + 6, int(on[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    pts = np.array([feet[k] for k in ks], np.int32)
    cv2.polylines(out, [pts], False, (255, 0, 255), 3)
    for k in ks[::15]:
        cv2.putText(out, f"{along(feet[k]):.0f}", (int(feet[k][0]), int(feet[k][1]) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    small = cv2.resize(cv2.cvtColor(dbg[0], cv2.COLOR_GRAY2BGR), (W // 3, H // 3))
    out[:H // 3, :W // 3] = small
    cv2.imwrite(sys.argv[3], out, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
