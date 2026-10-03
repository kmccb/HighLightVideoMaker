"""Follow one player from the spotlight clicks you place in the review page.

Each click is an anchor (time + box). From every anchor we run a CSRT tracker forward
to the next anchor and backward to the previous one, then fuse: a frame's position
comes from the nearer/more trustworthy anchor. Where neither run is trustworthy the
confidence drops, and the camera planner eases back to wide instead of guessing.

All coordinates are normalised to the full display frame (0..1), so they apply to the
720p review proxy and the original source alike.
"""
from __future__ import annotations

import cv2
import numpy as np

TRUST_S = 10.0         # anchor trust halves every this many seconds of tracking (drift)
MAX_STEP = 0.06        # a jump bigger than this (fraction of frame width) per frame = lost


def _csrt():
    p = cv2.TrackerCSRT_Params()
    p.padding = 3.0                 # bigger search window copes with fast pans
    return cv2.TrackerCSRT_create(p)


def _camera_moved(prev_f: np.ndarray, cur_f: np.ndarray, pt: tuple) -> tuple:
    """Where point pt in prev_f ends up in cur_f if only the camera moved (pan/zoom)."""
    from .motion import estimate_global
    s = 0.5
    g0 = cv2.cvtColor(cv2.resize(prev_f, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cv2.resize(cur_f, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    m = estimate_global(g0, g1)
    x, y = pt[0] * s, pt[1] * s
    return ((m[0, 0] * x + m[0, 1] * y + m[0, 2]) / s, (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / s)


def _hist(frame: np.ndarray, x: float, y: float, bw: float, bh: float):
    """Colour signature of the *player* in a box (jersey, pants, helmet), or None.

    Only non-grass pixels count: a loose box is mostly turf, and turf colour changes with
    lighting across the field, which used to make a correctly tracked player look 'lost'.
    """
    H, W = frame.shape[:2]
    x0, x1 = int(max(0, x + bw * 0.1)), int(min(W, x + bw * 0.9))
    y0, y1 = int(max(0, y + bh * 0.05)), int(min(H, y + bh * 0.95))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = frame[y0:y1, x0:x1]
    mask = _body_mask(patch)
    if int((mask > 0).sum()) < 20:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h)


def _body_frac(frame: np.ndarray, x: float, y: float, bw: float, bh: float) -> float:
    """Share of the box centre that is not grass, i.e. how much 'person' is in it."""
    H, W = frame.shape[:2]
    x0, x1 = int(max(0, x + bw * 0.2)), int(min(W, x + bw * 0.8))
    y0, y1 = int(max(0, y + bh * 0.1)), int(min(H, y + bh * 0.9))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    return float((_body_mask(frame[y0:y1, x0:x1]) > 0).mean())


def _similar(a, b) -> float:
    """1 = same colours as the clicked player, 0 = nothing alike."""
    if a is None or b is None:
        return 0.0
    return float(np.clip(1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA), 0, 1))


KEY_TOL = 0.06          # between two clicks, tracker answers further than this from the path are dropped
KEY_CONF = 0.6          # confidence when the ring just follows the path between clicks
STUCK_S = 1.5           # window for the "tracking the scenery" check
STUCK_PAN = 0.12        # ...camera moved at least this much (fraction of width) in the window
STUCK_STILL = 0.025     # ...while the box stayed this still relative to the field


def camera_motion(frames: np.ndarray) -> np.ndarray:
    """(N, 2, 3): affine from frame i-1 to frame i (camera pan/zoom), identity for frame 0."""
    from .motion import estimate_global
    n = len(frames)
    out = np.tile(np.array([[1, 0, 0], [0, 1, 0]], np.float32), (n, 1, 1))
    s = 0.5
    prev = None
    for i in range(n):
        g = cv2.cvtColor(cv2.resize(frames[i], None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            m = estimate_global(prev, g).astype(np.float32)
            m[:, 2] /= s                      # back to full-frame pixels
            out[i] = m
        prev = g
    return out


def _apply(m: np.ndarray, p: tuple) -> tuple:
    return (m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2], m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2])


def _step_matrix(cam: np.ndarray, i: int, step: int) -> np.ndarray:
    """Camera transform taking frame i-step to frame i."""
    if step > 0:
        return cam[i]
    m = np.vstack([cam[i - step], [0, 0, 1]])   # frame i -> i+1; we want i+1 -> i
    return np.linalg.inv(m)[:2]


def _run(frames: np.ndarray, start: int, stop: int, box_px: tuple, step: int,
         cam: np.ndarray | None = None) -> dict[int, tuple]:
    """Track from frame `start` toward `stop` (exclusive), step +1 or -1.
    Returns {frame: (cx, cy, w, h, frames_from_anchor, similarity)} in px, until the
    tracker fails."""
    h, w = frames.shape[1:3]
    out = {}
    win = int(STUCK_S * 30)
    ghosts: list[tuple] = []   # (frame, where a field-fixed point at the box would be now)
    tr = _csrt()
    tr.init(frames[start], tuple(int(round(v)) for v in box_px))
    x, y, bw, bh = box_px
    ref = _hist(frames[start], x, y, bw, bh)
    out[start] = (x + bw / 2, y + bh / 2, bw, bh, 0, 1.0)
    prev = (x + bw / 2, y + bh / 2)
    grass_run = 0
    for k, i in enumerate(range(start + step, stop, step), 1):
        ok, (x, y, bw, bh) = tr.update(frames[i])
        c = (x + bw / 2, y + bh / 2)
        if not ok or bw <= 2 or bh <= 2:
            break
        if np.hypot(c[0] - prev[0], c[1] - prev[1]) > MAX_STEP * w:
            # Big move: a fast camera pan moves everyone. Only call it lost if the box moved
            # far more than the camera did.
            exp = _camera_moved(frames[i - step], frames[i], prev)
            if np.hypot(c[0] - exp[0], c[1] - exp[1]) > MAX_STEP * w:
                break
        # Lost if the box is basically empty turf for a moment (tracker latched onto grass).
        if _body_frac(frames[i], x, y, bw, bh) < 0.08:
            grass_run += 1
            if grass_run >= 6:
                for j in range(i - step * 5, i, step):   # drop the frames already on grass
                    out.pop(j, None)
                break
        else:
            grass_run = 0
        # Tracking the scenery? (Off by default: a player holding a block while the camera
        # pans looks the same as a goalpost, so this hid the ring on good plays.)
        if cam is not None:
            m = _step_matrix(cam, i, step)
            ghosts = [(j, _apply(m, g), o) for j, g, o in ghosts]
            ghosts.append((i, c, c))
            if len(ghosts) > win:
                ghosts.pop(0)
                j, g, origin = ghosts[0]
                panned = np.hypot(g[0] - origin[0], g[1] - origin[1]) / w
                moved = np.hypot(c[0] - g[0], c[1] - g[1]) / w
                if panned > STUCK_PAN and moved < STUCK_STILL:
                    for jj, _, _ in ghosts:      # the whole window was on scenery
                        out.pop(jj, None)
                    break
        cur_h = _hist(frames[i], x, y, bw, bh)
        sim = _similar(ref, cur_h)
        out[i] = (c[0], c[1], bw, bh, k, sim)
        # Let the reference follow slow lighting changes while we're clearly on him.
        if sim > 0.6 and cur_h is not None and ref is not None:
            ref = cv2.normalize(0.92 * ref + 0.08 * cur_h, None)
        prev = c
    return out


DEFAULT_BOX = (0.032, 0.09)   # what the page sends for a plain click (no drag)


def is_click(a: dict) -> bool:
    return bool(a.get("auto")) or (abs(a["w"] - DEFAULT_BOX[0]) < 1e-3 and abs(a["h"] - DEFAULT_BOX[1]) < 1e-3)


def _body_mask(patch: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    body = cv2.morphologyEx(255 - green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))


def tighten_box(frame: np.ndarray, a: dict) -> dict:
    """A dragged box is usually loose: shrink it to the player inside (biggest non-grass
    shape whose centre is in the box). Keeps the box as drawn if nothing plausible."""
    H, W = frame.shape[:2]
    bx0, by0, bw0, bh0 = a["x"] * W, a["y"] * H, a["w"] * W, a["h"] * H
    # Only look inside the drawn box: the result may shrink, never grow.
    x0, y0 = int(max(0, bx0)), int(max(0, by0))
    x1, y1 = int(min(W, bx0 + bw0)), int(min(H, by0 + bh0))
    out = dict(a, refined=False)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return out
    n, _, stats, cents = cv2.connectedComponentsWithStats(_body_mask(frame[y0:y1, x0:x1]))
    best, best_area = None, 0
    for i in range(1, n):
        cx, cy = cents[i][0] + x0, cents[i][1] + y0
        if bx0 <= cx <= bx0 + bw0 and by0 <= cy <= by0 + bh0 and stats[i][4] > best_area:
            best, best_area = i, stats[i][4]
    if best is None:
        return out
    sx, sy, sw, sh, _ = stats[best]
    if sh < 0.3 * bh0 or sw < 0.15 * bw0 or sh < 6:
        return out
    pad_w, pad_h = 0.12 * sw, 0.06 * sh
    nx0, ny0 = max(bx0, x0 + sx - pad_w), max(by0, y0 + sy - pad_h)
    nx1, ny1 = min(bx0 + bw0, x0 + sx + sw + pad_w), min(by0 + bh0, y0 + sy + sh + pad_h)
    if (nx1 - nx0) * (ny1 - ny0) > 0.85 * bw0 * bh0:
        return out   # player merged with neighbours; the drawn box is as good as it gets
    out.update(x=nx0 / W, y=ny0 / H, w=(nx1 - nx0) / W, h=(ny1 - ny0) / H, refined=True)
    return out


def refine_anchor(frame: np.ndarray, a: dict) -> dict:
    """Fit a plain click to the actual player: the non-grass blob under/near the click.

    Players stand on green turf, so 'not green' finds them. Falls back to a modest
    default size when the blob is implausible (pile-up, merged with a yard line).
    """
    H, W = frame.shape[:2]
    cx, cy = (a["x"] + a["w"] / 2) * W, (a["y"] + a["h"] / 2) * H
    sw, sh = int(0.07 * W), int(0.18 * H)
    x0, y0 = int(max(0, cx - sw / 2)), int(max(0, cy - sh / 2))
    x1, y1 = int(min(W, cx + sw / 2)), int(min(H, cy + sh / 2))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(_body_mask(frame[y0:y1, x0:x1]))
    px, py = cx - x0, cy - y0
    best, best_d = None, 1e9
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < 12 or bh < 6:
            continue
        # distance from click to the blob's box (0 if inside)
        dx = max(bx - px, 0, px - (bx + bw))
        dy = max(by - py, 0, py - (by + bh))
        d = np.hypot(dx, dy)
        if d < best_d:
            best, best_d = i, d
    out = dict(a)
    ok = best is not None and best_d < 0.012 * W
    if ok:
        bx, by, bw, bh, _ = stats[best]
        # A player is taller than wide; reject lines, piles and crowd.
        ok = 0.8 * bw <= bh * 1.6 and bh <= 0.16 * H and bw <= 0.06 * W and bh >= 0.02 * H
    if ok:
        pad_w, pad_h = 0.15 * bw, 0.08 * bh
        out.update(x=(x0 + bx - pad_w) / W, y=(y0 + by - pad_h) / H,
                   w=(bw + 2 * pad_w) / W, h=(bh + 2 * pad_h) / H)
    else:
        w, h = 0.022, 0.06
        out.update(x=(cx / W) - w / 2, y=(cy / H) - h / 2, w=w, h=h)
    out["refined"] = bool(ok)
    return out


def fit_anchor(frame: np.ndarray, a: dict) -> dict:
    """Snap a spotlight click/box to the player's actual extent."""
    return refine_anchor(frame, a) if is_click(a) else tighten_box(frame, a)


def track_player(frames: np.ndarray, fps: float, anchors: list[dict],
                 cam: np.ndarray | None = None) -> np.ndarray:
    """frames: (N, h, w, 3) BGR. anchors: [{t, x, y, w, h}] normalised box (x, y = top-left).
    Returns (N, 5): cx, cy, w, h, conf - normalised, NaN where unknown."""
    n, H, W = frames.shape[:3]
    res = np.full((n, 5), np.nan, np.float32)
    if not anchors or n == 0:
        return res
    ks = sorted(anchors, key=lambda a: a["t"])
    idx = [int(np.clip(round(a["t"] * fps), 0, n - 1)) for a in ks]
    ks = [fit_anchor(frames[i], a) for a, i in zip(ks, idx)]
    runs = []  # (dict, anchor_idx)
    for j, (a, i0) in enumerate(zip(ks, idx)):
        box = (a["x"] * W, a["y"] * H, max(a["w"] * W, 6), max(a["h"] * H, 10))
        fwd_stop = idx[j + 1] + 1 if j + 1 < len(idx) else n
        bwd_stop = idx[j - 1] - 1 if j > 0 else -1
        runs.append(_run(frames, i0, fwd_stop, box, +1, cam))
        runs.append(_run(frames, i0, bwd_stop, box, -1, cam))

    # The tracker's own size estimate drifts (boxes balloon in pile-ups), which drags the
    # "feet" down. Keep size within +-25% of what was marked, interpolated between clicks.
    times = np.array([i for i in idx], float)
    exp_w = np.interp(np.arange(n), times, [a["w"] * W for a in ks])
    exp_h = np.interp(np.arange(n), times, [a["h"] * H for a in ks])
    for r in runs:
        for i, v in list(r.items()):
            cx, cy, bw, bh = v[:4]
            r[i] = (cx, cy, float(np.clip(bw, 0.8 * exp_w[i], 1.25 * exp_w[i])),
                    float(np.clip(bh, 0.8 * exp_h[i], 1.25 * exp_h[i]))) + tuple(v[4:])

    # Clicks are keyframes: between two clicks the player can't be far from the path
    # joining them. Tracker answers that stray off it (latched onto end-zone paint, swapped
    # in a pile) are dropped, and if none are left the ring glides along the path.
    acx = [(a["x"] + a["w"] / 2) * W for a in ks]
    acy = [(a["y"] + a["h"] / 2) * H for a in ks]
    seg_of = np.searchsorted(times, np.arange(n), side="right") - 1   # keyframe at/before i

    def on_path(i):
        j = seg_of[i]
        if j < 0 or j + 1 >= len(ks) or idx[j + 1] == idx[j]:
            return None
        u = (i - idx[j]) / (idx[j + 1] - idx[j])
        px, py = acx[j] + u * (acx[j + 1] - acx[j]), acy[j] + u * (acy[j + 1] - acy[j])
        tol = max(KEY_TOL * W, 0.5 * np.hypot(acx[j + 1] - acx[j], acy[j + 1] - acy[j]))
        return px, py, tol

    tau = TRUST_S * fps / np.log(2)
    for i in range(n):
        # Trust = slow decay with time since the click. Colour similarity is deliberately
        # NOT used: on real footage (same team colours, white pants on both sides, sunset
        # light) the player scored lower against himself than other players did, so it hid
        # the ring while tracking was fine. Real losses are caught by the tracker failing,
        # sudden jumps, an empty-grass box, or two clicks' runs disagreeing (below).
        cands = [(r[i][:5], np.exp(-r[i][4] / tau)) for r in runs if i in r]
        path = on_path(i)
        if path is not None:
            px, py, tol = path
            cands = [c for c in cands if np.hypot(c[0][0] - px, c[0][1] - py) <= tol]
            if not cands:
                res[i] = (px / W, py / H, exp_w[i] / W, exp_h[i] / H, KEY_CONF)
                continue
        if not cands:
            continue
        # Prefer the most trusted; if two disagree a lot, don't average them into nonsense.
        cands.sort(key=lambda c: -c[1])
        (cx, cy, bw, bh, _), wt = cands[0]
        if len(cands) > 1:
            (cx2, cy2, bw2, bh2, _), wt2 = cands[1]
            if np.hypot(cx - cx2, cy - cy2) < 2 * max(bw, bw2):
                s = wt + wt2
                cx, cy = (cx * wt + cx2 * wt2) / s, (cy * wt + cy2 * wt2) / s
                bw, bh = (bw * wt + bw2 * wt2) / s, (bh * wt + bh2 * wt2) / s
                wt = max(wt, wt2)
            else:
                wt *= 0.5  # conflicting evidence: be less sure
        res[i] = (cx / W, cy / H, bw / W, bh / H, wt)
    return res


def smooth_track(res: np.ndarray, fps: float, pos_s: float = 0.15, size_s: float = 0.5,
                 conf_s: float = 0.3, max_gap_s: float = 0.4) -> np.ndarray:
    """Remove frame-to-frame tracker wobble so a drawn highlight glides instead of jittering.

    Zero-phase (no lag): position lightly, box size heavily (the ring shouldn't breathe),
    confidence so the highlight fades rather than pops. Short dropouts are bridged; long
    ones stay unknown.
    """
    from scipy.ndimage import gaussian_filter1d
    out = res.copy()
    known = ~np.isnan(res[:, 0])
    if known.sum() < 2:
        return out
    idx = np.arange(len(res))
    # Bridge short gaps only.
    filled = res.copy()
    for c in range(4):
        filled[:, c] = np.interp(idx, idx[known], res[known, c])
    gap_ok = known.copy()
    run_start = None
    for i in range(len(res) + 1):
        if i < len(res) and not known[i]:
            run_start = i if run_start is None else run_start
        elif run_start is not None:
            if i - run_start <= max_gap_s * fps and run_start > 0 and i < len(res):
                gap_ok[run_start:i] = True
            run_start = None
    m = gap_ok.astype(np.float64)
    conf = np.where(known, np.nan_to_num(res[:, 4]), 0.0)
    for c, s in ((0, pos_s), (1, pos_s), (2, size_s), (3, size_s)):
        # Masked Gaussian: don't smear across long gaps.
        num = gaussian_filter1d(np.where(gap_ok, filled[:, c], 0) * m, s * fps, mode="nearest")
        den = gaussian_filter1d(m, s * fps, mode="nearest")
        out[:, c] = np.where(gap_ok, num / np.maximum(den, 1e-6), np.nan)
    out[:, 4] = np.where(gap_ok, gaussian_filter1d(conf, conf_s * fps, mode="nearest"), np.nan)
    return out
