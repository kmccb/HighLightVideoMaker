"""Big-play effects: the ghost trail and "silence, then the drop".

Ghost trail: on the hit the picture freezes, dims and moves in on the path, and copies of the
player pop in one by one where he was on his way there (the camera's own movement is removed, so
they stand on the field where he really ran). `render` makes that freeze as its own clip, cut into
the play at the hit by reel.build. Plays where he barely moved get none.

The drop: all sound cuts out for a moment before the hit, then a boom lands on it. `sound_track`
writes one track for the whole reel with the risers, booms, crowd swells and the chime of each
ghost; reel._assemble mixes it in and opens the gaps in the music and game sound.

Both apply to plays marked as a big play (review.big) with a hit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from . import camera, encoder, graphics
from .pipeline import CACHE_DIRNAME

VERSION = 2
FPS = 30
WIDE = (1920, 1080)
SR = 48000
GAP = 0.36              # silence before the hit
MAX_BACK = 8.0          # how far before the hit the trail can reach
WORK_W = 1920           # tracking the camera and cutting him out happen at this width
MAX_GHOSTS = 10
MIN_GHOSTS = 3          # fewer than this: he hardly moved, no trail
T_ZOOM, T_FIRST, T_OUT = 0.55, 0.25, 0.45


def seconds(period: float | None) -> float:
    """How long the freeze lasts: about four seconds, a whole number of beats when there's a song."""
    if not period or period <= 0:
        return 4.0
    n = max(4, int(round(4.0 / period)))
    return round(n * period * FPS) / FPS


def reel_view(clip: dict, plan: dict, t: float, size: tuple[int, int]) -> np.ndarray:
    """The reel's own framing of a moment of a play, in content px (x, y, w, h): the camera window
    (zoom), and for a tall reel its 9:16 window on the player, exactly as segment._render frames it."""
    from types import SimpleNamespace
    from . import segment
    info, crop = clip["probe"], clip.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    Wc, Hc = (crop["w"], crop["h"]) if crop else (info["width"], info["height"])
    if plan.get("rects"):
        rects = np.array(plan["rects"], np.float64) * [info["width"], info["height"], info["width"], info["height"]] - [ox, oy, 0, 0]
    else:
        fw = min(Wc, Hc * WIDE[0] / WIDE[1])
        rects = np.tile([(Wc - fw) / 2, (Hc - fw * WIDE[1] / WIDE[0]) / 2, fw, fw * WIDE[1] / WIDE[0]], (int(round(info["duration"] * FPS)) + 2, 1))
    rects[:, 2], rects[:, 3] = np.minimum(rects[:, 2], Wc), np.minimum(rects[:, 3], Hc)
    if size[0] < size[1]:
        player = np.array(plan["player"]) if plan.get("player") else None
        rects = segment._vertical_rects(rects, player, Wc, Hc, SimpleNamespace(width=info["width"], height=info["height"]), ox)
    return rects[min(len(rects) - 1, max(0, int(round(t * FPS))))].astype(np.float64)


def _key(root: Path, clip: dict, pl: dict | None, hit_t: float, secs: float, period: float | None,
         size: tuple[int, int] | None = None) -> str:
    from . import segment
    r = clip.get("review") or {}
    src = root / clip["source"]
    payload = {"v": VERSION, "src": [src.stat().st_size, src.stat().st_mtime], "win": segment._window(clip),
               "hit": round(float(hit_t), 3), "spot": r.get("spotlight"), "zoom": r.get("zoom"), "crop": clip.get("crop"),
               "secs": secs, "period": None if not period else round(period, 4), "color": (pl or {}).get("color")}
    if r.get("trail_start") is not None:
        payload["start"] = round(float(r["trail_start"]), 3)
    if size and tuple(size) != WIDE:
        payload["size"] = list(size)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _paths(root: Path, clip: dict, pl: dict | None, hit_t: float, secs: float, period: float | None,
           size: tuple[int, int] | None = None) -> tuple[Path, Path]:
    stem = root / CACHE_DIRNAME / "ghost" / f"{clip['id']}-{_key(root, clip, pl, hit_t, secs, period, size)}"
    return stem.with_suffix(".mp4"), stem.with_suffix(".json")


def status(root: Path, clip: dict, pl: dict | None, hit_t: float | None, secs: float, period: float | None) -> dict | None:
    """What's known without rendering: the saved result ({"pops": [...]} or {"none": reason}), or
    None if this play's trail hasn't been made yet."""
    if hit_t is None:
        return {"none": "no hit"}
    mp4, meta = _paths(root, clip, pl, hit_t, secs, period)
    if not meta.exists():
        return None
    d = json.loads(meta.read_text())
    return d if d.get("none") or mp4.exists() else None


def _read(src: Path, t0: float, n: int, crop: dict | None, size: tuple[int, int] | None, keep: set[int] | None = None) -> dict | np.ndarray:
    """n frames from t0 at 30 fps. size: scale to (w, h). keep: only these frame numbers (returned
    as a dict), so a second pass at full size doesn't hold every frame."""
    vf = [f"fps={FPS}"]
    if crop:
        vf.insert(0, f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}")
    w, h = size
    vf.append(f"scale={w}:{h}:flags=area")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{n / FPS + 0.5:.4f}", "-i", str(src),
                          "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)          # (it complains when we stop reading at the hit)
    out = {} if keep is not None else np.empty((n, h, w, 3), np.uint8)
    k = 0
    try:
        while k < n:
            buf = p.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            if keep is None:
                out[k] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            elif k in keep:
                out[k] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            k += 1
    finally:
        p.stdout.close()
        p.wait()
    return out if keep is not None else out[:k]


def _like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """img with its colours fitted to ref (the same frame, smaller). ffmpeg converts an HDR iPhone
    clip to slightly different colours at full size than when it also scales it down, so a full-size
    frame dropped into a picture built from the small ones showed as a yellow-green patch."""
    small = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    X = small.reshape(-1, 3)[::97].astype(np.float32)
    A, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X), np.float32)], ref.reshape(-1, 3)[::97].astype(np.float32), rcond=None)
    return cv2.transform(img, np.ascontiguousarray(A.T, np.float32))


def _homographies(frames: np.ndarray, ref: int) -> list[np.ndarray]:
    """For every frame, the homography taking it into frame `ref`'s picture (camera shake and pan removed)."""
    n, h, w = frames.shape[:3]
    sw, sh = w // 2, h // 2
    small = [cv2.cvtColor(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames]
    step = [np.eye(3)] * n                       # step[k]: frame k -> frame k+1
    for k in range(n - 1):
        pts = cv2.goodFeaturesToTrack(small[k], 700, 0.01, 12)
        Hm = None
        if pts is not None and len(pts) >= 12:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(small[k], small[k + 1], pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 12:
                Hm, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 1.5)
        step[k] = np.eye(3) if Hm is None else Hm
    S, Si = np.diag([sw / w, sh / h, 1.0]), np.diag([w / sw, h / sh, 1.0])
    out = [np.eye(3)] * n
    acc = np.eye(3)
    for k in range(ref - 1, -1, -1):             # earlier frames: chain forward to ref
        acc = acc @ step[k]
        out[k] = Si @ acc @ S
    acc = np.eye(3)
    for k in range(ref + 1, n):                  # later frames: chain backward
        acc = acc @ np.linalg.inv(step[k - 1])
        out[k] = Si @ acc @ S
    return out


def _pt(Hm: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = Hm @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def pick(pos: dict[int, tuple], end: tuple[float, float], gap: float, most: int = MAX_GHOSTS) -> list[int]:
    """Which frames become ghosts: walking back from the freeze, each at least `gap` px from where
    he ends up and from every ghost already chosen (so they never pile up where he dawdled or
    doubled back). Oldest first."""
    chosen: list[int] = []
    for i in sorted(pos, reverse=True):
        x, y = pos[i][:2]
        if np.hypot(x - end[0], y - end[1]) < gap:
            continue
        if any(np.hypot(x - pos[j][0], y - pos[j][1]) < gap for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= most:
            break
    return chosen[::-1]


def pop_times(n: int, secs: float, period: float | None) -> list[float]:
    """When each ghost appears, in seconds from the hit: on half-beats, all in by ~2 s."""
    if n <= 0:
        return []
    room = max(0.6, min(1.9, secs - T_OUT - 1.2 - T_FIRST))
    step = min(period / 2, room / n) if period else room / n
    return [round(T_FIRST + j * step, 4) for j in range(n)]


def render(root: Path, clip: dict, pl: dict | None, hit_t: float | None, secs: float, period: float | None,
           log=print, size: tuple[int, int] | None = None) -> Path | None:
    """Make (or reuse) the ghost-trail freeze for this play: a 16:9 clip `secs` long that starts and
    ends on the reel's own framing of the hit frame, silent (its sounds are in sound_track).
    None when there's nothing to show (no hit, he isn't tracked, or he hardly moved)."""
    from . import segment
    if hit_t is None:
        return None
    root = Path(root)
    size = tuple(size) if size else WIDE
    OW, OH = size
    mp4, meta = _paths(root, clip, pl, hit_t, secs, period, size)
    if meta.exists():
        known = json.loads(meta.read_text())
        if known.get("none") or mp4.exists():
            return None if known.get("none") else mp4
    mp4.parent.mkdir(parents=True, exist_ok=True)

    def none(why: str):
        meta.write_text(json.dumps({"none": why}))
        log(f"  no ghost trail for {clip['source']}: {why}")
        return None

    cache = root / CACHE_DIRNAME
    plan = camera.plan(cache, clip)
    if not plan.get("player"):
        return none("your player isn't marked on this play")
    player = np.array(plan["player"], np.float64)
    info = clip["probe"]
    crop = clip.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    Wc, Hc = (crop["w"], crop["h"]) if crop else (info["width"], info["height"])
    ws = min(1.0, WORK_W / Wc)                                  # content px -> working px
    W, H = int(round(Wc * ws / 2)) * 2, int(round(Hc * ws / 2)) * 2
    _, _, snap = segment._window(clip)
    t0, _ = segment.extents(clip, info["duration"])
    tF = round(float(hit_t) * FPS) / FPS
    mark = (clip.get("review") or {}).get("trail_start")       # where the trail begins: marked, else the snap
    start = max(t0, float(mark)) if mark is not None else max(t0, snap - 0.3)
    if tF - start > MAX_BACK:                                   # a long run: the camera travels along it (longtrail.py)
        from . import longtrail
        return longtrail.render(root, clip, pl, tF, start, secs, period, size, mp4, meta, plan, log)
    ta = round(max(start, tF - MAX_BACK) * FPS) / FPS
    iF = int(round((tF - ta) * FPS))
    if iF < 18:
        return none("the hit comes too soon after the snap")
    frames = _read(root / clip["source"], ta, iF + 1, crop, (W, H))
    if len(frames) < iF + 1:
        return none("the video ends before the hit")

    def row(i: int):
        cx, cy, bw, bh, conf = player[min(len(player) - 1, int(round((ta + i / FPS) * FPS)))]
        return (cx * info["width"] - ox) * ws, (cy * info["height"] - oy) * ws, bw * info["width"] * ws, bh * info["height"] * ws, conf, cx

    G = _homographies(frames, iF)

    # the field with nobody moving on it: the median over time, in the freeze frame's picture
    idx = list(range(0, iF + 1, max(3, (iF + 1) // 45)))
    stack = np.empty((len(idx), H, W, 3), np.uint8)
    for j, k in enumerate(idx):
        stack[j] = cv2.warpPerspective(frames[k], G[k], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    plate = np.empty((H, W, 3), np.uint8)
    for y in range(0, H, 120):                                  # in strips, to keep memory down
        plate[y:y + 120] = np.median(stack[:, y:y + 120], axis=0).astype(np.uint8)
    del stack

    pos = {}                                                    # frame -> (cx, cy, w, h) in the freeze picture
    for i in range(iF + 1):
        x, y, bw, bh, conf, raw = row(i)
        if raw < 0 or np.isnan(raw) or conf < 0.3 or bw <= 0:
            continue
        gx, gy = _pt(G[i], x, y)
        pos[i] = (gx, gy, bw, bh)
    last = max((i for i in pos if i >= iF - 6), default=None)
    if last is None:
        return none("he isn't found at the hit")
    fx, fy, fw, fh = pos[last]
    chosen = pick({i: p for i, p in pos.items() if i < last}, (fx, fy), 1.55 * fw)
    if len(chosen) < MIN_GHOSTS:
        return none("he doesn't cover enough ground before the hit")
    if any(not (0 <= pos[i][0] <= W and 0 <= pos[i][1] <= H) for i in chosen):   # the camera panned past where he was
        del frames
        from . import longtrail
        return longtrail.render(root, clip, pl, tF, start, secs, period, size, mp4, meta, plan, log)

    def matte(i: int, grow: float = 1.0) -> np.ndarray:
        """A soft mask of the player in frame i (freeze picture): what differs from the empty field around his box."""
        img = frames[i] if i == iF else cv2.warpPerspective(frames[i], G[i], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x, y, bw, bh = pos[i]
        diff = cv2.absdiff(img, plate).max(axis=2)
        m = (cv2.GaussianBlur(diff, (0, 0), 1.2) > 26).astype(np.uint8)
        box = np.zeros((H, W), np.uint8)
        cv2.ellipse(box, (int(x), int(y)), (int(bw * 0.66 * grow), int(bh * 0.66 * grow)), 0, 0, 360, 1, -1)
        m &= box
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        core = (slice(max(0, int(y - bh * 0.5)), int(y + bh * 0.5) + 1), slice(max(0, int(x - bw * 0.35)), int(x + bw * 0.35) + 1))
        for lb in range(1, nlab):
            if stats[lb, cv2.CC_STAT_AREA] >= 0.03 * bw * bh and (lab[core] == lb).any():
                keep[lab == lb] = 1
        if keep.sum() < 0.12 * bw * bh:                         # too little found: a soft oval on him instead
            keep[:] = 0
            cv2.ellipse(keep, (int(x), int(y)), (int(bw * 0.42), int(bh * 0.52)), 0, 0, 360, 1, -1)
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.3)

    mattes = {i: matte(i) for i in chosen}
    pos[iF] = pos[last]
    live_wk = matte(iF, 1.15)
    feet_i = sorted(i for i in pos if chosen[0] <= i <= last)
    feet = np.array([(pos[i][0], pos[i][1] + pos[i][3] * 0.5) for i in feet_i], np.float32)

    # ---- the camera: from the reel's own framing of this moment to the path ----
    A = OW / OH
    wide = list(reel_view(clip, plan, tF, size) * ws)
    wide[2], wide[3] = min(wide[2], W), min(wide[3], H)
    xs = [pos[i][0] for i in chosen] + [fx]
    ys = [pos[i][1] for i in chosen] + [fy]
    T_G = pop_times(len(chosen), secs, period)
    if OW >= OH:                                                # wide: the whole path in view
        cw = max(max(xs) - min(xs) + 5.0 * fw, (max(ys) - min(ys) + 3.2 * fh) * A, wide[2] * 0.52)
        cw = min(cw, W, H * A)
        ch = cw / A
        tight = [float(np.clip((max(xs) + min(xs)) / 2, cw / 2, W - cw / 2)) - cw / 2,
                 float(np.clip((max(ys) + min(ys)) / 2 - 0.1 * fh, ch / 2, H - ch / 2)) - ch / 2, cw, ch]
        tight_at = lambda t: tight    # noqa: E731
        span = tight
    else:                                                       # tall: too narrow for that, so the camera travels
        ch = min(H, max(max(ys) - min(ys) + 6.0 * fh, wide[3] * 0.62))   # along the path with the ghosts, and ends on him
        cw = min(ch * A, W)
        ch = cw / A
        ty = float(np.clip((max(ys) + min(ys)) / 2 - 0.1 * fh, ch / 2, H - ch / 2)) - ch / 2
        t_a, t_b = T_G[0], T_G[-1] + 0.45

        def tight_at(t: float):
            k = np.interp(t, [T_G[j] for j in range(len(chosen))] + [t_b], xs) if t > t_a else xs[0]
            return [float(np.clip(k, cw / 2, W - cw / 2)) - cw / 2, ty, cw, ch]
        lo = min(tight_at(t)[0] for t in np.linspace(0, secs, 60))
        hi = max(tight_at(t)[0] for t in np.linspace(0, secs, 60))
        span = [lo, ty, hi - lo + cw, ch]
        tight = [lo, ty, cw, ch]

    # ---- the canvas everything is drawn on: the part of the picture either view shows, at the
    #      sharpness the close view needs (from the full-size frames when the source is bigger) ----
    ux0, uy0 = int(np.floor(min(wide[0], span[0]))), int(np.floor(min(wide[1], span[1])))
    ux1 = int(np.ceil(max(wide[0] + wide[2], span[0] + span[2])))
    uy1 = int(np.ceil(max(wide[1] + wide[3], span[1] + span[3])))
    ux0, uy0, ux1, uy1 = max(0, ux0), max(0, uy0), min(W, ux1), min(H, uy1)
    cs = min(OW / tight[2], 1 / ws, 3840 / (ux1 - ux0))
    cs = max(cs, min(1.0, 3840 / (ux1 - ux0)))
    CW, CH = int(round((ux1 - ux0) * cs)), int(round((uy1 - uy0) * cs))
    C = np.array([[cs, 0, -ux0 * cs], [0, cs, -uy0 * cs], [0, 0, 1.0]])
    if ws < 1:                                                   # the chosen frames again, full size
        full = _read(root / clip["source"], ta, iF + 1, crop, (Wc, Hc), keep=set(chosen) | {iF})
        Sf = np.diag([ws, ws, 1.0])
        if any(i not in full for i in list(chosen) + [iF]):
            full, Sf = None, np.eye(3)
        else:
            full = {i: _like(f, frames[i]) for i, f in full.items()}
    else:
        full, Sf = None, np.eye(3)
    src_of = (lambda i: full[i]) if full is not None else (lambda i: frames[i])
    to_canvas = lambda img, Hm: cv2.warpPerspective(img, C @ Hm @ Sf, (CW, CH), flags=cv2.INTER_LINEAR,    # noqa: E731
                                                    borderMode=cv2.BORDER_REPLICATE)
    m_canvas = lambda m: cv2.warpAffine(m, C[:2], (CW, CH), flags=cv2.INTER_LINEAR)    # noqa: E731
    base = to_canvas(src_of(iF), np.eye(3))

    def roi_of(m: np.ndarray, pad: int) -> tuple[slice, slice]:
        ys_, xs_ = np.where(m > 0.01)
        if not len(ys_):
            return slice(0, 1), slice(0, 1)
        return (slice(max(0, ys_.min() - pad), min(CH, ys_.max() + pad + 1)),
                slice(max(0, xs_.min() - pad), min(CW, xs_.max() + pad + 1)))

    tc = (pl or {}).get("color") or "#1f4fd1"
    team = np.array([int(tc[j:j + 2], 16) for j in (5, 3, 1)], np.float32)          # BGR
    pad = int(14 * cs)
    ghosts = []
    for i in chosen:
        gm = m_canvas(mattes[i])
        sl = roi_of(gm, pad)
        gimg = to_canvas(src_of(i), G[i])[sl].astype(np.float32)
        gm = gm[sl]
        rim = np.clip(cv2.GaussianBlur(gm, (0, 0), 3.5 * cs) * 1.5 - gm, 0, 1) * 0.55
        ghosts.append((sl, gimg, gm[..., None], rim[..., None] * team))
    live_m = m_canvas(live_wk)
    live_sl = roi_of(live_m, 4)
    live_m, live_img = live_m[live_sl][..., None], base[live_sl].astype(np.float32)
    del frames, full, mattes

    # the path line, one layer per number of ghosts on screen (drawn up to the newest one)
    fc = feet * cs - [ux0 * cs, uy0 * cs]
    if len(fc) > 9:
        fc = cv2.GaussianBlur(fc.reshape(-1, 1, 2), (1, 9), 0).reshape(-1, 2)

    def line_layer(upto: int):
        pts = fc[[k for k, fi in enumerate(feet_i) if fi <= upto]]
        if len(pts) < 3:
            return None
        x0, y0 = np.floor(pts.min(0)).astype(int) - int(30 * cs)
        x1, y1 = np.ceil(pts.max(0)).astype(int) + int(30 * cs)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(CW, x1), min(CH, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        line = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.polylines(line, [np.round((pts - [x0, y0]) * 8).astype(np.int32)], False, 1.0, max(2, int(round(2 * cs))), cv2.LINE_AA, shift=3)
        glow = cv2.GaussianBlur(line, (0, 0), 5 * cs) * 2.2
        return (slice(y0, y1), slice(x0, x1)), glow[..., None] * team + line[..., None] * 150

    lines = [line_layer(i) for i in chosen] + [line_layer(last)]
    t_hold_end = secs - T_OUT
    white = np.full((OH, OW, 3), 255, np.uint8)

    def frame_at(t: float) -> np.ndarray:
        out_u = _ease((t - t_hold_end) / T_OUT)                 # 1 = back to the live picture
        zoom = _ease(t / T_ZOOM) * (1 - out_u)
        dim = 0.5 * _ease(t / 0.4) * (1 - out_u)
        img = cv2.convertScaleAbs(base, alpha=1 - dim)
        shown = [j for j, tg in enumerate(T_G) if t >= tg]
        if shown and out_u < 1:
            ln = lines[-1] if len(shown) == len(ghosts) and t >= T_G[-1] + 0.25 else lines[shown[-1]]
            if ln is not None:
                sl, layer = ln
                img[sl] = np.clip(img[sl] + layer * (0.9 * (1 - out_u)), 0, 255).astype(np.uint8)
        for j in shown:
            sl, gimg, gm, rim = ghosts[j]
            age = t - T_G[j]
            alpha = (0.55 + 0.4 * (j + 1) / len(ghosts)) * _ease(age / 0.12) * (1 - out_u)
            popv = max(0.0, 1 - age / 0.22)                     # a white flash as it lands
            patch = img[sl].astype(np.float32) + rim * (1 - out_u)
            g = gimg * (1 + 0.9 * popv) + 255 * 0.5 * popv if popv > 0 else gimg
            m = gm * alpha
            img[sl] = np.clip(patch * (1 - m) + g * m, 0, 255).astype(np.uint8)
        if dim > 0:                                             # he stays lit
            m = live_m * dim
            img[live_sl] = np.clip(img[live_sl].astype(np.float32) * (1 - m) + live_img * m, 0, 255).astype(np.uint8)
        tg = tight_at(t)
        x, y, w_, h_ = [wide[j] + (tg[j] - wide[j]) * zoom for j in range(4)]
        sc = OW / (w_ * cs)
        M = np.array([[sc, 0, -(x - ux0) * cs * sc], [0, OH / (h_ * cs), -(y - uy0) * cs * OH / (h_ * cs)]])
        out = cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_AREA if sc < 0.75 else cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        flash = max(0.0, 1 - t / 0.18) * 0.55                   # the hit itself
        return cv2.addWeighted(out, 1 - flash, white, flash, 0) if flash > 0 else out

    tmp = mp4.with_name(mp4.stem + ".tmp.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo",
                            *encoder.h264(17), "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(tmp)],
                           stdin=subprocess.PIPE)
    try:
        for j in range(int(round(secs * FPS))):
            enc.stdin.write(frame_at(j / FPS).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ghost trail render failed for {clip['source']}")
    tmp.replace(mp4)
    meta.write_text(json.dumps({"pops": T_G, "seconds": secs, "ghosts": len(ghosts)}))
    log(f"  ghost trail {clip['source']}: {len(ghosts)} ghosts")
    return mp4


def pops_of(mp4: Path) -> list[float]:
    return json.loads(mp4.with_suffix(".json").read_text()).get("pops") or []


# ---------------------------------------------------------------- sound
def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    from scipy import signal
    return signal.sosfilt(signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x, axis=0)


def sound_track(path: Path, total: float, events: list[dict]) -> Path:
    """One effects track for the whole reel. events: [{"t": hit time in the reel, "drop": bool,
    "pops": [seconds after the hit, ...] or None, "len": freeze seconds}]. A drop gets a riser that
    cuts dead GAP before the hit, then a sub boom, a crack and some air; a ghost trail gets a crowd
    swell under the freeze and a rising chime for each ghost."""
    N = int(round(total * SR))
    mix = np.zeros((N, 2), np.float32)
    rng = np.random.default_rng(7)

    def place(x: np.ndarray, at: float, gain: float):
        s = int(round(at * SR))
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        a0, b0 = max(0, s), min(N, s + len(x))
        if b0 > a0:
            mix[a0:b0] += (gain * x[a0 - s:b0 - s]).astype(np.float32)

    norm = lambda x: x / max(1e-9, np.abs(x).max())    # noqa: E731
    # the same sounds every time: build once
    rn = int(1.3 * SR)
    ru = np.arange(rn) / rn
    noise = rng.standard_normal(rn)
    riser = np.zeros(rn)
    for lo, hi, w0, w1 in ((400, 1500, 0.0, 0.5), (1500, 5000, 0.3, 0.85), (5000, 12000, 0.6, 1.0)):
        riser += _bandpass(noise, lo, hi) * np.clip((ru - w0) / (w1 - w0), 0, 1) ** 2
    riser *= ru ** 2.5
    riser[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))
    riser = norm(riser)
    bt = np.arange(int(1.6 * SR)) / SR
    boom = np.tanh(np.sin(2 * np.pi * np.cumsum(38 + 95 * np.exp(-bt / 0.07)) / SR) * np.exp(-bt / 0.42) * 2.2)
    ck = np.arange(int(0.25 * SR)) / SR
    crack = norm(_bandpass(rng.standard_normal(len(ck)), 1200, 7000) * np.exp(-ck / 0.035))
    at_ = np.arange(int(1.4 * SR)) / SR
    air = norm(_bandpass(rng.standard_normal(len(at_)), 150, 900) * np.exp(-at_ / 0.5))
    scale = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26, 28]
    ct = np.arange(int(0.35 * SR)) / SR
    tick = norm(_bandpass(rng.standard_normal(len(ct)), 2500, 9000) * np.exp(-ct / 0.03))

    wt = np.arange(int(0.45 * SR)) / SR
    ot = np.arange(int(2.2 * SR)) / SR
    ohh = np.stack([_bandpass(rng.standard_normal(len(ot)), 300, 1100) + 0.5 * _bandpass(rng.standard_normal(len(ot)), 1100, 2600) for _ in range(2)], 1)
    bank = {"boom": boom, "crack": crack, "air": air,
            "whoosh": norm(_bandpass(rng.standard_normal(len(wt)), 500, 6000) * np.sin(np.pi * wt / 0.45) ** 2 * (1 + 0.5 * np.sin(2 * np.pi * 30 * wt))),
            "ohh": norm(ohh * (np.clip(ot / 0.25, 0, 1) * np.exp(-np.clip(ot - 0.5, 0, None) / 0.6) * (1 + 0.2 * np.sin(2 * np.pi * 1.3 * ot)))[:, None])}

    for ev in events:
        th = float(ev["t"])
        if ev.get("sounds"):                                     # a hit effect (impact.py) brings its own
            if not ev.get("drop"):
                place(boom, th, 0.5)
                place(crack, th, 0.3)
            for name, at, gain in ev["sounds"]:
                place(bank[name], th + at, gain)
        if ev.get("drop"):
            place(riser, th - GAP - 1.3, 0.5)
            place(boom, th, 0.6)
            place(crack, th, 0.4)
            place(air, th + 0.01, 0.22)
        pops = ev.get("pops")
        if ev.get("swing"):                                      # the 3D freeze: air moving past as the camera goes round, a crowd under it
            ln = float(ev.get("len") or 4.0)
            zn = int(ln * SR)
            zu = np.arange(zn) / zn
            airp = norm(_bandpass(rng.standard_normal(zn), 250, 1800)) * (0.55 + 0.45 * np.sin(np.pi * zu)) \
                * np.clip(zu / 0.1, 0, 1) * np.clip((1 - zu) / 0.15, 0, 1)
            pan = 0.08 + 0.84 * zu
            place(np.stack([airp * (1 - pan), airp * pan], 1), th, 0.22)
            cn = int((ln + 1.5) * SR)
            cu = np.arange(cn) / SR
            crowd = np.stack([_bandpass(rng.standard_normal(cn), 350, 2600) for _ in range(2)], 1)
            crowd *= (np.clip(cu / 0.9, 0, 1) * np.clip((ln + 1.5 - cu) / 1.6, 0, 1) * (1 + 0.25 * np.sin(2 * np.pi * 0.7 * cu + 1.0)))[:, None]
            place(norm(crowd), th + 0.05, 0.14)
        if pops:
            ln = float(ev.get("len") or 4.0) + 1.5
            cn = int(ln * SR)
            cu = np.arange(cn) / SR
            crowd = np.stack([_bandpass(rng.standard_normal(cn), 350, 2600) for _ in range(2)], 1)
            wob = 1 + 0.25 * np.sin(2 * np.pi * 0.7 * cu + 1.0) + 0.15 * np.sin(2 * np.pi * 2.3 * cu)
            crowd *= (np.clip(cu / 0.9, 0, 1) * np.clip((ln - cu) / 1.6, 0, 1) * wob)[:, None]
            place(norm(crowd), th + 0.05, 0.16)
            for j, tg in enumerate(pops):
                f0 = 660 * 2 ** (scale[min(j, len(scale) - 1)] / 12)
                chime = (np.sin(2 * np.pi * f0 * ct) + 0.4 * np.sin(2 * np.pi * 2 * f0 * ct)) * np.exp(-ct / 0.09)
                place(chime, th + tg, 0.07)
                place(tick, th + tg, 0.06)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2), wf.setsampwidth(2), wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    return path
