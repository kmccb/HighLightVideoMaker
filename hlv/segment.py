"""Render one play as a finished 1080p30 segment for the reel.

Bakes in: trim window, camera (zoom) path, player highlight, and the freeze-frame intro:
at the snap the picture holds ~1.4 s, the field dims to a spotlight on the player, the
camera pushes in slightly and the name banner slides in; then play resumes, the
spotlight hands over to the chosen highlight style and the camera eases back.
Segments are cached by a hash of everything that affects them.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import camera, graphics, project, slowmo
from .autozoom import _smoothstep, draw_highlight
from .pipeline import CACHE_DIRNAME
from .probe import probe

RENDER_VERSION = 6
FPS = 30
OW, OH = graphics.W, graphics.H
FREEZE_S = 1.4           # nominal freeze; beat sync may use FREEZE_MIN..FREEZE_MAX
FREEZE_MIN, FREEZE_MAX = 1.0, 2.2
LEAD_MAX = 2.5           # footage kept before the snap so the reel can start earlier
TAIL_EXTRA = 2.0         # footage kept after the out-point
PUSH_IN = 1.08           # extra zoom during the freeze (completes in the first second)
RESUME_FRAMES = 15       # spotlight -> style handover and zoom ease-back after the freeze
HIT_FRAMES = 9           # zoom bump + flash on the hit
FREEZE_REACH = 1.5       # a click must be this close to the snap to freeze there (as the review strip)
FREEZE_CONF = 0.35       # tracker confidence needed at the snap to freeze on the player
FREEZE_DIM = 0.78        # how dark the field goes around him during the freeze (0 = not at all)
RING_FADE = 0.25         # highlight fades out before a "ring off" point and back in after "on"


def ring_visibility(ring_off: list, t: float) -> float:
    """How visible the highlight is at time t, given [[off, on or None], ...] ranges where
    you turned it off (e.g. once he's in a pile). Gone by `off`, back in by `on` + fade."""
    v = 1.0
    for a, b in ring_off or []:
        gone = float(_smoothstep((t - (a - RING_FADE)) / RING_FADE))
        if b is not None:
            gone *= 1 - float(_smoothstep((t - b) / RING_FADE))
        v *= 1 - gone
    return v


def _window(clip: dict) -> tuple[float, float, float]:
    """Nominal (in, out, snap) from review/proposals."""
    r, d = clip.get("review") or {}, clip["detection"]
    snap = r.get("snap") if r.get("snap") is not None else d["snap"]
    t0, t1 = project.final_window(clip)
    return max(0.0, t0), t1, snap


def extents(clip: dict, duration: float) -> tuple[float, float]:
    """What we actually render: extra lead and tail so the reel can re-time without re-rendering."""
    t_in, t_out, snap = _window(clip)
    return max(0.0, min(t_in, snap - LEAD_MAX)), min(duration, t_out + TAIL_EXTRA)


def freeze_time(player, snap: float, t0: float, t1: float, t_out: float,
                clicks: list[float] = ()) -> tuple[float | None, bool]:
    """When to freeze on the player: at the snap if he's found there AND you clicked him within
    FREEZE_REACH of it (tracking further back than that guesses - it has spotlit the sideline);
    else at your first click after the snap, the one moment we know for sure where he is, if
    that's before the play ends. Returns (time, at_snap)."""
    if player is None:
        return None, False
    ok = lambda g: g < len(player) and player[g][0] >= 0 and player[g][4] >= FREEZE_CONF   # noqa: E731
    g = int(round(snap * FPS))
    near = any(abs(t - snap) <= FREEZE_REACH for t in clicks) if clicks else True
    if t0 <= snap < t1 and near and ok(g):
        return g / FPS, True
    for t in sorted(clicks):
        k = int(round(t * FPS))
        if snap < t < min(t1, t_out) - 0.5 and ok(k):
            return k / FPS, False
    return None, False


def segment_key(clip: dict, pl: dict | None, settings: dict, plan: dict, src: Path) -> str:
    r = clip.get("review") or {}
    payload = {
        "v": RENDER_VERSION, "src": [src.stat().st_size, src.stat().st_mtime],
        "win": _window(clip), "zoom": r.get("zoom"), "spot": r.get("spotlight"),
        "style": settings.get("highlight"), "freeze": settings.get("freeze", True),
        "hit": r.get("hit"), "hit_fx": settings.get("hit_fx", True) and settings.get("_hit"),
        "slowmo": [settings.get("slowmo"), settings.get("slowmo_speed")] if settings.get("slowmo") else None,
        # Profile text only matters to plays that show the name banner; the accent colour
        # (highlight ring) matters to all of them.
        "banner": bool(settings.get("_banner", True)),
        "pl": {k: (pl or {}).get(k) for k in (("name", "number", "position", "class_year",
                                                "height_weight", "color", "accent")
                                               if settings.get("_banner", True) else ("accent",))},
        "plan": hashlib.sha1(json.dumps([plan.get("rects"), plan.get("player")]).encode()).hexdigest(),
    }
    if settings.get("_aspect") == "9:16":
        payload["aspect"] = "9:16"
    if r.get("ring_off"):                      # only keyed when used, so old renders stay valid
        payload["ring_off"] = r["ring_off"]
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def render(root: Path, clip: dict, pl: dict | None, settings: dict, log=print) -> Path:
    """Render (or reuse) this clip's extended segment. Returns the mp4 path; the layout
    (lead / freeze / post frame counts) is in the .json next to it."""
    from . import hits
    cache = root / CACHE_DIRNAME
    src = root / clip["source"]
    plan = camera.plan(cache, clip)
    settings = {**settings, "_hit": hits.hit_of(root, clip)[0]}
    out = cache / "segments" / f"{clip['id']}-{segment_key(clip, pl, settings, plan, src)}.mp4"
    if out.exists() and out.with_suffix(".json").exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    _render(src, clip, pl, settings, plan, out)
    return out


def layout(seg: Path) -> dict:
    return json.loads(seg.with_suffix(".json").read_text())


VERTICAL = (1080, 1920)


def _vertical_rects(rects: np.ndarray, player: np.ndarray | None, W: int, H: int, info) -> np.ndarray:
    """9:16 windows from the 16:9 camera windows: same height, following the player sideways
    (smoothed; the window's own centre where he isn't known), kept inside the picture."""
    from scipy.ndimage import gaussian_filter1d
    out = rects.copy()
    h = np.minimum(rects[:, 3], H)
    w = np.minimum(h * 9 / 16, W)
    cx = rects[:, 0] + rects[:, 2] / 2
    if player is not None:
        n = min(len(player), len(cx))
        p = player[:n]
        ok = (p[:, 0] >= 0) & ~np.isnan(p[:, 0]) & (np.nan_to_num(p[:, 4]) >= 0.3)
        if ok.sum() > 5:
            idx = np.arange(n)
            px = np.interp(idx, idx[ok], p[ok, 0] * info.width)
            # where he's unknown, drift back toward the camera window's centre
            near = np.interp(idx, idx[ok], idx[ok].astype(float))
            far = np.clip(np.abs(idx - near) / FPS / 1.0, 0, 1)
            cx[:n] = px * (1 - far) + cx[:n] * far
    cx = gaussian_filter1d(cx, 0.35 * FPS, mode="nearest")
    out[:, 0] = np.clip(cx - w / 2, 0, W - w)
    out[:, 2] = w
    out[:, 3] = h
    return out


def _render(src: Path, clip: dict, pl: dict | None, settings: dict, plan: dict, out: Path) -> None:
    vertical = settings.get("_aspect") == "9:16"
    OW, OH = VERTICAL if vertical else (graphics.W, graphics.H)
    info = probe(src)
    crop = clip.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    W, H = (crop["w"], crop["h"]) if crop else (info.width, info.height)
    _, _, snap = _window(clip)
    t0, t1 = extents(clip, info.duration)

    # Camera rects in content px. No zoom: the largest 16:9 window, centred.
    if plan.get("rects"):
        rects = np.array(plan["rects"]) * [info.width, info.height, info.width, info.height] - [ox, oy, 0, 0]
    else:
        fw = min(W, H * OW / OH)
        n = int(round(info.duration * FPS)) + 2
        rects = np.tile([(W - fw) / 2, (H - fw * OH / OW) / 2, fw, fw * OH / OW], (n, 1))
    rects[:, 2] = np.minimum(rects[:, 2], W)
    rects[:, 3] = np.minimum(rects[:, 3], H)
    player = np.array(plan["player"]) if plan.get("player") else None
    if vertical:
        rects = _vertical_rects(rects, player, W, H, info)
    style = settings.get("highlight", "ring")
    color = (pl or {}).get("accent") or settings.get("highlight_color", "#ffd23f")

    # Freeze at the snap if we know where the player is then (else as soon after as we do).
    freeze_t = None
    if settings.get("freeze", True):
        freeze_t, _ = freeze_time(player, snap, t0, t1, _window(clip)[1],
                                  [a["t"] for a in camera.anchors_of(clip)])
    nfz = int(round(FREEZE_MAX * FPS)) if freeze_t is not None else 0
    split_t = min(max(freeze_t if freeze_t is not None else snap, t0), t1)
    hit_t = settings.get("_hit")
    fx = hit_t is not None and settings.get("hit_fx", True)
    banner = graphics.name_banner(pl, (OW, OH)) if (pl and freeze_t is not None and settings.get("_banner", True)) else None

    # Slow-mo ramp into the hit (after the freeze), as a variable-speed output timeline.
    sspeed = float(settings.get("slowmo_speed", 0.4))
    smo = None
    if settings.get("slowmo") and hit_t is not None and split_t + 0.3 < hit_t < t1:
        smo = slowmo.window(hit_t, split_t + 0.15, t1)
    pre = slowmo.timeline(t0, split_t, FPS, None, 1.0)
    post = slowmo.timeline(split_t, t1, FPS, smo, sspeed)
    # Decode at 60 fps for slow-mo when the clip has real high-frame-rate frames.
    D = 60 if (smo and info.avg_fps >= 50) else FPS

    def lerp_row(arr, t):
        x = t * FPS
        n = int(np.floor(x))
        a, b = arr[min(max(n, 0), len(arr) - 1)], arr[min(max(n + 1, 0), len(arr) - 1)]
        if a[0] < 0 or b[0] < 0 or np.isnan(a[0]) or np.isnan(b[0]):
            return a if (x - n) < 0.5 else b
        return a + (b - a) * (x - n)

    def frame_out(f, t, extra_zoom=1.0):
        """Crop + scale one source frame at source time t; returns (img, rect) in content px."""
        x, y, w, h = lerp_row(rects, t)
        p = lerp_row(player, t) if player is not None else None
        if extra_zoom > 1 and p is not None and p[0] >= 0:
            px, py = p[0] * info.width - ox, p[1] * info.height - oy
            w2, h2 = w / extra_zoom, h / extra_zoom
            # push in toward the player, keeping the frame inside the picture
            x = float(np.clip(x + (px - x) * (1 - 1 / extra_zoom), 0, W - w2))
            y = float(np.clip(y + (py - y) * (1 - 1 / extra_zoom), 0, H - h2))
            w, h = w2, h2
        xi, yi = int(x), int(y)
        wi, hi = max(2, min(int(np.ceil(w)), W - xi)), max(2, min(int(np.ceil(h)), H - yi))
        img = cv2.resize(f[yi:yi + hi, xi:xi + wi], (OW, OH),
                         interpolation=cv2.INTER_AREA if wi >= OW else cv2.INTER_CUBIC)
        return img, (x, y, w, h)

    ring_off = (clip.get("review") or {}).get("ring_off") or []

    def highlight(img, t, rect, style_, alpha_mul=1.0, dim=0.5):
        if player is None or alpha_mul <= 0:
            return
        p = lerp_row(player, t)
        if p[0] < 0 or np.isnan(p[0]):
            return
        x, y, w, h = rect
        s = OW / w
        a = float(np.clip((p[4] - 0.2) / 0.3, 0, 1)) * alpha_mul
        draw_highlight(img, style_, (p[0] * info.width - ox - x) * s, (p[1] * info.height - oy - y) * s,
                       p[2] * info.width * s, p[3] * info.height * s, a, color, t, dim=dim)

    # ---- audio: lead | silence (freeze) | play, time-stretched through the slow-mo ----
    dur = t1 - t0

    def out_len(lo, hi):
        return sum(1 for t in post if lo <= t < hi) / FPS

    pieces = [(t0, split_t, len(pre) / FPS)]
    if nfz:
        pieces.append((None, None, nfz / FPS))
    if smo:
        a_, b_, c_, d_ = smo
        pieces += [(split_t, a_, out_len(split_t, a_)), (a_, b_, out_len(a_, b_)),
                   (b_, c_, out_len(b_, c_)), (c_, d_, out_len(c_, d_)), (d_, t1, out_len(d_, t1))]
    else:
        pieces.append((split_t, t1, len(post) / FPS))
    total_out = (len(pre) + nfz + len(post)) / FPS
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
           "-r", str(FPS), "-i", "pipe:0"]
    fc = []
    if info.has_audio:
        cmd += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
        live = [pc for pc in pieces if pc[0] is not None and pc[2] > 0]
        fc.append(f"[1:a:0]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{dur:.3f},"
                  f"asplit={max(len(live), 1)}" + "".join(f"[s{k}]" for k in range(max(len(live), 1))))
    labels, k = [], 0
    for n_pc, (lo, hi, od) in enumerate(pieces):
        if od <= 0:
            continue
        if lo is None or not info.has_audio:
            fc.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{od:.4f}[q{n_pc}]")
        else:
            tempo = (hi - lo) / od
            st = "" if abs(tempo - 1) < 0.01 else "," + slowmo.atempo(tempo)
            fc.append(f"[s{k}]atrim={lo - t0:.4f}:{hi - t0:.4f},asetpts=PTS-STARTPTS{st},"
                      f"apad,atrim=0:{od:.4f}[q{n_pc}]")
            k += 1
        labels.append(f"[q{n_pc}]")
    fc.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1,atrim=0:{total_out:.4f}[aout]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    vf = ([f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}"] if crop else []) + [f"fps={D}", "format=bgr24"]
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
                            "-an", "-vf", ",".join(vf), "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE)
    fbytes = W * H * 3
    buf: dict[int, np.ndarray] = {}
    decoded = [0]

    def src_frame(n):
        """Decoded frame n (at D fps from t0); keeps a small window, decodes forward."""
        while decoded[0] <= n:
            raw = dec.stdout.read(fbytes)
            if len(raw) < fbytes:
                break
            buf[decoded[0]] = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
            decoded[0] += 1
            for old in [i for i in buf if i < decoded[0] - 4]:
                del buf[old]
        if not buf:
            return None
        return buf.get(n) if n in buf else buf[max(buf)]

    interp = slowmo.Interpolator()

    def picture(t, extra_zoom=1.0):
        """Output picture at source time t, interpolating between real frames if needed."""
        x = (t - t0) * D
        n = int(np.floor(x + 1e-6))
        frac = x - n
        fa = src_frame(n)
        if fa is None:
            return None, None
        img, rect = frame_out(fa, t, extra_zoom)
        if 0.04 < frac < 0.96:
            fb = src_frame(n + 1)
            if fb is not None and fb is not fa:
                img_b, _ = frame_out(fb, t, extra_zoom)
                img = interp.between(img, img_b, frac, key=n)
        return img, rect

    written = 0
    hit_k = next((i for i, t in enumerate(post) if hit_t is not None and t >= hit_t), None) if fx else None
    try:
        for t in pre:
            img, rect = picture(t)
            if img is None:
                break
            highlight(img, t, rect, style, ring_visibility(ring_off, t))
            enc.stdin.write(img.tobytes())
            written += 1
        for jz in range(nfz):
            u = jz / FPS / (FREEZE_MIN - 0.05)   # push-in done before the shortest freeze ends
            img, rect = picture(split_t, 1 + (PUSH_IN - 1) * _smoothstep(u))
            if img is None:
                break
            highlight(img, split_t, rect, "shadow", _smoothstep(jz / 8), dim=FREEZE_DIM)
            if banner is not None:
                ba = _smoothstep((jz - 4) / 10)
                graphics.composite(img, banner, ba, slide_px=int(60 * (1 - ba)))
            enc.stdin.write(img.tobytes())
            written += 1
        for kk, t in enumerate(post):
            if nfz and kk < RESUME_FRAMES:
                # hand over: zoom eases back, spotlight fades, chosen style fades in, banner out
                v = _smoothstep(kk / RESUME_FRAMES)
                img, rect = picture(t, 1 + (PUSH_IN - 1) * (1 - v))
                if img is None:
                    break
                highlight(img, t, rect, "shadow", 1 - v, dim=FREEZE_DIM)
                if style != "shadow":
                    highlight(img, t, rect, style, v * ring_visibility(ring_off, t))
                if banner is not None:
                    graphics.composite(img, banner, 1 - v)
            else:
                hj = kk - hit_k if hit_k is not None else -1
                bump = 1 + 0.06 * np.sin(np.pi * hj / HIT_FRAMES) if 0 <= hj < HIT_FRAMES else 1.0
                img, rect = picture(t, bump)
                if img is None:
                    break
                highlight(img, t, rect, style, ring_visibility(ring_off, t))
                if 0 <= hj < 5:
                    a = 0.35 * (1 - hj / 5)                               # white flash
                    img[:] = (img.astype(np.float32) * (1 - a) + 255 * a).astype(np.uint8)
            enc.stdin.write(img.tobytes())
            written += 1
    finally:
        dec.stdout.close()
        dec.wait()
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0 or not out.exists():
        raise RuntimeError(f"segment render failed for {clip['source']}")
    t_in, t_out, _ = _window(clip)
    b_frames = max(0, written - len(pre) - nfz)
    out.with_suffix(".json").write_text(json.dumps({
        "fps": FPS, "t0": t0, "snap": snap, "split": split_t, "hit": hit_t,
        "a_frames": len(pre),             # output frames before the snap (lead, real time)
        "f_frames": nfz,                  # freeze frames (0 = no freeze)
        "b_frames": b_frames,             # output frames from the snap on (play + tail)
        # where things land inside the play part, in output seconds (slow-mo stretches it)
        "hit_b": None if hit_t is None else sum(1 for t in post if t < hit_t) / FPS,
        "post_nom": sum(1 for t in post if t < t_out) / FPS,
        "slowmo": None if smo is None else {"window": smo, "speed": sspeed},
    }))


def card(out: Path, img: np.ndarray, seconds: float, fade_in: bool) -> Path:
    """Still card (title/end) as a segment with silent audio."""
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    cv2.imwrite(str(png), img)
    vf = "format=yuv420p" + (",fade=t=in:st=0:d=0.6" if fade_in else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(png),
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{seconds:.3f}",
                    "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-r", str(FPS),
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
                   check=True)
    png.unlink(missing_ok=True)
    return out
