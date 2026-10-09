"""The intro's last frame blowing apart on the beat to reveal the first play.

The intro ends on a big hit; the reel hard-cuts to the first play and this overlay (the intro's
final picture, with alpha) plays on top of the play's first BURST_S seconds:

  bricks  the picture cracks into a brick wall that blasts out from the centre toward the camera,
          tumbling and falling, with dust and debris
  smoke   the picture bursts into smoke from the centre outwards: a ragged hole races out, a
          rolling smoke front at its edge, puffs drifting and thinning over the play

Frame 0 is the intact picture, so the cut is seamless. Written as a .mov (qtrle, with alpha),
cached by the intro file and style. Drawn here with OpenCV/numpy: free and offline.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np

FPS = 30
BURST_S = 0.9
STYLES = ("bricks", "smoke")
VERSION = 2


def frame_before(mp4: Path, t: float) -> np.ndarray:
    """The last frame shown before time t (BGR): the picture the reel cuts away from."""
    k = max(0, int(round(t * FPS)) - 1)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", rf"select=eq(n\,{k})", "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    if img is None:
        raise RuntimeError(f"couldn't read frame {k} of {mp4.name}")
    return img


def _ease_out(u):
    u = np.clip(u, 0, 1)
    return 1 - (1 - u) ** 3


def _over(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite premultiplied-free BGRA src onto BGRA float dst at (x, y), clipped."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    a = s[..., 3:4] / 255.0
    d[..., :3] = s[..., :3] * a + d[..., :3] * (1 - a)
    d[..., 3:4] = 255 * a + d[..., 3:4] * (1 - a)


# ---------------------------------------------------------------- bricks
def _bricks(img: np.ndarray, rng) -> list[np.ndarray]:
    H, W = img.shape[:2]
    bh = int(round(min(W, H) / 8.5))
    bw = 2 * bh
    cx, cy = W / 2, H * 0.47
    maxd = np.hypot(max(cx, W - cx), max(cy, H - cy))   # to the farthest corner: cy isn't the middle

    bricks = []
    for r, y in enumerate(range(0, H, bh)):
        off = -(bw // 2) if r % 2 else 0
        for x in range(off, W, bw):
            x0, x1 = max(0, x), min(W, x + bw)
            y1 = min(H, y + bh)
            if x1 - x0 < 8 or y1 - y < 8:
                continue
            patch = img[y:y1, x0:x1].astype(np.float32)
            h, w = patch.shape[:2]
            # a brick's face: mortar groove round the edge, lit from above
            shade = np.ones((h, w), np.float32)
            e = max(2, bh // 22)
            shade[:e, :] *= 1.35
            shade[-e:, :] *= 0.55
            shade[:, :e] *= 1.15
            shade[:, -e:] *= 0.7
            face = np.clip(patch * shade[..., None], 0, 255)
            alpha = np.full((h, w), 255, np.float32)
            bx, by = (x0 + x1) / 2, (y + y1) / 2
            dx, dy = bx - cx, by - cy
            d = float(np.hypot(dx, dy)) + 1e-3
            near = max(0.0, 1 - d / maxd)               # never negative: a negative ** 1.5 is NaN
            spd = rng.uniform(500, 800) + 1900 * near ** 1.5
            bricks.append({"face": face, "alpha": alpha, "c": np.array([bx, by]), "d": d,
                           "v": np.array([dx / d, dy / d - 0.35]) * spd + rng.normal(0, 120, 2),
                           "w": rng.uniform(-9, 9) * (0.5 + near), "delay": 0.05 * (d / maxd),
                           "grow": 0.4 + 2.2 * near ** 2})
    bricks.sort(key=lambda b: -b["d"])           # far ones first: the centre flies at the camera, on top
    return bricks


def _brick_frame(bricks, t: float, size, rng_dust, dust) -> np.ndarray:
    W, H = size
    out = np.zeros((H, W, 4), np.float32)
    for b in bricks:
        dt = max(0.0, t - b["delay"])
        pos = b["c"] + b["v"] * dt + np.array([0, 1800]) * dt * dt
        ang = np.degrees(b["w"] * dt)
        sc = 1 + b["grow"] * dt
        fade = 1 - np.clip((t - 0.45) / 0.4, 0, 1)
        h, w = b["face"].shape[:2]
        rgba = np.dstack([b["face"], b["alpha"] * fade])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        side = int(np.ceil(np.hypot(w, h) * sc)) + 4
        M[:, 2] += [side / 2 - w / 2, side / 2 - h / 2]
        warped = cv2.warpAffine(rgba, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        if t > 0:                                 # a touch of motion blur once they fly
            k = int(min(9, 1 + np.hypot(*b["v"]) * (1 / FPS) * 0.15))
            if k > 1:
                warped = cv2.blur(warped, (k, k))
        _over(out, warped, int(pos[0] - side / 2), int(pos[1] - side / 2))
    # dust and grit thrown out of the cracks
    for p in dust:
        dt = t - p["t0"]
        if dt <= 0 or dt > p["life"]:
            continue
        q = p["p"] + p["v"] * dt + np.array([0, 900]) * dt * dt
        a = (1 - dt / p["life"]) * p["a"]
        cv2.circle(out, (int(q[0]), int(q[1])), p["r"], (p["c"], p["c"], p["c"], 255 * a), -1, cv2.LINE_AA)
    if t < 0.12:                                  # impact flash
        f = 0.55 * (1 - t / 0.12)
        out[..., :3] = out[..., :3] * (1 - f) + 255 * f * (out[..., 3:4] / 255)
    return out


# ---------------------------------------------------------------- smoke
def _noise(shape, rng, scales=(48, 24, 12)) -> np.ndarray:
    """Soft billowy noise 0..1: big blobs plus smaller ones (cell sizes in px of this grid)."""
    H, W = shape
    n = np.zeros((H, W), np.float32)
    for k, s in enumerate(scales):
        g = rng.random((max(2, H // s + 2), max(2, W // s + 2))).astype(np.float32)
        n += cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC) / (1.8 ** k)
    n = cv2.GaussianBlur(n, (0, 0), 2)
    n -= n.min()
    return n / (n.max() + 1e-6)


def _smoke_setup(img, rng, team_bgr):
    H, W = img.shape[:2]
    q = 4                                          # masks at quarter size, upscaled
    h, w = H // q, W // q
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h * 0.47
    r = np.hypot((xx - cx) / (w / 2), (yy - cy) / (h / 2)) / np.sqrt(2)     # 0 centre .. ~1 corners
    ang = np.arctan2(yy - cy, xx - cx)
    return {"r": r, "ang": ang, "n1": _noise((h, w), rng), "n2": _noise((h, w), rng, (32, 16, 8)),
            "img": img.astype(np.float32), "team": np.array(team_bgr, np.float32), "size": (W, H)}


def _smoke_frame(s, t: float) -> np.ndarray:
    W, H = s["size"]
    r, n1, n2 = s["r"], s["n1"], s["n2"]
    u = t / BURST_S
    rho = 1.3 * _ease_out(t / 0.8)                # the hole races out from the centre
    # ragged edge: noise pushed outward over time, like billowing
    shift = int(6 * u * 4)
    n_t = np.roll(n1, shift, axis=0) * 0.6 + np.roll(n2, -shift, axis=1) * 0.4
    d = r + 0.35 * (n_t - 0.5)
    pic_a = np.clip((d - rho) / 0.06, 0, 1)                                   # what's left of the picture
    front = np.exp(-((d - rho - 0.02) / 0.13) ** 2)                           # rolling smoke at the edge
    inside = np.clip(1 - (rho - d) / 0.45, 0, 1) * (d < rho)                  # thinning smoke in the hole
    smoke_a = np.clip((1.0 * front + 0.45 * inside) * (0.45 + 0.75 * n2), 0, 1) * (1 - u) ** 1.2
    up = lambda m: cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    pic_a, smoke_a, shade = up(pic_a), up(smoke_a), up(0.78 + 0.32 * n2)
    grey = np.array([215, 215, 220], np.float32)
    col = np.clip((grey * 0.8 + s["team"] * 0.2)[None, None, :] * shade[..., None], 0, 255)
    if t < 0.1:                                   # flash at the blast
        col = col + (255 - col) * (0.6 * (1 - t / 0.1))
    a = smoke_a + pic_a * (1 - smoke_a)
    rgb = (col * smoke_a[..., None] + s["img"] * (pic_a * (1 - smoke_a))[..., None]) / np.maximum(a, 1e-4)[..., None]
    return np.dstack([np.clip(rgb, 0, 255), a * 255])


# ---------------------------------------------------------------- render
def cache_path(cache: Path, intro_mp4: Path, style: str, at: float) -> Path:
    st = intro_mp4.stat()
    key = hashlib.sha1(f"{VERSION}|{style}|{intro_mp4.name}|{st.st_size}|{st.st_mtime}|{at:.4f}".encode()).hexdigest()[:12]
    return Path(cache) / "burst" / f"burst-{style}-{key}.mov"


def render(cache: Path, intro_mp4: Path, style: str, at: float, team_hex: str | None = None, seed: int = 5) -> Path:
    """Render (or reuse) the burst overlay for this intro, cut at `at` seconds. Returns the .mov (with alpha)."""
    style = style if style in STYLES else "bricks"
    out = cache_path(cache, intro_mp4, style, at)
    if out.exists():
        return out
    img = frame_before(intro_mp4, at)
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    h = (team_hex or "#808080").lstrip("#")
    team = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)) if len(h) == 6 else (128, 128, 128)
    n = int(round(BURST_S * FPS))
    if style == "bricks":
        bricks = _bricks(img, rng)
        dust = [{"p": np.array([W / 2, H * 0.47]) + rng.normal(0, [W * 0.18, H * 0.12]),
                 "v": rng.normal(0, 900, 2), "t0": rng.uniform(0, 0.12), "life": rng.uniform(0.3, 0.8),
                 "r": int(rng.integers(2, 7)), "c": float(rng.uniform(150, 230)), "a": rng.uniform(0.5, 1)}
                for _ in range(260)]
        frame = lambda t: _brick_frame(bricks, t, (W, H), rng, dust)   # noqa: E731
    else:
        s = _smoke_setup(img, rng, team)
        frame = lambda t: _smoke_frame(s, t)                            # noqa: E731
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mov")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "pipe:0", "-c:v", "qtrle", "-pix_fmt", "argb", str(tmp)],
                           stdin=subprocess.PIPE)
    try:
        for k in range(n):
            f = frame(k / FPS)
            if k == 0:
                f = np.dstack([img.astype(np.float32), np.full((H, W), 255, np.float32)])   # seamless cut
            enc.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("burst render failed")
    tmp.replace(out)
    return out


def preview(intro_mp4: Path, cut: float, burst_mov: Path, play_src: Path, play_t0: float,
            music: Path | None, music_start: float, out: Path, play_s: float = 2.2) -> Path:
    """Quick look for the Build panel: the intro up to the cut, then the burst over the first
    play's opening (from its review copy), with the song underneath."""
    w, h = (int(v) for v in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(intro_mp4)], capture_output=True, text=True).stdout.strip().split(",")[:2])
    total = cut + play_s
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(intro_mp4), "-ss", f"{play_t0:.3f}", "-i", str(play_src),
           "-i", str(burst_mov)]
    fc = [f"[0:v]trim=end={cut:.4f},setpts=PTS-STARTPTS,fps={FPS},format=yuv420p,setsar=1[iv]",
          f"[1:v]trim=duration={play_s:.3f},setpts=PTS-STARTPTS,fps={FPS},"
          f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1[pv0]",
          f"[2:v]setpts=PTS-STARTPTS,fps={FPS},format=yuva420p[b]",
          "[pv0][b]overlay=0:0:eof_action=pass,format=yuv420p[pv]",
          "[iv][pv]concat=n=2:v=1:a=0[v]",
          f"[0:a]atrim=end={cut:.4f},asetpts=PTS-STARTPTS,apad,atrim=end={total:.4f}[ia]"]
    if music:
        cmd += ["-ss", f"{music_start:.3f}", "-i", str(music)]
        fc.append(f"[3:a]aresample=48000,aformat=channel_layouts=stereo,atrim=end={total:.4f},afade=t=in:d=0.5[m]")
        fc.append("[ia][m]amix=inputs=2:normalize=0:duration=longest[a]")
    else:
        fc.append("[ia]anull[a]")
    from . import encoder
    cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]", *encoder.h264(18),
            "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.3f}", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out
