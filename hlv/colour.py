"""Big-play effect: only him in colour. For the last second and a half before the hit everything
drains to black and white except your player; colour comes back on the hit.

It works on the play's own rendered segment (so the framing, ring and slow motion are exactly the
reel's): `render` takes those frames, finds his outline in each with the person-segmentation model
(parallax.py), and writes them out again as a short clip with the same sound, which reel.build
plays in place of that stretch. Where he is in each frame comes from the tracker's path and the
reel's framing (ghost.reel_view), so nothing is searched for.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import camera, encoder, ghost, parallax, slowmo
from .pipeline import CACHE_DIRNAME

VERSION = 1
FPS = 30
RUN = 1.5             # seconds before the hit
MIN_FRAMES = 12       # shorter than this isn't worth it
LEFT = 6              # frames of normal picture kept before it (the freeze's hand-over)


def frames(hit_b: float | None) -> int:
    """How many frames before the hit go black and white (0 = none): RUN seconds, or what there is."""
    if hit_b is None or hit_b <= 0:
        return 0
    n = min(int(round(RUN * FPS)), int(round(hit_b * FPS)) - LEFT)
    return n if n >= MIN_FRAMES else 0


def render(root: Path, clip: dict, seg: Path, size: tuple[int, int] | None = None, log=print) -> tuple[Path, int] | None:
    """Make (or reuse) the run-up for this play from its rendered segment `seg`: (clip, frames),
    or None when there's no hit, too little before it, or the model is missing."""
    from . import segment
    if not parallax.available():
        return None
    root, seg = Path(root), Path(seg)
    lay = segment.layout(seg)
    n = frames(lay.get("hit_b"))
    if not n or lay.get("hit") is None:
        return None
    OW, OH = tuple(size) if size else ghost.WIDE
    hbk = int(round(lay["hit_b"] * FPS))
    f0 = lay["a_frames"] + lay["f_frames"] + hbk - n
    out = root / CACHE_DIRNAME / "colour" / f"{seg.stem}-{f0}-{n}-v{VERSION}.mp4"
    if out.exists():
        return out, n
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = root / CACHE_DIRNAME
    plan = camera.plan(cache, clip)
    if not plan.get("player"):
        return None
    player = np.array(plan["player"])
    info, crop = clip["probe"], clip.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    # the source time of each of those frames (slow motion stretches the run into the hit)
    _, t1 = segment.extents(clip, info["duration"])
    smo = lay.get("slowmo") or {}
    ts = slowmo.timeline(lay["split"], t1, FPS, smo.get("window"), float(smo.get("speed") or 1.0))[hbk - n:hbk]
    if len(ts) < n:
        return None

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(seg), "-vf", f"trim=start_frame={f0}:end_frame={f0 + n},setpts=PTS-STARTPTS",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    if len(raw) < n * OW * OH * 3:
        return None
    fr = np.frombuffer(raw, np.uint8)[:n * OW * OH * 3].reshape(n, OH, OW, 3)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-ss", f"{f0 / FPS:.5f}", "-t", f"{n / FPS:.5f}", "-i", str(seg), "-map", "0:v", "-map", "1:a",
                            *encoder.h264(17), "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
    prev = None
    found = 0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    try:
        for k in range(n):
            f = fr[k]
            rx, ry, rw, rh = ghost.reel_view(clip, plan, ts[k], (OW, OH))
            p = player[min(len(player) - 1, int(round(ts[k] * FPS)))]
            m = None
            if p[0] >= 0 and not np.isnan(p[0]):
                hx, hy = (p[0] * info["width"] - ox - rx) * OW / rw, (p[1] * info["height"] - oy - ry) * OH / rh
                hw, hh = p[2] * info["width"] * OW / rw, p[3] * info["height"] * OH / rh
                # look only at the area round him, enlarged: quicker, and a finer outline
                R = int(max(5 * hh, 320))
                X0, Y0 = int(np.clip(hx - R / 2, 0, max(0, OW - R))), int(np.clip(hy - R / 2, 0, max(0, OH - R)))
                X1, Y1 = min(OW, X0 + R), min(OH, Y0 + R)
                kz = min(3.0, 960 / max(X1 - X0, Y1 - Y0))
                big = cv2.resize(f[Y0:Y1, X0:X1], None, fx=kz, fy=kz, interpolation=cv2.INTER_CUBIC)
                people = parallax._outlines(big, 960)
                if people:
                    boxes = np.array([b / kz + [X0, Y0, X0, Y0] for b, _, _ in people])
                    best = parallax.which(boxes, hx, hy, hw, hh)
                    if best is not None:
                        m = np.zeros((OH, OW), np.float32)
                        m[Y0:Y1, X0:X1] = cv2.resize(people[best][2].astype(np.uint8), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
                        m = cv2.dilate(m, k3)
                        found += 1
                if m is None and prev is None:                  # no outline yet: a soft oval on the tracker's box
                    m = np.zeros((OH, OW), np.float32)
                    cv2.ellipse(m, (int(hx), int(hy)), (max(4, int(hw * 0.7)), max(6, int(hh * 0.8))), 0, 0, 360, 1.0, -1)
            if m is None:
                m = prev if prev is not None else np.zeros((OH, OW), np.float32)
            elif prev is not None:
                m = 0.65 * m + 0.35 * prev                       # steadier from frame to frame
            prev = m
            keep = cv2.GaussianBlur(m, (0, 0), 2.0)[..., None]
            u = ghost._ease(k / 9)                              # the colour drains out over a third of a second
            bw = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32) * 0.86
            ff = f.astype(np.float32)
            enc.stdin.write(np.clip(ff + (bw - ff) * (u * (1 - keep)), 0, 255).astype(np.uint8).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"colour run-up failed for {clip['source']}")
    tmp.replace(out)
    log(f"  only him in colour {clip['source']}: {n} frames, outline found in {found}")
    return out, n
