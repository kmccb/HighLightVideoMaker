"""Prototype: The Journey. An opener that starts with him as a little kid in flag football and
cuts through to the player he is now, the camera locked on him the whole way.

journey.py <out.mp4> --born 2010 --then <project> <clip>@<time> ... --now <project> <clip>@<time> ...

Shots are plays where he's been clicked (the tracker's path keeps him in the same spot of the
frame in every shot, so the cuts match). "Then" shots get a home-video look, slow motion and the
year and his age; a year counter spins up to today on the song's drop; "now" shots cut on the
beat and the last one freezes under his name.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from hlv import beats, camera, ghost, graphics, library, profiles, project

FPS, OW, OH, SR = 30, 1920, 1080, 48000
THEN_F, THEN_RATE, XF = 78, 0.7, 8          # frames per old shot, its speed, cross-dissolve frames
TICK_F = 39                                  # the year counter
NOW_F, LAST_F, HOLD_F = 30, 66, 78           # quick shots, the last shot's run-up, its freeze
TITLE_F = 54
AT = (0.5, 0.58)                             # where he sits in the frame
TARGET_H = 0.20                              # how tall he is in the frame, when the footage allows
MAX_UP = 1.9                                 # never enlarge the source more than this


def text_layer(lines: list[tuple[str, int, tuple[int, int], str, tuple[int, int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    """lines: (text, size, (x, y), anchor, rgb). Returns (bgr float, alpha float) at output size, with a soft shadow."""
    img = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    sh = Image.new("L", (OW, OH), 0)
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(sh)
    for txt, size, xy, anchor, rgb in lines:
        f = graphics.font(size)
        ds.text((xy[0] + 3, xy[1] + 4), txt, font=f, fill=200, anchor=anchor)
        d.text(xy, txt, font=f, fill=(*rgb, 255), anchor=anchor)
    a = np.asarray(img, np.float32)
    shadow = cv2.GaussianBlur(np.asarray(sh, np.float32) / 255, (0, 0), 6) * 0.7
    alpha = a[..., 3] / 255
    return a[..., 2::-1].copy(), np.maximum(alpha, 0)[..., None], shadow[..., None]


def over(frame: np.ndarray, layer, opacity: float = 1.0) -> np.ndarray:
    bgr, alpha, shadow = layer
    out = frame.astype(np.float32) * (1 - shadow * opacity)
    return (out * (1 - alpha * opacity) + bgr * alpha * opacity).astype(np.uint8)


def shot(root: Path, clip_id: str, t_key: float, before: float, after: float, n_out: int) -> dict:
    """n_out frames covering [t_key - before, t_key + after] of the clip, the camera locked on him."""
    proj = project.load(root)
    c = next(c for c in proj["clips"] if c["id"] == clip_id)
    info, crop = c["probe"], c.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    Wc, Hc = (crop["w"], crop["h"]) if crop else (info["width"], info["height"])
    Wd = int(min(Wc, 2560) // 2 * 2)
    Hd = int(round(Hc * Wd / Wc / 2)) * 2
    t0 = round(max(0.0, t_key - before) * FPS) / FPS
    n_src = int(round((before + after) * FPS)) + 2
    fr = ghost._read(root / c["source"], t0, n_src, crop, (Wd, Hd))
    plan = camera.plan(root / ".hlv_cache", c)
    p = np.array(plan["player"])
    k0 = int(round(t0 * FPS))
    rows = p[np.clip(np.arange(k0, k0 + len(fr)), 0, len(p) - 1)].copy()
    ok = rows[:, 0] >= 0
    for j in (0, 1, 2, 3):                                       # fill gaps, then smooth: a locked camera mustn't jitter
        rows[:, j] = np.interp(np.arange(len(rows)), np.where(ok)[0], rows[ok, j]) if ok.any() else 0.5
        rows[:, j] = cv2.GaussianBlur(rows[:, j].reshape(-1, 1), (1, 0), 9, borderType=cv2.BORDER_REPLICATE).ravel() if len(rows) > 3 else rows[:, j]
    sx, sy = info["width"] * Wd / Wc, info["height"] * Hd / Hc
    cx, cy = rows[:, 0] * sx - ox * Wd / Wc, rows[:, 1] * sy - oy * Hd / Hc
    box_h = float(np.median(rows[:, 3])) * sy
    A = OW / OH
    fit_w = min(Wd, Hd * A)                                      # the widest 16:9 window the picture holds
    cw = float(np.clip(box_h / TARGET_H * A, OW / MAX_UP, fit_w))
    ch = cw / A
    idx = np.linspace(0, len(fr) - 1.001, n_out)
    out = np.empty((n_out, OH, OW, 3), np.uint8)
    spot = []
    for j, x in enumerate(idx):
        i = int(x)
        f = fr[i] if x - i < 0.02 else cv2.addWeighted(fr[i], 1 - (x - i), fr[i + 1], x - i, 0)
        px, py = cx[i] + (cx[i + 1] - cx[i]) * (x - i), cy[i] + (cy[i + 1] - cy[i]) * (x - i)
        rx = float(np.clip(px - cw * AT[0], 0, Wd - cw))
        ry = float(np.clip(py - ch * AT[1], 0, Hd - ch))
        M = np.array([[OW / cw, 0, -rx * OW / cw], [0, OH / ch, -ry * OH / ch]])
        out[j] = cv2.warpAffine(f, M, (OW, OH), flags=cv2.INTER_AREA if cw > OW * 1.3 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        spot.append(((px - rx) * OW / cw, (py - ry) * OH / ch, box_h * OH / ch))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{before + after:.4f}", "-i", str(root / c["source"]), "-vn",
                          "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    taken = c.get("taken")
    print(f"  {clip_id[:44]:44} at {t_key:5.1f}s  he is {box_h * OH / ch:4.0f} px tall on screen, source enlarged {OW / cw:.2f}x", flush=True)
    return {"frames": out, "spot": spot, "audio": np.frombuffer(raw, np.float32).reshape(-1, 2),
            "date": datetime.fromisoformat(taken) if taken else None}


def old_look(f: np.ndarray, vig: np.ndarray, rng) -> np.ndarray:
    x = f.astype(np.float32)
    g = x.mean(axis=2, keepdims=True)
    x = g * 0.38 + x * 0.62                                      # faded colour
    x *= np.array([0.90, 1.0, 1.08], np.float32)                 # warm
    x = x * 0.86 + 20                                            # lifted blacks
    x += cv2.GaussianBlur(x, (0, 0), 14) * 0.22                  # soft glow
    x *= vig
    x += rng.normal(0, 5.0, (OH // 2, OW // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
    return np.clip(x, 0, 255).astype(np.uint8)


def ring(f: np.ndarray, spot, colour, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return f
    x, y, h = spot
    o = f.copy()
    cv2.ellipse(o, (int(x), int(y + h * 0.55)), (int(max(26, h * 0.5)), int(max(9, h * 0.16))), 0, 0, 360, colour, 3, cv2.LINE_AA)
    return cv2.addWeighted(o, alpha, f, 1 - alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--born", type=int, required=True)
    ap.add_argument("--then", nargs="+", required=True, help="project then clip@time pairs")
    ap.add_argument("--now", nargs="+", required=True)
    ap.add_argument("--line", default="EVERY PLAYER STARTS SOMEWHERE.")
    a = ap.parse_args()
    rng = np.random.default_rng(11)

    def parse(v):
        root = Path(v[0])
        return root, [(s.rsplit("@", 1)[0], float(s.rsplit("@", 1)[1])) for s in v[1:]]

    then_root, then = parse(a.then)
    now_root, now = parse(a.now)
    nproj = project.load(now_root)
    settings = project.settings(nproj)
    pl = profiles.get(library.home(), settings.get("player")) or {}
    acc = pl.get("accent") or "#ffffff"
    accent = tuple(int(acc[j:j + 2], 16) for j in (1, 3, 5))
    print("then:", flush=True)
    olds = [shot(then_root, cid, t, 1.0, THEN_F * THEN_RATE / FPS - 1.0, THEN_F) for cid, t in then]
    print("now:", flush=True)
    quick = [shot(now_root, cid, t, (NOW_F - 6) / FPS, 6 / FPS, NOW_F) for cid, t in now[:-1]]
    last = shot(now_root, now[-1][0], now[-1][1], LAST_F / FPS, 0.0, LAST_F)
    year_then = olds[0]["date"].year if olds[0]["date"] else None
    year_now = (last["date"] or datetime.now()).year

    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    vig = (1 - 0.55 * np.clip(((xx - OW / 2) / (OW * 0.62)) ** 2 + ((yy - OH / 2) / (OH * 0.62)) ** 2, 0, 1) ** 1.3)[..., None]
    white = (245, 244, 240)

    def stamp(d: datetime | None, year: int):
        age = year - a.born
        lines = [(str(year), 190, (90, OH - 70), "ls", white), (f"AGE {age}", 62, (100 + int(graphics.font(190).getlength(str(year))), OH - 82), "ls", accent)]
        if d:
            lines.append((d.strftime("%b %d %Y").upper().replace(" 0", " "), 40, (OW - 80, 84), "rs", white))
        return text_layer(lines)

    frames: list[np.ndarray] = []
    # 1) the line
    title = text_layer([(a.line, 74, (OW // 2, OH // 2), "mm", white)])
    black = np.zeros((OH, OW, 3), np.uint8)
    for j in range(TITLE_F):
        frames.append(over(black, title, min(1.0, j / 12, (TITLE_F - j) / 10)))
    # 2) then
    prev_tail = None
    for s in olds:
        st = stamp(s["date"], s["date"].year if s["date"] else year_then)
        graded = [old_look(f, vig, rng) for f in s["frames"]]
        for j, f in enumerate(graded):
            f = ring(f, s["spot"][j], white[::-1], 0.8 * min(1.0, j / 10) * min(1.0, (len(graded) - j) / 30) if j < 54 else 0.0)
            f = over(f, st, min(1.0, j / 8))
            if prev_tail is not None and j < XF:
                f = cv2.addWeighted(prev_tail[j], 1 - (j + 1) / (XF + 1), f, (j + 1) / (XF + 1), 0)
            elif prev_tail is None and j < 10:
                f = (f * (j / 10)).astype(np.uint8)
            frames.append(f)
        # the next shot dissolves in over this one's last frames (held)
        prev_tail = [frames[-1]] * XF
    t_tick = len(frames)
    # 3) the years spin up to now
    hold = graded[-1]
    for j in range(TICK_F):
        u = j / (TICK_F - 1)
        yr = int(round(year_then + (year_now - year_then) * u ** 2.2))
        k = 1 + 0.5 * u ** 2                                     # push in, blur, burn to white
        M = cv2.getRotationMatrix2D((OW * AT[0], OH * AT[1]), 0, k)
        f = cv2.warpAffine(hold, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        f = cv2.GaussianBlur(f, (0, 0), 1 + 22 * u)
        f = cv2.addWeighted(f, 1 - 0.25 * u, np.full_like(f, 255), 0.25 * u, 0)
        big = text_layer([(str(yr), 420, (OW // 2, OH // 2 + 20), "mm", white), (f"AGE {yr - a.born}", 80, (OW // 2, OH // 2 + 250), "mm", accent)])
        frames.append(over(f, big, min(1.0, j / 5)))
    t_cut = len(frames)
    # 4) now: cuts on the beat, then the last play runs to its hit and holds
    now_stamp = text_layer([(str(year_now), 110, (90, OH - 70), "ls", white),
                            (f"AGE {year_now - a.born}", 44, (100 + int(graphics.font(110).getlength(str(year_now))), OH - 78), "ls", accent)])

    def now_look(f):
        x = f.astype(np.float32)
        return np.clip((x - 128) * 1.08 + 128 + (x - x.mean(axis=2, keepdims=True)) * 0.10, 0, 255).astype(np.uint8)

    for qi, s in enumerate(quick):
        for j, f in enumerate(s["frames"]):
            f = now_look(f)
            if qi == 0 and j < 5:
                f = cv2.addWeighted(f, j / 5 * 0.6 + 0.4, np.full_like(f, 255), 0.6 - j / 5 * 0.6, 0)
            f = ring(f, s["spot"][j], accent[::-1], 0.9)
            frames.append(over(f, now_stamp))
    for j, f in enumerate(last["frames"]):
        frames.append(over(ring(now_look(f), last["spot"][j], accent[::-1], 0.9), now_stamp))
    t_freeze = len(frames)
    stop = now_look(last["frames"][-1])
    name = (pl.get("name") or "").upper()
    detail = "  ·  ".join(x for x in (f"#{pl['number']}" if pl.get("number") else "", pl.get("position") or "",
                                     f"CLASS OF {pl['class_year']}" if pl.get("class_year") else "", pl.get("school") or "") if x)
    card = text_layer([(name, 200, (OW // 2, OH // 2 - 10), "mm", white), (detail, 54, (OW // 2, OH // 2 + 130), "mm", accent)])
    for j in range(HOLD_F):
        u = min(1.0, j / 14)
        k = 1 + 0.05 * j / HOLD_F
        M = cv2.getRotationMatrix2D((OW * AT[0], OH * AT[1]), 0, k)
        f = cv2.warpAffine(stop, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        f = (f.astype(np.float32) * (1 - 0.55 * u)).astype(np.uint8)
        flash = max(0.0, 1 - j / 5) * 0.6
        f = cv2.addWeighted(f, 1 - flash, np.full_like(f, 255), flash, 0)
        f = over(f, card, u)
        if HOLD_F - j <= 12:
            f = (f * ((HOLD_F - j) / 12)).astype(np.uint8)
        frames.append(f)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_v = out.with_suffix(".video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", str(tmp_v)],
                           stdin=subprocess.PIPE)
    for f in frames:
        enc.stdin.write(f.tobytes())
    enc.stdin.close()
    enc.wait()

    # ---- sound: the song far away under the old footage (and the kids' own game sound), silence, then the drop on "now" ----
    total = len(frames) / FPS
    N = int(round(total * SR))
    tt = np.arange(N) / SR
    tc, tf = t_cut / FPS, t_freeze / FPS
    mix = np.zeros((N, 2), np.float32)

    def place(x, at, gain, rate=1.0, fade=0.15):
        if rate != 1.0:                                          # slowed footage: slow its sound the same way
            n2 = int(len(x) / rate)
            x = np.stack([np.interp(np.linspace(0, len(x) - 1, n2), np.arange(len(x)), x[:, ch]) for ch in (0, 1)], 1)
        x = x.astype(np.float32).copy()
        nf = int(fade * SR)
        if len(x) > 2 * nf:
            x[:nf] *= np.linspace(0, 1, nf)[:, None]
            x[-nf:] *= np.linspace(1, 0, nf)[:, None]
        s = int(round(at * SR))
        e = min(N, s + len(x))
        mix[s:e] += gain * x[:e - s]

    t = TITLE_F / FPS
    for s in olds:
        place(s["audio"], t, 0.7, THEN_RATE)
        t += THEN_F / FPS
    t = tc
    for s in quick:
        place(s["audio"], t, 0.25, fade=0.03)
        t += NOW_F / FPS
    place(last["audio"], t, 0.25, fade=0.03)
    music = project.music_path(now_root, settings)
    if music:
        heavy = beats.heavy_hits(music, now_root / ".hlv_cache")
        mh = max((h for h in heavy if h[0] > tc + 15), key=lambda h: h[1])[0]
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{mh - tc:.4f}", "-t", f"{total:.4f}", "-i", str(music), "-vn", "-ac", "2",
                              "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
        song = np.zeros((N, 2), np.float32)
        x = np.frombuffer(raw, np.float32).reshape(-1, 2)
        song[:min(N, len(x))] = x[:N]
        from scipy import signal
        far = signal.sosfilt(signal.butter(4, 650, btype="low", fs=SR, output="sos"), song, axis=0).astype(np.float32)
        env_far = np.clip(tt / 2.0, 0, 1) * 0.75 * (tt < tc - ghost.GAP)
        env_now = (tt >= tc).astype(np.float32) * np.clip((total - tt) / 1.5, 0, 1)
        mix += far * env_far[:, None] + song * env_now[:, None]
        print(f"song hit {mh:.1f}s lands on the cut to now ({tc:.2f}s)", flush=True)
    fx = out.with_suffix(".fx.wav")
    ghost.sound_track(fx, total, [{"t": tc, "drop": True, "pops": None}, {"t": tf, "drop": False, "pops": None}])
    with wave.open(str(fx)) as wf:
        mix += np.frombuffer(wf.readframes(wf.getnframes()), np.int16).reshape(-1, 2)[:N] / 32768
    fx.unlink()
    mix[(tt >= tc - ghost.GAP) & (tt < tc)] = 0                  # dead silence before the drop
    mix = np.tanh(mix * 0.75) * 0.8
    tmp_a = out.with_suffix(".wav")
    with wave.open(str(tmp_a), "wb") as wf:
        wf.setnchannels(2), wf.setsampwidth(2), wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_v), "-i", str(tmp_a), "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-movflags", "+faststart", "-shortest", str(out)], check=True)
    tmp_v.unlink(), tmp_a.unlink()
    print(f"wrote {out}  {total:.1f}s: then until {t_tick / FPS:.1f}s, cut to now at {tc:.1f}s, name at {tf:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
