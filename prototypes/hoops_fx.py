"""Prototype: basketball effects on one made shot. The shot arc (a glowing line that follows the
ball from the release to the rim, in slow motion, and stays for a moment after the make) and stat
stamps ("3PT", "AND-1", ...).

hoops_fx.py <video> <out dir> --from T --release T --make T --to T [--word 3PT] [--song FILE --beat T]
  --from / --to   the stretch of the video to show (seconds)
  --release       when the ball leaves his hands;  --make  when it goes through the rim
Writes "<out>/Shot arc.mp4" and "<out>/Stat stamps.mp4" (720p).

The ball is found with the person/ball detector on every frame of the flight (60 fps source), the
camera's small movements are taken out (ghost._homographies), and a parabola is fitted through the
ball's positions, so the line is smooth and can be drawn ahead of / behind the ball exactly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

from hlv import detect, ghost

sys.path.insert(0, str(Path(__file__).parent))
from hit_fx import paste, stamp_layer   # noqa: E402  the stamp from the big-hit prototype

W, H, SRC_FPS, FPS, SR = 1280, 720, 60, 30, 48000
DW, DH = 1920, 1080                 # the ball is looked for at this size
GOLD = (60, 180, 255)               # BGR


def read(src: Path, t0: float, n: int, size) -> np.ndarray:
    w, h = size
    out = np.empty((n, h, w, 3), np.uint8)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", str(src), "-vf", f"fps={SRC_FPS},scale={w}:{h}:flags=area",
                          "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    k = 0
    while k < n:
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        out[k] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        k += 1
    p.stdout.close(), p.wait()
    return out[:k]


def find_ball(src: Path, t0: float, n: int) -> dict[int, tuple[float, float, float]]:
    """frame -> (x, y, width) of the ball in the air, in 720p px: the detection that continues the flight."""
    model = detect._import_ultralytics().YOLO(str(detect.MODEL))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", str(src), "-vf", f"fps={SRC_FPS},scale={DW}:{DH}:flags=area",
                          "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    cands = []
    for k in range(n):
        buf = p.stdout.read(DW * DH * 3)
        if len(buf) < DW * DH * 3:
            break
        f = np.frombuffer(buf, np.uint8).reshape(DH, DW, 3)
        b = model.predict(f, imgsz=DW, classes=[0, 32], conf=0.08, verbose=False)[0].boxes
        ppl = [x.xyxy[0].tolist() for x in b if int(x.cls) == 0 and float(x.conf) > 0.3]
        row = []
        for x in b:
            if int(x.cls) != 32:
                continue
            bb = x.xyxy[0].tolist()
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            near = [q for q in ppl if abs((q[0] + q[2]) / 2 - cx) < 3 * (q[2] - q[0])]
            clear = min(((q[1] - cy) / max(1, q[3] - q[1]) for q in near), default=9.0)
            row.append((cx, cy, bb[2] - bb[0], float(x.conf), clear))
        cands.append(row)
    p.stdout.close(), p.wait()
    # the flight: start from the surest ball that is clear of everyone, then follow it both ways
    best = max(((k, c) for k, row in enumerate(cands) for c in row if c[4] > 0.3), key=lambda kc: kc[1][3], default=None)
    if best is None:
        return {}
    s = W / DW
    out = {best[0]: best[1]}
    for step in (1, -1):
        k, last, miss = best[0], best[1], 0
        while 0 <= k + step < len(cands) and miss < 8:
            k += step
            near = [c for c in cands[k] if np.hypot(c[0] - last[0], c[1] - last[1]) < 70 + 30 * miss]
            if near:
                last = min(near, key=lambda c: np.hypot(c[0] - last[0], c[1] - last[1]))
                out[k], miss = last, 0
            else:
                miss += 1
    return {k: (c[0] * s, c[1] * s, c[2] * s) for k, c in out.items()}


def glow_line(img, pts, col, width, alpha=1.0):
    """A bright line with a soft glow, thinner at its tail."""
    if len(pts) < 2 or alpha <= 0:
        return img
    layer = np.zeros(img.shape[:2], np.float32)
    core = np.zeros(img.shape[:2], np.float32)
    n = len(pts)
    for i in range(n - 1):
        u = (i + 1) / n
        a, b = tuple(np.int32(np.round(pts[i] * 8))), tuple(np.int32(np.round(pts[i + 1] * 8)))
        cv2.line(layer, a, b, 1.0, max(1, int(round(width * (0.35 + 0.65 * u)))), cv2.LINE_AA, shift=3)
        cv2.line(core, a, b, 1.0, max(1, int(round(width * 0.35 * (0.4 + 0.6 * u)))), cv2.LINE_AA, shift=3)
    glow = cv2.GaussianBlur(layer, (0, 0), width * 1.6) * 1.8 + layer
    out = img.astype(np.float32) + (glow[..., None] * np.array(col, np.float32) + core[..., None] * 230) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def bp(x, lo, hi):
    return signal.sosfilt(signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x, axis=0)


def sounds():
    r = np.random.default_rng(5)
    t = lambda s: np.arange(int(s * SR)) / SR           # noqa: E731
    norm = lambda x: x / max(1e-9, np.abs(x).max())     # noqa: E731
    st = t(0.42)
    swish = norm(bp(r.standard_normal(len(st)), 2500, 9000) * np.sin(np.pi * np.clip(st / 0.42, 0, 1)) ** 1.5 * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * st)))
    bt = t(1.4)
    boom = np.tanh(np.sin(2 * np.pi * np.cumsum(40 + 90 * np.exp(-bt / 0.07)) / SR) * np.exp(-bt / 0.4) * 2.2)
    tt = t(0.5)
    thud = np.tanh(np.sin(2 * np.pi * np.cumsum(45 + 60 * np.exp(-tt / 0.03)) / SR) * np.exp(-tt / 0.12) * 2.5)
    ot = t(2.4)
    ohh = np.stack([bp(r.standard_normal(len(ot)), 300, 1100) + 0.5 * bp(r.standard_normal(len(ot)), 1100, 2600) for _ in range(2)], 1)
    ohh = norm(ohh * (np.clip(ot / 0.3, 0, 1) * np.exp(-np.clip(ot - 0.6, 0, None) / 0.7))[:, None])
    rt = t(1.2)
    ru = rt / rt[-1]
    riser = norm(bp(r.standard_normal(len(rt)), 800, 6000) * ru ** 2.2)
    return {"swish": swish, "boom": boom, "thud": thud, "ohh": ohh, "riser": riser}


def write(out: Path, frames, mix):
    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(2), wf.setsampwidth(2), wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-i", str(wav), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for f in frames:
        enc.stdin.write(np.ascontiguousarray(f).tobytes())
    enc.stdin.close(), enc.wait()
    wav.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"), ap.add_argument("out")
    ap.add_argument("--from", dest="t0", type=float, required=True)
    ap.add_argument("--release", type=float, required=True)
    ap.add_argument("--make", type=float, required=True)
    ap.add_argument("--to", dest="t1", type=float, required=True)
    ap.add_argument("--word", default="3PT")
    ap.add_argument("--song", default=None), ap.add_argument("--beat", type=float, default=None)
    a = ap.parse_args()
    src, outdir = Path(a.video), Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    n = int(round((a.t1 - a.t0) * SRC_FPS))
    V = read(src, a.t0, n, (W, H))
    n = len(V)
    iR, iM = int(round((a.release - a.t0) * SRC_FPS)), int(round((a.make - a.t0) * SRC_FPS))
    print(f"{n} frames; release at {iR}, make at {iM}", flush=True)

    # the ball during the flight (a little either side), with the camera's movement taken out
    f0, f1 = max(0, iR - 12), min(n, iM + 18)
    ball = {k + f0: v for k, v in find_ball(src, a.t0 + f0 / SRC_FPS, f1 - f0).items()}
    G = ghost._homographies(V[f0:], iM - f0)                   # frame -> the make frame's picture
    Hm = {f0 + j: g for j, g in enumerate(G)}
    fl = sorted(k for k in ball if iR <= k <= iM)
    if len(fl) < 6:
        sys.exit(f"the ball was only found in {len(fl)} frames of the flight")
    P = np.array([ghost._pt(Hm[k], ball[k][0], ball[k][1]) for k in fl])
    tt = (np.array(fl) - iR) / SRC_FPS
    cx, cy = np.polyfit(tt, P[:, 0], 1), np.polyfit(tt, P[:, 1], 2)        # x straight, y a parabola (gravity)
    res = float(np.sqrt(np.mean((np.polyval(cy, tt) - P[:, 1]) ** 2)))
    bw = float(np.median([ball[k][2] for k in fl]))
    print(f"ball found in {len(fl)} of {iM - iR + 1} flight frames; arc fits to {res:.1f} px; ball {bw:.0f} px wide", flush=True)
    T = (iM - iR) / SRC_FPS
    arc = lambda t: np.array([np.polyval(cx, t), np.polyval(cy, t)])      # noqa: E731  in the make frame's picture
    rim = arc(T)

    def to_frame(k, pts):
        Hi = np.linalg.inv(Hm[min(max(k, f0), n - 1)])
        v = np.c_[pts, np.ones(len(pts))] @ Hi.T
        return v[:, :2] / v[:, 2:3]

    # ---- the timeline: live to the release, the flight at half speed, live again ----
    order = list(range(0, iR, 2)) + list(range(iR, iM)) + list(range(iM, n, 2))
    k_make = len(range(0, iR, 2)) + (iM - iR)                 # output frame of the make
    HOLD, FADE = 0.9, 0.5
    lay = stamp_layer(a.word, 130)
    frames = []
    for j, k in enumerate(order):
        f = V[k].copy()
        since = (j - k_make) / FPS
        if iR <= k and since < HOLD + FADE:
            t_now = min(T, (k - iR) / SRC_FPS)
            ts = np.linspace(0, t_now, max(2, int(60 * t_now / max(T, 1e-6)) + 2))
            pts = to_frame(k, np.array([arc(t) for t in ts]))
            alpha = 1.0 if since < HOLD else max(0.0, 1 - (since - HOLD) / FADE)
            f = glow_line(f, pts, GOLD, max(3.0, bw * 0.45), alpha)
            if k < iM:                                         # a hot spot on the ball itself
                c = tuple(np.int32(pts[-1]))
                spot = np.zeros((H, W), np.float32)
                cv2.circle(spot, c, int(bw * 0.9), 1.0, -1, cv2.LINE_AA)
                spot = cv2.GaussianBlur(spot, (0, 0), bw * 0.7)
                f = np.clip(f + spot[..., None] * np.array([120, 200, 255], np.float32), 0, 255).astype(np.uint8)
        if 0 <= since < 0.6:                                   # the make: rings off the rim, a flash, a small punch
            c = to_frame(k, rim[None])[0]
            for q, r0 in enumerate((0.0, 0.12)):
                u = (since - r0) / 0.45
                if 0 < u < 1:
                    cv2.circle(f, tuple(np.int32(c)), int(bw * (1.2 + 5 * u)), (255, 255, 255), max(1, int(4 * (1 - u))), cv2.LINE_AA)
            fz = 1 + 0.035 * np.exp(-since / 0.12)
            M = np.float32([[fz, 0, c[0] * (1 - fz)], [0, fz, c[1] * (1 - fz)]])
            f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            fl_a = 0.45 * max(0.0, 1 - since / 0.12)
            f = cv2.addWeighted(f, 1 - fl_a, np.full_like(f, 255), fl_a, 0)
        s = since - 0.25                                       # the stamp
        if 0 <= s < 1.6:
            if s < 0.15:
                f = paste(f, lay, W / 2, H - 130, 2.6 - 1.6 * (s / 0.15), 0.4 + 0.6 * s / 0.15)
            elif s < 1.4:
                f = paste(f, lay, W / 2, H - 130)
            else:
                u = (s - 1.4) / 0.2
                f = paste(f, lay, W / 2 + 480 * u * u, H - 130, 1.0, 1 - u)
        frames.append(f)
    nf = len(frames)
    for q in range(8):
        frames[q] = (frames[q] * (q / 8)).astype(np.uint8)
        frames[nf - 1 - q] = (frames[nf - 1 - q] * (q / 8)).astype(np.uint8)

    # ---- sound ----
    N = int(nf / FPS * SR)
    mix = np.zeros((N, 2), np.float32)
    tm = k_make / FPS
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a.t0:.4f}", "-t", f"{a.t1 - a.t0:.4f}", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True).stdout
    game = np.frombuffer(raw, np.float32).reshape(-1, 2)
    spf = SR // FPS
    for j, k in enumerate(order):                               # the gym's own sound wherever it plays at normal speed
        if not (iR <= k < iM):
            s0 = int(k / SRC_FPS * SR)
            chunk = game[s0:s0 + spf]
            mix[j * spf:j * spf + len(chunk)] += chunk[:max(0, N - j * spf)] * 0.5
    S = sounds()

    def place(x, at, gain):
        s0 = int(round(at * SR))
        x = np.stack([x, x], 1) if x.ndim == 1 else x
        a0, b0 = max(0, s0), min(N, s0 + len(x))
        if b0 > a0:
            mix[a0:b0] += (gain * x[a0 - s0:b0 - s0]).astype(np.float32)
    place(S["riser"], tm - 1.2, 0.3), place(S["swish"], tm - 0.05, 0.6), place(S["boom"], tm, 0.6)
    place(S["thud"], tm + 0.4, 0.7), place(S["ohh"], tm + 0.15, 0.3)
    if a.song:
        beat = a.beat if a.beat is not None else 60.0
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, beat - tm):.4f}", "-t", f"{N / SR:.4f}", "-i", a.song, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                             capture_output=True).stdout
        song = np.frombuffer(raw, np.float32).reshape(-1, 2)
        env = np.clip(np.arange(N) / SR / 0.5, 0, 1) * np.clip((N - np.arange(N)) / SR / 1.0, 0, 1) * 0.7
        mix[:len(song)] += song[:N] * env[:len(song), None]
    mix = np.tanh(mix * 1.1) / 1.1
    write(outdir / "Shot arc.mp4", frames, mix)
    print("Shot arc.mp4", f"{nf / FPS:.1f}s, make at {tm:.2f}s", flush=True)

    # ---- the stat stamps, one after another over the made shot ----
    words = ["3PT", "AND-1", "STEAL", "BLOCK", "ASSIST", "BUCKET"]
    base = cv2.convertScaleAbs(V[min(n - 1, iM + 20)], alpha=0.55)
    per = int(1.1 * FPS)
    sf = []
    for wi, word in enumerate(words):
        lay2 = stamp_layer(word, 130)
        for q in range(per):
            f = base.copy()
            s = q / FPS
            if s < 0.15:
                f = paste(f, lay2, W / 2, H / 2, 2.6 - 1.6 * (s / 0.15), 0.4 + 0.6 * s / 0.15)
            elif s < 0.95:
                d = 10 * np.exp(-(s - 0.15) / 0.08)
                f = paste(f, lay2, W / 2 + d * np.sin(s * 90), H / 2 + d * np.cos(s * 70))
            else:
                u = (s - 0.95) / 0.15
                f = paste(f, lay2, W / 2 + 480 * u * u, H / 2, 1.0, 1 - u)
            sf.append(f)
    N2 = int(len(sf) / FPS * SR)
    mix = np.zeros((N2, 2), np.float32)
    N = N2
    for wi in range(len(words)):
        place(S["thud"], wi * per / FPS + 0.14, 0.8)
    write(outdir / "Stat stamps.mp4", sf, mix)
    print("Stat stamps.mp4", f"{len(sf) / FPS:.1f}s", flush=True)


if __name__ == "__main__":
    main()
