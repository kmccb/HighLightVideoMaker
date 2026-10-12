"""Prototype: "silence, then the drop" + "the whole run in one frame" (ghost trail) on one play.

ghost_drop.py <project> <clip id> <out.mp4> [--freeze T] [--start T] [--debug DIR]
Plays the clip live up to the hit, cuts all sound for a beat, then on the hit: boom, the frame
freezes, the camera pulls back and copies of the player appear along the path he ran.
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

from hlv import beats, camera, hits, profiles, library, project, segment, track

FPS, W, H = 30, 1920, 1080
SR = 48000
GAP = 0.36            # silence before the hit
LEAD = 1.0            # footage before the snap


def read_video(src: Path, t0: float, dur: float) -> np.ndarray:
    n = int(round(dur * FPS))
    out = np.empty((n, H, W, 3), np.uint8)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{dur + 0.5:.4f}", "-i", str(src),
                          "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    k = 0
    while k < n:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        out[k] = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        k += 1
    p.stdout.close()
    p.wait()
    return out[:k]


def read_audio(src: Path, t0: float, dur: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{dur:.4f}", "-i", str(src), "-vn",
                          "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float32)
    out = np.zeros((int(round(dur * SR)), 2), np.float32)
    out[:min(len(a), len(out))] = a[:len(out)]
    return out


def homographies(frames: np.ndarray, ref: int) -> list[np.ndarray]:
    """For every frame, the homography taking it into frame `ref`'s picture (camera shake / pan removed)."""
    small = [cv2.cvtColor(cv2.resize(f, (W // 2, H // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in frames]
    step = [np.eye(3)] * len(frames)            # step[k]: frame k -> frame k+1
    for k in range(len(frames) - 1):
        pts = cv2.goodFeaturesToTrack(small[k], 700, 0.01, 12)
        Hm = None
        if pts is not None and len(pts) >= 12:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(small[k], small[k + 1], pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 12:
                Hm, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 1.5)
        step[k] = np.eye(3) if Hm is None else Hm
    S, Si = np.diag([0.5, 0.5, 1.0]), np.diag([2.0, 2.0, 1.0])   # half-res <-> full-res
    out = [np.eye(3)] * len(frames)
    acc = np.eye(3)
    for k in range(ref - 1, -1, -1):            # earlier frames: chain forward to ref
        acc = acc @ step[k]
        out[k] = Si @ acc @ S
    acc = np.eye(3)
    for k in range(ref + 1, len(frames)):       # later frames: chain backward
        acc = acc @ np.linalg.inv(step[k - 1])
        out[k] = Si @ acc @ S
    return out


def warp_pt(Hm: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = Hm @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project"), ap.add_argument("clip"), ap.add_argument("out")
    ap.add_argument("--freeze", type=float, default=None, help="source time of the freeze (default: the hit)")
    ap.add_argument("--start", type=float, default=None, help="source time the ghost trail starts (default: the snap)")
    ap.add_argument("--music-hit", type=float, default=None, help="song time that lands on the freeze")
    ap.add_argument("--debug", default=None)
    a = ap.parse_args()
    root = Path(a.project).resolve()
    proj = project.load(root)
    settings = project.settings(proj)
    c = next(c for c in proj["clips"] if c["id"] == a.clip)
    cache = root / ".hlv_cache"
    src = root / c["source"]
    t_in, t_out, snap = segment._window(c)
    hit = hits.hit_of(root, c)[0]
    tF = a.freeze if a.freeze is not None else hit
    t0 = round(max(0.0, snap - LEAD) * FPS) / FPS
    t1 = min(c["probe"]["duration"] - 0.1, t_out)
    trail0 = a.start if a.start is not None else snap
    print(f"play {t0:.2f}-{t1:.2f}  snap {snap:.2f}  freeze {tF:.2f}", flush=True)

    frames = read_video(src, t0, t1 - t0)
    n = len(frames)
    iF = min(n - 1, int(round((tF - t0) * FPS)))
    player = np.nan_to_num(track.smooth_track(np.load(camera._track_file(cache, c)), FPS), nan=-1)
    row = lambda i: player[min(len(player) - 1, int(round((t0 + i / FPS) * FPS)))]    # noqa: E731
    print(f"{n} frames, freeze at frame {iF}", flush=True)

    G = homographies(frames, iF)
    drift = max(np.hypot(*np.subtract(warp_pt(G[k], W / 2, H / 2), (W / 2, H / 2))) for k in range(0, iF + 1))
    print(f"camera moved up to {drift:.0f} px before the freeze", flush=True)

    # ---- clean plate: the field with nobody moving on it (median over time, in the freeze frame's picture) ----
    idx = list(range(0, n, 4))
    stack = np.empty((len(idx), H, W, 3), np.uint8)
    for j, k in enumerate(idx):
        stack[j] = cv2.warpPerspective(frames[k], G[k], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    plate = np.empty((H, W, 3), np.uint8)
    for y in range(0, H, 120):                  # in strips, to keep memory down
        plate[y:y + 120] = np.median(stack[:, y:y + 120], axis=0).astype(np.uint8)
    del stack

    # ---- where he was, in the freeze frame's picture ----
    pos = {}                                    # frame -> (cx, cy, w, h) in freeze-frame px
    for i in range(int(round((trail0 - t0) * FPS)), iF + 1):
        cx, cy, bw, bh, conf = row(i)
        if cx < 0 or conf < 0.3:
            continue
        x, y = warp_pt(G[i], cx * W, cy * H)
        pos[i] = (x, y, bw * W, bh * H)
    fx, fy, fw, fh = pos[max(pos)]
    # ghosts: far enough apart not to overlap, the newest not on top of him
    gap_px = 1.55 * fw
    chosen = []
    for i in sorted(pos, reverse=True):         # walk back from the freeze
        x, y = pos[i][:2]
        last = (fx, fy) if not chosen else pos[chosen[-1]][:2]
        if np.hypot(x - last[0], y - last[1]) >= gap_px:
            chosen.append(i)
    chosen = chosen[::-1]
    while len(chosen) > 12:                     # too many: thin them out evenly
        chosen = chosen[::2] if len(chosen) > 24 else [c_ for j, c_ in enumerate(chosen) if j % 3 != 1]
    print(f"{len(chosen)} ghosts", flush=True)

    def matte(i: int, grow: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Frame i in the freeze picture, and a soft mask of the player there (what differs from the empty field)."""
        img = cv2.warpPerspective(frames[i], G[i], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
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
        if keep.sum() < 0.12 * bw * bh:         # the difference found too little: fall back to a soft oval on him
            keep[:] = 0
            cv2.ellipse(keep, (int(x), int(y)), (int(bw * 0.42), int(bh * 0.52)), 0, 0, 360, 1, -1)
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return img, cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.3)

    ghosts = [(i, *matte(i)) for i in chosen]
    _, live_m = matte(iF, 1.15)
    base = frames[iF].copy()

    pl = profiles.get(library.home(), settings.get("player")) or {}
    tc = pl.get("color") or "#1f4fd1"
    team = tuple(int(tc[j:j + 2], 16) for j in (5, 3, 1))           # BGR
    feet = [(pos[i][0], pos[i][1] + pos[i][3] * 0.5) for i in sorted(pos)]
    feet_i = sorted(pos)

    # ---- the camera: from the full frame to the path, 16:9 ----
    xs = [pos[i][0] for i in chosen] + [fx]
    ys = [pos[i][1] for i in chosen] + [fy]
    cw = max((max(xs) - min(xs)) + 5.0 * fw, (max(ys) - min(ys) + 3.2 * fh) * 16 / 9, W * 0.52)
    cw = min(cw, W)
    chh = cw * 9 / 16
    ccx = float(np.clip((max(xs) + min(xs)) / 2, cw / 2, W - cw / 2))
    ccy = float(np.clip((max(ys) + min(ys)) / 2 - 0.1 * fh, chh / 2, H - chh / 2))
    wide = (0.0, 0.0, float(W), float(H))
    tight = (ccx - cw / 2, ccy - chh / 2, cw, chh)
    print(f"freeze view {cw:.0f} px wide ({W / cw:.2f}x)", flush=True)

    def view(img: np.ndarray, u: float) -> np.ndarray:
        x, y, w_, h_ = [wide[j] + (tight[j] - wide[j]) * u for j in range(4)]
        M = np.array([[W / w_, 0, -x * W / w_], [0, H / h_, -y * H / h_]])
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

    # ---- the freeze: timings (seconds from the hit) ----
    T_ZOOM, T_G0, T_HOLD_END, T_OUT = 0.55, 0.25, 3.5, 0.5
    beat = 60.0 / 118.7 / 2                                         # ghosts pop on half-beats
    T_G = [T_G0 + j * min(beat, 1.6 / max(1, len(ghosts))) for j in range(len(ghosts))]
    T_LAST = T_G[-1] if T_G else T_G0
    T_HOLD_END = max(T_HOLD_END, T_LAST + 1.6)
    FREEZE_S = T_HOLD_END + T_OUT

    def freeze_frame(t: float) -> np.ndarray:
        out_u = ease((t - T_HOLD_END) / T_OUT)                      # 1 = back to the live picture
        zoom = ease(t / T_ZOOM) * (1 - out_u)
        dim = 0.5 * ease(t / 0.4) * (1 - out_u)
        img = base.astype(np.float32) * (1 - dim)
        # the path, drawn up to the newest ghost on screen
        shown = [j for j, tg in enumerate(T_G) if t >= tg]
        if shown and out_u < 1:
            upto = chosen[shown[-1]] if len(shown) < len(chosen) or t < T_LAST + 0.25 else feet_i[-1]
            pts = np.array([feet[k] for k, fi in enumerate(feet_i) if fi <= upto], np.float32)
            if len(pts) > 2:
                pts = cv2.GaussianBlur(pts.reshape(-1, 1, 2), (1, 9), 0).reshape(-1, 2)
                line = np.zeros((H, W), np.float32)
                cv2.polylines(line, [np.round(pts * 8).astype(np.int32)], False, 1.0, 2, cv2.LINE_AA, shift=3)
                glow = cv2.GaussianBlur(line, (0, 0), 5) * 2.2
                a_l = (1 - out_u) * 0.9
                for ch in range(3):
                    img[..., ch] += a_l * (glow * team[ch] + line * 150)
        for j in shown:
            i, gimg, gm = ghosts[j]
            age = t - T_G[j]
            alpha = (0.55 + 0.4 * (j + 1) / len(ghosts)) * ease(age / 0.12) * (1 - out_u)
            pop = max(0.0, 1 - age / 0.22)                         # a white flash as it lands
            rim = np.clip(cv2.GaussianBlur(gm, (0, 0), 3.5) * 1.5 - gm, 0, 1) * 0.55 * (1 - out_u)
            for ch in range(3):
                img[..., ch] += rim * team[ch]
            m3 = (gm * alpha)[..., None]
            g = gimg.astype(np.float32) * (1 + 0.9 * pop) + 255 * 0.5 * pop
            img = img * (1 - m3) + g * m3
        # he stays lit
        m3 = (live_m * (dim / max(1e-6, 1 - dim + dim)))[..., None] if dim > 0 else 0
        img = img * (1 - m3) + base.astype(np.float32) * m3
        flash = max(0.0, 1 - t / 0.18) * 0.55                      # the hit itself
        img = img * (1 - flash) + 255 * flash
        return view(np.clip(img, 0, 255).astype(np.uint8), zoom)

    # ---- write the video ----
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_v = out.with_suffix(".video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", str(tmp_v)],
                           stdin=subprocess.PIPE)
    nfz = int(round(FREEZE_S * FPS))
    total_frames = iF + nfz + (n - iF)
    k_out = 0
    for i in range(iF):                                             # live, up to the hit
        f = frames[i]
        if k_out < 8:
            f = (f * (k_out / 8)).astype(np.uint8)
        enc.stdin.write(f.tobytes())
        k_out += 1
    dbg = Path(a.debug) if a.debug else None
    for j in range(nfz):
        f = freeze_frame(j / FPS)
        if dbg and j in (int(0.3 * FPS), int((T_LAST + 0.6) * FPS), int((T_LAST + 1.3) * FPS)):
            cv2.imwrite(str(dbg / f"freeze-{j:03d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
        enc.stdin.write(f.tobytes())
        k_out += 1
    tail = n - iF
    for j in range(tail):                                           # live again
        f = frames[iF + j]
        if tail - j <= 15:
            f = (f * ((tail - j) / 15)).astype(np.uint8)
        enc.stdin.write(f.tobytes())
        k_out += 1
    enc.stdin.close()
    enc.wait()
    if dbg:
        cv2.imwrite(str(dbg / "plate.jpg"), plate, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---- sound ----
    dur = total_frames / FPS
    N = int(round(dur * SR))
    tt = np.arange(N) / SR
    th = iF / FPS                                                   # the hit, in output time
    rng = np.random.default_rng(7)
    mix = np.zeros((N, 2), np.float32)

    def place(x: np.ndarray, at: float, gain: float = 1.0):
        s = int(round(at * SR))
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        a0, b0 = max(0, s), min(N, s + len(x))
        if b0 > a0:
            mix[a0:b0] += gain * x[a0 - s:b0 - s]

    # game sound: live before and after, nothing while frozen
    game = read_audio(src, t0, n / FPS)
    gs = np.zeros((N, 2), np.float32)
    cut = int(round(th * SR))
    gs[:cut] = game[:cut]
    after = int(round((th + FREEZE_S) * SR))
    rest = game[cut:cut + (N - after)]
    gs[after:after + len(rest)] = rest
    genv = np.ones(N, np.float32)
    genv[(tt > th - GAP) & (tt < th + FREEZE_S)] = 0.0
    genv = np.minimum(genv, np.clip((tt - (th + FREEZE_S)) / 0.4, 0, 1) + (tt < th - GAP))
    mix += gs * genv[:, None] * 0.55

    # the song: a heavy hit of it lands on the freeze
    music = project.music_path(root, settings)
    if music and settings.get("music_source") != "compose":
        heavy = beats.heavy_hits(music, cache)
        mh = a.music_hit if a.music_hit is not None else max((h_ for h_ in heavy if h_[0] > th + 20), key=lambda h_: h_[1])[0]
        song = read_audio(music, mh - th, dur)
        menv = np.full(N, 0.8, np.float32)
        menv *= np.clip(tt / 0.5, 0, 1)                             # fade in
        menv[(tt >= th - GAP) & (tt < th)] = 0.0                    # the silence
        edge = (tt >= th - GAP - 0.03) & (tt < th - GAP)
        menv[edge] *= np.clip((th - GAP - tt[edge]) / 0.03, 0, 1)
        menv[tt >= th] = 1.0                                        # ... and back, louder
        menv *= np.clip((dur - tt) / 1.2, 0, 1)                     # fade out
        mix += song * menv[:, None]
        print(f"song hit {mh:.2f}s lands on the freeze", flush=True)

    # riser into the silence: noise sweeping up, cut dead
    rl = 1.3
    rn = int(rl * SR)
    ru = np.arange(rn) / rn
    noise = rng.standard_normal(rn)
    riser = np.zeros(rn)
    for lo, hi, w0, w1 in ((400, 1500, 0.0, 0.5), (1500, 5000, 0.3, 0.85), (5000, 12000, 0.6, 1.0)):
        riser += bandpass(noise, lo, hi) * np.clip((ru - w0) / (w1 - w0), 0, 1) ** 2
    riser *= ru ** 2.5
    riser[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))
    place(riser / np.abs(riser).max(), th - GAP - rl, 0.5)

    # the drop: sub boom + crack + a low thud
    bl = 1.6
    bt = np.arange(int(bl * SR)) / SR
    f_inst = 38 + 95 * np.exp(-bt / 0.07)
    boom = np.sin(2 * np.pi * np.cumsum(f_inst) / SR) * np.exp(-bt / 0.42)
    boom = np.tanh(boom * 2.2)
    place(boom, th, 0.6)
    crack = bandpass(rng.standard_normal(int(0.25 * SR)), 1200, 7000) * np.exp(-np.arange(int(0.25 * SR)) / SR / 0.035)
    place(crack / np.abs(crack).max(), th, 0.4)
    tail_n = int(1.4 * SR)                                          # air after the hit
    air = bandpass(rng.standard_normal(tail_n), 150, 900) * np.exp(-np.arange(tail_n) / SR / 0.5)
    place(air / np.abs(air).max(), th + 0.01, 0.22)

    # crowd swell under the freeze
    cn = int((FREEZE_S + 1.5) * SR)
    cu = np.arange(cn) / SR
    crowd = np.stack([bandpass(rng.standard_normal(cn), 350, 2600) for _ in range(2)], 1)
    wob = 1 + 0.25 * np.sin(2 * np.pi * 0.7 * cu + 1.0) + 0.15 * np.sin(2 * np.pi * 2.3 * cu)
    cenv = np.clip(cu / 0.9, 0, 1) * np.clip((FREEZE_S + 1.5 - cu) / 1.6, 0, 1) * wob
    crowd *= cenv[:, None]
    place((crowd / np.abs(crowd).max()).astype(np.float32), th + 0.05, 0.16)

    # each ghost: a short rising chime
    scale = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26, 28]
    for j, tg in enumerate(T_G):
        f0 = 660 * 2 ** (scale[min(j, len(scale) - 1)] / 12)
        ct = np.arange(int(0.35 * SR)) / SR
        chime = (np.sin(2 * np.pi * f0 * ct) + 0.4 * np.sin(2 * np.pi * 2 * f0 * ct)) * np.exp(-ct / 0.09)
        wh = bandpass(rng.standard_normal(len(ct)), 2500, 9000) * np.exp(-ct / 0.03)
        place(chime, th + tg, 0.07)
        place(wh / np.abs(wh).max(), th + tg, 0.06)

    mix = np.tanh(mix * 0.75) * 0.8
    tmp_a = out.with_suffix(".wav")
    with wave.open(str(tmp_a), "wb") as wf:
        wf.setnchannels(2), wf.setsampwidth(2), wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_v), "-i", str(tmp_a), "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-movflags", "+faststart", "-shortest", str(out)], check=True)
    tmp_v.unlink(), tmp_a.unlink()
    print(f"wrote {out}  {dur:.1f}s  hit at {th:.2f}s, freeze {FREEZE_S:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
