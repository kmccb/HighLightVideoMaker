"""Big-play effects for a HIT (a collision, not a run): shockwave, impact frames, triple hit on the
beat, instant replay.

Like colour.py these are made from the play's own rendered segment (so the framing, ring, slow
motion and hit punch are exactly the reel's, wide or tall): `render` reads the frames round the
hit, and writes a short clip that reel.build plays in place of the first `rep` frames after the
hit. Triple hit and instant replay also hold the reel up (`extra`), by a whole number of beats so
the plays after them stay on the beat. `spec` says how long each is and which sounds go with it
without rendering anything (the Build preview uses it).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import camera, encoder, slowmo
from .pipeline import CACHE_DIRNAME

VERSION = 3
FPS = 30
KINDS = ("shock", "frames", "triple", "replay")
NAMES = {"shock": "shockwave", "frames": "impact frames", "triple": "triple hit", "replay": "instant replay"}
REP = {"shock": 22, "frames": 19, "triple": 22, "replay": 22}     # frames of the play, from the hit, the clip stands in for
LEAD = {"triple": 5, "replay": 18}                                # of those, the ones that play as they are before the extra part
TAIL = 1.25               # seconds of play kept after the hit (reel.plan_timing), so the clip always fits
HOLD = 7                  # impact frames: how many
REWIND = 12               # instant replay: frames of tape running back
SPAN = (0.6, 0.4)         # ... and what it shows again: this long before the hit to this long after (real seconds)
REPLAY_S = 3.4            # ... in about this long

_SHAKE = np.random.default_rng(3).uniform(-1, 1, (64, 2))


def _step(period: float | None) -> int:
    """Frames per beat."""
    return int(np.clip(round((period or 0.5) * FPS), 10, 24))


def spec(kind: str, period: float | None) -> dict:
    """{"rep": frames of the play replaced, "extra": frames the reel is held up, "at": frames after the
    hit where the extra part starts, "sounds": [(name, seconds after the hit, level), ...]}."""
    rep, at, extra = REP[kind], LEAD.get(kind, 0), 0
    if kind == "shock":
        snd = [("crack", 0.0, 0.3), ("air", 0.01, 0.25), ("ohh", 0.18, 0.2)]
    elif kind == "frames":
        snd = [("crack", 2 / FPS, 0.3), ("crack", 4 / FPS, 0.3), ("boom", HOLD / FPS, 0.35), ("ohh", 0.3, 0.18)]
    elif kind == "triple":
        st = _step(period)
        extra = 2 * st
        snd = [("boom", st / FPS, 0.55), ("crack", st / FPS, 0.4), ("boom", 2 * st / FPS, 0.85), ("crack", 2 * st / FPS, 0.6),
               ("air", 2 * st / FPS, 0.25), ("ohh", 2 * st / FPS + 0.2, 0.2)]
    else:
        p = period if period and period > 0 else 0.5
        extra = max(REWIND + 45, int(round(max(4, round(REPLAY_S / p)) * p * FPS)))
        again = (at + REWIND + (extra - REWIND) * SPAN[0] / sum(SPAN)) / FPS       # the hit, the second time
        snd = [("whoosh", at / FPS - 0.05, 0.45), ("boom", again, 0.75), ("air", again, 0.3), ("ohh", again + 0.3, 0.22)]
    return {"rep": rep, "extra": extra, "at": at, "sounds": [(n, round(float(t), 4), g) for n, t, g in snd]}


def path(root: Path, seg: Path, kind: str, period: float | None, hbk: int) -> Path:
    sp = spec(kind, period)
    return Path(root) / CACHE_DIRNAME / "impact" / f"{Path(seg).stem}-{kind}-{hbk}-{sp['extra']}-v{VERSION}.mp4"


def render(root: Path, clip: dict, seg: Path, kind: str, period: float | None,
           size: tuple[int, int] | None = None, log=print) -> Path | None:
    """Make (or reuse) the effect clip for this play from its rendered segment `seg`, or None when
    the play has no hit or there's too little of it round the hit."""
    from . import ghost, segment
    root, seg = Path(root), Path(seg)
    lay = segment.layout(seg)
    if kind not in KINDS or lay.get("hit") is None or lay.get("hit_b") is None:
        return None
    sp = spec(kind, period)
    rep, extra, lead = sp["rep"], sp["extra"], sp["at"]
    OW, OH = tuple(size) if size else ghost.WIDE
    S = min(OW, OH) / 720.0
    hbk = int(round(lay["hit_b"] * FPS))
    if hbk + rep + 2 > lay["b_frames"]:
        return None
    out = path(root, seg, kind, period, hbk)
    if out.exists():
        return out

    # the real time of every frame after the snap (slow motion stretches the run into the hit), and where he is in each
    info, crop = clip["probe"], clip.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    _, t1 = segment.extents(clip, info["duration"])
    smo = lay.get("slowmo") or {}
    ts = np.array(slowmo.timeline(lay["split"], t1, FPS, smo.get("window"), float(smo.get("speed") or 1.0)))
    if len(ts) < hbk + rep:
        return None
    plan = camera.plan(root / CACHE_DIRNAME, clip)
    player = np.array(plan["player"]) if plan.get("player") else None

    def where(k: float) -> tuple[float, float]:
        """Him in the picture at frame k after the snap (the middle of the picture if he isn't tracked there)."""
        t = float(np.interp(k, np.arange(len(ts)), ts))
        if player is not None:
            p = player[min(len(player) - 1, int(round(t * FPS)))]
            if p[0] >= 0 and not np.isnan(p[0]):
                rx, ry, rw, rh = ghost.reel_view(clip, plan, t, (OW, OH))
                x, y = (p[0] * info["width"] - ox - rx) * OW / rw, (p[1] * info["height"] - oy - ry) * OH / rh
                return float(np.clip(x, OW * 0.1, OW * 0.9)), float(np.clip(y, OH * 0.1, OH * 0.9))
        return OW / 2, OH * 0.55

    st = _step(period)
    x0 = x1 = float(hbk)
    if kind == "replay":
        x0 = float(np.interp(lay["hit"] - SPAN[0], ts, np.arange(len(ts))))
        x1 = min(float(np.interp(lay["hit"] + SPAN[1], ts, np.arange(len(ts)))), hbk + rep - 1.0)
        if x1 - x0 < 8:
            return None
    back = {"triple": st - lead, "replay": hbk - int(np.floor(x0))}.get(kind, 0)
    if back > hbk:
        return None                                        # the hit comes too soon after the snap to show it again
    post0 = lay["a_frames"] + lay["f_frames"]
    f0, n = post0 + hbk - back, back + rep
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(seg), "-vf", f"trim=start_frame={f0}:end_frame={f0 + n},setpts=PTS-STARTPTS",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    if len(raw) < n * OW * OH * 3:
        return None
    fr = np.frombuffer(raw, np.uint8)[:n * OW * OH * 3].reshape(n, OH, OW, 3)
    F = lambda j: fr[int(np.clip(back + j, 0, n - 1))]      # noqa: E731  frame j after the hit (negative = before)
    c = where(hbk)
    YY, XX = np.mgrid[0:OH, 0:OW].astype(np.float32)

    def remap(img, mx, my):
        return cv2.remap(img, np.float32(mx), np.float32(my), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def move(f, ctr, zoom=1.0, dx=0.0, dy=0.0):
        M = np.float32([[zoom, 0, ctr[0] * (1 - zoom) + dx], [0, zoom, ctr[1] * (1 - zoom) + dy]])
        return cv2.warpAffine(f, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def flash(f, a):
        return cv2.addWeighted(f, 1 - a, np.full_like(f, 255), a, 0) if a > 0 else f

    def punch(f, ctr, k, amp=20.0, bump=0.07, fl=0.7):
        """k frames after a hit: shake, a zoom bump and a flash, dying away."""
        if k < 0 or k > 22:
            return f
        d = _SHAKE[k] * amp * S * np.exp(-k / 5.0)
        return flash(move(f, ctr, 1 + bump * np.exp(-k / 4.0), d[0], d[1]), fl * (1.0, 0.45, 0.15, 0)[min(k, 3)])

    def ripple(f, tt):
        d = np.hypot(XX - c[0], YY - c[1]) + 1e-3
        r0, wd, amp = (50 + 1500 * tt) * S, 70.0 * S, 34 * S * np.exp(-tt / 0.28)
        z = (d - r0) / wd
        g = np.exp(-z * z)
        ux, uy = (XX - c[0]) / d, (YY - c[1]) / d
        o = np.empty_like(f)
        for ch, m in enumerate((1.3, 1.0, 0.75)):                    # the colours split a little on the wave
            k = amp * m * g * z * 1.6
            o[..., ch] = remap(f[..., ch], XX - ux * k, YY - uy * k)
        return np.clip(o.astype(np.float32) + (g * amp / S * 2.2)[..., None], 0, 255).astype(np.uint8)

    def speed_lines(seed, count=80):
        r = np.random.default_rng(seed)
        m = np.zeros((OH, OW), np.uint8)
        far = 2.0 * max(OW, OH)
        for a in r.uniform(0, 2 * np.pi, count):
            r_in, hw = r.uniform(170, 420) * S, np.deg2rad(r.uniform(0.35, 1.3))
            cv2.fillPoly(m, [np.int32([(c[0] + r_in * np.cos(a), c[1] + r_in * np.sin(a)),
                                       (c[0] + far * np.cos(a - hw), c[1] + far * np.sin(a - hw)),
                                       (c[0] + far * np.cos(a + hw), c[1] + far * np.sin(a + hw))])], 255, cv2.LINE_AA)
        return m

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flows: dict[int, np.ndarray] = {}

    def between(x: float):
        """The picture at a fractional frame x after the snap, by motion (optical flow at half size)."""
        j = x - hbk
        i, u = int(np.floor(j)), float(j - np.floor(j))
        if u < 0.03:
            return F(i)
        a, b = F(i), F(i + 1)
        if i not in flows:
            sm = lambda im: cv2.cvtColor(cv2.resize(im, (OW // 2, OH // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)   # noqa: E731
            flows[i] = cv2.resize(dis.calc(sm(a), sm(b), None), (OW, OH), interpolation=cv2.INTER_LINEAR) * 2.0
        fl = flows[i]
        return cv2.addWeighted(remap(a, XX - u * fl[..., 0], YY - u * fl[..., 1]), 1 - u,
                               remap(b, XX + (1 - u) * fl[..., 0], YY + (1 - u) * fl[..., 1]), u, 0)

    def frames():
        if kind == "shock":
            for j in range(rep):
                yield punch(ripple(F(j), j / FPS), c, j, amp=26, bump=0.09)
        elif kind == "frames":
            g0 = cv2.cvtColor(F(0), cv2.COLOR_BGR2GRAY).astype(np.float32)
            xr = np.clip(np.abs(g0 - cv2.GaussianBlur(g0, (0, 0), 28 * S)) * 5.5, 0, 255).astype(np.uint8)   # people light up, the field goes dark
            for j in range(HOLD):
                g = np.maximum(xr, speed_lines(10 + j // 2))
                if j in (2, 3):
                    f = cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR)                       # black on white
                elif j >= 4:
                    a = g[..., None] / 255.0
                    f = (a * np.array([255, 190, 120]) + (1 - a) * np.array([40, 12, 4])).astype(np.uint8)
                else:
                    f = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)                             # white on black
                yield move(f, c, 1.04 + 0.03 * j, *(_SHAKE[j] * 10 * S))
            for j in range(HOLD, rep):
                yield punch(F(j), c, j - HOLD, amp=18, fl=0.55)
        elif kind == "triple":
            for j in range(lead):
                yield F(j)
            for j in range(lead - st, lead):                                             # again, closer
                f = punch(move(F(j), c, 1.16), c, j, amp=16, bump=0.05, fl=0.3)
                yield flash(f, 0.4) if j == lead - st else f
            for j in range(lead - st, rep):                                              # and again, closer still; then it eases back out
                z = 1 + 0.36 * (1 - _ease((j - 8) / 14))
                f = punch(move(F(j), c, z), c, j, amp=30, bump=0.09, fl=0.5)
                yield flash(f, 0.4) if j == lead - st else f
        else:
            for j in range(lead):
                yield F(j)
            rng = np.random.default_rng(11)
            for i in range(REWIND):                                                      # the tape runs back
                f = F(int(round(lead + (x0 - hbk - lead) * (i + 1) / REWIND)))
                f = cv2.addWeighted(f, 0.35, cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), 0.65, 0)
                for y in rng.integers(0, OH - int(14 * S), 7):
                    f[y:y + int(12 * S)] = np.roll(f[y:y + int(12 * S)], int(rng.integers(-60, 60) * S), 1)
                cv2.putText(f, "<<", (OW - int(150 * S), int(80 * S)), cv2.FONT_HERSHEY_SIMPLEX, 1.6 * S, (255, 255, 255), max(2, int(4 * S)), cv2.LINE_AA)
                yield f
            nrep = extra - REWIND
            vig = np.clip(1.15 - np.hypot((XX - OW / 2) / (OW * 0.62), (YY - OH / 2) / (OH * 0.62)) ** 2.2 * 0.75, 0.35, 1)[..., None]
            for i in range(nrep):
                x = x0 + (x1 - x0) * i / nrep
                cc, u = where(x), _ease(i / 26)
                f = move(between(x), cc, 1 + 0.42 * u, (OW / 2 - cc[0]) * 0.5 * u, (OH * 0.52 - cc[1]) * 0.5 * u)
                f = punch(f, (OW / 2, OH / 2), int((x - hbk) * nrep / (x1 - x0) / 2) if x >= hbk else -1, amp=9, bump=0.03, fl=0.35)
                f = (f * vig).astype(np.uint8)
                cv2.circle(f, (OW - int(214 * S), int(56 * S)), int(9 * S), (60, 60, 255), -1, cv2.LINE_AA)
                cv2.putText(f, "REPLAY", (OW - int(194 * S), int(68 * S)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * S, (255, 255, 255), max(2, int(3 * S)), cv2.LINE_AA)
                yield flash(f, 0.6 * (1 - i / 3)) if i < 3 else f
            for j in range(lead, rep):                                                   # back to the play
                yield flash(F(j), 0.6 * (1 - (j - lead) / 4)) if j - lead < 4 else F(j)

    # sound: the play's own for the frames it stands in for, nothing under the part that holds the reel up
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    tA, tB, tC = (post0 + hbk) / FPS, (post0 + hbk + lead) / FPS, (post0 + hbk + rep) / FPS
    fmt = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
    if extra:
        af = (f"[1:a]{fmt},asplit[x][y];[x]atrim=start={tA:.5f}:end={tB:.5f},asetpts=PTS-STARTPTS[a0];"
              f"anullsrc=r=48000:cl=stereo,atrim=0:{extra / FPS:.5f}[a1];"
              f"[y]atrim=start={tB:.5f}:end={tC:.5f},asetpts=PTS-STARTPTS[a2];[a0][a1][a2]concat=n=3:v=0:a=1[a]")
    else:
        af = f"[1:a]{fmt},atrim=start={tA:.5f}:end={tC:.5f},asetpts=PTS-STARTPTS[a]"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-i", str(seg), "-filter_complex", af, "-map", "0:v", "-map", "[a]",
                            *encoder.h264(17, software=True), "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)],
                           stdin=subprocess.PIPE)
    wrote = 0
    try:
        for f in frames():
            enc.stdin.write(np.ascontiguousarray(f).tobytes())
            wrote += 1
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0 or not tmp.exists() or wrote != rep + extra:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{NAMES[kind]} failed for {clip['source']} ({wrote} of {rep + extra} frames)")
    tmp.replace(out)
    log(f"  {NAMES[kind]} {clip['source']}: {rep + extra} frames ({extra} extra)")
    return out


def _ease(u: float) -> float:
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)
