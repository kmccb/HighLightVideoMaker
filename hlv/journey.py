"""The Journey: an opener that starts with him as a little kid and cuts through to the player he is
now, the camera locked on him the whole way. It takes the place of the intro (settings "journey").

"Then" shots come from the player's OTHER projects (same player name) whose footage is from earlier
years: any play there where he's been clicked and that isn't left out, one to three per year. They
get a home-video look, slow motion, the year and his age (settings "journey_born"). A year counter
then spins up to today - the reel's music is muffled until here, cuts out, and drops on the cut
(reel._assemble does that; `plan` puts the cut on one of the song's heavy hits) - and up to three of
this project's plays cut on the beat, the last one freezing under his name.

The tracker's path keeps him at the same spot of the frame in every shot, so the cuts match.
`plan` works everything out without touching video (for the Build preview); `build` renders it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import camera, encoder, ghost, graphics, library, profiles, project
from .pipeline import CACHE_DIRNAME

VERSION = 1
FPS, SR = 30, 48000
TITLE_F = 54                                 # the opening line (longer when the cut needs to reach a heavy hit)
THEN_F, THEN_RATE, XF = 78, 0.7, 8           # frames per old shot, its speed, cross-dissolve frames
TICK_F = 39                                  # the year counter
LAST_F, HOLD_F = 66, 78                      # the last shot's run-up, and its freeze under his name
AT = (0.5, 0.58)                             # where he sits in the frame
TARGET_H = 0.20                              # how tall he is in the frame, when the footage allows
MAX_UP = 1.9                                 # never enlarge the source more than this
LINE = "EVERY PLAYER STARTS SOMEWHERE."
MAX_THEN = 6
_then_cache: dict[str, tuple] = {}


def _year(clip: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(clip["taken"]) if clip.get("taken") else None
    except ValueError:
        return None


def _key_time(root: Path, clip: dict) -> float | None:
    """The moment of a play to show: its hit if one is marked, else the middle one of his clicks."""
    r = clip.get("review") or {}
    if r.get("hit") is not None:
        return float(r["hit"])
    ts = sorted(a["t"] for a in camera.anchors_of(clip))
    return float(ts[len(ts) // 2]) if ts else None


def then_shots(root: Path, pl: dict | None, before_year: int) -> list[dict]:
    """Plays of this player from earlier years, out of his other projects: where he's clicked and not
    left out. One per year with three or more years, two with two, three with one; oldest first."""
    root = Path(root).resolve()
    hit = _then_cache.get(str(root))                             # (other projects can be on the NAS: don't re-read them every time)
    if hit and time.time() - hit[0] < 60 and hit[1] == ((pl or {}).get("id"), before_year):
        return hit[2]
    lib = library.home()
    name = ((pl or {}).get("name") or "").strip().casefold()
    by_year: dict[int, list[dict]] = {}
    for entry in library.projects():
        folder = Path(entry["folder"])
        try:
            if folder.resolve() == root or not (folder / project.FILENAME).exists():
                continue
            proj = project.load(folder)
        except OSError:
            continue                                             # e.g. the NAS is off
        other = profiles.get(lib, project.settings(proj).get("player") or entry.get("player"))
        if not other or not name or (other.get("id") != (pl or {}).get("id") and (other.get("name") or "").strip().casefold() != name):
            continue
        for c in proj["clips"]:
            r = c.get("review") or {}
            d, t = _year(c), _key_time(folder, c)
            if d is None or t is None or d.year >= before_year or r.get("keep") is False or not camera.anchors_of(c):
                continue
            by_year.setdefault(d.year, []).append({"root": str(folder), "clip": c["id"], "t": t, "date": d.isoformat(), "year": d.year,
                                                   "score": (2 if r.get("hit") is not None else 0) + len(camera.anchors_of(c))})
    per = 1 if len(by_year) >= 3 else 2 if len(by_year) == 2 else 3
    out = []
    for y in sorted(by_year):
        out += sorted(sorted(by_year[y], key=lambda s: -s["score"])[:per], key=lambda s: s["date"])
    out = out[-MAX_THEN:]
    _then_cache[str(root)] = (time.time(), ((pl or {}).get("id"), before_year), out)
    return out


def now_shots(root: Path, proj: dict) -> list[dict]:
    """Up to three of this project's plays to stand for "now": in the reel, clicked, with a hit;
    big plays and starred ones first. The best one comes last (it freezes under his name)."""
    from . import hits, reel
    picks = []
    for k, c in enumerate(reel.ordered_clips(proj)):
        r = c.get("review") or {}
        if not reel.included(c) or not camera.anchors_of(c):
            continue
        t = hits.hit_of(root, c)[0]
        if t is None:
            continue
        picks.append((-(2 * bool(r.get("big")) + bool(r.get("star")) + (r.get("hit") is not None)), k,
                      {"root": str(root), "clip": c["id"], "t": float(t), "date": c.get("taken")}))
    picks = [p[2] for p in sorted(picks, key=lambda p: p[:2])[:3]]
    return picks[1:] + picks[:1]


def plan(root: Path, pl: dict | None, settings: dict, period: float | None, music: Path | None, music_start: float) -> dict | None:
    """What the Journey will be with these settings, or None when there's nothing to make it from
    (no earlier footage of him, or no play of this project to end on)."""
    from . import beats
    root = Path(root).resolve()
    proj = project.load(root)
    now = now_shots(root, proj)
    if not now:
        return None
    d_now = _year({"taken": now[-1]["date"]})
    year_now = (d_now or datetime.now()).year
    then = then_shots(root, pl, year_now)
    if not then:
        return None
    now_f = int(np.clip(round(2 * period * FPS), 24, 40)) if period else 30
    title_f = TITLE_F
    cut_nom = (title_f + len(then) * THEN_F + TICK_F) / FPS
    if music and settings.get("music_source") != "compose":      # put the cut to "now" on one of the song's heavy hits
        cands = [(t - music_start, s) for t, s in beats.heavy_hits(music, root / CACHE_DIRNAME)]
        cands = [(t, s) for t, s in cands if 0 <= t - cut_nom <= 2.5]
        if cands:
            t_hit = max(cands, key=lambda c: c[1] - 0.15 * (c[0] - cut_nom))[0]
            title_f += int(round((t_hit - cut_nom) * FPS))
    cut_f = title_f + len(then) * THEN_F + TICK_F
    freeze_f = cut_f + (len(now) - 1) * now_f + LAST_F
    born = settings.get("journey_born")
    years = {}
    for s in then:
        years[s["year"]] = years.get(s["year"], 0) + 1
    return {"then": then, "now": now, "title_f": title_f, "now_f": now_f, "cut": cut_f / FPS, "freeze": freeze_f / FPS,
            "seconds": (freeze_f + HOLD_F) / FPS, "year_now": year_now, "born": int(born) if born else None,
            "line": (settings.get("journey_line") or LINE).strip() or LINE, "years": years}


def available(root: Path, pl: dict | None, settings: dict) -> bool:
    return bool(settings.get("journey")) and plan(root, pl, settings, None, None, 0.0) is not None


# ---------------------------------------------------------------- rendering
def _fit(txt: str, size: int, max_w: float):
    f = graphics.font(size)
    w = f.getlength(txt)
    return graphics.font(max(12, int(size * max_w / w))) if w > max_w else f


def _text(size: tuple[int, int], lines: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lines: (text, font, (x, y), anchor, rgb). Returns (bgr, alpha, shadow) layers at `size`."""
    OW, OH = size
    img, sh = Image.new("RGBA", (OW, OH), (0, 0, 0, 0)), Image.new("L", (OW, OH), 0)
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(sh)
    for txt, f, xy, anchor, rgb in lines:
        ds.text((xy[0] + 3, xy[1] + 4), txt, font=f, fill=200, anchor=anchor)
        d.text(xy, txt, font=f, fill=(*rgb, 255), anchor=anchor)
    a = np.asarray(img, np.float32)
    return a[..., 2::-1].copy(), a[..., 3:] / 255, (cv2.GaussianBlur(np.asarray(sh, np.float32) / 255, (0, 0), 6) * 0.7)[..., None]


def _over(frame: np.ndarray, layer, opacity: float = 1.0) -> np.ndarray:
    bgr, alpha, shadow = layer
    out = frame.astype(np.float32) * (1 - shadow * opacity)
    return (out * (1 - alpha * opacity) + bgr * alpha * opacity).astype(np.uint8)


def _shot(s: dict, before: float, after: float, n_out: int, size: tuple[int, int]) -> dict:
    """n_out frames covering [t - before, t + after] of the play, the camera locked on him, plus its sound."""
    OW, OH = size
    root = Path(s["root"])
    c = next(c for c in project.load(root)["clips"] if c["id"] == s["clip"])
    info, crop = c["probe"], c.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    Wc, Hc = (crop["w"], crop["h"]) if crop else (info["width"], info["height"])
    Wd = int(min(Wc, 2560) // 2 * 2)
    Hd = int(round(Hc * Wd / Wc / 2)) * 2
    t0 = round(max(0.0, s["t"] - before) * FPS) / FPS
    fr = ghost._read(root / c["source"], t0, int(round((before + after) * FPS)) + 2, crop, (Wd, Hd))
    if len(fr) < 4:
        raise RuntimeError(f"couldn't read {c['source']}")
    p = np.array(camera.plan(root / CACHE_DIRNAME, c)["player"])
    k0 = int(round(t0 * FPS))
    rows = p[np.clip(np.arange(k0, k0 + len(fr)), 0, len(p) - 1)].copy()
    ok = (rows[:, 0] >= 0) & ~np.isnan(rows[:, 0])
    sure = np.where(ok, rows[:, 4], 0.0) >= 0.35                 # where the tracker really has him (the ring only shows there)
    for j in range(4):                                           # fill gaps, then smooth: a locked camera mustn't jitter
        rows[:, j] = np.interp(np.arange(len(rows)), np.where(ok)[0], rows[ok, j]) if ok.any() else 0.5
        rows[:, j] = cv2.GaussianBlur(rows[:, j].reshape(-1, 1), (1, 0), 9, borderType=cv2.BORDER_REPLICATE).ravel()
    sx, sy = info["width"] * Wd / Wc, info["height"] * Hd / Hc
    cx, cy = rows[:, 0] * sx - ox * Wd / Wc, rows[:, 1] * sy - oy * Hd / Hc
    box_h = float(np.median(rows[:, 3])) * sy
    A = OW / OH
    fit_w = min(Wd, Hd * A)                                      # the widest window of this shape the picture holds
    cw = float(min(fit_w, max(box_h / TARGET_H * A, OW / MAX_UP)))
    ch = cw / A
    out = np.empty((n_out, OH, OW, 3), np.uint8)
    spot = []
    for j, x in enumerate(np.linspace(0, len(fr) - 1.001, n_out)):
        i = int(x)
        f = fr[i] if x - i < 0.02 else cv2.addWeighted(fr[i], 1 - (x - i), fr[i + 1], x - i, 0)
        px, py = cx[i] + (cx[i + 1] - cx[i]) * (x - i), cy[i] + (cy[i + 1] - cy[i]) * (x - i)
        rx, ry = float(np.clip(px - cw * AT[0], 0, Wd - cw)), float(np.clip(py - ch * AT[1], 0, Hd - ch))
        M = np.array([[OW / cw, 0, -rx * OW / cw], [0, OH / ch, -ry * OH / ch]])
        out[j] = cv2.warpAffine(f, M, (OW, OH), flags=cv2.INTER_AREA if cw > OW * 1.3 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        spot.append(((px - rx) * OW / cw, (py - ry) * OH / ch, box_h * OH / ch, bool(sure[i])))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{before + after:.4f}", "-i", str(root / c["source"]), "-vn",
                          "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return {"frames": out, "spot": spot, "audio": np.frombuffer(raw, np.float32).reshape(-1, 2),
            "date": _year({"taken": s.get("date")})}


def _ring(f: np.ndarray, spot, colour, alpha: float) -> np.ndarray:
    x, y, h, sure = spot
    if alpha <= 0 or not sure:
        return f
    o = f.copy()
    cv2.ellipse(o, (int(x), int(y + h * 0.45)), (int(max(26, h * 0.5)), int(max(9, h * 0.16))), 0, 0, 360, colour, 3, cv2.LINE_AA)
    return cv2.addWeighted(o, alpha, f, 1 - alpha, 0)


def _key(pl: dict | None, pln: dict, size) -> str:
    def stat(s):
        root = Path(s["root"])
        c = next(c for c in project.load(root)["clips"] if c["id"] == s["clip"])
        src = root / c["source"]
        return [s["root"], s["clip"], round(s["t"], 3), src.stat().st_size, src.stat().st_mtime, (c.get("review") or {}).get("spotlight")]
    pl = pl or {}
    payload = {"v": VERSION, "size": list(size), "then": [stat(s) for s in pln["then"]], "now": [stat(s) for s in pln["now"]],
               "f": [pln["title_f"], pln["now_f"]], "born": pln["born"], "line": pln["line"], "year": pln["year_now"],
               "pl": [pl.get(k) for k in ("name", "number", "position", "class_year", "school", "accent")]}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def build(root: Path, pl: dict | None, pln: dict, size: tuple[int, int] | None = None, log=print) -> Path:
    """Render (or reuse) the Journey for a plan from `plan`. The clip carries the plays' own sound
    (the kids' game, slowed with the picture, then today's); the song is the reel's."""
    size = tuple(size) if size else (graphics.W, graphics.H)
    OW, OH = size
    root = Path(root).resolve()
    out = root / CACHE_DIRNAME / "journey" / f"journey-{_key(pl, pln, size)}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    pl = pl or {}
    rng = np.random.default_rng(11)
    k = min(OW, OH) / 1080                                       # text scale
    acc = pl.get("accent") or "#ffffff"
    accent = tuple(int(acc[j:j + 2], 16) for j in (1, 3, 5))
    white = (245, 244, 240)
    born = pln["born"]
    log(f"  the Journey: {len(pln['then'])} old plays, {len(pln['now'])} from now")
    olds = [_shot(s, 1.0, THEN_F * THEN_RATE / FPS - 1.0, THEN_F, size) for s in pln["then"]]
    quick = [_shot(s, (pln["now_f"] - 6) / FPS, 6 / FPS, pln["now_f"], size) for s in pln["now"][:-1]]
    last = _shot(pln["now"][-1], LAST_F / FPS, 0.0, LAST_F, size)
    year_then, year_now = pln["then"][-1]["year"], pln["year_now"]
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    vig = (1 - 0.55 * np.clip(((xx - OW / 2) / (OW * 0.62)) ** 2 + ((yy - OH / 2) / (OH * 0.62)) ** 2, 0, 1) ** 1.3)[..., None]

    def stamp(year: int, d: datetime | None, big: int):
        fy = graphics.font(int(big * k))
        lines = [(str(year), fy, (int(90 * k), OH - int(70 * k)), "ls", white)]
        if born:
            lines.append((f"AGE {year - born}", graphics.font(int(big * k / 3)), (int(100 * k) + int(fy.getlength(str(year))), OH - int(80 * k)), "ls", accent))
        if d:
            lines.append((d.strftime("%b %d %Y").upper().replace(" 0", " "), graphics.font(int(40 * k)), (OW - int(80 * k), int(84 * k)), "rs", white))
        return _text(size, lines)

    def old_look(f: np.ndarray) -> np.ndarray:
        x = f.astype(np.float32)
        x = x.mean(axis=2, keepdims=True) * 0.38 + x * 0.62      # faded colour
        x *= np.array([0.90, 1.0, 1.08], np.float32)             # warm
        x = x * 0.86 + 20                                        # lifted blacks
        x += cv2.GaussianBlur(x, (0, 0), 14) * 0.22              # soft glow
        x *= vig
        x += cv2.resize(rng.normal(0, 5.0, (OH // 2, OW // 2)).astype(np.float32), (OW, OH), interpolation=cv2.INTER_NEAREST)[..., None]
        return np.clip(x, 0, 255).astype(np.uint8)

    def now_look(f: np.ndarray) -> np.ndarray:
        x = f.astype(np.float32)
        return np.clip((x - 128) * 1.08 + 128 + (x - x.mean(axis=2, keepdims=True)) * 0.10, 0, 255).astype(np.uint8)

    tmp_v = out.with_name(out.stem + ".video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS),
                            "-i", "-", *encoder.h264(17), str(tmp_v)], stdin=subprocess.PIPE)
    put = lambda f: enc.stdin.write(f.tobytes())    # noqa: E731
    n_frames = 0
    try:
        # 1) the line
        title = _text(size, [(pln["line"], _fit(pln["line"], int(74 * k), OW * 0.86), (OW // 2, OH // 2), "mm", white)])
        black = np.zeros((OH, OW, 3), np.uint8)
        tf = pln["title_f"]
        for j in range(tf):
            put(_over(black, title, min(1.0, j / 12, (tf - j) / 10)))
        n_frames += tf
        # 2) then
        prev = None
        hold = None
        for s in olds:
            d = s["date"]
            st = stamp(d.year if d else year_then, d, 190)
            for j, f in enumerate(s["frames"]):
                f = old_look(f)
                f = _ring(f, s["spot"][j], white[::-1], 0.8 * min(1.0, j / 10) * min(1.0, (54 - j) / 12) if j < 54 else 0.0)
                f = _over(f, st, min(1.0, j / 8))
                if prev is not None and j < XF:                  # dissolves in over the last shot's final frame
                    f = cv2.addWeighted(prev, 1 - (j + 1) / (XF + 1), f, (j + 1) / (XF + 1), 0)
                elif prev is None and j < 10:
                    f = (f * (j / 10)).astype(np.uint8)
                put(f)
                hold = f
            prev = hold
            n_frames += len(s["frames"])
        # 3) the years spin up to now
        for j in range(TICK_F):
            u = j / (TICK_F - 1)
            yr = int(round(year_then + (year_now - year_then) * u ** 2.2))
            M = cv2.getRotationMatrix2D((OW * AT[0], OH * AT[1]), 0, 1 + 0.5 * u ** 2)       # push in, blur, burn to white
            f = cv2.GaussianBlur(cv2.warpAffine(hold, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), (0, 0), 1 + 22 * u)
            f = cv2.addWeighted(f, 1 - 0.25 * u, np.full_like(f, 255), 0.25 * u, 0)
            lines = [(str(yr), _fit(str(yr), int(420 * k), OW * 0.8), (OW // 2, OH // 2 + int(20 * k)), "mm", white)]
            if born:
                lines.append((f"AGE {yr - born}", graphics.font(int(80 * k)), (OW // 2, OH // 2 + int(250 * k)), "mm", accent))
            put(_over(f, _text(size, lines), min(1.0, j / 5)))
        n_frames += TICK_F
        # 4) now: cuts on the beat, then the last play runs to its hit and holds under his name
        now_stamp = stamp(year_now, None, 110)
        for qi, s in enumerate(quick):
            for j, f in enumerate(s["frames"]):
                f = _ring(now_look(f), s["spot"][j], accent[::-1], 0.9)
                if qi == 0 and j < 5:
                    f = cv2.addWeighted(f, j / 5 * 0.6 + 0.4, np.full_like(f, 255), 0.6 - j / 5 * 0.6, 0)
                put(_over(f, now_stamp))
            n_frames += len(s["frames"])
        for j, f in enumerate(last["frames"]):
            f = _ring(now_look(f), last["spot"][j], accent[::-1], 0.9)
            if not quick and j < 5:
                f = cv2.addWeighted(f, j / 5 * 0.6 + 0.4, np.full_like(f, 255), 0.6 - j / 5 * 0.6, 0)
            put(_over(f, now_stamp))
        n_frames += len(last["frames"])
        stop = now_look(last["frames"][-1])
        name = (pl.get("name") or "").upper()
        detail = "  ·  ".join(x for x in (f"#{pl['number']}" if pl.get("number") else "", pl.get("position") or "",
                                         f"CLASS OF {pl['class_year']}" if pl.get("class_year") else "", pl.get("school") or "") if x)
        card = _text(size, [(name, _fit(name, int(200 * k), OW * 0.86), (OW // 2, OH // 2 - int(10 * k)), "mm", white),
                            (detail, _fit(detail, int(54 * k), OW * 0.86), (OW // 2, OH // 2 + int(130 * k)), "mm", accent)] if name else [])
        for j in range(HOLD_F):
            u = min(1.0, j / 14)
            M = cv2.getRotationMatrix2D((OW * AT[0], OH * AT[1]), 0, 1 + 0.05 * j / HOLD_F)
            f = cv2.warpAffine(stop, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            f = (f.astype(np.float32) * (1 - 0.55 * u)).astype(np.uint8)
            flash = max(0.0, 1 - j / 5) * 0.6
            f = _over(cv2.addWeighted(f, 1 - flash, np.full_like(f, 255), flash, 0), card, u)
            put((f * ((HOLD_F - j) / 12)).astype(np.uint8) if HOLD_F - j <= 12 else f)
        n_frames += HOLD_F
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0 or not tmp_v.exists():
        raise RuntimeError("the Journey didn't render")

    # ---- its own sound: the old plays' (slowed with the picture), then today's, quietly ----
    N = int(round(n_frames / FPS * SR))
    mix = np.zeros((N, 2), np.float32)

    def place(x, at, gain, rate=1.0, fade=0.15):
        if not len(x):
            return
        if rate != 1.0:
            n2 = int(len(x) / rate)
            x = np.stack([np.interp(np.linspace(0, len(x) - 1, n2), np.arange(len(x)), x[:, c_]) for c_ in (0, 1)], 1)
        x = x.astype(np.float32).copy()
        nf = int(fade * SR)
        if len(x) > 2 * nf:
            x[:nf] *= np.linspace(0, 1, nf)[:, None]
            x[-nf:] *= np.linspace(1, 0, nf)[:, None]
        s0 = int(round(at * SR))
        e = min(N, s0 + len(x))
        if e > s0:
            mix[s0:e] += gain * x[:e - s0]

    t = pln["title_f"] / FPS
    for s in olds:
        place(s["audio"], t, 0.7, THEN_RATE)
        t += THEN_F / FPS
    t = pln["cut"]
    for s in quick:
        place(s["audio"], t, 0.25, fade=0.03)
        t += pln["now_f"] / FPS
    place(last["audio"], t, 0.25, fade=0.03)
    tmp_a = out.with_suffix(".wav")
    with wave.open(str(tmp_a), "wb") as wf:
        wf.setnchannels(2), wf.setsampwidth(2), wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    tmp = out.with_name(out.stem + ".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_v), "-i", str(tmp_a), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(tmp)], check=True)
    tmp_v.unlink(), tmp_a.unlink()
    tmp.replace(out)
    return out
