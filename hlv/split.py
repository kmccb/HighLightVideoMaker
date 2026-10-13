"""A video that is several plays already cut together (a game's highlights made in another app):
find its hard cuts and split it into one video per play, so each play can be trimmed, marked and
given effects like any other.

`cuts` finds the cuts from ffmpeg's scene score on the review copy (the play detector's own cut finder
misses cuts between two shots from the same seat in the same gym). `split` writes "<name> - play 01.mp4",
... next to the video (re-encoded at high quality: the cuts are not on keyframes, so a plain copy
would start each play on the wrong frame) and moves the original into "split originals", which the
app does not look in.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from . import encoder, proxy
from .pipeline import CACHE_DIRNAME

VERSION = 2
SPIKE, QUIET, ALONE = 0.20, 0.08, 0.5   # a cut's score; what counts as a change at all; how long it must stand alone (s)
MIN_PLAY = 1.2            # seconds: cuts closer together than this are one cut (a flash, a dissolve)
EDGE = 2 / 60             # trimmed off each side of a cut, so no frame of the neighbouring play shows
ASIDE = "split originals"


def cuts(root: Path, clip: dict) -> list[float]:
    """Seconds into the video where one play ends and the next begins ([] = it is one play)."""
    root = Path(root)
    src = root / clip["source"]
    dur = float(clip["probe"]["duration"])
    f = root / CACHE_DIRNAME / "cuts" / f"{clip['id']}.json"
    stamp = [VERSION, src.stat().st_size, src.stat().st_mtime]
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("stamp") == stamp:
            return d["cuts"]
    px = proxy.proxy_path(root / CACHE_DIRNAME, clip["id"])
    scan = px if px.exists() else src
    # ffmpeg's scene score for every frame that changes at all. A hard cut is one frame that stands
    # alone (the frame after it may echo it); a fast pan or a flash raises the score for a run of frames.
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(scan), "-vf", f"select='gt(scene,{QUIET})',metadata=print:file=-", "-an", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    ts = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stdout)]
    sc = [float(m) for m in re.findall(r"lavfi\.scene_score=([0-9.]+)", r.stdout)]
    out: list[float] = []
    for t, v in zip(ts, sc):
        if v < SPIKE or t < MIN_PLAY or t > dur - MIN_PLAY or (out and t - out[-1] < MIN_PLAY):
            continue
        near = [u for u, w in zip(ts, sc) if u != t and abs(u - t) < ALONE and not 0 < u - t < 0.05]
        if not near:
            out.append(round(t, 3))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"stamp": stamp, "cuts": out}))
    return out


def split(root: Path, clip: dict, log=print) -> list[Path]:
    """Write one video per play and move the original aside. Returns the new files ([] = nothing to split)."""
    root = Path(root)
    src = root / clip["source"]
    marks = cuts(root, clip)
    if not marks:
        return []
    dur = float(clip["probe"]["duration"])
    bounds = [0.0, *marks, dur]
    made: list[Path] = []
    try:
        for k in range(len(bounds) - 1):
            a = bounds[k] + (EDGE if k else 0.0)
            b = bounds[k + 1] - (EDGE if k + 1 < len(bounds) - 1 else 0.0)
            out = src.with_name(f"{src.stem} - play {k + 1:02d}.mp4")
            tmp = out.with_name(out.stem + ".tmp.mp4")
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.4f}", "-i", str(src), "-t", f"{b - a:.4f}",
                   "-map", "0:v:0", "-map", "0:a:0?", *encoder.h264(17), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)]
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if r.returncode != 0 or not tmp.exists():                    # the hardware encoder now and then quits: once more in software
                cmd[cmd.index("-map") + 4:cmd.index("-c:a")] = encoder.h264(17, software=True)
                r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if r.returncode != 0 or not tmp.exists():
                raise RuntimeError(f"could not write {out.name}: {r.stderr.strip()[-300:]}")
            tmp.replace(out)
            made.append(out)
            log(f"  split {src.name}: play {k + 1} of {len(bounds) - 1} ({b - a:.1f} s)")
    except Exception:
        for p in made:
            p.unlink(missing_ok=True)
        raise
    aside = src.parent / ASIDE
    aside.mkdir(exist_ok=True)
    shutil.move(str(src), str(aside / src.name))
    return made
