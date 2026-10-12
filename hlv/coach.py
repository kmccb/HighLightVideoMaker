"""A copy of the recruiting cut for each coach: the same video, opening on a card addressed to
that coach and school ("COACH SMITH / YOUNGSTOWN STATE / Caleb Scott #12 ...").

The schools are a list in the project's settings ("coaches": [{"school", "coach"}], typed into
the Build screen one per line). `versions` makes the card as a short clip encoded exactly like
the reel and joins it to the finished recruiting cut without re-encoding, so twenty versions take
seconds. If a join doesn't decode cleanly it is redone the slow way (re-encoded).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from PIL import ImageDraw

from . import encoder, graphics, profiles

CARD_S = 3.5
MAX = 30
FPS = 30


def clean(raw) -> list[dict]:
    """The list as stored, from whatever the page sent: text with one "School, Coach" per line, or a list."""
    rows = raw.splitlines() if isinstance(raw, str) else (raw or [])
    out, seen = [], set()
    for r in rows:
        if isinstance(r, dict):
            school, coach = str(r.get("school") or ""), str(r.get("coach") or "")
        else:
            school, _, coach = str(r).partition(",")
        school, coach = " ".join(school.split())[:60], " ".join(coach.split())[:60]
        if school and slug(school) not in seen:
            seen.add(slug(school))
            out.append({"school": school, "coach": coach})
    return out[:MAX]


def slug(school: str) -> str:
    return profiles.slug(school)[:40] or "school"


def name_for(base: Path, school: str) -> str:
    """caleb_recruiting.mp4 -> caleb_recruiting_for_youngstown-state.mp4 (the _vertical ending kept last)."""
    stem = base.stem
    tail = "_vertical" if stem.endswith("_vertical") else ""
    return f"{stem[:len(stem) - len(tail)]}_for_{slug(school)}{tail}.mp4"


def school_of(file: str, coaches: list[dict]) -> dict | None:
    m = re.search(r"_for_([a-z0-9-]+?)(_vertical)?\.mp4$", file)
    return next((c for c in coaches if m and slug(c["school"]) == m.group(1)), None)


def card(lib: Path, pl: dict, size: tuple[int, int], school: str, coach: str):
    """The opening card for one coach, in the player's colours."""
    W, H = size
    img = graphics._background(pl, size)
    dr = ImageDraw.Draw(img)
    accent = graphics._rgb(pl.get("accent"), (255, 210, 63))
    tall = H > W
    k = W / 1080 if tall else H / 1080
    who = coach.upper() if coach else ""
    if who and not who.startswith("COACH"):
        who = "COACH " + who
    name = (pl.get("name") or "").upper() + (f"  #{pl['number']}" if pl.get("number") else "")
    spec = [("PREPARED FOR", 44, "SemiBold Condensed", (190, 198, 210), 26)]
    if who:
        spec.append((who, 150, "Bold Condensed", (255, 255, 255), 18))
    spec.append((school.upper(), 150 if not who else 84, "Bold Condensed", accent if who else (255, 255, 255), 70))
    spec.append((name, 78, "Bold Condensed", (255, 255, 255), 14))
    det = graphics._detail_line(pl, custom=False)
    if det:
        spec.append((det, 48, "SemiBold Condensed", (225, 230, 238), 12))
    if pl.get("school"):
        spec.append((pl["school"].upper(), 48, "SemiBold Condensed", accent, 0))
    max_w = W * (0.88 if tall else 0.8)
    rows = []
    for t, size_, style, c, g in spec:
        f = graphics.font(int(size_ * k), style)
        while dr.textlength(t, font=f) > max_w and f.size > 20:
            f = graphics.font(int(f.size * 0.92), style)
        rows.append((t, f, c, int(g * k)))
    heights = [f.getbbox(t)[3] for t, f, _, _ in rows]
    y = (H - sum(heights) - sum(g for *_, g in rows)) // 2
    for (t, f, c, g), h in zip(rows, heights):
        dr.text(((W - dr.textlength(t, font=f)) / 2, y), t, font=f, fill=c)
        y += h + g
    return graphics._bgr(img)


def _decodes(f: Path, want_s: float) -> bool:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-f", "null", "-"], capture_output=True, text=True)
    d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout
    try:
        return r.returncode == 0 and not r.stderr.strip() and abs(float(d) - want_s) < 0.5
    except ValueError:
        return False


def versions(lib: Path, pl: dict, base: Path, coaches: list[dict], size: tuple[int, int], log=print) -> list[dict]:
    """One copy of the finished recruiting cut `base` per coach, next to it. Returns [{"path", "school", "coach"}]."""
    import cv2
    base = Path(base)
    work = base.parent / f"{base.stem}_coach_cards"
    work.mkdir(parents=True, exist_ok=True)
    base_s = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(base)],
                                  capture_output=True, text=True).stdout)
    out = []
    for c in coaches:
        dst = base.with_name(name_for(base, c["school"]))
        png, clip, lst = work / f"{slug(c['school'])}.png", work / f"{slug(c['school'])}.mp4", work / f"{slug(c['school'])}.txt"
        cv2.imwrite(str(png), card(lib, pl, size, c["school"], c["coach"]))
        # encoded like the reel itself (reel._assemble), so the two can be joined as they are
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(png),
                        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{CARD_S:.3f}",
                        "-vf", f"format=yuv420p,fade=t=in:st=0:d=0.5,fade=t=out:st={CARD_S - 0.4:.2f}:d=0.4",
                        *encoder.h264(18, "medium"), "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-shortest", str(clip)], check=True)
        lst.write_text(f"file '{clip.as_posix()}'\nfile '{base.as_posix()}'\n", encoding="utf-8")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(dst)], capture_output=True)
        if not dst.exists() or not _decodes(dst, base_s + CARD_S):
            log(f"  {c['school']}: joining as-is didn't hold, re-encoding")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-i", str(base), "-filter_complex",
                            "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]",
                            *encoder.h264(18, "medium"), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)], check=True)
        out.append({"path": str(dst), "school": c["school"], "coach": c["coach"]})
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    log(f"  recruiting cut: {len(out)} coach version{'s' if len(out) != 1 else ''}")
    return out
