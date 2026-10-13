"""A watermarked preview of a finished reel.

Tom sells reels on a "pay when you're happy" promise: the customer sees a marked, smaller copy
first, and gets the clean file once they've paid. This makes that copy next to the reel
(`caleb_reel.mp4` -> `caleb_reel_PREVIEW.mp4`). The reel itself is never touched."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

SUFFIX = "_PREVIEW"
SITE = "reels.scottforge.ai"          # where customers order; shown under PREVIEW
LONG_SIDE = 1280                      # previews are 720p at most: enough to judge, not worth keeping
FONT = "C\\:/Windows/Fonts/arialbd.ttf"


def preview_path(reel: Path) -> Path:
    return reel.with_name(reel.stem + SUFFIX + ".mp4")


def is_preview(f: Path) -> bool:
    return f.stem.endswith(SUFFIX)


def site_text(url: str) -> str:
    """'https://ScottForge.ai/' -> 'ScottForge.ai': how the site reads on the mark."""
    return re.sub(r"^https?://", "", (url or "").strip()).rstrip("/")


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def filter_chain(w: int, h: int, site: str) -> str:
    """The ffmpeg video filter: shrink to 720p (never enlarge), then PREVIEW across the middle,
    the site under it, and a small 'not for sharing' line in two corners."""
    k = min(1.0, LONG_SIDE / max(w, h))
    ow, oh = _even(w * k), _even(h * k)
    short = min(ow, oh)
    site = re.sub(r"[^A-Za-z0-9 ._/-]", "", site or "").strip()   # nothing that could end the quoted text
    big, mid, small = round(short * 0.21), round(short * 0.064), round(short * 0.036)

    def text(t: str, size: int, alpha: float, x: str, y: str) -> str:
        return (f"drawtext=fontfile='{FONT}':text='{t}':fontsize={size}:fontcolor=white@{alpha}"
                f":shadowcolor=black@{alpha * 0.6:.2f}:shadowx=2:shadowy=2:x={x}:y={y}")

    parts = [f"scale={ow}:{oh}", text("PREVIEW", big, 0.3, "(w-tw)/2", f"(h-th)/2-{round(big * 0.3)}")]
    if site:
        parts.append(text(site, mid, 0.42, "(w-tw)/2", f"(h/2)+{round(big * 0.5)}"))
    pad = round(short * 0.03)
    parts.append(text("PREVIEW - not for sharing", small, 0.5, str(pad), str(pad)))
    parts.append(text("PREVIEW - not for sharing", small, 0.5, f"w-tw-{pad}", f"h-th-{pad}"))
    return ",".join(parts)


def _size(src: Path) -> tuple[int, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "json", str(src)], capture_output=True, text=True)
    try:
        s = json.loads(r.stdout)["streams"][0]
        return int(s["width"]), int(s["height"])
    except (ValueError, KeyError, IndexError):
        raise RuntimeError(f"{src.name} could not be read as a video: {r.stderr.strip()[-200:] or 'no picture in it'}") from None


def make(src: Path, site: str = SITE) -> Path:
    """Write the marked copy next to `src` and return its path. Raises RuntimeError with ffmpeg's
    own words if it can't; a half-written file is never left behind."""
    w, h = _size(src)
    out = preview_path(src)
    tmp = out.with_name(out.stem + ".part.mp4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", filter_chain(w, h, site),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "27", "-maxrate", "1400k", "-bufsize", "2800k",
                        "-pix_fmt", "yuv420p",   # small enough to send: about 10 MB a minute
                        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"the preview of {src.name} could not be made: {r.stderr.strip()[-300:]}")
    tmp.replace(out)
    return out
