"""ffprobe wrapper."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi"}


@dataclass
class ClipInfo:
    path: str
    duration: float
    fps: float          # nominal (r_frame_rate)
    avg_fps: float
    vfr: bool
    width: int          # display-oriented (rotation applied)
    height: int
    rotation: int
    codec: str
    has_audio: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path")
        return d


def _rate(s: str) -> float:
    try:
        f = Fraction(s)
        return float(f) if f else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path) -> ClipInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])

    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if not rotation and "rotate" in v.get("tags", {}):
        rotation = int(v["tags"]["rotate"])

    w, h = int(v["width"]), int(v["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w

    fps = _rate(v.get("r_frame_rate", "0/1"))
    avg = _rate(v.get("avg_frame_rate", "0/1")) or fps
    duration = float(data["format"].get("duration") or v.get("duration") or 0)
    return ClipInfo(
        path=str(path), duration=duration, fps=round(fps, 3), avg_fps=round(avg, 3),
        vfr=abs(fps - avg) > 0.01 * max(fps, 1), width=w, height=h, rotation=rotation,
        codec=v.get("codec_name", "?"), has_audio=has_audio,
    )


def find_clips(root: Path) -> list[Path]:
    skip = {"normalized", ".hlv_cache", "debug", "previews", "reel", "assets"}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not (skip & set(p.relative_to(root).parts))
    )
