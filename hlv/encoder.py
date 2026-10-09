"""H.264 encoder settings for everything the app renders.

Uses Intel Quick Sync (h264_qsv) when this PC has it: on Tom's 4-core laptop it encodes 1080p
about 4x faster than libx264 "fast" (10 s of video: 3.5 s vs 13.5 s) at the same file size and
look. Anything else falls back to libx264. Set HLV_ENCODER=x264 to force software encoding.
Decoding stays in software: Quick Sync decodes iPhone HEVC faster, but its frames drifted up to
two frames from ffmpeg's own over a few seconds, which would put the spotlight off the player.
"""
from __future__ import annotations

import functools
import os
import subprocess


@functools.lru_cache(maxsize=None)
def qsv() -> bool:
    """Can this PC encode with Quick Sync? (Tried once per run on a tiny clip.)"""
    if os.environ.get("HLV_ENCODER", "").lower() == "x264":
        return False
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
                            "-c:v", "h264_qsv", "-f", "null", "-"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def h264(crf: int = 17, preset: str = "fast") -> list[str]:
    """ffmpeg output args (codec, quality, pixel format) for x264-style quality crf."""
    if qsv():
        return ["-c:v", "h264_qsv", "-global_quality", str(crf + 1), "-preset", "medium", "-pix_fmt", "nv12"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
