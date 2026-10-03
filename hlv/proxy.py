"""Browser-friendly 720p H.264 proxies for the review page.

Chrome can't play HEVC and scrubbing 4K is sluggish. The fps filter (same as analysis)
keeps proxy time == source time, and a short GOP makes frame-stepping snappy.
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROXY_DIRNAME = "proxy"


def proxy_path(cache_dir: Path, clip_id: str) -> Path:
    return cache_dir / PROXY_DIRNAME / f"{clip_id}.mp4"


def make_proxy(src: Path, dst: Path, height: int = 720, fps: int = 30) -> Path:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-vf", f"fps={fps},scale=-2:'min({height},ih)'",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "15", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart", str(tmp)],
        check=True)
    tmp.replace(dst)
    return dst


def make_all(items: list[tuple[Path, Path]], jobs: int = 4, log=print) -> None:
    todo = [(s, d) for s, d in items if not (d.exists() and d.stat().st_mtime >= s.stat().st_mtime)]
    if not todo:
        return
    log(f"making {len(todo)} review proxies (one-time) ...")
    with ThreadPoolExecutor(jobs) as ex:
        futs = {ex.submit(make_proxy, s, d): s for s, d in todo}
        for f in as_completed(futs):
            try:
                f.result()
                log(f"  proxy ok   {futs[f].name}")
            except subprocess.CalledProcessError as e:
                log(f"  proxy FAIL {futs[f].name}: {e}")
