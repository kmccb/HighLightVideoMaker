"""A project's song: copy one in, or download a YouTube video's audio as MP3."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from . import profiles

MUSIC_DIR = "music"
_YT = re.compile(r"^https?://(www\.|m\.|music\.)?(youtube\.com/(watch\?.*v=|shorts/|live/)|youtu\.be/)[\w-]{6,}", re.I)


def is_youtube(url: str) -> bool:
    return bool(_YT.match((url or "").strip()))


def import_file(root: Path, src: Path) -> str:
    """Copy a song into <project>/music/ (once) and return its project-relative path."""
    src = Path(src)
    if src.suffix.lower() not in profiles.AUDIO_EXTS:
        raise ValueError(f"music must be one of: {', '.join(sorted(profiles.AUDIO_EXTS))}")
    if not src.is_file():
        raise ValueError(f"file not found: {src}")
    d = root / MUSIC_DIR
    d.mkdir(exist_ok=True)
    try:
        if src.resolve().parent == d.resolve():
            return f"{MUSIC_DIR}/{src.name}"          # already there
    except OSError:
        pass
    stem = profiles.slug(src.stem)[:60]
    dst, n = d / f"{stem}{src.suffix.lower()}", 2
    while dst.exists() and dst.stat().st_size != src.stat().st_size:
        dst, n = d / f"{stem}-{n}{src.suffix.lower()}", n + 1
    if not dst.exists():
        shutil.copy2(src, dst)
    return f"{MUSIC_DIR}/{dst.name}"


def friendly_error(msg: str) -> str:
    """yt-dlp's errors, in words Tom can act on."""
    m = (msg or "").lower()
    if "private" in m:
        return "That video is private, so it can't be downloaded."
    if "sign in to confirm your age" in m or "age-restricted" in m or "inappropriate for some users" in m:
        return "That video is age-restricted, so it can't be downloaded. Try another upload of the song."
    if "video unavailable" in m or "not available" in m or "has been removed" in m:
        return "That video isn't available (removed, or blocked in your country)."
    if "getaddrinfo" in m or "unable to download webpage" in m or "connection" in m or "timed out" in m:
        return "Couldn't reach YouTube. Check the internet connection and try again."
    if "ffmpeg" in m or "ffprobe" in m:
        return "Downloaded, but converting to MP3 failed (ffmpeg problem)."
    return "The download didn't work. Try another link, or choose a song file instead."


def download_youtube(root: Path, url: str, progress=lambda frac, text: None, log=print) -> str:
    """Save the video's audio as <project>/music/<title>.mp3; returns the project-relative path."""
    if not is_youtube(url):
        raise ValueError("That doesn't look like a YouTube link (youtube.com/watch?v=… or youtu.be/…).")
    import yt_dlp
    d = root / MUSIC_DIR
    d.mkdir(exist_ok=True)

    def hook(h):
        if h.get("status") == "downloading":
            tot = h.get("total_bytes") or h.get("total_bytes_estimate") or 0
            progress(min(0.95, h.get("downloaded_bytes", 0) / tot) if tot else 0.0, "downloading")
        elif h.get("status") == "finished":
            progress(0.96, "converting to MP3")

    opts = {
        "format": "bestaudio/best", "noplaylist": True, "quiet": True, "no_warnings": True,
        "restrictfilenames": True, "outtmpl": str(d / "%(title).80s.%(ext)s"),
        "progress_hooks": [hook],
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url.strip(), download=True)
            out = Path(ydl.prepare_filename(info)).with_suffix(".mp3")
    except Exception as e:  # noqa: BLE001 - yt-dlp raises many kinds; show one plain line
        log(f"youtube download failed: {e}")
        raise ValueError(friendly_error(str(e))) from None
    if not out.is_file():
        raise ValueError(friendly_error("ffmpeg"))
    progress(1.0, "done")
    return f"{MUSIC_DIR}/{out.name}"
