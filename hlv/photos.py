"""Photos in a project folder: finding them, when they (and the plays) were taken, and
keeping highlights.json["photos"] in step with the folder.

A photo entry: {"id": "photo-…", "type": "photo", "source": relative path, "group": sub-folder,
"taken": ISO local time, "keep": include in the reel}.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

from .signals import clip_id

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
SKIP = {".hlv_cache", "reel", "music", "previews", "debug", "normalized", "assets"}
_SCREEN_REC = re.compile(r"ScreenRecording_(\d\d)-(\d\d)-(\d{4}) (\d\d)-(\d\d)-(\d\d)")
_heif = False


def is_photo(item: dict) -> bool:
    return item.get("type") == "photo"


def find(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in PHOTO_EXTS and not (SKIP & set(p.relative_to(root).parts)))


def _register_heif() -> None:
    global _heif
    if not _heif:
        import pillow_heif
        pillow_heif.register_heif_opener()
        _heif = True


def open_image(p: Path) -> Image.Image:
    """The photo upright (EXIF rotation applied), RGB. HEIC works too."""
    _register_heif()
    with Image.open(p) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0, tzinfo=None).isoformat()


def _mtime(p: Path) -> str:
    return _iso(datetime.fromtimestamp(p.stat().st_mtime))


def photo_taken(p: Path) -> str:
    """When the photo was taken (EXIF), else the file's modified time."""
    _register_heif()
    try:
        with Image.open(p) as im:
            exif = im.getexif()
            raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        if raw:
            return _iso(datetime.strptime(str(raw).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S"))
    except Exception:  # noqa: BLE001 - no/odd EXIF: fall back to the file date
        pass
    return _mtime(p)


def clip_taken(root: Path, clip: dict) -> str:
    """When a play was recorded: the stored value, a screen recording's file-name date (its file
    dates are when it was copied), the video's creation tag, else the file's modified time."""
    if clip.get("taken"):
        return clip["taken"]
    p = Path(root) / clip["source"]
    m = _SCREEN_REC.search(p.name)
    if m:
        mo, d, y, hh, mi, ss = map(int, m.groups())
        return _iso(datetime(y, mo, d, hh, mi, ss))
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(p)],
                             capture_output=True, text=True, timeout=20).stdout
        tags = {k.lower(): v for k, v in (json.loads(out or "{}").get("format", {}).get("tags") or {}).items()}
        for k in ("com.apple.quicktime.creationdate", "creation_time"):
            if tags.get(k):
                dt = datetime.fromisoformat(tags[k].replace("Z", "+00:00"))
                return _iso(dt.astimezone() if dt.tzinfo else dt)
    except Exception:  # noqa: BLE001 - unreadable: use the file date
        pass
    return _mtime(p)


def sync(root: Path, proj: dict) -> bool:
    """Bring proj["photos"] in line with the folder (new photos in, deleted ones out, your
    tick boxes kept) and give every play a date. True if anything changed."""
    root = Path(root)
    old = {p["id"]: p for p in proj.get("photos") or []}
    new = []
    for path in find(root):
        pid = "photo-" + clip_id(root, path)
        if pid in old:
            new.append(old[pid])
            continue
        try:
            open_image(path)
        except Exception as e:  # noqa: BLE001 - can't be decoded: leave it out
            print(f"  photo skipped ({path.name}): {e}", flush=True)
            continue
        rel = path.relative_to(root)
        new.append({"id": pid, "type": "photo", "source": rel.as_posix(),
                    "group": rel.parts[0] if len(rel.parts) > 1 else "", "taken": photo_taken(path), "keep": True})
    changed = new != (proj.get("photos") or [])
    if changed or "photos" not in proj:
        proj["photos"] = new
    for c in proj.get("clips") or []:
        if not c.get("taken"):
            c["taken"] = clip_taken(root, c)
            changed = True
    return changed
