"""Find the same footage (or photo) added to a project more than once.

Each play's review copy is sampled 4 times a second (inside a screen recording's app chrome),
shrunk to 64x64 grey and reduced to a 255-bit perceptual hash (low DCT frequencies above their
median). Two plays are duplicates when, at some time offset, they share a run of at least
MIN_RUN_S seconds of near-identical frames (median distance <= MAX_DIST). On Tom's projects real
copies share 7-16 s at distance 0-0.05, while different plays - even from the same fixed Hudl
camera - never share more than 0.8 s. That catches file copies, a Hudl clip recorded twice, or a
recording that starts a little later. Photos match when their hashes are within PHOTO_DIST.

Fingerprints are saved in .hlv_cache/dupes/ (by review copy / photo file and date), so only new
items are looked at again. Free and offline: ffmpeg + OpenCV.
"""
from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import camera, photos, proxy

FPS = 4
SIZE = 64
MIN_RUN_S = 2.0
MAX_DIST = 0.12
PHOTO_DIST = 0.12
VERSION = 1


def _hash(gray: np.ndarray) -> np.ndarray:
    dct = cv2.dct(gray.astype(np.float32))[:16, :16].flatten()[1:]
    return dct > np.median(dct)


def _cache_file(cache: Path, item_id: str, src: Path) -> Path:
    st = src.stat()
    return Path(cache) / "dupes" / f"{item_id}-v{VERSION}-{st.st_size}-{int(st.st_mtime)}.npy"


def clip_print(cache: Path, clip: dict) -> np.ndarray | None:
    """(n, 255) bool hashes of a play's review copy at FPS, or None if it has none yet."""
    src = proxy.proxy_path(cache, clip["id"])
    if not src.exists():
        return None
    f = _cache_file(cache, clip["id"], src)
    if f.exists():
        return np.load(f)
    vf = []
    if clip.get("crop"):
        wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                             "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip().split(",")
        x, y, w, h = camera.content_rect(clip, int(wh[0]), int(wh[1]))
        vf.append(f"crop={int(w) // 2 * 2}:{int(h) // 2 * 2}:{int(x)}:{int(y)}")
    vf += [f"fps={FPS}", f"scale={SIZE}:{SIZE}", "format=gray"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", ",".join(vf), "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE)
    out = np.array([_hash(g) for g in frames], bool).reshape(-1, 255)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, out)
    return out


def photo_print(root: Path, cache: Path, item: dict) -> np.ndarray | None:
    src = root / item["source"]
    if not src.exists():
        return None
    f = _cache_file(cache, item["id"], src)
    if f.exists():
        return np.load(f)
    try:
        img = np.array(photos.open_image(src).convert("L"))
    except Exception:  # noqa: BLE001 - an unreadable photo just isn't compared
        return None
    out = _hash(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, out)
    return out


def _shared_run(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    """Longest run of matching frames over all offsets: (seconds, offset s, median distance)."""
    if len(A) < 2 or len(B) < 2:
        return 0.0, 0.0, 1.0
    D = (A[:, None, :] != B[None, :, :]).mean(axis=2)
    best = (0, 0, 1.0)
    for off in range(-len(A) + 1, len(B)):
        i = np.arange(max(0, -off), min(len(A), len(B) - off))
        if len(i) < 2:
            continue
        dd = D[i, i + off]
        ok = dd <= 0.2
        run = m = end = 0
        for k, v in enumerate(ok):
            run = run + 1 if v else 0
            if run > m:
                m, end = run, k
        if m > best[0]:
            best = (m, off, float(np.median(dd[end - m + 1:end + 1])))
    return best[0] / FPS, best[1] / FPS, best[2]


def find(root: Path, items: list[dict], cache: Path | None = None) -> list[dict]:
    """Pairs of items (plays and photos, by id) that look like the same footage:
    [{"a": id, "b": id, "kind": "play"|"photo", "seconds": shared s, "offset": s, "same": bool}]
    "same" = identical copies (the whole shorter clip matches)."""
    cache = cache or root / ".hlv_cache"
    clips = [c for c in items if not photos.is_photo(c)]
    pics = [c for c in items if photos.is_photo(c)]
    out = []
    prints = {c["id"]: clip_print(cache, c) for c in clips}
    for a, b in itertools.combinations(clips, 2):
        A, B = prints[a["id"]], prints[b["id"]]
        if A is None or B is None:
            continue
        s, off, med = _shared_run(A, B)
        if s >= MIN_RUN_S and med <= MAX_DIST:
            out.append({"a": a["id"], "b": b["id"], "kind": "play", "seconds": round(s, 1), "offset": round(off, 2),
                        "same": s >= 0.9 * min(len(A), len(B)) / FPS and abs(off) < 0.3})
    pp = {p["id"]: photo_print(root, cache, p) for p in pics}
    for a, b in itertools.combinations(pics, 2):
        A, B = pp[a["id"]], pp[b["id"]]
        if A is not None and B is not None and float((A != B).mean()) <= PHOTO_DIST:
            out.append({"a": a["id"], "b": b["id"], "kind": "photo", "seconds": 0.0, "offset": 0.0, "same": True})
    return out
