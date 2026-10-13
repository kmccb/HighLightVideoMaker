"""Turn a folder of clips into a project: preview copies, then find each play's snap/end.

This is what `hlv detect` does, run from the New project screen with progress.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import library, project, proxy
from .config import DetectConfig
from .pipeline import CACHE_DIRNAME, analyze, load_clip
from .probe import VIDEO_EXTS, find_clips
from .signals import clip_id
from . import photos as photo_files

PHOTO_EXTS = photo_files.PHOTO_EXTS


def scan(folder: Path) -> dict:
    """What's in a folder the user picked: video and photo counts, and whether it's a project."""
    folder = Path(folder)
    videos = find_clips(folder)
    photos = photo_files.find(folder)
    # (Live Photo movies count with their photo, not as plays: find_clips leaves them out.)
    known = library.find(folder)
    return {"folder": str(folder.resolve()), "name": folder.name, "videos": len(videos), "photos": len(photos),
            "project": bool(known) or (folder / project.FILENAME).exists()}


def _signals(root: str, path: str, cfg: dict):
    t0 = time.time()
    load_clip(Path(root), Path(path), DetectConfig.from_dict(cfg), False)
    return path, time.time() - t0


def new_videos(root: Path, proj: dict) -> list[Path]:
    """Videos in the project's folder that aren't plays of the project yet (put there after it was set up)."""
    root = Path(root)
    have = {c["source"].replace("\\", "/").lower() for c in proj.get("clips") or []}
    return [p for p in find_clips(root) if p.relative_to(root).as_posix().lower() not in have]


def run(folder: Path, player: str | None, music: str | None, progress=lambda step, done, total: None,
        jobs: int = 3, log=print, only: list[Path] | None = None, lock=None) -> dict:
    """Set the folder up as a project and register it. Clips that can't be read are skipped
    (listed in the result); the rest carry on.
    only: just these videos, added to a project that already exists (its settings are left alone).
    lock: held while the project file is read, changed and saved (the app saves review marks under it)."""
    from contextlib import nullcontext
    from .cli import CONFIG_FILENAME
    root = Path(folder).resolve()
    paths = list(only) if only is not None else find_clips(root)
    if not paths:
        raise ValueError("No videos in that folder (looked for .mov .mp4 .m4v .mkv .avi).")
    cfg = DetectConfig.load(root / CONFIG_FILENAME)
    cache = root / CACHE_DIRNAME
    n = len(paths)
    total = 2 * n + 1
    log(f"setup: {n} videos in {root}")

    # 1) small preview copies for the review page
    done = [0]
    skipped: list[str] = []

    def plog(msg: str):
        if msg.strip().startswith(("proxy ok", "proxy FAIL")):
            done[0] += 1
            progress("making preview copies", done[0], total)
        log(msg)
    progress("making preview copies", 0, total)
    proxy.make_all([(p, proxy.proxy_path(cache, clip_id(root, p))) for p in paths], jobs, log=plog)
    done[0] = n

    # 2) motion + sound signals per clip (the slow part), in parallel
    ok_paths = []
    progress("finding the plays", done[0], total)
    log(f"finding the plays in {n} videos (the slow part) ...")
    with ProcessPoolExecutor(jobs) as ex:
        futs = {ex.submit(_signals, str(root), str(p), cfg.to_dict()): p for p in paths}
        for f in as_completed(futs):
            p = futs[f]
            try:
                _, dt = f.result()
                ok_paths.append(p)
                log(f"  plays found in {p.name}  ({dt:.0f}s)")
            except Exception as e:  # noqa: BLE001 - one bad file mustn't stop the rest
                skipped.append(p.name)
                log(f"  skipped {p.name}: {e}")
            done[0] += 1
            progress(f"finding the plays ({p.name})", done[0], total)

    # 3) detection from the cached signals; review data starts empty
    log("saving the project ...")
    found = [(p, *load_clip(root, p, cfg, False)) for p in sorted(ok_paths)]
    new = [analyze(root, p, info, cid, sig, cfg) for p, info, cid, sig in found]
    with lock or nullcontext():
        proj = project.load(root)
        if only is None:
            proj["config"] = cfg.to_dict()
        for c in new:
            project.merge_clip(proj, c)
        if not proj["clips"]:
            raise ValueError("None of the videos could be read.")
        if only is None:
            s = proj.get("settings") or {}
            s.update({"player": player, "music": music, "scottforge_card": s.get("scottforge_card", True)})
            proj["settings"] = s
        project.save(root, proj)
    if only is None:
        library.add_project(root, player)
    progress("done", total, total)
    return {"folder": str(root), "clips": len(proj["clips"]), "added": len(new), "skipped": skipped}
