"""Glue: probe -> signals (cached) -> classify -> detect -> clip record."""
from __future__ import annotations

from pathlib import Path

from . import plays
from .audio import whistle_events
from .config import DetectConfig
from .motion import scene_cuts
from .probe import ClipInfo, probe
from .signals import classify_kind, clip_id, extract

CACHE_DIRNAME = ".hlv_cache"


def load_clip(root: Path, path: Path, cfg: DetectConfig, force: bool = False) -> tuple[ClipInfo, str, dict]:
    info = probe(path)
    cid = clip_id(root, path)
    sig = extract(path, info, root / CACHE_DIRNAME, cid, fps=cfg.analysis_fps,
                  width=cfg.analysis_width, force=force)
    return info, cid, sig


def analyze(root: Path, path: Path, info: ClipInfo, cid: str, sig: dict, cfg: DetectConfig) -> dict:
    meta = sig["meta"]
    fps = meta["fps"]
    cuts = scene_cuts(sig["frame_diff"], fps, cfg.scene_cut_ratio)
    kind = classify_kind(info, path.name, cuts, meta["crop"], cfg.min_scene_cuts)
    whistles = whistle_events(sig["whistle_t"], sig["whistle_s"])
    act = plays.activity(sig, fps, cfg)
    det = plays.detect_play(act, fps, info.duration, cfg, whistles)
    det["scenes"] = cuts
    det["whistle_candidates"] = [[round(t, 2), round(s, 2)] for t, s in whistles]
    proposed, status = plays.propose(det, info.duration, kind, cfg)
    rel = path.relative_to(root)
    clip = {
        "id": cid,
        "source": rel.as_posix(),
        "group": rel.parts[0] if len(rel.parts) > 1 else "",
        "probe": info.to_dict(),
        "kind": kind,
        "crop": meta["crop"],
        "detection": det,
        "proposed": proposed,
        "status": status,
    }
    if kind == "pre_edited":
        clip["note"] = "Edited footage with scene cuts; may show other players. Trim manually."
    return clip
