"""highlights.json read/write.

`review` is owned by the Phase 2 review page; re-running detection must never clobber it.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1
FILENAME = "highlights.json"


HIGHLIGHT_STYLES = ("ring", "arrow", "shadow", "off")


def settings(project: dict) -> dict:
    """Reel-wide look settings (same highlight style on every clip)."""
    return {"highlight": "ring", "highlight_color": "#ffd23f", "transition": "crossfade",
            "freeze": True, "order": [], "player": None, "game_audio": 0.3, "music_volume": 0.9,
            "beat_sync": True, "hit_fx": True, "music_start": 0.0,
            "slowmo": False, "slowmo_speed": 0.4, "name_banner": "first",
            "music": None, "scottforge_card": True,
            **(project.get("settings") or {})}


def music_path(root: Path, settings: dict) -> Path | None:
    """The project's song (relative to the project folder, or absolute); None if unset/missing."""
    m = settings.get("music")
    if not m:
        return None
    p = Path(m)
    p = p if p.is_absolute() else root / p
    return p if p.is_file() else None


def empty_review() -> dict:
    # snap/end/labeled: human-confirmed play times (also exported as ground truth).
    # in/out: final trim window; keep: include in the reel.
    # spotlight: [{t, x, y, w, h}] clicks on your player (normalised); zoom: 0 = off, else level.
    return {"snap": None, "end": None, "labeled": False, "notes": "",
            "in": None, "out": None, "keep": None, "spotlight": [], "zoom": 0, "players": [],
            "hit": None, "no_hit": False}


def load(root: Path) -> dict:
    p = root / FILENAME
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema_version": SCHEMA_VERSION, "source_dir": str(root), "clips": []}


def merge_clip(project: dict, clip: dict) -> None:
    for i, old in enumerate(project["clips"]):
        if old["id"] == clip["id"]:
            clip["review"] = {**empty_review(), **(old.get("review") or {})}
            project["clips"][i] = clip
            return
    clip.setdefault("review", empty_review())
    project["clips"].append(clip)


def save(root: Path, project: dict) -> Path:
    project["schema_version"] = SCHEMA_VERSION
    project["updated"] = datetime.now().isoformat(timespec="seconds")
    project["clips"].sort(key=lambda c: c["source"].lower())
    p = root / FILENAME
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(project, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def final_window(clip: dict) -> tuple[float, float]:
    """What later phases should use: reviewed values win over proposals."""
    r, p = clip.get("review") or {}, clip["proposed"]
    return (r.get("in") if r.get("in") is not None else p["in"],
            r.get("out") if r.get("out") is not None else p["out"])
