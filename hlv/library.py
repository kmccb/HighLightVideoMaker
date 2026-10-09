"""The app folder shared by all projects: player profiles, the project list, branding.

Lives in ~/HighlightReels (override with HLV_HOME), outside the code so updates never
touch it. A project is one clips folder (its review data stays in its highlights.json)
plus the player it's about.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from . import profiles, project

PROJECTS = "projects.json"
BRANDING = "branding.json"
BRANDING_DEFAULTS = {
    "logo": "", "bg": "#0a0a0c", "accent": "#E85D2F", "title": "Created by ScottForge",
    "url": "https://ScottForge.ai", "invite": "Want a highlight reel like this? Reach out at ScottForge.ai",
    "video": "assets/scottforge-forge.mp4",     # background of the animated ending (the website's forge clip)
    "style": "forge",                           # which animated ending: forge, anvil, molten, minimal, hammer, quench
}


def home() -> Path:
    h = Path(os.environ.get("HLV_HOME") or Path.home() / "HighlightReels")
    (h / profiles.ASSETS).mkdir(parents=True, exist_ok=True)
    return h


def _key(folder: Path) -> str:
    return os.path.normcase(str(Path(folder).resolve()))


def projects() -> list[dict]:
    """Known projects, most recently opened first."""
    p = home() / PROJECTS
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    out = [x for x in data.get("projects", []) if isinstance(x, dict) and x.get("folder")]
    return sorted(out, key=lambda x: x.get("last_opened", ""), reverse=True)


def _save(items: list[dict]) -> None:
    p = home() / PROJECTS
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"projects": items}, indent=2), encoding="utf-8")
    tmp.replace(p)


def find(folder: Path) -> dict | None:
    k = _key(folder)
    return next((x for x in projects() if _key(Path(x["folder"])) == k), None)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def add_project(folder: Path, player: str | None) -> dict:
    """Register a folder (once). Re-adding keeps the entry and updates the player if given."""
    items, k = projects(), _key(folder)
    hit = next((x for x in items if _key(Path(x["folder"])) == k), None)
    if hit is None:
        hit = {"folder": str(Path(folder).resolve()), "player": player, "last_opened": ""}
        items.append(hit)
    elif player:
        hit["player"] = player
    _save(items)
    return hit


def touch(folder: Path) -> None:
    items, k = projects(), _key(folder)
    for x in items:
        if _key(Path(x["folder"])) == k:
            x["last_opened"] = _now()
    _save(items)


def remove_project(folder: Path) -> None:
    """Forget a project. Never deletes any files."""
    k = _key(folder)
    _save([x for x in projects() if _key(Path(x["folder"])) != k])


def migrate(root: Path) -> None:
    """Bring a project from the old layout (players.json + assets/ in the clips folder) into
    the shared library: players copied in (ids already there win), their song becomes the
    project's music, and the folder is registered. Safe to run every time it's opened;
    the old files are left as a backup."""
    lib = home()
    old = profiles.load(root) if profiles.path(root).exists() else []
    proj = project.load(root)
    s = proj.get("settings") or {}
    changed = False
    if old:
        have = {p["id"] for p in profiles.load(lib)}
        new = []
        for p in old:
            if p["id"] in have:
                continue
            src = profiles.asset(root, p.get("logo", ""))
            if src is not None:
                p["logo"] = profiles.store_upload(lib, src.name, src.read_bytes(), "logo")
            new.append(p)
        if new:
            profiles.save(lib, profiles.load(lib) + new)
        # Music used to belong to the player; it's per project now.
        raw = json.loads(profiles.path(root).read_text(encoding="utf-8-sig")).get("players", [])
        want = s.get("player") or old[0]["id"]
        song = next((r.get("music") for r in raw if r.get("id") == want and r.get("music")), None)
        if "music" not in s:
            s["music"], changed = song, True
        if not s.get("player"):
            s["player"], changed = old[0]["id"], True
        if "scottforge_card" not in s:
            s["scottforge_card"], changed = False, True    # existing reels keep their ending
    if changed and (root / project.FILENAME).exists():
        proj["settings"] = s
        project.save(root, proj)
    if find(root) is None:
        add_project(root, s.get("player"))


def branding() -> dict:
    """Tom's ScottForge closing card: logo (relative to the app folder), colours, text."""
    p = home() / BRANDING
    raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {k: str(raw.get(k, v) if raw.get(k) is not None else v) for k, v in BRANDING_DEFAULTS.items()}


def save_branding(patch: dict) -> dict:
    b = branding()
    for k in BRANDING_DEFAULTS:
        if k in patch and isinstance(patch[k], str):
            v = patch[k].strip()
            if k in ("bg", "accent") and not (len(v) == 7 and v.startswith("#")):
                continue
            if k == "style" and v not in ("forge", "anvil", "molten", "minimal", "hammer", "quench"):
                continue
            b[k] = v
    p = home() / BRANDING
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(b, indent=2), encoding="utf-8")
    tmp.replace(p)
    return b


def branding_ready(b: dict) -> bool:
    """Worth showing a card: a logo that exists, or some text."""
    return bool(profiles.asset(home(), b.get("logo", "")) or b.get("title") or b.get("url"))
