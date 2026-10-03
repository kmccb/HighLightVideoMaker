"""Player profiles: who the reel is about and how it's branded.

players.json lives next to the clips; logo/music paths are relative to that folder
(drop them in assets/). Several players are supported (Caleb and his friends).
"""
from __future__ import annotations

import json
from pathlib import Path

FILENAME = "players.json"
ASSETS = "assets"

TEMPLATE = {
    "players": [
        {
            "id": "caleb",
            "name": "Caleb",
            "number": "",
            "position": "",
            "class_year": "",
            "school": "",
            "height_weight": "",
            "contact": "",
            "logo": "assets/logo.png",
            "music": "assets/music.mp3",
            "color": "#1f4fd1",
            "accent": "#ffd23f",
        }
    ],
    "_help": {
        "id": "short name used for the output file, no spaces",
        "number": "jersey number, e.g. 22",
        "position": "e.g. LB / RB",
        "class_year": "graduation year, e.g. 2029",
        "height_weight": "optional, e.g. 6'1\" 185",
        "contact": "optional line for the end card: email, phone or Hudl link",
        "logo": "PNG (transparent background looks best), relative to this folder",
        "music": "MP3/M4A/WAV, relative to this folder",
        "color": "team colour for the title card and name banner",
        "accent": "highlight colour (ring/arrow) and banner accent",
    },
}

FIELDS = [k for k in TEMPLATE["players"][0]]


def path(root: Path) -> Path:
    return root / FILENAME


def ensure_template(root: Path) -> Path:
    """Create players.json and assets/ if missing. Never overwrites."""
    (root / ASSETS).mkdir(exist_ok=True)
    p = path(root)
    if not p.exists():
        p.write_text(json.dumps(TEMPLATE, indent=2), encoding="utf-8")
    return p


def load(root: Path) -> list[dict]:
    p = path(root)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    out = []
    for raw in data.get("players", []):
        pl = {k: "" for k in FIELDS}
        pl.update({k: (str(v) if v is not None else "") for k, v in raw.items() if k in FIELDS})
        if not pl["id"]:
            continue
        out.append(pl)
    return out


def get(root: Path, player_id: str | None) -> dict | None:
    players = load(root)
    if not players:
        return None
    if player_id:
        return next((p for p in players if p["id"] == player_id), None)
    return players[0]


def asset(root: Path, rel: str) -> Path | None:
    """Resolve a logo/music path; None if not set or missing."""
    if not rel:
        return None
    p = Path(rel)
    p = p if p.is_absolute() else root / p
    return p if p.is_file() else None


def status(root: Path, pl: dict) -> dict:
    """What's ready and what's missing, for the review page."""
    return {
        "logo": str(asset(root, pl["logo"]) or ""),
        "music": str(asset(root, pl["music"]) or ""),
        "logo_missing": bool(pl["logo"]) and asset(root, pl["logo"]) is None,
        "music_missing": bool(pl["music"]) and asset(root, pl["music"]) is None,
    }


AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def slug(text: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "-" for ch in (text or "").strip())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "player"


def save(root: Path, players: list[dict]) -> list[dict]:
    """Write players.json (keeps the _help notes). Unknown fields dropped, ids unique."""
    clean, seen = [], set()
    for raw in players:
        pl = {k: str(raw.get(k) or "").strip() for k in FIELDS}
        pl["id"] = slug(pl["id"] or pl["name"])
        base, n = pl["id"], 2
        while pl["id"] in seen:
            pl["id"], n = f"{base}-{n}", n + 1
        seen.add(pl["id"])
        for c in ("color", "accent"):
            if not (len(pl[c]) == 7 and pl[c].startswith("#")):
                pl[c] = TEMPLATE["players"][0][c]
        clean.append(pl)
    p = path(root)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"players": clean, "_help": TEMPLATE["_help"]}, indent=2), encoding="utf-8")
    tmp.replace(p)
    return clean


def asset_files(root: Path) -> dict:
    """Songs and images already in the assets folder (relative paths)."""
    d = root / ASSETS
    files = sorted(f for f in d.iterdir() if f.is_file()) if d.is_dir() else []
    rel = lambda f: f"{ASSETS}/{f.name}"  # noqa: E731
    return {"audio": [rel(f) for f in files if f.suffix.lower() in AUDIO_EXTS],
            "images": [rel(f) for f in files if f.suffix.lower() in IMAGE_EXTS]}


def store_upload(root: Path, filename: str, data: bytes, kind: str) -> str:
    """Save an uploaded logo/song into assets/ and return its path relative to root."""
    ext = Path(filename).suffix.lower()
    allowed = AUDIO_EXTS if kind == "music" else IMAGE_EXTS
    if ext not in allowed:
        raise ValueError(f"{kind} must be one of: {', '.join(sorted(allowed))}")
    d = root / ASSETS
    d.mkdir(exist_ok=True)
    stem = slug(Path(filename).stem)[:60]
    dst, n = d / f"{stem}{ext}", 2
    while dst.exists() and dst.read_bytes() != data:   # same name, different file
        dst, n = d / f"{stem}-{n}{ext}", n + 1
    if not dst.exists():
        dst.write_bytes(data)
    return f"{ASSETS}/{dst.name}"
