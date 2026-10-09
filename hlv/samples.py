"""Ready-made samples of every intro and ScottForge ending style, so the Build panel can show
them straight away instead of rendering one on demand (~45 s each).

Intros use a made-up player (SAMPLE) on the plain team-colour background, the default beat
timing and the impact sounds, so one set serves every project. Endings use the real branding
(they're the same in every reel anyway) and go to the shared outro cache, so a reel built with
that ending reuses the sample. Rendered one at a time in a background thread, waiting while a
reel builds; anything already on disk is kept. Everything lives in ~/HighlightReels/cache.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

from . import library

SAMPLE = {"id": "sample", "name": "Jordan Taylor", "number": "7", "position": "WR / DB", "class_year": "2027",
          "school": "Sample High Eagles", "height_weight": "6' 0\" - 175 lbs", "logo": "",
          "color": "#1f4fd1", "accent": "#ffd23f"}
VERSION = 1
POSTER_W = 480

_lock = threading.Lock()
_state = {"running": False, "current": None, "error": None}


def intro_styles() -> list[str]:
    from . import intro
    return list(intro.STYLES) + ["classic"]


def outro_styles() -> list[str]:
    from . import outro
    return list(outro.STYLES)


def folder() -> Path:
    return library.home() / "cache" / "samples"


def intro_file(style: str) -> Path:
    from . import intro
    key = hashlib.sha1(json.dumps({"v": VERSION, "iv": intro.INTRO_VERSION, "pl": SAMPLE, "s": style},
                                  sort_keys=True).encode()).hexdigest()[:10]
    return folder() / f"intro-{style}-{key}.mp4"


def outro_file(style: str) -> Path:
    from . import outro
    lib = library.home()
    b = {**library.branding(), "style": style}
    return lib / "cache" / "outro" / f"outro-{outro.cache_key(b, outro.video_path(lib, b), outro.OUTRO_S)}.mp4"


def poster(mp4: Path) -> Path:
    return mp4.with_suffix(".jpg")


def _make_poster(mp4: Path, at: float) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(mp4), "-frames:v", "1",
                    "-vf", f"scale={POSTER_W}:-2", "-q:v", "4", str(poster(mp4))], check=True)


def _render_intro(style: str, out: Path) -> None:
    from . import graphics, intro, segment
    lib = library.home()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    if style == "classic":
        segment.card(tmp, graphics.title_card(lib, SAMPLE), 3.0, fade_in=True)
        tmp.with_suffix(".png").unlink(missing_ok=True)
    else:
        with intro.frame_size(graphics.W, graphics.H):
            if style == "animated":
                intro.render(tmp, lib, SAMPLE, None, 0.0, None, sfx=True)
            else:
                from . import intro_fx
                intro_fx.render(style, tmp, lib, SAMPLE, None, 0.0, None, sfx=True)
    tmp.replace(out)
    _make_poster(out, 1.5 if style == "classic" else intro.beat_times(None, 0.0)[1] - 1.0)


def _render_outro(style: str, out: Path) -> None:
    from . import outro
    lib = library.home()
    mp4 = outro.build(lib / "cache", lib, {**library.branding(), "style": style})
    if mp4 != out:                                   # branding changed while we were waiting
        return
    _make_poster(out, 3.6)


def _todo() -> list[tuple[str, str, Path]]:
    items = [("intro", s, intro_file(s)) for s in intro_styles()] + [("outro", s, outro_file(s)) for s in outro_styles()]
    return [(kind, s, f) for kind, s, f in items if not (f.exists() and poster(f).exists())]


def status() -> dict:
    def entry(kind: str, style: str, f: Path) -> dict:
        ok = f.exists() and poster(f).exists()
        base = f"/media/samples/{kind}/{f.name}"
        return {"style": style, "video": base if ok else None, "poster": base[:-4] + ".jpg" if ok else None,
                "rendering": _state["current"] == (kind, style)}
    return {"intros": [entry("intro", s, intro_file(s)) for s in intro_styles()],
            "outros": [entry("outro", s, outro_file(s)) for s in outro_styles()],
            "running": _state["running"], "error": _state["error"]}


def media_path(kind: str, name: str) -> Path | None:
    """The file behind /media/samples/<kind>/<name>, if it's one of ours."""
    files = [intro_file(s) for s in intro_styles()] if kind == "intro" else \
            [outro_file(s) for s in outro_styles()] if kind == "outro" else []
    for f in files:
        for p in (f, poster(f)):
            if p.name == name and p.exists():
                return p
    return None


def warm(busy=lambda: False) -> None:
    """Render every missing sample in a background thread (one at a time; waits while busy())."""
    with _lock:
        if _state["running"] or not _todo():
            return
        _state.update(running=True, error=None)

    def run():
        failed = set()
        try:
            while True:
                todo = [t for t in _todo() if t[2] not in failed]
                if not todo:
                    break
                while busy():
                    time.sleep(5)
                kind, style, f = todo[0]
                _state["current"] = (kind, style)
                try:
                    (_render_intro if kind == "intro" else _render_outro)(style, f)
                except Exception as e:  # noqa: BLE001 - skip this one, keep the rest coming
                    _state["error"] = f"{kind} {style}: {e}"
                    failed.add(f)
        finally:
            _state.update(running=False, current=None)
    threading.Thread(target=run, daemon=True, name="samples").start()
