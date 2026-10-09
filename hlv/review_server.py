"""Local review page: confirm/correct snap & end per clip, keep/skip.

Saves into highlights.json `review` and (for confirmed clips) ground_truth.csv.
Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import camera, library, photo_slide, photos, profiles, project, proxy, reel, score, segment
from .config import DetectConfig
from .pipeline import CACHE_DIRNAME

STATIC = Path(__file__).parent / "static"
EDITABLE = {"snap", "end", "labeled", "keep", "notes", "spotlight", "zoom", "hit", "no_hit", "ring_off", "side", "limit",
            "star"}
ZOOM_LEVELS = (0, 1.25, 1.5, 1.8)


class NoProject(Exception):
    """A project page/API was used before a project was opened."""


class ReviewApp:
    """The running app. Starts on the project list; open_project() points it at one folder."""

    def __init__(self, jobs: int = 4):
        self.lock = threading.Lock()
        self.root: Path | None = None
        self.cache: Path | None = None
        self.cfg: DetectConfig | None = None
        self.jobs = jobs
        self.job = {"state": "idle"}
        self._gen = 0          # bumps on every project switch; background work for older ones stops
        self.setup = {"state": "idle"}     # new-project setup job
        self.music_job = {"state": "idle"}  # YouTube download job

    def _need(self) -> None:
        if self.root is None:
            raise NoProject()

    # ---------------- projects ----------------
    def building(self) -> bool:
        return self.job.get("state") == "running"

    def open_project(self, folder: Path) -> dict:
        from .cli import CONFIG_FILENAME
        folder = Path(folder).resolve()
        if self.building():
            raise ValueError("a reel is building - switch projects when it's finished")
        if not folder.is_dir():
            raise ValueError(f"folder not found: {folder}")
        if not (folder / project.FILENAME).exists():
            raise ValueError(f"{folder.name} hasn't been set up as a project yet")
        library.migrate(folder)
        cfg = DetectConfig.load(folder / CONFIG_FILENAME)
        with self.lock:
            proj = project.load(folder)
            if photos.sync(folder, proj):        # pick up photos added since last time
                project.save(folder, proj)
            self.root, self.cache, self.cfg = folder, folder / CACHE_DIRNAME, cfg
            self.job = {"state": "idle"}
            self._gen += 1
            gen = self._gen
        library.touch(folder)
        self._background(folder, gen)
        return {"folder": str(folder)}

    def _background(self, root: Path, gen: int) -> None:
        """Preview copies for any new clips, then pre-track clicked clips."""
        from . import camera
        from .probe import find_clips
        from .signals import clip_id
        stop = lambda: self._gen != gen          # noqa: E731
        busy = lambda: self.building()           # noqa: E731
        cache = root / CACHE_DIRNAME

        def run():
            todo = [(p, proxy.proxy_path(cache, clip_id(root, p))) for p in find_clips(root)]
            todo = [t for t in todo if not t[1].exists()]
            if todo and not stop():
                proxy.make_all(todo, self.jobs)
            if not stop():
                camera.warm(cache, project.load(root)["clips"], busy, stop=stop)
            if not stop():
                from . import detect
                detect.warm(cache, project.load(root)["clips"], busy, stop=stop)
            if not stop():
                from . import hitmodel
                hitmodel.schedule(5)                      # learn from any hits marked meanwhile
        threading.Thread(target=run, daemon=True).start()

    def projects_info(self) -> dict:
        lib = library.home()
        players = {p["id"]: p for p in profiles.load(lib)}
        out = []
        for x in library.projects():
            f = Path(x["folder"])
            ready = (f / project.FILENAME).exists()
            pid, clips, confirmed, last = x.get("player"), 0, 0, None
            if ready:
                proj = project.load(f)
                clips = len(proj["clips"])
                confirmed = sum(1 for c in proj["clips"] if (c.get("review") or {}).get("labeled"))
                pid = project.settings(proj).get("player") or pid
                reels = sorted((f / reel.OUT_DIRNAME).glob("*.mp4"), key=lambda r: r.stat().st_mtime)
                if reels:
                    last = datetime.fromtimestamp(reels[-1].stat().st_mtime).isoformat(timespec="minutes")
            pl = players.get(pid)
            out.append({
                "folder": str(f), "name": f.name, "exists": f.is_dir(), "ready": ready,
                "clips": clips, "confirmed": confirmed, "last_reel": last,
                "player": {k: pl[k] for k in ("id", "name", "number", "position", "color")} if pl else None,
                "logo": bool(pl and profiles.asset(lib, pl["logo"])),
                "current": self.root is not None and f == self.root,
            })
        return {"current": str(self.root) if self.root else None, "building": self.building(), "projects": out}

    # ---------------- new project ----------------
    def pick_folder(self) -> dict:
        from . import pickers, setup_job
        f = pickers.folder()
        return {"cancelled": True} if f is None else setup_job.scan(f)

    def _set_music(self, folder: Path, rel: str | None) -> None:
        """If folder is the open project, make rel its song."""
        if self.root is not None and Path(folder).resolve() == self.root:
            with self.lock:
                proj = project.load(self.root)
                proj["settings"] = {**project.settings(proj), "music": rel}
                project.save(self.root, proj)

    def pick_music(self, folder: Path) -> dict:
        from . import music, pickers
        f = pickers.music_file()
        if f is None:
            return {"cancelled": True}
        rel = music.import_file(Path(folder), f)
        self._set_music(folder, rel)
        return {"music": rel, "name": Path(rel).name}

    def use_music(self, folder: Path, src: Path) -> dict:
        from . import music
        rel = music.import_file(Path(folder), Path(src))
        self._set_music(folder, rel)
        return {"music": rel, "name": Path(rel).name}

    def music_suggestion(self, player: str) -> dict:
        """The song from this player's most recent other project, if any."""
        for x in library.projects():
            f = Path(x["folder"])
            if not (f / project.FILENAME).exists():
                continue
            s = project.settings(project.load(f))
            if (s.get("player") or x.get("player")) != player:
                continue
            m = project.music_path(f, s)
            if m:
                return {"path": str(m), "name": m.name, "from": f.name}
        return {}

    def start_youtube(self, folder: Path, url: str) -> dict:
        from . import music
        if self.music_job.get("state") == "running":
            return self.music_job
        if not music.is_youtube(url):
            raise ValueError("That doesn't look like a YouTube link (youtube.com/watch?v=… or youtu.be/…).")
        self.music_job = {"state": "running", "frac": 0.0, "step": "starting", "music": None, "error": None}
        job = self.music_job

        def run():
            try:
                rel = music.download_youtube(Path(folder), url,
                                             progress=lambda f, t: job.update(frac=f, step=t))
                self._set_music(folder, rel)
                job.update(state="done", music=rel, name=Path(rel).name)
            except Exception as e:  # noqa: BLE001
                job.update(state="error", error=str(e))
        threading.Thread(target=run, daemon=True).start()
        return job

    def create_project(self, folder: Path, player: str | None, music_rel: str | None) -> dict:
        from . import setup_job
        folder = Path(folder).resolve()
        if self.setup.get("state") == "running":
            raise ValueError("a project is already being set up")
        if self.building():
            raise ValueError("a reel is building - create the project when it's finished")
        if not folder.is_dir():
            raise ValueError(f"folder not found: {folder}")
        if library.find(folder) or (folder / project.FILENAME).exists():
            raise ValueError("that folder is already a project")
        if player and profiles.get(library.home(), player) is None:
            raise ValueError("pick a player first")
        self.setup = {"state": "running", "step": "starting", "done": 0, "total": 1, "error": None,
                      "folder": str(folder)}
        job = self.setup

        def run():
            try:
                res = setup_job.run(folder, player, music_rel, jobs=min(self.jobs, 3),
                                    progress=lambda st, d, t: job.update(step=st, done=d, total=t))
                self.open_project(folder)
                job.update(state="done", result=res)
            except Exception as e:  # noqa: BLE001 - report to the page
                job.update(state="error", error=str(e))
        threading.Thread(target=run, daemon=True).start()
        return job

    def remove_project(self, folder: Path) -> dict:
        if self.root is not None and Path(folder).resolve() == self.root:
            raise ValueError("that project is open - switch to another one first")
        library.remove_project(Path(folder))
        return self.projects_info()

    def clips(self) -> dict:
        self._need()
        with self.lock:
            proj = project.load(self.root)
        out = []
        for c in proj["clips"]:
            r = {**project.empty_review(), **(c.get("review") or {})}
            out.append({
                "id": c["id"], "source": c["source"], "group": c.get("group", ""), "kind": c["kind"],
                "status": c["status"], "note": c.get("note", ""), "duration": c["probe"]["duration"],
                "detection": c["detection"], "proposed": c["proposed"], "review": r,
                "proxy": f"/proxy/{c['id']}.mp4",
                "proxy_ready": proxy.proxy_path(self.cache, c["id"]).exists(),
            })
        pics = [{"id": ph["id"], "source": ph["source"], "group": ph.get("group", ""), "taken": ph.get("taken"),
                 "keep": ph.get("keep") is not False, "thumb": f"/api/photos/{ph['id']}/thumb"}
                for ph in proj.get("photos") or []]
        return {"pad_before": self.cfg.pad_before, "pad_after": self.cfg.pad_after,
                "settings": project.settings(proj), "clips": out, "photos": pics}

    # ---------------- reel ----------------
    @staticmethod
    def _credit_on(s: dict) -> bool:
        return bool(s.get("scottforge_card")) and library.branding_ready(library.branding())

    def branding_info(self) -> dict:
        b = library.branding()
        return {"branding": b, "logo": bool(profiles.asset(library.home(), b["logo"])),
                "images": profiles.asset_files(library.home())["images"]}

    def credit_png(self) -> bytes:
        import cv2
        from . import outro
        img = outro.still(library.home(), library.branding())
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".png", img)[1].tobytes()

    def outro_preview(self) -> str:
        """Render (or reuse) the animated ScottForge ending; returns its file name."""
        from . import outro
        lib = library.home()
        return outro.build(lib / "cache", lib, library.branding()).name

    def reel_info(self) -> dict:
        self._need()
        lib = library.home()
        with self.lock:
            proj = project.load(self.root)
        s = project.settings(proj)
        players = profiles.load(lib)
        pl = profiles.get(lib, s.get("player"))
        rows = []
        recruit = {c["id"] for c in reel.recruit_plays(proj)}
        for c in reel.ordered_items(proj):
            if photos.is_photo(c):
                rows.append({"id": c["id"], "type": "photo", "source": c["source"], "group": c.get("group", ""),
                             "taken": c.get("taken"), "included": reel.included(c), "keep": c.get("keep"),
                             "seconds": float(s.get("photo_s", 4)), "thumb": f"/api/photos/{c['id']}/thumb"})
                continue
            r = {**project.empty_review(), **(c.get("review") or {})}
            t0, t1 = project.final_window(c)
            spot = bool(r.get("spotlight"))
            rows.append({
                "id": c["id"], "source": c["source"], "group": c.get("group", ""), "kind": c["kind"],
                "included": reel.included(c), "keep": r.get("keep"), "labeled": bool(r.get("labeled")),
                "spotlight": spot, "zoom": r.get("zoom") or 0, "hit_set": r.get("hit") is not None, "no_hit": bool(r.get("no_hit")),
                "star": bool(r.get("star")), "recruit": c["id"] in recruit,
                # False: he isn't found at the snap, so this play won't freeze (None: unknown / not wanted)
                "freeze_ok": camera.freeze_found(self.cache, c) if spot and s.get("freeze", True) else None,
                "seconds": round(max(0.0, t1 - max(0.0, t0)) + (segment.FREEZE_S if spot and s.get("freeze") else 0), 2),
            })
        return {
            "settings": s, "players": players, "player": pl,
            "assets": {**profiles.status(lib, pl), "music": str(project.music_path(self.root, s) or ""),
                       "music_missing": bool(s.get("music")) and project.music_path(self.root, s) is None}
            if pl else None,
            "profile_path": str(profiles.path(lib)), "assets_dir": str(lib / profiles.ASSETS),
            "transitions": list(reel.TRANSITIONS),
            "card_s": reel.TITLE_S + reel.END_S + (reel.CREDIT_S if self._credit_on(s) else 0),
            "credit_ready": library.branding_ready(library.branding()),
            "clips": rows, "job": self.job, "photo_count": sum(1 for r in rows if r.get("type") == "photo"),
            "last_reel": self._reel_entry(s), "last_vertical": self._reel_entry(s, True),
            "outputs": self._outputs(s), "recruit_max_s": reel.RECRUIT_MAX_S,
        }

    def _outputs(self, s: dict) -> list[dict]:
        """Every finished video this project has for its player (reel / recruiting cut, wide /
        vertical), newest first."""
        pl = s.get("player")
        out = []
        for rec in (False, True):
            for vert in (False, True):
                f = self.root / reel.OUT_DIRNAME / reel.output_name(pl, vert, rec) if pl else None
                if f and f.exists():
                    out.append({"file": f.name, "path": str(f), "vertical": vert, "recruiting": rec,
                                "label": ("Recruiting cut" if rec else "Highlight reel") + (" · vertical 9:16" if vert else ""),
                                "built": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")})
        return sorted(out, key=lambda o: o["built"], reverse=True)

    # ---------------- player profiles ----------------
    def players_info(self) -> dict:
        lib = library.home()
        players = profiles.load(lib)
        s = {}
        if self.root is not None:
            with self.lock:
                s = project.settings(project.load(self.root))
        return {"players": players, "selected": s.get("player") or (players[0]["id"] if players else None),
                "status": {p["id"]: profiles.status(lib, p) for p in players},
                "assets": profiles.asset_files(lib), "fields": profiles.FIELDS,
                "assets_dir": str(lib / profiles.ASSETS), "project_open": self.root is not None}

    def save_players(self, players: list) -> dict:
        if not isinstance(players, list):
            raise ValueError("players must be a list")
        with self.lock:
            clean = profiles.save(library.home(), [p for p in players if isinstance(p, dict)])
            if self.root is not None:
                proj = project.load(self.root)
                s = project.settings(proj)
                if s.get("player") not in {p["id"] for p in clean}:
                    s["player"] = clean[0]["id"] if clean else None
                    proj["settings"] = s
                    project.save(self.root, proj)
        return self.players_info()

    def card_png(self, player_id: str, which: str) -> bytes:
        import cv2
        import numpy as np
        from . import graphics
        lib = library.home()
        pl = profiles.get(lib, player_id)
        if pl is None:
            raise KeyError(player_id)
        if which == "title":
            img = graphics.title_card(lib, pl)
        elif which == "end":
            img = graphics.end_card(lib, pl)
        else:   # banner over a dim field-ish background, cropped to the lower third
            img = np.full((graphics.H, graphics.W, 3), (40, 70, 45), np.uint8)
            graphics.composite(img, graphics.name_banner(pl), 1.0)
            img = img[780:1020, 0:1100]
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".png", img)[1].tobytes()

    def asset_path(self, player_id: str, kind: str) -> Path | None:
        if kind == "music":    # the song belongs to the open project now
            if self.root is None:
                return None
            with self.lock:
                return project.music_path(self.root, project.settings(project.load(self.root)))
        pl = profiles.get(library.home(), player_id)
        return profiles.asset(library.home(), pl["logo"]) if pl else None

    def store_music(self, filename: str, data: bytes) -> str:
        """Save a song into <project>/music/ and make it the project's music."""
        self._need()
        ext = Path(filename).suffix.lower()
        if ext not in profiles.AUDIO_EXTS:
            raise ValueError(f"music must be one of: {', '.join(sorted(profiles.AUDIO_EXTS))}")
        d = self.root / "music"
        d.mkdir(exist_ok=True)
        stem = profiles.slug(Path(filename).stem)[:60]
        dst, n = d / f"{stem}{ext}", 2
        while dst.exists() and dst.read_bytes() != data:
            dst, n = d / f"{stem}-{n}{ext}", n + 1
        if not dst.exists():
            dst.write_bytes(data)
        rel = f"music/{dst.name}"
        with self.lock:
            proj = project.load(self.root)
            proj["settings"] = {**project.settings(proj), "music": rel}
            project.save(self.root, proj)
        return rel

    def set_order(self, order: list) -> dict:
        self._need()
        with self.lock:
            proj = project.load(self.root)
            ids = {c["id"] for c in proj["clips"] + (proj.get("photos") or [])}
            s = project.settings(proj)
            s["order"] = [i for i in order if isinstance(i, str) and i in ids]
            proj["settings"] = s
            project.save(self.root, proj)
        return {"order": s["order"]}

    def render_segment(self, clip_id: str) -> str:
        self._need()
        with self.lock:
            proj = project.load(self.root)
        clip = next((c for c in proj["clips"] if c["id"] == clip_id), None)
        if clip is None:
            raise KeyError(clip_id)
        s = project.settings(proj)
        pl = profiles.get(library.home(), s.get("player"))
        banner = clip_id in reel.banner_ids([c for c in reel.ordered_clips(proj) if reel.included(c)], s)
        return segment.render(self.root, clip, pl, {**s, "_banner": banner}).name

    def _photo(self, photo_id: str) -> tuple[dict, dict]:
        self._need()
        with self.lock:
            proj = project.load(self.root)
        ph = next((p for p in proj.get("photos") or [] if p["id"] == photo_id), None)
        if ph is None:
            raise KeyError(photo_id)
        return proj, ph

    def photo_thumb(self, photo_id: str) -> Path:
        _, ph = self._photo(photo_id)
        return photo_slide.thumb(self.root, ph)

    def photo_slide(self, photo_id: str) -> str:
        """Render (or reuse) one slide for Preview, as it will look in the reel."""
        proj, ph = self._photo(photo_id)
        s = project.settings(proj)
        pics = [p for p in reel.ordered_items(proj) if photos.is_photo(p) and reel.included(p)]
        k = next((i for i, p in enumerate(pics) if p["id"] == photo_id), 0)
        return photo_slide.render(self.root, ph, photo_slide.slide_length(float(s.get("photo_s", 4))), k % 2 == 1).name

    def set_photo_keep(self, photo_id: str, keep: bool) -> dict:
        self._need()
        with self.lock:
            proj = project.load(self.root)
            ph = next((p for p in proj.get("photos") or [] if p["id"] == photo_id), None)
            if ph is None:
                raise KeyError(photo_id)
            ph["keep"] = bool(keep)
            project.save(self.root, proj)
        return ph

    def _last_reel(self, s: dict | None = None, vertical: bool = False) -> Path | None:
        """The (wide or vertical) reel this project built most recently: this session's build,
        else the file on disk."""
        self._need()
        if not vertical:
            out = (self.job.get("result") or {}).get("output") if self.job.get("state") == "done" else None
            if out and Path(out).exists() and not out.endswith("_vertical.mp4"):
                return Path(out)
        if s is None:
            with self.lock:
                s = project.settings(project.load(self.root))
        pl = s.get("player")
        f = self.root / reel.OUT_DIRNAME / f"{pl}_reel{'_vertical' if vertical else ''}.mp4" if pl else None
        return f if f and f.exists() else None

    def _reel_entry(self, s: dict, vertical: bool = False) -> dict | None:
        f = self._last_reel(s, vertical)
        if f is None:
            return None
        return {"path": str(f), "file": f.name,
                "built": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")}

    def reveal_reel(self, action: str, vertical: bool = False, file: str | None = None) -> dict:
        """Open a finished video in Windows' video player, or show it in File Explorer."""
        if file:
            self._need()
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.mp4", file):
                raise ValueError("bad file name")
            f = self.root / reel.OUT_DIRNAME / file
            f = f if f.exists() else None
        else:
            f = self._last_reel(vertical=vertical)
        if f is None:
            raise ValueError("no reel built yet")
        if action == "open":
            os.startfile(str(f))  # noqa: S606 - the user's own file, default app
        elif action == "folder":
            subprocess.Popen(["explorer", "/select,", str(f)])
        else:
            raise ValueError("unknown action")
        return {"ok": True, "path": str(f)}

    def intro_preview(self) -> str:
        """Render (or reuse) the animated intro with the song underneath; returns its file name."""
        from . import beats as beatmod, intro
        self._need()
        with self.lock:
            proj = project.load(self.root)
        s = project.settings(proj)
        lib = library.home()
        pl = profiles.get(lib, s.get("player"))
        if pl is None:
            raise ValueError("add a player on the Player profiles page first")
        clips = [c for c in reel.ordered_clips(proj) if reel.included(c)]
        music = project.music_path(self.root, s)
        ms = float(s.get("music_start") or 0)
        grid = beatmod.cached(music, self.cache) if music else None
        mp4, _ = intro.build(self.root, lib, pl, clips, grid, ms, sfx=bool(s.get("intro_sfx", False)),
                             style=s.get("intro", "animated") if s.get("intro") != "classic" else "animated")
        return intro.preview(self.root, mp4, music, ms, s).name

    def start_build(self) -> dict:
        self._need()
        if self.job.get("state") == "running":
            return self.job
        self.job = {"state": "running", "step": "starting", "done": 0, "total": 1, "result": None, "error": None}
        root = self.root

        def progress(step, done, total):
            self.job.update(step=step, done=done, total=total)

        def run():
            try:
                with self.lock:
                    s = project.settings(project.load(root))
                fmt, cut = s.get("format", "wide"), s.get("cut", "reel")
                res, outs = None, []
                for rec in ([False] if cut == "reel" else [True] if cut == "recruit" else [False, True]):
                    for vert in ([False] if fmt == "wide" else [True] if fmt == "vertical" else [False, True]):
                        r = reel.build(root, s.get("player"), s.get("transition"), progress=progress,
                                       vertical=vert, recruiting=rec)
                        outs.append(r)
                        if res is None:
                            res = r
                res = {**res, "outputs": [{"file": Path(r["output"]).name, "path": r["output"], "vertical": r["vertical"],
                                           "recruiting": r["recruiting"], "plays": r["plays"], "duration": r["duration"],
                                           "label": ("Recruiting cut" if r["recruiting"] else "Highlight reel")
                                           + (" · vertical 9:16" if r["vertical"] else "")} for r in outs]}
                self.job.update(state="done", result=res, step="done")
            except Exception as e:  # noqa: BLE001 - report any failure to the page
                self.job.update(state="error", error=str(e))
        threading.Thread(target=run, daemon=True).start()
        return self.job

    def update_settings(self, patch: dict) -> dict:
        self._need()
        with self.lock:
            proj = project.load(self.root)
            s = project.settings(proj)
            if patch.get("transition") in reel.TRANSITIONS:
                s["transition"] = patch["transition"]
            if "freeze" in patch:
                s["freeze"] = bool(patch["freeze"])
            if patch.get("music_source") in ("song", "compose"):
                s["music_source"] = patch["music_source"]
            if patch.get("compose_style") in ("rock", "trap", "cinematic"):
                s["compose_style"] = patch["compose_style"]
            if "compose_seed" in patch:
                s["compose_seed"] = int(patch["compose_seed"]) % 1000
            if patch.get("format") in ("wide", "vertical", "both"):
                s["format"] = patch["format"]
            if patch.get("cut") in reel.CUTS:
                s["cut"] = patch["cut"]
            if patch.get("intro") in ("animated", "inferno", "storm", "shatter", "stadium",
                                      "glitch", "smoke", "neon", "warp", "gold", "classic"):
                s["intro"] = patch["intro"]
            if "intro_sfx" in patch:
                s["intro_sfx"] = bool(patch["intro_sfx"])
            if patch.get("name_banner") in reel.BANNER_MODES:
                s["name_banner"] = patch["name_banner"]
            if "photo_s" in patch and float(patch["photo_s"]) in (3, 4, 5):
                s["photo_s"] = int(float(patch["photo_s"]))
            if "slowmo_speed" in patch and float(patch["slowmo_speed"]) in (0.5, 0.4, 0.3):
                s["slowmo_speed"] = float(patch["slowmo_speed"])
            for k in ("beat_sync", "hit_fx", "slowmo", "scottforge_card", "freeze_ring"):
                if k in patch:
                    s[k] = bool(patch[k])
            if "music_start" in patch:
                s["music_start"] = max(0.0, float(patch["music_start"] or 0))
            if "game_audio" in patch:
                s["game_audio"] = float(min(max(float(patch["game_audio"]), 0.0), 1.0))
            if "player" in patch and any(p["id"] == patch["player"] for p in profiles.load(library.home())):
                s["player"] = patch["player"]
            if patch.get("highlight") in project.HIGHLIGHT_STYLES:
                s["highlight"] = patch["highlight"]
            if isinstance(patch.get("highlight_color"), str) and re.fullmatch(r"#[0-9a-fA-F]{6}", patch["highlight_color"]):
                s["highlight_color"] = patch["highlight_color"]
            proj["settings"] = s
            project.save(self.root, proj)
            return s

    def hit_suggestion(self, clip_id: str) -> dict:
        self._need()
        from . import hits
        with self.lock:
            proj = project.load(self.root)
        clip = next((c for c in proj["clips"] if c["id"] == clip_id), None)
        if clip is None:
            raise KeyError(clip_id)
        return hits.suggest(self.root, clip)

    def frame_jpg(self, clip_id: str, t: float, width: int = 1280) -> bytes:
        """One frame of a clip's review copy (for the Mark plays page)."""
        self._need()
        src = proxy.proxy_path(self.cache, clip_id)
        if not src.exists():
            raise KeyError(clip_id)
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1",
                            "-vf", f"scale={width}:-2", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                           capture_output=True)
        if not r.stdout:
            raise KeyError(clip_id)
        return r.stdout

    def strip_jpg(self, clip_id: str, t0: float, t1: float, n: int) -> bytes:
        """n frames evenly from t0 to t1 side by side (the hit picker's filmstrip)."""
        self._need()
        src = proxy.proxy_path(self.cache, clip_id)
        if not src.exists():
            raise KeyError(clip_id)
        n = max(2, min(int(n), 30))
        span = max(0.2, t1 - t0)
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t0):.3f}", "-i", str(src), "-t", f"{span:.3f}",
                            "-vf", f"fps={n / span:.5f},scale=256:-2,tile={n}x1", "-frames:v", "1", "-q:v", "4",
                            "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True)
        if not r.stdout:
            raise KeyError(clip_id)
        return r.stdout

    def prepare(self, clip_id: str) -> None:
        """You're about to click your player (F): load the clip for tracking now, in the
        background, so the ring shows straight after the click."""
        from . import camera
        if self.root is None or not proxy.proxy_path(self.cache, clip_id).exists():
            return
        threading.Thread(target=camera.prepare, args=(self.cache, clip_id), daemon=True).start()

    def plan(self, clip_id: str, progressive: bool = False) -> dict:
        from . import camera
        self._need()
        with self.lock:
            proj = project.load(self.root)
        clip = next((c for c in proj["clips"] if c["id"] == clip_id), None)
        if clip is None:
            raise KeyError(clip_id)
        return camera.plan(self.cache, clip, progressive=progressive)

    def update(self, clip_id: str, patch: dict) -> dict:
        self._need()
        patch = {k: v for k, v in patch.items() if k in EDITABLE}
        with self.lock:
            proj = project.load(self.root)
            clip = next((c for c in proj["clips"] if c["id"] == clip_id), None)
            if clip is None:
                raise KeyError(clip_id)
            if "spotlight" in patch:
                patch["spotlight"] = _clean_anchors(patch["spotlight"])
            if "ring_off" in patch:
                patch["ring_off"] = _clean_ring_off(patch["ring_off"], clip["probe"]["duration"])
            if "side" in patch and patch["side"] not in ("O", "D", None):
                patch.pop("side")
            if "star" in patch:
                patch["star"] = bool(patch["star"])
            if "no_hit" in patch:
                patch["no_hit"] = bool(patch["no_hit"])
            if "hit" in patch:
                patch["hit"] = None if patch["hit"] is None else round(float(patch["hit"]), 3)
            if "zoom" in patch:
                z = float(patch["zoom"] or 0)
                patch["zoom"] = min(ZOOM_LEVELS, key=lambda v: abs(v - z))
            r = {**project.empty_review(), **(clip.get("review") or {}), **patch}
            dur = clip["probe"]["duration"]
            for k in ("snap", "end", "hit"):
                if r[k] is not None:
                    r[k] = round(min(max(float(r[k]), 0.0), dur), 3)
            if r["snap"] is not None and r["end"] is not None:
                r["in"] = round(max(0.0, r["snap"] - self.cfg.pad_before), 3)
                r["out"] = round(min(dur, r["end"] + self.cfg.pad_after), 3)
            clip["review"] = r
            project.save(self.root, proj)
            if {"hit", "no_hit", "side"} & set(patch):
                from . import hitmodel
                hitmodel.schedule()                           # relearn from your marks
            labeled = bool(r["labeled"])
            warning = None
            try:
                score.upsert_gt(self.root, clip["source"],
                                r["snap"] if labeled else None, r["end"] if labeled else None,
                                r.get("notes", ""), [c["source"] for c in proj["clips"]])
            except PermissionError:
                # Typically open in Excel. highlights.json already has it; score reads both.
                warning = f"{score.GT_FILENAME} is open in another program, so only highlights.json was updated"
            return {"review": r, "warning": warning}


def _clean_anchors(v) -> list[dict]:
    """Spotlight clicks: [{t, x, y, w, h}] normalised to the full frame."""
    out = []
    for a in v if isinstance(v, list) else []:
        try:
            a = {k: float(a[k]) for k in ("t", "x", "y", "w", "h")}
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= a["x"] <= 1 and 0 <= a["y"] <= 1 and 0 < a["w"] <= 1 and 0 < a["h"] <= 1 and a["t"] >= 0:
            out.append({k: round(val, 4) for k, val in a.items()})
    return sorted(out, key=lambda a: a["t"])


def _clean_ring_off(v, duration: float) -> list[list]:
    """Where the highlight is turned off: [[off, on or None (to the end)], ...], sorted and
    merged, within the clip."""
    spans = []
    for r in v if isinstance(v, list) else []:
        try:
            a = max(0.0, round(float(r[0]), 3))
            b = None if r[1] is None else round(float(r[1]), 3)
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if b is not None and b >= duration:
            b = None
        if a >= duration or (b is not None and b <= a):
            continue
        spans.append([a, b])
    out = []
    for a, b in sorted(spans, key=lambda r: r[0]):
        if out and (out[-1][1] is None or a <= out[-1][1]):
            if out[-1][1] is not None:
                out[-1][1] = None if b is None else max(out[-1][1], b)
            continue
        out.append([a, b])
    return out


def make_handler(app: ReviewApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                return self._get()
            except NoProject:
                return self._json({"error": "no project open"}, HTTPStatus.CONFLICT)

        def do_POST(self):
            try:
                return self._post()
            except NoProject:
                return self._json({"error": "no project open"}, HTTPStatus.CONFLICT)

        def _redirect(self, to: str):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", to)
            self.end_headers()

        def _get(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                page = "review.html" if app.root is not None else "start.html"
                return self._file(STATIC / page, "text/html; charset=utf-8")
            if path == "/projects":
                return self._file(STATIC / "start.html", "text/html; charset=utf-8")
            m = re.fullmatch(r"/static/((?:fonts/)?[A-Za-z0-9_.-]+\.(css|woff2|svg))", path)
            if m:   # shared theme (theme.css), bundled fonts and the app mark
                ctype = {"css": "text/css; charset=utf-8", "woff2": "font/woff2", "svg": "image/svg+xml"}[m.group(2)]
                return self._file(STATIC / m.group(1), ctype)
            if path == "/review":
                if app.root is None:
                    return self._redirect("/")
                return self._file(STATIC / "review.html", "text/html; charset=utf-8")
            if path == "/api/projects":
                return self._json(app.projects_info())
            if path == "/api/projects/setup":
                return self._json(app.setup)
            if path == "/api/music/status":
                return self._json(app.music_job)
            m = re.fullmatch(r"/api/music/suggest/([a-z0-9-]+)", path)
            if m:
                return self._json(app.music_suggestion(m.group(1)))
            if path == "/api/clips":
                return self._json(app.clips())
            if path in ("/guide", "/guide.html"):
                return self._file(STATIC / "guide.html", "text/html; charset=utf-8")
            if path in ("/mark", "/mark.html"):
                if app.root is None:
                    return self._redirect("/")
                return self._file(STATIC / "mark.html", "text/html; charset=utf-8")
            m = re.fullmatch(r"/api/clips/([a-z0-9-]+)/(frame|strip)\.jpg", path)
            if m:
                from urllib.parse import parse_qs
                q = {k: v[0] for k, v in parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "").items()}
                try:
                    body = (app.frame_jpg(m.group(1), float(q.get("t", 0))) if m.group(2) == "frame" else
                            app.strip_jpg(m.group(1), float(q.get("t0", 0)), float(q.get("t1", 1)), int(q.get("n", 12))))
                except (KeyError, ValueError):
                    return self.send_error(HTTPStatus.NOT_FOUND)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=300")
                self.end_headers()
                return self.wfile.write(body)
            if path in ("/players", "/players.html"):
                return self._file(STATIC / "players.html", "text/html; charset=utf-8")
            if path == "/api/players":
                return self._json(app.players_info())
            if path == "/api/branding":
                return self._json(app.branding_info())
            if path == "/media/branding/logo":
                f = profiles.asset(library.home(), library.branding()["logo"])
                if f is None:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                return self._file(f, {".png": "image/png", ".webp": "image/webp"}.get(f.suffix.lower(), "image/jpeg"))
            m = re.fullmatch(r"/api/players/([a-z0-9-]+)/(title|end|banner)\.png", path)
            if m or path == "/api/branding/card.png":
                try:
                    body = app.credit_png() if not m else app.card_png(m.group(1), m.group(2))
                except KeyError:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(body)
            m = re.fullmatch(r"/media/player/([a-z0-9-]+)/(logo|music)", path)
            if m:
                f = app.asset_path(m.group(1), m.group(2))
                if f is None:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                         ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac", ".wav": "audio/wav",
                         ".flac": "audio/flac", ".ogg": "audio/ogg"}
                return self._file(f, types.get(f.suffix.lower(), "application/octet-stream"))
            if path == "/api/reel":
                return self._json(app.reel_info())
            m = re.fullmatch(r"/api/clips/([a-z0-9-]+)/hit", path)
            if m:
                try:
                    return self._json(app.hit_suggestion(m.group(1)))
                except KeyError:
                    return self.send_error(HTTPStatus.NOT_FOUND)
            if path == "/api/hitmodel":
                from . import hitmodel
                m = hitmodel.load() or {}
                return self._json({k: m.get(k) for k in ("fitted", "n", "tol", "all", "O", "D")})
            if path == "/api/reel/status":
                return self._json(app.job)
            m = re.fullmatch(r"/media/compose/(demo-[a-z]+-\d+-v\d+\.wav)", path)
            if m:
                return self._file(library.home() / "cache" / "compose" / m.group(1), "audio/wav")
            if path == "/api/samples":
                from . import samples
                samples.warm(app.building)                # renders whatever's missing, in the background
                return self._json(samples.status())
            m = re.fullmatch(r"/media/samples/(intro|outro)/([a-z0-9.-]+\.(?:mp4|jpg))", path)
            if m:
                from . import samples
                f = samples.media_path(m.group(1), m.group(2))
                if f is None:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                return self._file(f, "video/mp4" if f.suffix == ".mp4" else "image/jpeg")
            m = re.fullmatch(r"/media/outro/(outro-[a-z0-9]+\.mp4)", path)
            if m:
                return self._file(library.home() / "cache" / "outro" / m.group(1), "video/mp4")
            m = re.fullmatch(r"/media/intro/(intro-[a-z0-9-]+\.mp4)", path)
            if m:
                app._need()
                return self._file(app.cache / "intro" / m.group(1), "video/mp4")
            m = re.fullmatch(r"/api/photos/(photo-[a-z0-9-]+)/thumb", path)
            if m:
                try:
                    return self._file(app.photo_thumb(m.group(1)), "image/jpeg")
                except Exception:  # noqa: BLE001 - unknown or unreadable photo
                    return self.send_error(HTTPStatus.NOT_FOUND)
            m = re.fullmatch(r"/media/photo/(photo-[a-z0-9-]+\.mp4)", path)
            if m:
                app._need()
                return self._file(app.cache / "photos" / m.group(1), "video/mp4")
            m = re.fullmatch(r"/media/segment/([a-z0-9-]+\.mp4)", path)
            if m:
                app._need()
                return self._file(app.cache / "segments" / m.group(1), "video/mp4")
            m = re.fullmatch(r"/media/reel/([A-Za-z0-9_.-]+\.mp4)", path)
            if m:
                app._need()
                return self._file(app.root / reel.OUT_DIRNAME / m.group(1), "video/mp4")
            m = re.fullmatch(r"/proxy/([a-z0-9-]+)\.mp4", path)
            if m:
                app._need()
                return self._file(proxy.proxy_path(app.cache, m.group(1)), "video/mp4")
            self.send_error(HTTPStatus.NOT_FOUND)

        def _post(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if self.path in ("/api/pick/folder", "/api/pick/music", "/api/music/use", "/api/music/youtube",
                             "/api/projects/create"):
                body = json.loads(self.rfile.read(n) or b"{}") or {}
                folder = body.get("folder") or (str(app.root) if app.root else None)
                try:
                    if self.path == "/api/pick/folder":
                        return self._json(app.pick_folder())
                    if not folder:
                        return self._json({"error": "choose the clips folder first"}, HTTPStatus.BAD_REQUEST)
                    if self.path == "/api/pick/music":
                        return self._json(app.pick_music(Path(folder)))
                    if self.path == "/api/music/use":
                        return self._json(app.use_music(Path(folder), Path(str(body.get("path") or ""))))
                    if self.path == "/api/music/youtube":
                        return self._json(app.start_youtube(Path(folder), str(body.get("url") or "")))
                    return self._json(app.create_project(Path(folder), body.get("player") or None,
                                                         body.get("music") or None))
                except ValueError as e:
                    return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            if self.path in ("/api/projects/open", "/api/projects/remove"):
                folder = (json.loads(self.rfile.read(n) or b"{}") or {}).get("folder")
                if not isinstance(folder, str) or not folder:
                    return self._json({"error": "folder missing"}, HTTPStatus.BAD_REQUEST)
                try:
                    if self.path.endswith("open"):
                        return self._json(app.open_project(Path(folder)))
                    return self._json(app.remove_project(Path(folder)))
                except ValueError as e:
                    return self._json({"error": str(e)}, HTTPStatus.CONFLICT)
            if self.path == "/api/branding":
                body = json.loads(self.rfile.read(n) or b"{}") or {}
                library.save_branding(body if isinstance(body, dict) else {})
                return self._json(app.branding_info())
            if self.path == "/api/reel/order":
                return self._json(app.set_order(json.loads(self.rfile.read(n) or b"{}").get("order", [])))
            if self.path == "/api/reel/reveal":
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    return self._json(app.reveal_reel(str(body.get("action")), bool(body.get("vertical")),
                                                      body.get("file") or None))
                except ValueError as e:
                    return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            if self.path == "/api/reel/build":
                return self._json(app.start_build())
            if self.path == "/api/players":
                try:
                    return self._json(app.save_players(json.loads(self.rfile.read(n) or b"{}").get("players")))
                except (ValueError, TypeError) as e:
                    return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            m = re.fullmatch(r"/api/upload/(logo|music|brandlogo)", self.path)
            if m:
                limit = 80 * 1024 * 1024 if m.group(1) == "music" else 15 * 1024 * 1024
                if n <= 0 or n > limit:
                    return self._json({"error": f"file too large (max {limit // 1024 // 1024} MB)"},
                                      HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                from urllib.parse import unquote
                name = unquote(self.headers.get("X-Filename", "upload"))
                data = self.rfile.read(n)
                try:
                    if m.group(1) in ("logo", "brandlogo"):
                        from io import BytesIO
                        from PIL import Image
                        try:
                            Image.open(BytesIO(data)).verify()     # reject non-images early
                        except Exception:  # noqa: BLE001
                            raise ValueError("that file isn't an image I can read - use a PNG, JPG or WEBP")
                    if m.group(1) == "music":
                        rel = app.store_music(name, data)
                    elif m.group(1) == "brandlogo":
                        rel = profiles.store_upload(library.home(), name, data, "logo")
                        library.save_branding({"logo": rel})
                    else:
                        rel = profiles.store_upload(library.home(), name, data, "logo")
                except NoProject:
                    raise
                except Exception as e:  # noqa: BLE001 - tell the page what was wrong
                    return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return self._json({"path": rel, "assets": profiles.asset_files(library.home())})
            if self.path == "/api/compose/demo":
                from . import composer
                body = json.loads(self.rfile.read(n) or b"{}")
                style = body.get("style") if body.get("style") in composer.STYLES else "rock"
                seed = int(body.get("seed", 1)) % 1000
                f = library.home() / "cache" / "compose" / f"demo-{style}-{seed}-v{composer.COMPOSER_VERSION}.wav"
                if not f.exists():
                    composer.render(f, style, composer.demo_timeline(style), seed)
                return self._json({"url": "/media/compose/" + f.name})
            if self.path == "/api/branding/preview":
                try:
                    return self._json({"url": "/media/outro/" + app.outro_preview()})
                except Exception as e:  # noqa: BLE001 - show the reason on the page
                    return self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            if self.path == "/api/intro/preview":
                if app.building():
                    return self._json({"error": "a reel is building - preview the intro when it's done"},
                                      HTTPStatus.CONFLICT)
                try:
                    return self._json({"url": "/media/intro/" + app.intro_preview()})
                except Exception as e:  # noqa: BLE001 - show the reason on the page
                    return self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            m = re.fullmatch(r"/api/photos/(photo-[a-z0-9-]+)/slide", self.path)
            if m:
                try:
                    return self._json({"url": "/media/photo/" + app.photo_slide(m.group(1))})
                except KeyError:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                except Exception as e:  # noqa: BLE001
                    return self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            m = re.fullmatch(r"/api/photos/(photo-[a-z0-9-]+)", self.path)
            if m:
                try:
                    return self._json(app.set_photo_keep(m.group(1), json.loads(self.rfile.read(n) or b"{}").get("keep", True)))
                except KeyError:
                    return self.send_error(HTTPStatus.NOT_FOUND)
            m = re.fullmatch(r"/api/clips/([a-z0-9-]+)/segment", self.path)
            if m:
                try:
                    return self._json({"url": "/media/segment/" + app.render_segment(m.group(1))})
                except KeyError:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                except Exception as e:  # noqa: BLE001
                    return self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            if self.path == "/api/settings":
                return self._json(app.update_settings(json.loads(self.rfile.read(n) or b"{}")))
            m = re.fullmatch(r"/api/clips/([a-z0-9-]+)/prepare", self.path)
            if m:
                app.prepare(m.group(1))
                return self._json({"ok": True})
            m = re.fullmatch(r"/api/clips/([a-z0-9-]+)/plan(\?progressive=1)?", self.path)
            if m:
                try:
                    return self._json(app.plan(m.group(1), progressive=bool(m.group(2))))
                except KeyError:
                    return self.send_error(HTTPStatus.NOT_FOUND)
            m = re.fullmatch(r"/api/clips/([a-z0-9-]+)", self.path)
            if not m:
                return self.send_error(HTTPStatus.NOT_FOUND)
            try:
                n = int(self.headers.get("Content-Length", 0))
                patch = json.loads(self.rfile.read(n) or b"{}")
                return self._json(app.update(m.group(1), patch))
            except KeyError:
                return self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, TypeError) as e:
                return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

        def _file(self, p: Path, ctype: str):
            """Static file with HTTP Range support (needed for video seeking)."""
            if not p.exists():
                return self.send_error(HTTPStatus.NOT_FOUND)
            size = p.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    return self.end_headers()
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                with p.open("rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass  # browser cancelled a range request while seeking

    return Handler


def serve(root: Path | None, port: int = 8765, open_browser: bool = True, jobs: int = 4) -> None:
    """root: open this project straight away; None starts on the project list."""
    app = ReviewApp(jobs)
    if root is not None:
        try:
            app.open_project(root)
        except ValueError as e:
            raise SystemExit(f"{e} - run `detect` on it first.")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    from . import samples
    samples.warm(app.building)                           # intro/ending style samples for the Build panel
    url = f"http://127.0.0.1:{port}/"
    print(f"review page: {url}   (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
