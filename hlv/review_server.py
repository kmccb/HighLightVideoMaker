"""Local review page: confirm/correct snap & end per clip, keep/skip.

Saves into highlights.json `review` and (for confirmed clips) ground_truth.csv.
Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import json
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import profiles, project, proxy, reel, score, segment
from .config import DetectConfig
from .pipeline import CACHE_DIRNAME

STATIC = Path(__file__).parent / "static"
EDITABLE = {"snap", "end", "labeled", "keep", "notes", "spotlight", "zoom", "hit", "no_hit"}
ZOOM_LEVELS = (0, 1.25, 1.5, 1.8)


class ReviewApp:
    def __init__(self, root: Path, cfg: DetectConfig):
        self.root, self.cfg = root, cfg
        self.lock = threading.Lock()
        self.cache = root / CACHE_DIRNAME
        if not (root / project.FILENAME).exists():
            raise SystemExit(f"No {project.FILENAME} in {root} - run `detect` first.")
        profiles.ensure_template(root)
        self.job = {"state": "idle"}

    def clips(self) -> dict:
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
        return {"pad_before": self.cfg.pad_before, "pad_after": self.cfg.pad_after,
                "settings": project.settings(proj), "clips": out}

    # ---------------- reel ----------------
    def reel_info(self) -> dict:
        with self.lock:
            proj = project.load(self.root)
        s = project.settings(proj)
        players = profiles.load(self.root)
        pl = profiles.get(self.root, s.get("player"))
        rows = []
        for c in reel.ordered_clips(proj):
            r = {**project.empty_review(), **(c.get("review") or {})}
            t0, t1 = project.final_window(c)
            spot = bool(r.get("spotlight"))
            rows.append({
                "id": c["id"], "source": c["source"], "group": c.get("group", ""), "kind": c["kind"],
                "included": reel.included(c), "keep": r.get("keep"), "labeled": bool(r.get("labeled")),
                "spotlight": spot, "zoom": r.get("zoom") or 0, "hit_set": r.get("hit") is not None, "no_hit": bool(r.get("no_hit")),
                "seconds": round(max(0.0, t1 - max(0.0, t0)) + (segment.FREEZE_S if spot and s.get("freeze") else 0), 2),
            })
        return {
            "settings": s, "players": players, "player": pl,
            "assets": profiles.status(self.root, pl) if pl else None,
            "profile_path": str(profiles.path(self.root)), "assets_dir": str(self.root / profiles.ASSETS),
            "transitions": list(reel.TRANSITIONS), "card_s": reel.TITLE_S + reel.END_S,
            "clips": rows, "job": self.job,
        }

    # ---------------- player profiles ----------------
    def players_info(self) -> dict:
        players = profiles.load(self.root)
        with self.lock:
            s = project.settings(project.load(self.root))
        return {"players": players, "selected": s.get("player") or (players[0]["id"] if players else None),
                "status": {p["id"]: profiles.status(self.root, p) for p in players},
                "assets": profiles.asset_files(self.root), "fields": profiles.FIELDS,
                "assets_dir": str(self.root / profiles.ASSETS)}

    def save_players(self, players: list) -> dict:
        if not isinstance(players, list):
            raise ValueError("players must be a list")
        with self.lock:
            clean = profiles.save(self.root, [p for p in players if isinstance(p, dict)])
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
        pl = profiles.get(self.root, player_id)
        if pl is None:
            raise KeyError(player_id)
        if which == "title":
            img = graphics.title_card(self.root, pl)
        elif which == "end":
            img = graphics.end_card(self.root, pl)
        else:   # banner over a dim field-ish background, cropped to the lower third
            img = np.full((graphics.H, graphics.W, 3), (40, 70, 45), np.uint8)
            graphics.composite(img, graphics.name_banner(pl), 1.0)
            img = img[780:1020, 0:1100]
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".png", img)[1].tobytes()

    def asset_path(self, player_id: str, kind: str) -> Path | None:
        pl = profiles.get(self.root, player_id)
        return profiles.asset(self.root, pl[kind]) if pl and kind in ("logo", "music") else None

    def set_order(self, order: list) -> dict:
        with self.lock:
            proj = project.load(self.root)
            ids = {c["id"] for c in proj["clips"]}
            s = project.settings(proj)
            s["order"] = [i for i in order if isinstance(i, str) and i in ids]
            proj["settings"] = s
            project.save(self.root, proj)
        return {"order": s["order"]}

    def render_segment(self, clip_id: str) -> str:
        with self.lock:
            proj = project.load(self.root)
        clip = next((c for c in proj["clips"] if c["id"] == clip_id), None)
        if clip is None:
            raise KeyError(clip_id)
        s = project.settings(proj)
        pl = profiles.get(self.root, s.get("player"))
        banner = clip_id in reel.banner_ids([c for c in reel.ordered_clips(proj) if reel.included(c)], s)
        return segment.render(self.root, clip, pl, {**s, "_banner": banner}).name

    def start_build(self) -> dict:
        if self.job.get("state") == "running":
            return self.job
        self.job = {"state": "running", "step": "starting", "done": 0, "total": 1, "result": None, "error": None}

        def progress(step, done, total):
            self.job.update(step=step, done=done, total=total)

        def run():
            try:
                with self.lock:
                    s = project.settings(project.load(self.root))
                res = reel.build(self.root, s.get("player"), s.get("transition"), progress=progress)
                self.job.update(state="done", result=res, step="done")
            except Exception as e:  # noqa: BLE001 - report any failure to the page
                self.job.update(state="error", error=str(e))
        threading.Thread(target=run, daemon=True).start()
        return self.job

    def update_settings(self, patch: dict) -> dict:
        with self.lock:
            proj = project.load(self.root)
            s = project.settings(proj)
            if patch.get("transition") in reel.TRANSITIONS:
                s["transition"] = patch["transition"]
            if "freeze" in patch:
                s["freeze"] = bool(patch["freeze"])
            if patch.get("name_banner") in reel.BANNER_MODES:
                s["name_banner"] = patch["name_banner"]
            if "slowmo_speed" in patch and float(patch["slowmo_speed"]) in (0.5, 0.4, 0.3):
                s["slowmo_speed"] = float(patch["slowmo_speed"])
            for k in ("beat_sync", "hit_fx", "slowmo"):
                if k in patch:
                    s[k] = bool(patch[k])
            if "music_start" in patch:
                s["music_start"] = max(0.0, float(patch["music_start"] or 0))
            if "game_audio" in patch:
                s["game_audio"] = float(min(max(float(patch["game_audio"]), 0.0), 1.0))
            if "player" in patch and any(p["id"] == patch["player"] for p in profiles.load(self.root)):
                s["player"] = patch["player"]
            if patch.get("highlight") in project.HIGHLIGHT_STYLES:
                s["highlight"] = patch["highlight"]
            if isinstance(patch.get("highlight_color"), str) and re.fullmatch(r"#[0-9a-fA-F]{6}", patch["highlight_color"]):
                s["highlight_color"] = patch["highlight_color"]
            proj["settings"] = s
            project.save(self.root, proj)
            return s

    def hit_suggestion(self, clip_id: str) -> dict:
        from . import hits
        with self.lock:
            proj = project.load(self.root)
        clip = next((c for c in proj["clips"] if c["id"] == clip_id), None)
        if clip is None:
            raise KeyError(clip_id)
        return hits.suggest(self.root, clip)

    def plan(self, clip_id: str) -> dict:
        from . import camera
        with self.lock:
            proj = project.load(self.root)
        clip = next((c for c in proj["clips"] if c["id"] == clip_id), None)
        if clip is None:
            raise KeyError(clip_id)
        return camera.plan(self.cache, clip)

    def update(self, clip_id: str, patch: dict) -> dict:
        patch = {k: v for k, v in patch.items() if k in EDITABLE}
        with self.lock:
            proj = project.load(self.root)
            clip = next((c for c in proj["clips"] if c["id"] == clip_id), None)
            if clip is None:
                raise KeyError(clip_id)
            if "spotlight" in patch:
                patch["spotlight"] = _clean_anchors(patch["spotlight"])
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
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._file(STATIC / "review.html", "text/html; charset=utf-8")
            if path == "/api/clips":
                return self._json(app.clips())
            if path in ("/players", "/players.html"):
                return self._file(STATIC / "players.html", "text/html; charset=utf-8")
            if path == "/api/players":
                return self._json(app.players_info())
            m = re.fullmatch(r"/api/players/([a-z0-9-]+)/(title|end|banner)\.png", path)
            if m:
                try:
                    body = app.card_png(m.group(1), m.group(2))
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
            if path == "/api/reel/status":
                return self._json(app.job)
            m = re.fullmatch(r"/media/segment/([a-z0-9-]+\.mp4)", path)
            if m:
                return self._file(app.cache / "segments" / m.group(1), "video/mp4")
            m = re.fullmatch(r"/media/reel/([A-Za-z0-9_.-]+\.mp4)", path)
            if m:
                return self._file(app.root / reel.OUT_DIRNAME / m.group(1), "video/mp4")
            m = re.fullmatch(r"/proxy/([a-z0-9-]+)\.mp4", path)
            if m:
                return self._file(proxy.proxy_path(app.cache, m.group(1)), "video/mp4")
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if self.path == "/api/reel/order":
                return self._json(app.set_order(json.loads(self.rfile.read(n) or b"{}").get("order", [])))
            if self.path == "/api/reel/build":
                return self._json(app.start_build())
            if self.path == "/api/players":
                try:
                    return self._json(app.save_players(json.loads(self.rfile.read(n) or b"{}").get("players")))
                except (ValueError, TypeError) as e:
                    return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            m = re.fullmatch(r"/api/upload/(logo|music)", self.path)
            if m:
                limit = 80 * 1024 * 1024 if m.group(1) == "music" else 15 * 1024 * 1024
                if n <= 0 or n > limit:
                    return self._json({"error": f"file too large (max {limit // 1024 // 1024} MB)"},
                                      HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                from urllib.parse import unquote
                name = unquote(self.headers.get("X-Filename", "upload"))
                data = self.rfile.read(n)
                try:
                    if m.group(1) == "logo":
                        from io import BytesIO
                        from PIL import Image
                        try:
                            Image.open(BytesIO(data)).verify()     # reject non-images early
                        except Exception:  # noqa: BLE001
                            raise ValueError("that file isn't an image I can read - use a PNG, JPG or WEBP")
                    rel = profiles.store_upload(app.root, name, data, m.group(1))
                except Exception as e:  # noqa: BLE001 - tell the page what was wrong
                    return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return self._json({"path": rel, "assets": profiles.asset_files(app.root)})
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
            m = re.fullmatch(r"/api/clips/([a-z0-9-]+)/plan", self.path)
            if m:
                try:
                    return self._json(app.plan(m.group(1)))
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


def serve(root: Path, cfg: DetectConfig, port: int = 8765, open_browser: bool = True,
          jobs: int = 4) -> None:
    from .probe import find_clips
    from .signals import clip_id
    app = ReviewApp(root, cfg)
    proxy.make_all([(p, proxy.proxy_path(app.cache, clip_id(root, p))) for p in find_clips(root)], jobs)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"review page: {url}   (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
