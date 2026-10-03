# Projects part 1 (shared library, migration, start screen, switching) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The app opens on a start screen listing projects; player profiles live in one shared list; Caleb_Footbal migrates automatically; the review page can switch projects.

**Architecture:** A new `hlv/library.py` owns the app folder (`~/HighlightReels`): `players.json` (existing `profiles` code, pointed at the app folder), `projects.json`, and migration from a project's old `players.json`. Music moves to `highlights.json` → `settings.music`. `ReviewApp` gets an optional current project and an `open_project(folder)` that re-points root, cache and config and restarts the tracking warm-up. A new `static/start.html` lists projects.

**Tech Stack:** Python 3.12 stdlib http.server, plain HTML/JS, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-projects-and-branding-design.md`

## Global Constraints
- App folder: `Path.home() / "HighlightReels"`, overridable with env var `HLV_HOME` (tests use a tmp dir).
- Logos live in `<app folder>/assets/` (the existing `profiles.store_upload` layout, relative paths `assets/<file>`). This replaces the spec's `players/<id>/` sub-folders, which was a simplification agreed while planning.
- The `music` field is removed from player profiles. `settings.music` holds a path relative to the project folder, or an absolute path.
- Never move, rename or delete clips, the old `players.json` or the old `assets/`.
- Migration must not change any play's `segment.segment_key`.
- Switching project is refused while a reel build is running.
- Keep `python -m hlv review <folder>` working.

## File map
- Create `hlv/library.py`: app folder, projects registry, migration.
- Modify `hlv/profiles.py`: drop `music` from fields/template/status/upload kinds.
- Modify `hlv/project.py`: add `music_path(root, settings)`.
- Modify `hlv/reel.py`, `hlv/cli.py`: players from `library.home()`, music from `project.music_path`.
- Modify `hlv/camera.py`: `warm(..., stop=...)` so a project switch ends the old warm-up.
- Modify `hlv/review_server.py`: optional project, projects API, start page routing, library-backed players.
- Create `hlv/static/start.html`; modify `hlv/static/review.html` (Switch project link) and `hlv/static/players.html` (remove song section).
- Modify `.claude/launch.json`: start with no folder.
- Tests: `tests/test_library.py`.

---

### Task 1: library module + music path + migration

**Files:** Create `hlv/library.py`, `tests/test_library.py`; modify `hlv/project.py`, `hlv/profiles.py`.

**Interfaces (produces):**
- `library.home() -> Path` (creates it)
- `library.projects() -> list[dict]` with `{"folder": str, "player": str|None, "last_opened": str}`
- `library.add_project(folder: Path, player: str|None) -> dict`, `library.touch(folder: Path) -> None`, `library.remove_project(folder: Path) -> None`, `library.find(folder: Path) -> dict|None`
- `library.migrate(root: Path) -> None` (idempotent)
- `project.music_path(root: Path, settings: dict) -> Path|None`

- [ ] Step 1: write failing tests (`tests/test_library.py`): registry add/find/remove/touch round trip with `HLV_HOME` set to `tmp_path`; adding the same folder twice keeps one entry; `music_path` resolves relative and absolute and returns None for missing; migration of a tmp project that has `players.json` (one player with `logo: assets/logo.png`, `music: assets/song.mp3`) and `highlights.json`: the player lands in the library with the logo copied to `<home>/assets/`, `settings.music == "assets/song.mp3"`, `settings.player` set, project registered; running migrate twice changes nothing; an id already in the library is not overwritten.
- [ ] Step 2: run, expect failures (module missing).
- [ ] Step 3: implement `library.py`, `project.music_path`, and drop `music` from `profiles.TEMPLATE`/`status`. `profiles.load` must still tolerate (ignore) a `music` key in old files, since `FIELDS` filters unknown keys.
- [ ] Step 4: run all tests and expect them to pass.
- [ ] Step 5: commit.

### Task 2: render path uses the library and project music

**Files:** modify `hlv/reel.py` (lines ~92, 120-121, 132), `hlv/cli.py` (`cmd_reel`), `hlv/review_server.py` (players/cards/assets use `library.home()`; logo upload goes to the library; music upload goes to `<project>/music/` and sets `settings.music`), `hlv/static/players.html` (remove the song block and its JS; keep logo).

- [ ] Step 1: write a script check (scratchpad): compute `segment.segment_key` for every Caleb_Footbal play using the current code and save it to JSON.
- [ ] Step 2: change `reel.build`: `lib = library.home()`; `pl = profiles.get(lib, ...)`; cards `graphics.*_card(lib, pl)`; `music = project.music_path(root, settings)`. `cmd_reel`: call `library.migrate(root)` first; the error text points to the Player profiles page.
- [ ] Step 3: server: `players_info`, `save_players`, `card_png`, `asset_path(logo)`, `reel_info` (players and logo from the library; `assets.music` from `project.music_path`), and `update_settings` player validation all use `library.home()`. `/media/player/<id>/music` is replaced by `/media/music` (the project's song).
- [ ] Step 4: run the migration on Caleb_Footbal, recompute the keys, and assert they're identical. Run pytest.
- [ ] Step 5: commit.

### Task 3: switchable project in the server

**Files:** modify `hlv/review_server.py`, `hlv/camera.py` (`warm` gains `stop=lambda: False`, checked before each clip and while waiting), `hlv/cli.py` (`review` dir optional), `.claude/launch.json`.

**Interfaces:** `ReviewApp(cfg_loader)`, `app.root: Path|None`, `app.open_project(folder: Path) -> dict` (raises `ValueError` with a message for: building, no `highlights.json`, folder missing), `app.projects_info() -> dict` with `{"current": str|None, "building": bool, "projects": [{folder, exists, player:{id,name,number}|None, clips, confirmed, last_reel: iso|None, logo_url}]}`.

- [ ] Step 1: refactor `ReviewApp.__init__` so the project is optional; `open_project` runs `library.migrate`, loads the config via `DetectConfig.load(root/CONFIG_FILENAME)`, builds proxies for missing clips (`proxy.make_all`) in a background thread, sets root/cache, `library.touch`, and stops/restarts the warm-up (generation counter).
- [ ] Step 2: endpoints: `GET /api/projects`, `POST /api/projects/open {folder}`, `POST /api/projects/remove {folder}`; `GET /` serves `start.html` when no project is open, otherwise `review.html`; `GET /projects` is always the start screen; `/review` serves review.html. Any project API called with no project open → 409 `{"error": "no project open"}`.
- [ ] Step 3: `serve(root|None, ...)`; `cmd_review` with `nargs="?"`; launch.json drops the folder argument.
- [ ] Step 4: pytest; start the server with no folder and check `/api/projects` lists Caleb_Footbal (registered by the migration in task 2); open it; check `/api/clips` returns 23.
- [ ] Step 5: commit.

### Task 4: start screen + switch link

**Files:** create `hlv/static/start.html`; modify `hlv/static/review.html` (sidebar: "◂ Projects" link, disabled with a tooltip while building), `hlv/static/players.html` (back link goes to the review page or start screen).

- [ ] Step 1: `start.html`: dark theme matching review.html; cards (logo, name #number, folder, "N clips · M confirmed", "last reel <date>" or "no reel yet"); missing folder → dim card + "folder not found" + Remove from list button; click → POST open → `location = '/review'`; a "+ New project" button, disabled with "coming next" until part 2.
- [ ] Step 2: verify in the browser: the start screen shows Caleb's card; clicking it opens the review page with all clicks; "◂ Projects" goes back; screenshot.
- [ ] Step 3: commit; update STATUS.md.
