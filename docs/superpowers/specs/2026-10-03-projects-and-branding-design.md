# Projects, shared players, music and ScottForge card: design

_2026-10-03. Approved in conversation with Tom; the next step is an implementation plan._

## Goal
Tom makes reels for more than one player (Caleb, Brody, …). Today the app opens a single clip folder chosen on the command line, and each folder has its own `players.json` and music. This work adds:

1. A **start screen** listing projects, plus a **New project** flow that picks a player, a clips folder and music.
2. **One shared player list** for the whole app.
3. **Music per project**: pick a file, or paste a YouTube link and have the app download it as MP3.
4. An optional **ScottForge closing card** at the end of each reel.

Photos in the reel (Brody's folder has JPG/HEIC files) is a **separate follow-up job** with its own design. In this work, photos are counted and shown but otherwise ignored.

## Decisions (from Tom)
- **One folder per player.** A project is one player plus one clips folder. Review data stays in the folder's `highlights.json`, as today. No sharing of footage between players.
- **Shared player list.** Players live in one app-wide list, and a project links to one player by id.
- **Folder picking** uses the native Windows folder dialog, opened by the local server.
- **ScottForge credit** is a closing card after the player's end card, not a watermark.

## Storage

### App folder: `C:\Users\tom\HighlightReels\` (`Path.home() / "HighlightReels"`)
Kept outside the repo, so code updates never touch it.
- `players.json`: `{"players": [ {id, name, number, position, class_year, school, height_weight, contact, logo, color, accent} ]}`. `logo` is relative to the app folder (`players/<id>/logo.png`). The `music` field moves to projects.
- `players/<id>/`: the player's logo file.
- `projects.json`: `{"projects": [ {"folder": "C:\\...\\Caleb_Footbal", "player": "caleb", "last_opened": "2026-10-03T…"} ]}`. The folder is the project's identity, and the same folder can't be added twice.
- `branding.json`: `{"logo": "branding/logo.png", "bg": "#0a0a0c", "accent": "#ff5a1f", "title": "Created by ScottForge", "url": "https://ScottForge.ai", "invite": "Want a highlight reel like this? Reach out at ScottForge.ai"}`.
- `branding/`: Tom's uploaded logo.

### Project folder (unchanged layout plus `music/`)
- Clips, `highlights.json`, `.hlv_cache/`, `reel/` exactly as today. Clips are never moved or renamed.
- `music/`: the project's song, picked (copied in) or downloaded.
- New keys in `highlights.json` → `settings`: `music` (path relative to the project folder) and `scottforge_card` (bool; defaults to true for new projects and false for migrated ones until Tom ticks it). `settings.player` already exists and is reused.

### Migration (first start, automatic, idempotent)
For `TestClips\Caleb_Footbal` (and any folder passed on the command line that still has a `players.json`):
- Copy each player in its `players.json` into the shared list (skip ids already there), copying the logo to `players/<id>/`.
- Set the project's `settings.music` to that player's old `music` path. The file stays where it is, at its absolute or relative location. No copy is needed, because the render only needs a readable path.
- Register the folder in `projects.json`.
- Leave the old `players.json` and `assets/` in place as a backup.
- **No re-render:** the segment cache key doesn't include the music or the logo path. Verify this (the logo is in the title/end card, not the segments).

## App behaviour

### Starting
- `python -m hlv review` with no folder shows the start screen. `.claude/launch.json` changes to this.
- `python -m hlv review <folder>` still opens that folder directly (registering it if it's new).

### Start screen (`/` when no project is open; also at `/projects`)
- One card per project: player name, number and logo; folder path; clip count; confirmed count; date of the last reel. Click a card to open it.
- If a project's folder no longer exists, its card shows "folder not found" with **Remove from list**. That only removes it from `projects.json`; files are never deleted.
- **+ New project** opens a three-step panel:
  1. **Player:** pick one from the shared list, or **Create new**, which opens the existing profile form (now backed by the shared list).
  2. **Clips folder:** **Browse…** opens the native folder dialog. The panel shows "N videos, M photos found (photos aren't used yet)". With 0 videos the panel says so and won't continue. If the folder is already a project, it offers **Open that project**.
  3. **Music:** **Browse…** opens the native file dialog (MP3/M4A/WAV), and the file is copied into `<project>/music/`. Or **YouTube link**: paste it and click **Download**, which saves `<project>/music/<title>.mp3` with progress. A failure shows a plain message (not a valid link / private or age-restricted / no internet) and the file option stays available. Pre-filled with the player's most recent project's song when there is one. Music can be skipped and set later in the Build reel panel.
- **Create** registers the project and runs setup in the background, with a progress bar: 720p proxies, then snap/end detection (the same code path as `hlv detect`). It then opens the review page.

### Review page
- The header gets a **Switch project** link to the start screen. It's disabled, with an explanation, while a reel is building.
- Opening a project re-points the server (`ReviewApp`) at that folder: clips, plans, the build and the background tracking warm-up (restarted for the new folder).
- The Build reel panel gets a **Music** row (current song, Browse…, YouTube link) and a **ScottForge card** tick box.

### Player profiles page
- Reads and writes the shared list. Logo upload goes to `players/<id>/`.
- The music field is removed (it's per project now).
- A new **Branding** section edits `branding.json` (logo upload, colours, title, URL, invite line) with a live card preview.

### ScottForge closing card
- 3.5 s, after the player's end card: dark background (`bg`), logo, title, URL in the accent colour, and the invite line. Fades in from the end card. Music continues underneath, and the music fade-out moves to the end of this card.
- Rendered like the existing end card (`graphics.py`), cached separately, so toggling it re-renders only the ending, not the play segments.
- Included in the FCPXML as its own clip on the main lane.
- Shown only when `settings.scottforge_card` is true and a branding logo or title exists.

## Components (new / changed)
| Unit | Responsibility |
|---|---|
| `hlv/library.py` (new) | App folder location; load/save `players.json`, `projects.json`, `branding.json`; migration from a per-folder `players.json`. Pure file I/O, unit-tested. |
| `hlv/pickers.py` (new) | Native folder/file dialogs via tkinter (topmost, run on its own thread, returns a path or None). |
| `hlv/music.py` (new) | Import a music file into a project; YouTube download via `yt-dlp` (Python package, new dependency) plus ffmpeg → MP3; URL validation; friendly error mapping. |
| `hlv/setup_job.py` (new) | Background new-project setup: proxies, then detection, reporting progress (reuses `cli._analyze_all` logic, moved somewhere importable). |
| `hlv/profiles.py` | Backed by `library` instead of the folder; `music` removed from fields. |
| `hlv/review_server.py` | `ReviewApp` can open/switch projects; new endpoints for projects, pickers, music, setup progress and branding; switching refused while building. |
| `hlv/reel.py`, `hlv/graphics.py` | Music from `settings.music`; ScottForge card render and placement; FCPXML clip. |
| `static/start.html` (new) | Start screen and New project steps. |
| `static/review.html`, `static/players.html` | Switch project link, Music row, ScottForge tick box; Branding section, music field removed. |

## Errors
- No videos in the folder → can't create. An unreadable video → listed as skipped during setup, and the rest continue.
- YouTube download failures map to one-line messages. Raw `yt-dlp` output goes to the server log only.
- A missing project folder → "folder not found" card. Missing music → the existing "music missing" warning in the Build panel.
- Folder dialog cancelled → nothing changes.

## Testing
- **Unit:**
  - `library` round trips
  - migration on a copy of Caleb_Footbal's `highlights.json` + `players.json`: player imported once, music set, project registered, running it twice changes nothing
  - YouTube URL validation and error mapping with `yt-dlp` mocked
  - ScottForge card timing in the reel plan, and the card disabled when no branding exists
- **Real footage:**
  - New project from `assets\brody`: setup finishes, plays are found, and photos are counted and ignored
  - Caleb_Footbal opens after migration with all clicks, hits and settings, and **migration alone causes no re-render**: `segment.segment_key` for every play is identical before and after migration (computed directly, not by building). Plays with 2+ clicks will still re-render once because of the 2026-10-03 tracking change; that's unrelated to this work.
  - Screenshot of the ScottForge card from a rendered reel

## Build order
1. `library` + migration + start screen + project switching.
2. New project steps: folder picker, setup job, music (file + YouTube).
3. Branding section + ScottForge closing card.

Photos in the reel: separate design afterwards.

## Note for Tom
Commercial songs downloaded from YouTube can get a reel muted or blocked if it's posted to YouTube or Instagram. Sending it to coaches or family directly is fine.
