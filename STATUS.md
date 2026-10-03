# Project status: Highlight Reel Maker

_Last updated: 2026-10-03. This is the hand-off note for the next session; read it first, then README.md for usage details._

## What this is
A local Python tool that turns a folder of football clips into a finished highlight reel for Tom's son **Caleb Scott (#12, WR / DB, class of 2027)** and his friends. The clips are iPhone 4K video shot from the stands (lots of pan and zoom), plus Hudl screen recordings and one TV highlight. It runs on Windows 11 with Python 3.12 in `.venv`, and needs ffmpeg/ffprobe on PATH. Tom edits in **free DaVinci Resolve**, so there's no scripting there; the tool exports FCPXML instead.

Test project folder: `TestClips\Caleb_Footbal\` (23 clips across Scrimmage, Week 1, Week 2 and Week 5).

## How to run
```
.venv\Scripts\python -m hlv review "TestClips\Caleb_Footbal"     # main UI: http://127.0.0.1:8765
.venv\Scripts\python -m hlv reel   "TestClips\Caleb_Footbal"     # build the reel from the command line (same as the Build button)
.venv\Scripts\python -m pytest -q tests                          # 24 tests, all passing
```
In the Claude desktop app, `.claude/launch.json` has a `review` preview config (use `preview_start name=review`). **Restart the server after any Python change**; static HTML changes only need a page reload. **Never restart while a reel is building** (check `/api/reel/status`); the build runs inside the server process.

**Git:** private repo https://github.com/kmccb/HighLightVideoMaker (branch `main`). Media, caches, `reel/`, `previews/` and `assets/` are git-ignored, because the footage and song are personal.

## Built so far (all working)
| Phase | What | Key files |
|---|---|---|
| 1 Detection | Snap/end detection by camera-compensated motion, plus whistle detection; VFR→CFR; pads ±1 s; spots tight Hudl clips and edited TV footage; scoring and parameter sweep against ground truth | `plays.py motion.py audio.py signals.py score.py` |
| 2 Review page | Confirm/correct snap and end (S/E/Enter); keep/leave-out tick boxes; 720p proxies | `review_server.py static/review.html proxy.py` |
| 2b Spotlight + zoom | F click/drag on Caleb starts a CSRT tracker; clicks act as **keyframes**; magnifier while picking; zoom levels 1.25/1.5/1.8× follow him; highlight styles ring/arrow/spot shadow (H) | `track.py camera.py autozoom.py` |
| 3 Player profiles | `/players` page: name, number, position, class, school, height/weight, contact, colours, logo and song upload with live card previews → `players.json` | `profiles.py static/players.html graphics.py` |
| 4 Reel builder | Build reel panel: drag order, transitions (crossfade/cut/dip), freeze intro at the snap with spot shadow, name banner (first/every/off), title and end cards, music bed, FCPXML for Resolve | `reel.py segment.py` |
| 2c Tracking speed + feedback (2026-10-03) | "Tracking your player…" badge on the video, and the stale ring is hidden while re-tracking. Per-click tracker runs are reused, so a new click only tracks between its neighbouring clicks (~1-2 s instead of 6-19 s); runs are saved to `.hlv_cache\track\<id>-runs-r1.pkl`. Tracking is limited to the renderable window (`segment.extents`). On server start a background warm-up tracks every clicked clip (paused while a build runs), so clicked clips open in <0.5 s | `camera.py track.py review_server.py` |
| 2d Better tracking in crowds (2026-10-03) | Between two clicks, frames where the forward and backward runs meet, and the stretches before/after they meet, are trusted and skip the straight-path check. That check wrongly rejected correct answers during camera pans. Held-out-click benchmark: within 1 player-width 78%→93% (every other click kept), 64%→78% (every 3rd kept). Strip under the timeline goes amber beyond 1.5 s from the outer clicks | `track.py static/review.html` |
| 4b Beat sync | Beat detection (~5 ms accurate); hit marked with K, auto-suggested, or N for "no hit"; hits land on strong beats (measured 22/22 within 1 frame); hit punch (zoom bump + flash); optional slow-mo ramp into the hit (optical-flow interpolation for 30 fps clips) | `beats.py hits.py slowmo.py` |

## Current data state (Caleb_Footbal)
- 23 clips, all confirmed. **22 are in the reel**: IMG_0761 is currently unticked, and it's unclear whether Tom meant that.
- **Spotlight clicks on 12 clips:** 0954 (18), 0945 (14), 6590 (12), BigHit (8), 0926 (7), 0791 (3), 1420 (3), and one each on 0761, 0869, 0881, 1126 and ScreenRecording. The other 11 have none, so no freeze or highlight on those.
- **Hits marked** on 0761, 0869, 0881 and 0945; 0954 is set to "no hit"; the rest use suggestions.
- **Profile:** Caleb Scott #12, with logo and song both found. The song is at `C:\Users\tom\Claude\HighLightVideoMaker\assets\music.mp3` (118.6 BPM); `players.json` points there by absolute path.
- **Settings:** ring highlight, crossfade, name banner on the first play only, freeze on, beat sync on, hit punch on, **slow-mo on (0.4×)**, game audio 5%.
- **Last full reel (2026-10-03):** `reel\caleb_reel.mp4` (22 plays, 4:25, 371 MB) plus `caleb_reel.fcpxml`; 21 of 22 plays on the beat. The build took ~24 min. It was built **before** the crowd-tracking change was live in the server, so plays with 2+ clicks will re-render on the next build.

## Open items / known issues
1. **Tracking still loses him on very fast pans** (e.g. IMG_0954 ~14-15 s): both runs swap players, so nothing can confirm him there and a click is needed. Past the last click the tracker is on its own and can jump players within about a second (the amber strip shows where). The next step for fewer clicks would be a better tracker or re-detection, not more fusion rules. The benchmark is in the session scratchpad: it hides every 2nd/3rd click, measures distance to the hidden clicks in player-widths, and uses `track.py` from commit 7a1d935 as the "before".
   - First click on an unclicked clip still tracks from scratch (10-20 s on a long play).
   - IMG_0945: Tom asked about the ring before the first click. That's backward tracking (it feeds the freeze at the snap). Suggested: click Caleb at the snap.
2. **FCPXML import into Resolve has never been tested.** It uses cuts plus a music lane; transitions get added in Resolve with Ctrl+T.
3. **Detection accuracy.** Snap is good. End-of-play is weak on long plays such as kickoff returns, where spread-out players read as "play over". Cross-validation showed the parameter sweep overfits at about 20 clips, so the defaults are kept honest. Tom reviews every clip anyway.
4. **Ideas Tom hasn't asked for yet:** syncing the cuts between plays to the beat; a "first N seconds only" spot-shadow mode; a freeze-frame title with the player's photo.
5. **IMG_0761 is still unticked**, so it's out of the reel. Unclear if intended.

## Lessons learned / gotchas (please read)
- **Memory:** the box has 16 GB. Never load whole clips into RAM unnecessarily; `camera.plan` only decodes frames when tracking or zoom needs them, and `frames.read_frames` pre-allocates. Renders run with `-j 3`.
- **Disk:** C: was down to 1.4 GB during one build (it's at 58 GB now). The reel folder hard-links segments rather than copying them. Old segment versions in `.hlv_cache\segments` can be deleted safely.
- **Excel locks `ground_truth.csv`.** Saving still works through `highlights.json`, and the page warns about it.
- **Tracking confidence:** colour similarity was tried and **removed**. On this footage (same team colours, white pants on both teams, sunset light) it hid the ring while tracking was correct. A "stuck on scenery" check was also tried and disabled, because it gave false alarms on blocks during pans. Current loss signals: the tracker fails, a jump bigger than the camera's own pan, an empty-grass box, or clicks disagreeing. Plus the keyframe path between clicks.
- **Track caches:** the raw track `.npy` key is `"v13"` + anchors + play window (`camera.player_track`). Bump it when the fusion in `track_player` changes. Bump `camera.RUNS_VERSION` when `track._run` changes, so saved runs are dropped. Renders re-key automatically because `segment_key` hashes the plan.
- **Track coordinates are centres:** `plan.player` rows are `cx, cy, w, h, conf`, not top-left. A debug sheet drawn from the corner looks wrong (this misled one investigation).
- **Exact timing:** pieces are trimmed by frame/sample number, and xfade offsets sit just under the exact frame time. Rounding to milliseconds previously pushed hits a frame or two off the beat.
- **Segment cache key** (`segment.segment_key`) includes everything that changes a render. The name-banner text only keys the play that shows the banner, so profile edits re-render one play. The accent colour keys all plays.
- **Dev workflow:** editing the big HTML through bash heredocs breaks on backticks and quotes. Write a small Python patch script to the scratchpad and run that.
- Tom tests by watching the result. Verify visually (contact sheets, rendered frames) and measure (beat offsets, tracking coverage) before claiming something works.
