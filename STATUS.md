# Project status: Highlight Reel Maker

_Last updated: 2026-10-02. This is the hand-off note for the next session; read it first, then README.md for usage details._

## What this is
A local Python tool that turns a folder of football clips into a finished highlight reel for Tom's son **Caleb Scott (#12, WR / DB, class of 2027)** and his friends. The clips are iPhone 4K video shot from the stands (lots of pan and zoom), plus Hudl screen recordings and one TV highlight. It runs on Windows 11 with Python 3.12 in `.venv`, and needs ffmpeg/ffprobe on PATH. Tom edits in **free DaVinci Resolve**, so there's no scripting there; the tool exports FCPXML instead.

Test project folder: `TestClips\Caleb_Footbal\` (23 clips across Scrimmage, Week 1, Week 2 and Week 5).

## How to run
```
.venv\Scripts\python -m hlv review "TestClips\Caleb_Footbal"     # main UI: http://127.0.0.1:8765
.venv\Scripts\python -m hlv reel   "TestClips\Caleb_Footbal"     # build the reel from the command line (same as the Build button)
.venv\Scripts\python -m pytest -q tests                          # 22 tests, all passing
```
In the Claude desktop app, `.claude/launch.json` has a `review` preview config (use `preview_start name=review`). **Restart the server after any Python change**; static HTML changes only need a page reload.

## Built so far (all working)
| Phase | What | Key files |
|---|---|---|
| 1 Detection | Snap/end detection by camera-compensated motion, plus whistle detection; VFR→CFR; pads ±1 s; spots tight Hudl clips and edited TV footage; scoring and parameter sweep against ground truth | `plays.py motion.py audio.py signals.py score.py` |
| 2 Review page | Confirm/correct snap and end (S/E/Enter); keep/leave-out tick boxes; 720p proxies | `review_server.py static/review.html proxy.py` |
| 2b Spotlight + zoom | F click/drag on Caleb starts a CSRT tracker; clicks act as **keyframes**; magnifier while picking; zoom levels 1.25/1.5/1.8× follow him; highlight styles ring/arrow/spot shadow (H) | `track.py camera.py autozoom.py` |
| 3 Player profiles | `/players` page: name, number, position, class, school, height/weight, contact, colours, logo and song upload with live card previews → `players.json` | `profiles.py static/players.html graphics.py` |
| 4 Reel builder | Build reel panel: drag order, transitions (crossfade/cut/dip), freeze intro at the snap with spot shadow, name banner (first/every/off), title and end cards, music bed, FCPXML for Resolve | `reel.py segment.py` |
| 4b Beat sync | Beat detection (~5 ms accurate); hit marked with K, auto-suggested, or N for "no hit"; hits land on strong beats (measured 22/22 within 1 frame); hit punch (zoom bump + flash); optional slow-mo ramp into the hit (optical-flow interpolation for 30 fps clips) | `beats.py hits.py slowmo.py` |

## Current data state (Caleb_Footbal)
- 23 clips, all confirmed. **22 are in the reel**: IMG_0761 is currently unticked, and it's unclear whether Tom meant that.
- **Spotlight clicks on 8 clips:** 0761, 0791 (3 clicks), 0869, 0881, 0926 (4), 0945, BigHit, ScreenRecording. The other 15 have none, so no freeze or highlight on those.
- **Hits marked** on 0761, 0869 and 0881; the rest use suggestions.
- **Profile:** Caleb Scott #12, with logo and song both found. The song is at `C:\Users\tom\Claude\HighLightVideoMaker\assets\music.mp3` (118.6 BPM); `players.json` points there by absolute path.
- **Settings:** ring highlight, crossfade, name banner on the first play only, freeze on, beat sync on, hit punch on, slow-mo off, game audio 5%.
- **Last full reel:** `reel\caleb_reel.mp4` (~3:52, 358 MB) plus `caleb_reel.fcpxml`. That build predates the profile and tracking fixes, so rebuild it.

## Open items / known issues
1. **Tracking needs more clicks on two plays:**
   - IMG_0945: night kickoff, Caleb blue on blue end-zone paint. Click him about every second, especially once he's on green.
   - IMG_0791: lost in the tackle pile at 5.0 s. Click once just after it.
2. **FCPXML import into Resolve has never been tested.** It uses cuts plus a music lane; transitions get added in Resolve with Ctrl+T.
3. **Detection accuracy.** Snap is good. End-of-play is weak on long plays such as kickoff returns, where spread-out players read as "play over". Cross-validation showed the parameter sweep overfits at about 20 clips, so the defaults are kept honest. Tom reviews every clip anyway.
4. **Ideas Tom hasn't asked for yet:** syncing the cuts between plays to the beat; a "first N seconds only" spot-shadow mode; a freeze-frame title with the player's photo.
5. Not a git repo yet. Suggest `git init` and a first commit before big changes. Ignore the media and caches listed in `.gitignore`, and also `reel/`, `previews/` and `assets/`.

## Lessons learned / gotchas (please read)
- **Memory:** the box has 16 GB. Never load whole clips into RAM unnecessarily; `camera.plan` only decodes frames when tracking or zoom needs them, and `frames.read_frames` pre-allocates. Renders run with `-j 3`.
- **Disk:** C: was down to 1.4 GB during one build (it's at 58 GB now). The reel folder hard-links segments rather than copying them. Old segment versions in `.hlv_cache\segments` can be deleted safely.
- **Excel locks `ground_truth.csv`.** Saving still works through `highlights.json`, and the page warns about it.
- **Tracking confidence:** colour similarity was tried and **removed**. On this footage (same team colours, white pants on both teams, sunset light) it hid the ring while tracking was correct. A "stuck on scenery" check was also tried and disabled, because it gave false alarms on blocks during pans. Current loss signals: the tracker fails, a jump bigger than the camera's own pan, an empty-grass box, or clicks disagreeing. Plus the keyframe path between clicks.
- **Exact timing:** pieces are trimmed by frame/sample number, and xfade offsets sit just under the exact frame time. Rounding to milliseconds previously pushed hits a frame or two off the beat.
- **Segment cache key** (`segment.segment_key`) includes everything that changes a render. The name-banner text only keys the play that shows the banner, so profile edits re-render one play. The accent colour keys all plays.
- **Dev workflow:** editing the big HTML through bash heredocs breaks on backticks and quotes. Write a small Python patch script to the scratchpad and run that.
- Tom tests by watching the result. Verify visually (contact sheets, rendered frames) and measure (beat offsets, tracking coverage) before claiming something works.
