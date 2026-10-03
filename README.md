# Highlight Reel Maker

Turns a folder of football clips into a highlight reel. **Phase 1 (done):** snap / end-of-play detection.

## Setup
```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
Needs `ffmpeg` / `ffprobe` on PATH.

## Use
```
.venv\Scripts\python -m hlv detect  "TestClips\Caleb_Footbal" --plots   # -> highlights.json (+ debug\*.png)
.venv\Scripts\python -m hlv review  "TestClips\Caleb_Footbal"           # review page: confirm/correct snap & end
.venv\Scripts\python -m hlv score   "TestClips\Caleb_Footbal" [--sweep] # accuracy; --sweep tunes + saves hlv_config.json
.venv\Scripts\python -m hlv zoom-preview "TestClips\Caleb_Footbal" IMG_0761.MOV  # auto-zoom prototype -> previews\
.venv\Scripts\python -m hlv reel   "TestClips\Caleb_Footbal"            # build the finished reel (or use the Build reel button)
.venv\Scripts\python -m hlv normalize "TestClips\Caleb_Footbal"         # optional CFR H.264 copies for Resolve
```
First `detect` decodes every clip (about 3 min for 23 4K clips). Results are cached in `.hlv_cache\`, so re-runs, scoring and sweeps take seconds.

**Review page** (`review`): opens http://127.0.0.1:8765. The first run makes 720p proxies (about 2 min). Each clip opens just before the auto-detected snap.
- Step through it: `←` / `→` moves one frame, `Shift` moves 0.5s, `1` / `2` / `3` changes speed.
- Mark the play: `S` / `E` set snap / end at the playhead, or drag the handles on the timeline.
- Check and decide: `P` previews the trimmed clip, `X` toggles keep / skip.
- `Enter` confirms and moves to the next clip.

Confirmed clips are saved in `highlights.json` (`review`) and mirrored to `ground_truth.csv`. `score` uses them automatically. Only clips you confirm count as ground truth. If the CSV is open in Excel, saving still works and the page tells you the CSV wasn't updated.

**Debug plots** (`debug\*.png`): the top panel shows player motion after camera compensation (blue) and camera pan (orange). The middle panel shows the combined activity curve. The bottom panel shows the whistle score. Solid lines are what was detected, dashed lines are your ground truth, and the green band is the proposed in/out window.

## How it works
1. **Probe:** ffprobe reads fps, VFR, rotation and audio.
2. **CFR decode:** ffmpeg `fps=30` produces a 640px proxy, so frame *i* = *i*/30 s on the source timeline.
3. **Camera compensation:** grid features are tracked with LK flow and a RANSAC similarity transform is fitted (pan/zoom/roll). The previous frame is warped onto the current one, and what still moves *on the field* (green mask) counts as players.
4. **Whistle:** a tonal peak at 2.8–4.5 kHz that stands out from broadband crowd noise.
5. **Snap:** the biggest burst of activity that follows a quiet "set". **End:** activity settles after the peak, or a whistle overrides it.
6. **Clip kinds:**
   - `iphone` gets 1s padding on each side.
   - `screen_recording` (Hudl): the video rectangle is auto-cropped and the clip is trimmed only lightly.
   - `pre_edited` (TV packages, detected from hard cuts) is passed through whole as `needs_review`.

## highlights.json
Each clip has `probe`, `kind`, `crop`, `detection`, `proposed {in,out}`, `status`, and `review`. The `review` block belongs to the Phase 2 review page, and re-running `detect` never overwrites it. Later phases use `review` values where set and `proposed` otherwise (`hlv.project.final_window`).

## Spotlight + zoom (review page)
- **F**, then click your player on a paused frame, or drag a box around him. A CSRT tracker follows him forward and backward from every click.
- **Hard plays** (night, far away, blue on blue paint): drag a tight box using the **magnifier** that appears in spotlight mode, and click him again every second or so. Clicks act as keyframes: between two clicks the ring never strays far from the path joining them.
- **Check the tracking.** The strip under the timeline shows tracking quality: green is solid, amber is unsure, red means lost. Where it goes red, scrub there and click him again. Each extra click re-anchors the tracker. **Del** removes the click nearest the playhead.
- **Z** cycles zoom: off, 1.25x, 1.5x, 1.8x. With a spotlight, the camera follows him. Without one, it frames the play gently (max 1.4x). Wherever tracking confidence drops, the camera eases back to wide instead of guessing.
- **H** picks the highlight style for the whole reel: ring (glowing halo at his feet), arrow (marker above his head), spot shadow (everything dims except a pool of light on him), or off. It fades out wherever tracking is unsure. Plain clicks are auto-fitted to the player, and dragged boxes are tightened to him. Box size is held to what you marked, so the highlight doesn't drift in pile-ups.
- **V** switches between the full frame (yellow camera box + ring under the player) and the zoomed preview.
- Clicks are saved as `review.spotlight` ([{t, x, y, w, h}] normalised) and the level as `review.zoom`. The render uses the same planner as the page (`hlv/camera.py`): `zoom-preview <dir> IMG_0761.MOV` writes `previews\<clip>_zoom.mp4` and a side-by-side `_compare.mp4`.

## Reel builder
1. **Set up the player** on the **Player profiles** page (sidebar link, or http://127.0.0.1:8765/players): name, number, position, class year, school, height/weight, contact line, team and accent colours. **Choose image…** and **Choose song…** copy the logo and song into `assets\`. Live previews show the title card, name banner and end card. Add a player per kid; everything is saved to `players.json`.
2. **Open the Build reel panel** (in the review page sidebar):
   - pick the player and the transition (crossfade, cut or dip to black),
   - turn the freeze intro on or off and set the game-audio level under the music,
   - drag plays into order and untick any to leave them out,
   - use **Preview** on a play to see its finished render.
3. **Press Build reel.** Each play is rendered at 1080p30 with its trim, zoom and highlight. Plays with a spotlight click also get a freeze intro at the snap (spot shadow, slight push-in, name banner). Then come the title card, the plays, the end card, the transitions, and the music bed with a fade in and out. Rendered plays are cached, so rebuilding after a re-order or transition change is quick.
4. **Output** goes to `reel\<player>_reel.mp4`, plus `reel\<player>_reel.fcpxml` for DaVinci Resolve (File > Import > Timeline). The Resolve timeline uses the rendered plays as cuts with the music on its own track; add transitions in Resolve with Ctrl+T.

## Beat sync
- **No hit:** press **N** or the **✕** on the Hit button for plays with nothing to sync (no punch, no slow-mo, natural timing).
- **Mark the hit** (tackle, catch, big block) with **K** on the review page. **J** jumps to it and **Shift+K** goes back to the suggestion. Unmarked plays use a suggestion: with a spotlight, the moment your player's sudden stop *begins*; otherwise the motion peak. The suggestion shows as a dashed orange line.
- **Beat detection** (`hlv/beats.py`) finds the song's tempo and beat grid, weighted to the kick and bass, and flags the strongest beats (the bar's "1"). On synthetic tracks it is accurate to about 5 ms. Results are cached per song.
- **Fitting:** each play is rendered once with extra lead-in, tail and a 2.2 s freeze. The builder then picks, play by play, the strongest reachable beat for its hit. It adjusts the lead-in (0.4-2.5 s) and the freeze (1.0-2.2 s) so the hit lands on it. Re-timing never re-renders, so a new song or order rebuilds quickly.
- **Hit punch:** a 0.3 s zoom bump and a white flash on the hit frame.
- **Reel panel options:** Sync hits to beat, Hit punch, and Music starts at (to skip a slow intro). After a build it reports how many hits landed on the beat.

## Slow-mo into the hit (option)
- Turn it on in the Reel panel with **Slow-mo into hit** and pick a speed: 0.5x, 0.4x or 0.3x. It's off by default.
- **Speed curve:** full speed, then a 0.5 s smooth ramp down, then slow motion from 0.4 s before the hit to 0.25 s after it, then a 0.35 s ramp back to full speed.
- **Smoothness:** 60/120 fps clips use their real extra frames. 30 fps clips get optical-flow in-between frames (`hlv/slowmo.py`), so there is no stutter.
- **Audio:** game audio is time-stretched through the slow section, so the hit sound stays on the hit.
- **Beat sync** accounts for the stretch, so the slowed hit still lands on the beat.
