# Photos in the reel: design

_2026-10-03. Approved in conversation with Tom; the next step is an implementation plan._

## Goal
Put the player's photos into the reel as slow pan-and-zoom ("Ken Burns") slides, mixed in with the plays. Brody's folder (`assets\brody`) has JPG, HEIC and PNG photos; Caleb's test folder has none.

## Decisions (from Tom)
- **Found automatically and listed in the app.** Photos in the project folder are picked up on their own and appear in the Build panel with a tick box, like plays. Unticked = left out.
- **Mixed in, and movable.** By default each photo is placed **by date** next to the plays from the same day. Tom can drag / ↑↓ them anywhere.
- **Not rushed.** Default 4 s on screen, adjustable.
- **Tall photos: blurred fill.** The whole photo in the middle, with a blurred, darkened copy filling the sides.

## Finding photos
- `photos.find(root)` lists files with `setup_job.PHOTO_EXTS` (`.jpg .jpeg .png .heic .heif .webp`), skipping the same folders as `probe.find_clips` (`.hlv_cache`, `reel`, `music`, `previews`, `debug`, `normalized`, `assets`). `setup_job.scan` uses the same function so the counts match.
- Run whenever a project is loaded by the server (cheap: a file walk plus EXIF reads for new files only). New photos are added to `highlights.json`, and photos whose file is gone are dropped. No re-setup needed.
- **HEIC** is read with `pillow-heif` (new dependency in `requirements.txt`). A photo that can't be decoded is skipped with a warning in the log and not listed.
- A Live Photo's still (HEIC/JPG) is a normal photo; its movie is already left out of the plays by `probe.is_live_photo`.

### Stored in `highlights.json`
A new top-level list beside `clips`:
```json
"photos": [ {"id": "photo-img-2795", "source": "IMG_2795.JPG", "group": "", "taken": "2026-09-12T19:04:11", "keep": true} ]
```
- `id`: `"photo-" + clip_id(path)` (same slug rule as plays, with a prefix so it can't collide with a play id).
- `taken`: EXIF `DateTimeOriginal` (PIL `getexif`, HEIC via pillow-heif), else the file's modified time. ISO string, local time.
- `keep`: default `true`.

### Dates for plays
By-date placement also needs a date for each play. `probe` gains `taken`: the ffprobe `com.apple.quicktime.creationdate` tag, else `creation_time`, else the file's modified time. For screen recordings named `ScreenRecording_MM-DD-YYYY HH-MM-SS`, the date in the name wins (the file's own dates are when it was copied). Existing projects get `taken` filled in lazily the first time it's needed (one ffprobe per clip, saved).

## Order
`settings.order` stays a single list of ids and can now hold photo ids too. `reel.ordered_clips` becomes `ordered_items`, returning plays and photos (each item has a `type`: `"play"` or `"photo"`):
1. Items in the saved order, as today.
2. New plays: appended in game/file order, as today.
3. New photos: each one is inserted **after the last item (play or photo) whose date is on or before the photo's date**, or at the very start if it's older than everything. Photos with the same insertion point stay in date order. If nothing in the reel has a date, photos go at the end in file order.

So a saved order Tom made is never rearranged; new photos only slot in.

`included(item)` for a photo is `keep is not False`.

## Build panel (review.html)
- The list shows photos in the same rows as plays: a 40 px thumbnail, a 📷 tag, the file name, the date, a tick box, grip/↑/↓, and **Preview** (plays the rendered slide in the existing preview modal; renders it first if needed, with a spinner). No **Edit** button for photos.
- The header reads e.g. `22 plays · 6 photos`, and the length estimate includes photos.
- New setting **Photo length: 3 / 4 / 5 s** (segmented control, default 4) → `settings.photo_s`.
- Thumbnails come from `GET /api/photo/thumb/<id>` (JPEG, 160 px long side, cached in `.hlv_cache\photos\thumbs\`).
- Ticking/unticking a photo posts to `/api/photo/keep` `{id, keep}`.
- The review sidebar (clip list for snap/end/spotlight) stays plays only.

## Rendering a slide (`hlv/photo_slide.py`)
`render(root, photo, seconds, out_size) -> Path`: an MP4 the same size, frame rate (`segment.FPS` = 30) and codec as play segments, with a silent audio track so `_assemble` treats it like any other piece.
- **Load:** PIL (plus pillow-heif), apply EXIF orientation (`ImageOps.exif_transpose`), convert to RGB.
- **Background:** the photo scaled to cover the frame, Gaussian blur (radius ≈ 40 px at 1080p), darkened to ~45% brightness. Static.
- **Foreground:** the photo scaled to fit inside the frame (whole photo visible). A wide photo that already fills the frame simply covers the background.
- **Movement:** the foreground scales steadily from 1.00 → 1.10 (alternate slides 1.10 → 1.00) about a point at the photo's horizontal centre and 40% of its height, where faces and numbers usually are. Face detection was planned, but OpenCV 5 no longer ships the Haar cascades, and its YuNet detector needs a downloaded model, so it was left out (2026-10-03). Each frame is rendered with `cv2.warpAffine` at sub-pixel precision (no `zoompan` jitter).
- **Length:** rendered `seconds + 1.0` long, so the reel timing can trim it.
- **Cache:** `.hlv_cache\photos\<id>-<key>.mp4`, where the key hashes the file's size + mtime, the rendered length, output size and a `PHOTO_VERSION` constant. Rebuilds reuse it.
- Rendered in the same process pool as the plays during a build.

## Reel timing (reel.py)
- Photo items join the `items` list in their place: one piece `(slide, 0.0, length)`.
- `plan_timing` walks plays and photos in order. A photo has no hit. Its nominal length is `photo_s`. **With a beat grid**, the length is adjusted (within 0.75 × to 1.25 × of `photo_s`) so the cut out of the photo lands on the nearest beat. Like a play's tail, the photo's length can also absorb a following play's spill, within the same limits.
- Game-audio ducking: photo slides have no game audio, so the music plays straight through.
- The name banner, freeze intro, hit punch and slow-mo are play-only. `banner_ids` ("first play") skips photos. The intro background still comes from the first spotlighted play.
- The FCPXML lists photo slides as ordinary clips.
- The build's result counts `photos` alongside `plays`. "No plays to include" still applies only if there are no plays at all (a reel of only photos is not a goal).

## Errors
- A photo that fails to load at build time is skipped, with a line in the build log and in the build result (`skipped_photos`). The build carries on.
- A missing thumbnail shows a grey placeholder.

## Testing
- Unit tests (pytest, `HLV_HOME` pointed at a temp folder as the others do):
  - finding photos skips the cache/reel folders, drops deleted files, and keeps `keep` flags when rescanning.
  - date reading: EXIF date, mtime fallback, screen-recording file-name date.
  - by-date insertion: into a saved order without moving saved items; older-than-everything goes first; no dates → end.
  - `plan_timing` with a photo between two plays: the photo's cut lands on a beat, and its length stays within limits.
  - slide render: the right frame count and size; a portrait photo has the blurred sides (side columns aren't black) and the whole photo is visible.
- By eye (as Tom would): make a project from Brody's folder and build a reel. Check a contact sheet from each slide (tall, wide, HEIC, PNG), check for smooth movement with no jitter, check the photos' places against their dates, and check the beat offsets for the cuts after photos.

## Out of scope
Captions on photos; picking the pan direction or focus point per photo; photos in the animated intro; photo-only reels.
