# Photos in the Reel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Photos in a project folder become slow pan-and-zoom slides that Tom can tick/untick and reorder alongside the plays, placed by date by default.

**Architecture:** `hlv/photos.py` finds photos, reads dates (photos and plays) and keeps `highlights.json["photos"]` in sync. `reel.ordered_items` merges plays and photos into one order (saved order first, new photos slotted in by date). `hlv/photo_slide.py` renders a cached MP4 slide per photo (blurred fill + Ken Burns via `cv2.warpAffine`). `reel.build`/`plan_timing` treat a photo as an item with a beat-snapped length. The server and Build panel list photos with thumbnails, tick boxes and Preview.

**Tech Stack:** Python 3.12, Pillow + pillow-heif, OpenCV (Haar face cascade), numpy, ffmpeg, vanilla JS page.

Spec: `docs/superpowers/specs/2026-10-03-photos-in-reel-design.md`.

## Global Constraints
- Photo extensions: `.jpg .jpeg .png .heic .heif .webp`.
- Skipped folders (relative to the project): `.hlv_cache reel music previews debug normalized assets`.
- Photo id: `"photo-" + signals.clip_id(root, path)`. Photo entries have `"type": "photo"`.
- Photo length setting `settings.photo_s` ∈ {3, 4, 5}, default 4. Beat-fitted length within 0.75×–1.25× of it.
- Slides: 1920×1080 (`graphics.W, graphics.H`), 30 fps (`segment.FPS`), H.264 + silent stereo AAC 48 kHz, cached in `.hlv_cache\photos\`.
- Zoom 1.00 ↔ 1.10; background = cover-scaled, blurred, ~45% brightness.
- Tests use `HLV_HOME` pointed at a temp folder (never the real library).
- Never rearrange items already in `settings.order`.

---

### Task 1: Find photos and read dates (`hlv/photos.py`)

**Files:**
- Create: `hlv/photos.py`
- Modify: `hlv/setup_job.py` (use `photos.find` / `photos.PHOTO_EXTS`), `requirements.txt` (+ `pillow-heif`)
- Test: `tests/test_photos.py`

**Interfaces:**
- Produces: `PHOTO_EXTS`, `find(root) -> list[Path]`, `open_image(path) -> PIL.Image (RGB, upright)`, `photo_taken(path) -> str` (ISO, seconds), `clip_taken(root, clip) -> str`, `sync(root, proj) -> bool` (mutates `proj["photos"]` and fills `clip["taken"]`; True if anything changed), `is_photo(item) -> bool`.

- [ ] **Step 1: Tests**: finds photos and skips cache/reel folders; EXIF `DateTimeOriginal` is read; mtime fallback; screen-recording name date; `sync` adds new photos with `keep: True`, keeps `keep: False` across rescans, drops deleted files, skips undecodable files, fills `clip["taken"]` from the file-name date or mtime.
- [ ] **Step 2: Run, expect import failure.**
- [ ] **Step 3: Implement** (code below).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `Photos: find, dates, sync into highlights.json`.

```python
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
SKIP = {".hlv_cache", "reel", "music", "previews", "debug", "normalized", "assets"}
_SR = re.compile(r"ScreenRecording_(\d\d)-(\d\d)-(\d{4}) (\d\d)-(\d\d)-(\d\d)")

def find(root): sorted files with PHOTO_EXTS not under SKIP
def open_image(p): pillow_heif.register_heif_opener() once; Image.open; ImageOps.exif_transpose; convert("RGB")
def photo_taken(p): exif.get_ifd(0x8769)[36867] or exif[306] parsed "%Y:%m:%d %H:%M:%S", else mtime
def clip_taken(root, clip): clip.get("taken") or name date or ffprobe format tags
    ("com.apple.quicktime.creationdate", "creation_time") -> local ISO, else mtime
def sync(root, proj): rebuild proj["photos"] keeping existing entries (by id) whose file exists;
    new ones verified by open_image (skip on error), {"id","source","group","taken","keep":True,"type":"photo"};
    fill clip["taken"] where missing. Return changed.
```

### Task 2: One order for plays and photos (`reel.py`)

**Files:** Modify `hlv/reel.py`. Test: `tests/test_photos.py`.

**Interfaces:**
- Consumes: `photos.is_photo`.
- Produces: `reel.ordered_items(proj) -> list[dict]` (plays + photos), `reel.ordered_clips(proj)` (plays only, unchanged behaviour), `reel.included(item)` (photo → `keep is not False`).

- [ ] **Step 1: Tests**: saved order kept and new photos inserted after the last item dated on or before them; a photo older than everything goes first; no dated items → photos at the end; `ordered_clips` has no photos; an unticked photo isn't `included`.
- [ ] **Step 2–4: Fail, implement, pass.**

```python
def ordered_items(proj):
    plays, photos_ = proj["clips"], proj.get("photos") or []
    by_id = {c["id"]: c for c in plays + photos_}
    order = [i for i in settings.order if i in by_id]
    seq = [by_id[i] for i in order] + sorted(new plays, key=(group, source))
    for ph in sorted(new photos, key=(taken, source)):
        dated = [k for k, it in enumerate(seq) if it.get("taken")]
        if not dated: seq.append(ph); continue
        before = [k for k in dated if seq[k]["taken"] <= ph["taken"]]
        seq.insert(before[-1] + 1 if before else 0, ph)
    return seq
```

- [ ] **Step 5: Commit** `Reel order: plays and photos, new photos placed by date`.

### Task 3: Render a slide (`hlv/photo_slide.py`)

**Files:** Create `hlv/photo_slide.py`. Test: `tests/test_photos.py`.

**Interfaces:**
- Consumes: `photos.open_image`, `segment.FPS`, `graphics.W/H`.
- Produces: `render(root, photo, seconds, zoom_out=False) -> Path`, `thumb(root, photo) -> Path`, `slide_length(photo_s) -> float` (= `photo_s * 1.25 + 0.2`), `MAX_ZOOM = 1.10`, `PHOTO_VERSION`.

- [ ] **Step 1: Tests** (small synthetic photos, in tmp): a portrait photo gives an MP4 of `round(seconds * 30)` ± 1 frames at 1920×1080 with an audio stream; its first frame's outer columns aren't black (blurred fill) and the middle column matches the photo's colour; a second call returns the same cached file without re-rendering; `thumb` is a JPEG with a long side ≤ 160.
- [ ] **Step 2–4: Fail, implement, pass.**

Implementation notes:
- `fit = min(W / w, H / h)`. Background: cover-scale (`max`), centre crop, `GaussianBlur` (sigma 30), ×0.45.
- Foreground hi-res: the photo resized by `fit * MAX_ZOOM` (LANCZOS). Its zoom-1 top-left is `off = ((W - w*fit)/2, (H - h*fit)/2)`.
- Focus `F`: the largest Haar face (`haarcascade_frontalface_default.xml`, on a ≤ 800 px grey copy, `minNeighbors=6`), mapped to frame coords; else the frame centre.
- Frame k, `u = k / (n - 1)`, `z = 1 + (MAX_ZOOM - 1) * (1 - u if zoom_out else u)` (steady, linear motion). `M = [[z/MAX_ZOOM, 0, F.x + z*(off.x - F.x)], [0, z/MAX_ZOOM, F.y + z*(off.y - F.y)]]`, `cv2.warpAffine(fg, M, (W, H), dst=bg.copy(), flags=INTER_LINEAR, borderMode=BORDER_TRANSPARENT)`.
- Pipe raw BGR into ffmpeg with `anullsrc`; write to `<name>.tmp.mp4`, then rename. Cache key: sha1 of (file size, mtime, seconds, W, H, zoom_out, PHOTO_VERSION).
- With a face, always zoom in (`zoom_out` is ignored).

- [ ] **Step 5: Commit** `Photo slides: blurred fill + Ken Burns, cached`.

### Task 4: Photos in the reel build (`reel.py`)

**Files:** Modify `hlv/reel.py`, `hlv/project.py` (`photo_s: 4` default). Test: `tests/test_photos.py`.

**Interfaces:**
- `plan_timing` metas may contain `{"photo": True, "len": photo_s}`; the output entry for it is `{"photo": True, "lead": 0, "fz": 0, "post": L, "beat": b|None, "err": e|None, "shift": L - photo_s}`.

- [ ] **Step 1: Tests**: play, photo, play with a beat grid every 0.5 s: the photo's cut point (`start + L - d/2`) is within one frame of a beat, `0.75*4 <= L <= 1.25*4`, and the following play's hit still lands on a beat; without a grid L = photo_s.
- [ ] **Step 2–4: Fail, implement, pass.** In the loop, a photo item: `lo, hi = 0.75*P, 1.25*P`; with beats, choose the beat nearest `t + P - d/2` within `[t + lo - d/2, t + hi - d/2]`, `L = _fr(beat - t + d/2)`; `prev_grow, prev_shrink = hi - L, L - lo`; `t += L - d`.
- `build()`: `photos.sync` (save if changed); `items_ = [i for i in ordered_items(proj) if included(i)]`; plays = non-photos (error if none); render plays and photo slides in the same pool (`photo_slide.render(root, ph, slide_length(P), zoom_out=(k % 2 == 1))`, where k = photo index); a photo that fails is dropped and listed in `skipped_photos`; segment files numbered across all items; `metas` include photo entries; `items` get `[(slide, 0.0, tm["post"])]` for photos; the result has `photos` and `skipped_photos`; the beat summary counts plays only.
- [ ] **Step 5: Commit** `Reel build: photo slides with beat-fitted length`.

### Task 5: Server and Build panel

**Files:** Modify `hlv/review_server.py`, `hlv/static/review.html`.

- Server: `open_project` runs `photos.sync` and saves. `reel_info` rows come from `ordered_items`; photo rows are `{id, type: "photo", source, group, taken, included, keep, seconds: photo_s, thumb: "/api/photos/<id>/thumb"}`; the response adds `photo_count`. `set_order` accepts photo ids. `update_settings`: `photo_s` ∈ (3, 4, 5). New routes: `GET /api/photos/<id>/thumb` (JPEG), `POST /api/photos/<id>` `{keep}`, `POST /api/photos/<id>/slide` → `{url: "/media/photo/<file>"}`, `GET /media/photo/<file>.mp4`.
- Page: photo rows show a thumbnail, the 📷 tag, the date, a tick box, grip/↑/↓ and Preview (no Edit); ticking posts to `/api/photos/<id>`; Preview posts `/slide`; the header reads `N plays · M photos`; plays are numbered across all included items; new `Photo length` control `#rPhotoS` 3 / 4 / 5 s; the order heading becomes "Plays and photos in reel order…".
- [ ] Verify in the browser on a project with photos: thumbnails show, untick saves, drag saves, Preview plays a slide, Photo length saves; no console errors.
- [ ] **Commit** `Build panel: photos with thumbnails, tick boxes, preview, photo length`.

### Task 6: End-to-end check on Brody's folder + STATUS
- [ ] Set up `assets\brody` as a project (in the app or with `setup_job.run`), build a reel, and render a contact sheet of every slide (first, middle and last frames). Check tall, wide, HEIC and PNG photos; smooth motion; dates against placement; cut offsets after photos.
- [ ] Update STATUS.md (built table, data state, test count), commit.
