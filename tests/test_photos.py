"""Photos in the reel: finding, dates, order, slides and timing."""
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from hlv import photos, reel


def _jpg(p: Path, size=(300, 400), color=(200, 40, 40), taken: str | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    exif = Image.Exif()
    if taken:
        exif.get_ifd(0x8769)[36867] = taken
    img.save(p, exif=exif)
    return p


def _set_mtime(p: Path, iso: str) -> None:
    t = time.mktime(datetime.fromisoformat(iso).timetuple())
    os.utime(p, (t, t))


# ---------------- finding + dates ----------------
def test_find_skips_cache_and_output_folders(tmp_path):
    _jpg(tmp_path / "a.jpg")
    _jpg(tmp_path / "Week 1" / "b.JPG")
    (tmp_path / "c.png").write_bytes(b"")
    for skip in (".hlv_cache", "reel", "music", "previews"):
        _jpg(tmp_path / skip / "x.jpg")
    (tmp_path / "notes.txt").write_text("hi")
    names = [p.relative_to(tmp_path).as_posix() for p in photos.find(tmp_path)]
    assert sorted(names) == ["Week 1/b.JPG", "a.jpg", "c.png"]


def test_photo_taken_reads_exif_then_falls_back_to_mtime(tmp_path):
    a = _jpg(tmp_path / "a.jpg", taken="2026:09:12 19:04:11")
    assert photos.photo_taken(a) == "2026-09-12T19:04:11"
    b = _jpg(tmp_path / "b.jpg")
    _set_mtime(b, "2026-08-01T10:00:00")
    assert photos.photo_taken(b) == "2026-08-01T10:00:00"


def test_clip_taken_prefers_screen_recording_name(tmp_path):
    v = tmp_path / "ScreenRecording_09-19-2026 08-40-29_1.mp4"
    v.write_bytes(b"")
    clip = {"id": "x", "source": v.name}
    assert photos.clip_taken(tmp_path, clip) == "2026-09-19T08:40:29"


def test_clip_taken_falls_back_to_mtime_and_reuses_stored(tmp_path):
    v = tmp_path / "IMG_1.MOV"
    v.write_bytes(b"not a video")
    _set_mtime(v, "2026-09-05T18:30:00")
    assert photos.clip_taken(tmp_path, {"source": "IMG_1.MOV"}) == "2026-09-05T18:30:00"
    assert photos.clip_taken(tmp_path, {"source": "IMG_1.MOV", "taken": "2020-01-01T00:00:00"}) == "2020-01-01T00:00:00"


def test_sync_adds_keeps_drops_and_skips_broken(tmp_path):
    _jpg(tmp_path / "a.jpg", taken="2026:09:12 19:00:00")
    _jpg(tmp_path / "b.jpg", taken="2026:09:13 19:00:00")
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    v = tmp_path / "IMG_1.MOV"
    v.write_bytes(b"x")
    _set_mtime(v, "2026-09-12T18:00:00")
    proj = {"clips": [{"id": "img-1", "source": "IMG_1.MOV"}]}
    assert photos.sync(tmp_path, proj) is True
    assert [p["id"] for p in proj["photos"]] == ["photo-a", "photo-b"]
    assert proj["photos"][0] == {"id": "photo-a", "type": "photo", "source": "a.jpg", "group": "",
                                 "taken": "2026-09-12T19:00:00", "keep": True}
    assert proj["clips"][0]["taken"] == "2026-09-12T18:00:00"
    proj["photos"][0]["keep"] = False
    (tmp_path / "b.jpg").unlink()
    assert photos.sync(tmp_path, proj) is True
    assert [(p["id"], p["keep"]) for p in proj["photos"]] == [("photo-a", False)]
    assert photos.sync(tmp_path, proj) is False


# ---------------- order ----------------
def _proj(order=None):
    clips = [{"id": f"p{i}", "source": f"G/p{i}.mov", "group": "G", "kind": "iphone",
              "taken": f"2026-09-0{i}T12:00:00", "review": {}} for i in (1, 2, 3)]
    return {"clips": clips, "photos": [], "settings": {"order": order or []}}


def _photo(pid, taken, keep=True):
    return {"id": pid, "type": "photo", "source": pid + ".jpg", "group": "", "taken": taken, "keep": keep}


def test_new_photos_slot_in_by_date_without_moving_saved_order():
    proj = _proj(order=["p3", "p1", "p2"])
    proj["photos"] = [_photo("photo-x", "2026-09-01T13:00:00"), _photo("photo-y", "2026-09-01T14:00:00"),
                      _photo("photo-old", "2025-01-01T00:00:00")]
    ids = [i["id"] for i in reel.ordered_items(proj)]
    assert ids == ["photo-old", "p3", "p1", "photo-x", "photo-y", "p2"]
    assert [c["id"] for c in reel.ordered_clips(proj)] == ["p3", "p1", "p2"]


def test_saved_photo_position_is_kept():
    proj = _proj(order=["photo-x", "p1", "p2", "p3"])
    proj["photos"] = [_photo("photo-x", "2026-09-02T13:00:00")]
    assert [i["id"] for i in reel.ordered_items(proj)][0] == "photo-x"


def test_photos_go_last_when_nothing_has_a_date():
    proj = _proj()
    for c in proj["clips"]:
        del c["taken"]
    proj["photos"] = [_photo("photo-x", "2026-09-02T13:00:00")]
    assert [i["id"] for i in reel.ordered_items(proj)] == ["p1", "p2", "p3", "photo-x"]


def test_unticked_photo_is_not_included():
    assert reel.included(_photo("photo-x", "", keep=True))
    assert not reel.included(_photo("photo-x", "", keep=False))


# ---------------- slides ----------------
def _frames(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f)


def test_portrait_slide_has_blurred_sides_and_is_cached(tmp_path):
    from hlv import photo_slide
    src = Image.new("RGB", (300, 400), (40, 40, 200))       # blue, tall
    src.paste((250, 250, 250), (0, 0, 300, 60))               # white band: the blurred sides aren't uniform
    src.save(tmp_path / "tall.jpg")
    ph = {"id": "photo-tall", "type": "photo", "source": "tall.jpg"}
    out = photo_slide.render(tmp_path, ph, 2.0)
    fr = _frames(out)
    assert abs(len(fr) - 60) <= 1
    f0 = fr[0]
    assert f0.shape == (1080, 1920, 3)
    assert f0[540, 40].sum() > 30                               # side filled (blurred), not black
    b, g, r = f0[700, 960].astype(int)
    assert b > 150 and r < 90                                    # middle shows the photo itself
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(out)], capture_output=True, text=True).stdout.split()
    assert "audio" in streams
    mtime = out.stat().st_mtime
    assert photo_slide.render(tmp_path, ph, 2.0) == out and out.stat().st_mtime == mtime


def test_slide_zooms_in_over_time(tmp_path):
    from hlv import photo_slide
    img = Image.new("RGB", (1600, 900), (0, 0, 0))
    img.paste((255, 255, 255), (700, 350, 900, 550))            # white square in the middle
    img.save(tmp_path / "wide.jpg")
    out = photo_slide.render(tmp_path, {"id": "photo-wide", "source": "wide.jpg"}, 1.0)
    fr = _frames(out)
    white = lambda f: int((f.min(axis=2) > 200).sum())          # noqa: E731
    assert white(fr[-1]) > white(fr[0]) * 1.12                  # ~1.1² more area at the end


def test_thumb_is_small_jpeg(tmp_path):
    from hlv import photo_slide
    _jpg(tmp_path / "a.jpg", size=(1200, 800))
    t = photo_slide.thumb(tmp_path, {"id": "photo-a", "source": "a.jpg"})
    with Image.open(t) as im:
        assert im.format == "JPEG" and max(im.size) <= 160


# ---------------- timing ----------------
def _lay():
    return {"a_frames": 75, "f_frames": 66, "b_frames": 300, "post_nom": 4.0}


def test_photo_cut_lands_on_a_beat_and_next_hit_still_does():
    grid = {"beats": list(np.arange(0, 120, 0.5)), "strength": [1.0] * 240}
    d = 1 / 3
    play = {"lay": _lay(), "lead_nom": 1.0, "post_nom": 4.0, "hit": 2.0}
    metas = [play, {"photo": True, "len": 4.0}, dict(play)]
    t = reel.plan_timing(metas, grid, 0.0, 3.65, d)
    start = 3.65 + t[0]["lead"] + t[0]["fz"] + t[0]["post"] - d
    ph = t[1]
    assert ph["photo"] and 3.0 <= ph["post"] <= 5.0
    cut = start + ph["post"] - d / 2
    assert min(abs(cut - b) for b in grid["beats"]) <= 1 / 30 + 1e-6
    assert abs(t[2]["err"]) <= 1 / 30 + 1e-6


def test_photo_length_is_nominal_without_beats():
    t = reel.plan_timing([{"photo": True, "len": 5.0}], None, 0.0, 3.65, 1 / 3)
    assert t[0]["post"] == pytest.approx(5.0)
