"""The watermarked preview: a smaller, marked copy of a finished reel to send before the customer pays."""
import json
import re
import shutil
import subprocess

import pytest

from hlv import watermark

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _probe(f):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                          "-of", "json", str(f)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    return v["width"], v["height"], float(d["format"]["duration"]), any(s["codec_type"] == "audio" for s in d["streams"])


def _clip(path, size):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=green:s={size}:r=30:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_preview_sits_next_to_the_reel_under_a_name_the_app_can_open(tmp_path):
    src = tmp_path / "caleb_reel_vertical.mp4"
    out = watermark.preview_path(src)
    assert out == tmp_path / "caleb_reel_vertical_PREVIEW.mp4"
    assert re.fullmatch(r"[A-Za-z0-9_.-]+\.mp4", out.name), "the reveal endpoint only opens plain names"
    assert watermark.is_preview(out) and not watermark.is_preview(src)


def test_the_mark_says_preview_and_carries_the_site_safely():
    f = watermark.filter_chain(1920, 1080, "reels.scottforge.ai")
    assert "PREVIEW" in f and "reels.scottforge.ai" in f and f.startswith("scale=1280:720")
    # anything that could break out of the ffmpeg filter is dropped from the site text
    g = watermark.filter_chain(1920, 1080, "evil':x=0,drawtext=text='x")
    texts = re.findall(r"text='([^']*)'", g)
    assert len(texts) == 4 and texts[1].startswith("evil"), texts
    assert not (set("':,=") & set(texts[1])), "no quote, colon, comma, equals or backslash survives"
    assert "text=''" not in watermark.filter_chain(1920, 1080, "")


def test_a_small_reel_is_never_enlarged_and_a_tall_one_stays_tall():
    assert watermark.filter_chain(640, 360, "").startswith("scale=640:360")
    assert watermark.filter_chain(1080, 1920, "").startswith("scale=720:1280")


@needs_ffmpeg
def test_make_writes_a_marked_copy_and_leaves_the_reel_alone(tmp_path):
    src = tmp_path / "x_reel.mp4"
    _clip(src, "1920x1080")
    before = src.read_bytes()
    out = watermark.make(src, "reels.scottforge.ai")
    assert out == watermark.preview_path(src) and out.exists()
    assert src.read_bytes() == before
    w, h, dur, audio = _probe(out)
    assert (w, h) == (1280, 720) and abs(dur - 1.0) < 0.2 and audio
    assert not list(tmp_path.glob("*.part.mp4")), "no half-written file left behind"
    # the mark is really in the picture: the middle of a plain green frame is no longer plain green
    px = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-vf", "crop=600:120:340:260,scale=1:1", "-frames:v", "1",
                         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    plain = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", "crop=900:180:510:390,scale=1:1", "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    assert sum(abs(a - b) for a, b in zip(px, plain)) > 12


@needs_ffmpeg
def test_make_says_why_when_the_video_cannot_be_read(tmp_path):
    bad = tmp_path / "broken_reel.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="broken_reel.mp4"):
        watermark.make(bad, "")
    assert not watermark.preview_path(bad).exists()


def test_the_mark_names_the_sales_site_by_default():
    assert watermark.SITE == "reels.scottforge.ai"
    assert "reels.scottforge.ai" in watermark.filter_chain(1920, 1080, watermark.SITE)


def test_site_text_drops_the_scheme():
    assert watermark.site_text("https://ScottForge.ai/") == "ScottForge.ai"
    assert watermark.site_text("reels.scottforge.ai") == "reels.scottforge.ai"
    assert watermark.site_text("") == ""
