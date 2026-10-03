import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HLV_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_youtube_link_check():
    from hlv import music
    for ok in ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ",
               "https://music.youtube.com/watch?v=dQw4w9WgXcQ&si=x", "https://youtube.com/shorts/abcdefghijk",
               "  https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ  "):
        assert music.is_youtube(ok), ok
    for bad in ("", "youtube", "https://vimeo.com/123456", "https://www.youtube.com/", "C:\\song.mp3"):
        assert not music.is_youtube(bad), bad


def test_youtube_errors_become_plain_words():
    from hlv import music
    assert "private" in music.friendly_error("ERROR: [youtube] x: Private video. Sign in")
    assert "age-restricted" in music.friendly_error("Sign in to confirm your age")
    assert "internet" in music.friendly_error("Unable to download webpage: <urlopen error getaddrinfo failed>")
    assert "didn't work" in music.friendly_error("something new and strange")


def _fake_ytdlp(monkeypatch, fail: str | None = None):
    class YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            if fail:
                raise RuntimeError(fail)
            for h in self.opts["progress_hooks"]:
                h({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
                h({"status": "finished"})
            out = Path(self.opts["outtmpl"].replace("%(title).80s", "Hype_Song").replace("%(ext)s", "webm"))
            out.with_suffix(".mp3").write_bytes(b"mp3")
            return {"title": "Hype Song"}

        def prepare_filename(self, info):
            return self.opts["outtmpl"].replace("%(title).80s", "Hype_Song").replace("%(ext)s", "webm")
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=YDL))


def test_youtube_download_saves_mp3_in_project(tmp_path, monkeypatch):
    from hlv import music
    _fake_ytdlp(monkeypatch)
    seen = []
    rel = music.download_youtube(tmp_path, "https://youtu.be/dQw4w9WgXcQ", progress=lambda f, t: seen.append(f))
    assert rel == "music/Hype_Song.mp3" and (tmp_path / rel).read_bytes() == b"mp3"
    assert seen[-1] == 1.0 and 0.4 < seen[0] < 0.6


def test_youtube_download_failure_is_friendly(tmp_path, monkeypatch):
    from hlv import music
    _fake_ytdlp(monkeypatch, fail="ERROR: Private video")
    with pytest.raises(ValueError, match="private"):
        music.download_youtube(tmp_path, "https://youtu.be/dQw4w9WgXcQ", log=lambda m: None)
    with pytest.raises(ValueError, match="YouTube link"):
        music.download_youtube(tmp_path, "https://vimeo.com/1")


def test_import_music_copies_once(tmp_path):
    from hlv import music
    src = tmp_path / "My Song.MP3"
    src.write_bytes(b"abc")
    proj = tmp_path / "proj"
    proj.mkdir()
    assert music.import_file(proj, src) == "music/my-song.mp3"
    assert music.import_file(proj, src) == "music/my-song.mp3"            # same file: no copy 2
    assert music.import_file(proj, proj / "music" / "my-song.mp3") == "music/my-song.mp3"
    with pytest.raises(ValueError):
        music.import_file(proj, tmp_path / "notes.txt")


def test_scan_counts_videos_and_photos(home, tmp_path):
    from hlv import library, setup_job
    f = tmp_path / "Brody"
    (f / "Week 1").mkdir(parents=True)
    for n in ("a.mov", "Week 1/b.MP4", "c.jpg", "d.HEIC", "e.txt", "d.MP4"):   # d.MP4: Live Photo movie
        (f / n).write_bytes(b"x")
    r = setup_job.scan(f)
    assert (r["videos"], r["photos"], r["project"]) == (2, 2, False)
    library.add_project(f, "brody")
    assert setup_job.scan(f)["project"] is True
