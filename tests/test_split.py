import json

from hlv import split


def _clip(tmp_path, dur=20.0):
    src = tmp_path / "game.mov"
    src.write_bytes(b"x")
    return {"id": "game", "source": "game.mov", "probe": {"duration": dur}}


def test_cuts_are_read_from_the_cache_while_the_video_is_unchanged(tmp_path):
    c = _clip(tmp_path)
    src = tmp_path / "game.mov"
    f = tmp_path / ".hlv_cache" / "cuts" / "game.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"stamp": [split.VERSION, src.stat().st_size, src.stat().st_mtime], "cuts": [4.5, 11.0]}))
    assert split.cuts(tmp_path, c) == [4.5, 11.0]


def test_a_video_with_no_cuts_is_left_alone(tmp_path, monkeypatch):
    c = _clip(tmp_path)
    monkeypatch.setattr(split, "cuts", lambda root, clip: [])
    assert split.split(tmp_path, c) == []
    assert (tmp_path / "game.mov").exists() and not (tmp_path / split.ASIDE).exists()


def test_the_app_does_not_look_in_the_folder_the_originals_are_moved_to(tmp_path):
    from hlv.probe import find_clips
    (tmp_path / split.ASIDE).mkdir()
    (tmp_path / split.ASIDE / "game.mov").write_bytes(b"x")
    (tmp_path / "game - play 01.mp4").write_bytes(b"x")
    assert [p.name for p in find_clips(tmp_path)] == ["game - play 01.mp4"]
