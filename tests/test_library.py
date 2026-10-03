import json
from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("HLV_HOME", str(h))
    return h


def _old_project(root: Path) -> Path:
    """A project folder in the pre-library layout: players.json + assets/ next to the clips."""
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "logo.png").write_bytes(b"png")
    (root / "assets" / "song.mp3").write_bytes(b"mp3")
    (root / "players.json").write_text(json.dumps({"players": [
        {"id": "caleb", "name": "Caleb Scott", "number": "12", "logo": "assets/logo.png",
         "music": "assets/song.mp3", "color": "#1f4fd1", "accent": "#f5f4f0"}]}), encoding="utf-8")
    (root / "highlights.json").write_text(json.dumps(
        {"schema_version": 1, "clips": [], "settings": {"transition": "cut"}}), encoding="utf-8")
    return root


def test_project_registry_round_trip(home, tmp_path):
    from hlv import library
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    library.add_project(a, "caleb")
    library.add_project(b, None)
    library.add_project(a, "caleb")                       # same folder twice: one entry
    assert [Path(p["folder"]) for p in library.projects()] == [a, b]
    assert library.find(a)["player"] == "caleb"
    library.touch(b)
    assert Path(library.projects()[0]["folder"]) == b      # most recently opened first
    library.remove_project(a)
    assert library.find(a) is None and len(library.projects()) == 1


def test_music_path_relative_absolute_missing(tmp_path):
    from hlv import project
    (tmp_path / "music").mkdir()
    song = tmp_path / "music" / "s.mp3"
    song.write_bytes(b"x")
    assert project.music_path(tmp_path, {"music": "music/s.mp3"}) == song
    assert project.music_path(tmp_path, {"music": str(song)}) == song
    assert project.music_path(tmp_path, {"music": "music/nope.mp3"}) is None
    assert project.music_path(tmp_path, {}) is None


def test_migrate_moves_player_and_music_once(home, tmp_path):
    from hlv import library, profiles, project
    root = _old_project(tmp_path / "Caleb")
    library.migrate(root)
    players = profiles.load(library.home())
    assert [p["id"] for p in players] == ["caleb"] and "music" not in players[0]
    assert profiles.asset(library.home(), players[0]["logo"]).read_bytes() == b"png"
    s = project.settings(project.load(root))
    assert s["music"] == "assets/song.mp3" and s["player"] == "caleb" and s["transition"] == "cut"
    assert s["scottforge_card"] is False
    assert library.find(root)["player"] == "caleb"
    before = (home / "players.json").read_text(), (root / "highlights.json").read_text()
    library.migrate(root)                                   # idempotent
    assert ((home / "players.json").read_text(), (root / "highlights.json").read_text()) == before
    assert (root / "players.json").exists() and (root / "assets" / "logo.png").exists()   # backup kept


def test_migrate_keeps_existing_library_player(home, tmp_path):
    from hlv import library, profiles
    profiles.save(library.home(), [{"id": "caleb", "name": "Caleb (edited)"}])
    library.migrate(_old_project(tmp_path / "Caleb"))
    assert profiles.load(library.home())[0]["name"] == "Caleb (edited)"


def test_branding_defaults_save_and_card(home):
    from hlv import graphics, library
    b = library.branding()
    assert b["accent"] == "#ff5a1f" and b["url"] == "https://ScottForge.ai" and library.branding_ready(b)
    b = library.save_branding({"accent": "orange", "invite": "  DM me  ", "bogus": "x"})
    assert b["accent"] == "#ff5a1f" and b["invite"] == "DM me" and "bogus" not in b
    assert library.branding() == b                                  # persisted
    img = graphics.scottforge_card(library.home(), b)
    assert img.shape == (graphics.H, graphics.W, 3)
    assert tuple(img[graphics.H - 5, 10]) == (31, 90, 255)          # accent stripe, BGR
    empty = library.save_branding({"title": "", "url": "", "invite": ""})
    assert not library.branding_ready(empty)                        # nothing to show: no card
