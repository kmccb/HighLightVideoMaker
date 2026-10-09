from datetime import datetime

import pytest

from hlv import youtube


def test_default_title():
    pl = {"name": "Caleb Scott", "number": "12", "tagline": ""}
    when = datetime(2026, 10, 9)
    assert youtube.default_title(pl, False, False, when) == "Caleb Scott #12 – 2026 Highlights"
    assert youtube.default_title(pl, True, True, when) == "Caleb Scott #12 – 2026 Highlights – Recruiting #Shorts"
    assert youtube.default_title({**pl, "number": "", "tagline": "8th Grade"}, False, False, when) == "Caleb Scott – 8th Grade"


def test_clean_strips_angle_brackets_and_limits():
    assert youtube.clean("a <b> c", 100) == "a b c"
    assert len(youtube.clean("x" * 300, 100)) == 100


def test_store_secret_rejects_wrong_client(tmp_path, monkeypatch):
    monkeypatch.setattr(youtube.library, "home", lambda: tmp_path)
    with pytest.raises(ValueError, match="Desktop app"):
        youtube.store_secret(b'{"web": {}}')
    youtube.store_secret(b'{"installed": {"client_id": "x"}}')
    assert youtube.has_secret()


def test_visibility_defaults_to_unlisted_and_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(youtube.library, "home", lambda: tmp_path)
    assert youtube._load()["privacy"] == "unlisted"
    with pytest.raises(ValueError, match="visibility"):
        youtube.publish([], "", "everyone")
