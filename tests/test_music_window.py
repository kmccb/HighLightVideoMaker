"""Trimming the song and its fades in the finished reel."""
from hlv.reel import music_window


def test_default_plays_to_end_with_fades():
    assert music_window(120.0, {}) == (120.0, 0.5, 3.0)


def test_music_end_stops_song_early_in_reel_time():
    stop, fin, fout = music_window(120.0, {"music_start": 10, "music_end": 70, "music_fade_out": 4})
    assert stop == 60.0 and fout == 4.0


def test_end_after_reel_is_capped_and_compose_ignores_trim():
    assert music_window(50.0, {"music_end": 400})[0] == 50.0
    assert music_window(50.0, {"music_end": 20, "music_source": "compose"})[0] == 50.0


def test_fades_never_longer_than_the_music():
    stop, fin, fout = music_window(4.0, {"music_fade_in": 10, "music_fade_out": 10})
    assert fin <= 2.0 and fin + fout <= stop
