import wave

import numpy as np

from hlv import ghost, journey, parallax, reel


def test_big_kind_reads_old_and_new_marks():
    k = lambda v: reel.big_kind({"review": {"big": v}})    # noqa: E731
    assert k(None) is None and k(False) is None and reel.big_kind({}) is None
    assert k(True) == "trail" and k("trail") == "trail"    # True = plays marked before the 3D freeze existed
    assert k("3d") == "3d"
    assert reel.big_play({"review": {"big": "3d"}}) and not reel.big_play({"review": {}})


def test_big_meta_carries_the_kind_of_freeze():
    m = {"lay": {"b_frames": 300}, "hit": 3.0}
    reel._big_meta(m, True, "f.mp4", None, 4.0, "3d")
    assert m["kind"] == "3d" and m["pops"] == [] and m["extra"] == 4.0
    m = {"lay": {"b_frames": 300}, "hit": 3.0}
    reel._big_meta(m, True, "f.mp4", [0.25], 4.0)
    assert m["kind"] == "trail"


def test_3d_freeze_starts_and_ends_on_the_live_picture():
    secs = 4.033
    for t in (0.0, secs):
        zoom, orbit, lift, _ = parallax.swing(t, secs)
        assert abs(zoom) < 1e-6 and abs(orbit) < 1e-6 and abs(lift) < 1e-6
    mid = [parallax.swing(t, secs) for t in np.linspace(0.6, secs - 0.5, 30)]
    assert all(z > 0.99 for z, *_ in mid)                   # fully in for the whole pass
    orbits = [o for _, o, *_ in mid]
    assert orbits == sorted(orbits) and orbits[0] < -0.8 and orbits[-1] > 0.8    # one pass, left to right


def test_which_picks_the_box_holding_the_trackers_box():
    boxes = np.array([[100, 100, 160, 260, 0.9],            # him: the tracker's chest box sits inside
                      [150, 110, 210, 270, 0.9],            # the tackler next to him
                      [600, 100, 660, 260, 0.9]], np.float32)
    assert parallax.which(boxes, 128, 170, 30, 50) == 0
    assert parallax.which(boxes, 400, 400, 30, 50) is None


def test_3d_freeze_has_its_own_sound(tmp_path):
    p = ghost.sound_track(tmp_path / "fx.wav", 10.0, [{"t": 3.0, "drop": False, "pops": None, "len": 4.0, "swing": True}])
    with wave.open(str(p)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2) / 32768
    rms = lambda ch, t0, t1: float(np.sqrt((a[int(t0 * ghost.SR):int(t1 * ghost.SR), ch] ** 2).mean()))    # noqa: E731
    assert rms(0, 0, 2.9) == 0 and rms(0, 4.0, 6.0) > 0.005
    assert rms(0, 3.3, 4.0) > rms(1, 3.3, 4.0)             # the air starts on the left ...
    assert rms(1, 6.0, 6.7) > rms(0, 6.0, 6.7)             # ... and ends on the right, with the camera


def test_journey_key_moment_is_the_hit_else_the_middle_click():
    clip = {"review": {"hit": 7.5, "spotlight": [{"t": 1.0}, {"t": 2.0}, {"t": 9.0}]}}
    assert journey._key_time(None, clip) == 7.5
    clip["review"]["hit"] = None
    assert journey._key_time(None, clip) == 2.0
    assert journey._key_time(None, {"review": {"spotlight": []}}) is None
