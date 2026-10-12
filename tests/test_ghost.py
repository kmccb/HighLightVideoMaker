import wave

import numpy as np

from hlv import ghost, reel, segment


def test_freeze_length_is_whole_beats():
    assert ghost.seconds(None) == 4.0
    secs = ghost.seconds(60 / 118.7)                       # 8 beats at 118.7 BPM
    assert abs(secs - 8 * 60 / 118.7) <= 0.5 / ghost.FPS
    assert abs(secs * ghost.FPS - round(secs * ghost.FPS)) < 1e-6
    assert 3.4 < ghost.seconds(60 / 70) < 4.6              # slow song: still about four seconds


def test_ghosts_never_pile_up():
    # he stands still for 30 frames, runs right for 60, then doubles back over his own path
    pos = {i: (100.0, 500.0, 40, 80) for i in range(30)}
    pos |= {30 + i: (100.0 + 10 * i, 500.0, 40, 80) for i in range(60)}
    pos |= {90 + i: (700.0 - 10 * i, 500.0, 40, 80) for i in range(20)}
    end = (500.0, 500.0)
    chosen = ghost.pick(pos, end, gap=60)
    assert chosen == sorted(chosen) and 3 <= len(chosen) <= ghost.MAX_GHOSTS
    pts = [pos[i][:2] for i in chosen] + [end]
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            assert np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) >= 60


def test_ghosts_are_all_in_before_the_hold():
    for n in (3, 10):
        pops = ghost.pop_times(n, 4.03, 0.5055)
        assert len(pops) == n and pops == sorted(pops) and pops[0] == ghost.T_FIRST
        assert pops[-1] <= 4.03 - ghost.T_OUT - 1.2
    assert np.allclose(np.diff(ghost.pop_times(4, 4.03, 0.5055)), 0.5055 / 2, atol=1e-3)   # on half-beats when they fit


def test_a_ghost_trail_holds_the_reel_up_but_hits_stay_on_beats():
    lay = {"a_frames": 75, "f_frames": 0, "b_frames": 300, "banner": False}
    metas = [{"lay": dict(lay), "lead_nom": 1.0, "post_nom": 6.0, "hit": 3.0} for _ in range(3)]
    grid = {"beats": [0.5 * k for k in range(200)], "strength": [1.0] * 200}
    plain = reel.plan_timing([dict(m) for m in metas], grid, 0.0, 4.0, 0.0)
    metas[0]["extra"] = 4.0                                 # 8 beats
    held = reel.plan_timing(metas, grid, 0.0, 4.0, 0.0)
    assert held[0]["hit_reel"] == plain[0]["hit_reel"]      # the freeze comes after its own hit
    for p, h in zip(plain[1:], held[1:]):
        assert abs(h["hit_reel"] - p["hit_reel"] - 4.0) < 1e-6 and abs(h["err"]) <= 1 / segment.FPS


def test_big_meta_needs_a_hit_and_room_after_it():
    lay = {"b_frames": 300}
    m = {"lay": lay, "hit": -1.0}
    reel._big_meta(m, True, "trail.mp4", [0.25], 4.0)
    assert "drop" not in m and "ghost" not in m             # no hit: nothing
    m = {"lay": lay, "hit": 3.0}
    reel._big_meta(m, True, "trail.mp4", [0.25], 4.0)
    assert m["drop"] and m["extra"] == 4.0 and m["pops"] == [0.25]
    m = {"lay": lay, "hit": 9.9}                            # the hit is on the play's last frames
    reel._big_meta(m, False, "trail.mp4", [0.25], 4.0)
    assert m["drop"] is False and "ghost" not in m


def test_sound_track_is_silent_in_the_gap_and_loud_on_the_hit(tmp_path):
    p = ghost.sound_track(tmp_path / "fx.wav", 12.0, [{"t": 5.0, "drop": True, "pops": [0.25, 0.5], "len": 4.0}])
    with wave.open(str(p)) as w:
        assert (w.getframerate(), w.getnchannels()) == (ghost.SR, 2)
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)[:, 0] / 32768
    rms = lambda t0, t1: float(np.sqrt((a[int(t0 * ghost.SR):int(t1 * ghost.SR)] ** 2).mean()))    # noqa: E731
    assert rms(5.0 - ghost.GAP + 0.01, 4.99) == 0.0         # dead silence before the hit
    assert rms(4.2, 4.6) > 0.01                             # the riser before it
    assert rms(5.0, 5.3) > 0.2                              # the boom
    assert rms(0.0, 3.0) == 0.0
