import wave

import numpy as np

from hlv import ghost, impact, reel


def test_hit_effects_are_big_play_kinds():
    for k in impact.KINDS:
        assert reel.big_kind({"review": {"big": k}}) == k
    assert reel.big_kind({"review": {"big": "nonsense"}}) == "trail"


def test_spec_fits_after_the_hit_and_holds_the_reel_for_whole_beats():
    for period in (0.506, 0.4, 0.75, None):
        for k in impact.KINDS:
            sp = impact.spec(k, period)
            assert sp["rep"] / impact.FPS < impact.TAIL          # the play always has that much left after its hit
            assert sp["at"] <= sp["rep"]
            assert all(name in ("boom", "crack", "air", "whoosh", "ohh") and t >= -0.1 for name, t, _ in sp["sounds"])
        assert impact.spec("shock", period)["extra"] == 0 and impact.spec("frames", period)["extra"] == 0
        if period:
            beats = impact.spec("triple", period)["extra"] / impact.FPS / period
            assert abs(beats - 2) < 0.07                         # two more hits, a beat apart (to the frame)
            beats = impact.spec("replay", period)["extra"] / impact.FPS / period
            assert abs(beats - round(beats)) < 0.07 and beats >= 4


def test_big_meta_takes_a_hit_effect():
    sp = impact.spec("triple", 0.5)
    m = {"lay": {"b_frames": 300}, "hit": 3.0}
    reel._big_meta(m, True, None, None, 4.0, "triple", None, ("t.mp4", "triple", sp))
    assert m["kind"] == "triple" and m["extra"] == sp["extra"] / 30 and m["tail"] == impact.TAIL and "ghost" not in m
    m = {"lay": {"b_frames": 300}, "hit": 3.0}
    reel._big_meta(m, False, None, None, 4.0, "shock", None, ("s.mp4", "shock", impact.spec("shock", 0.5)))
    assert m["fx"][1] == "shock" and "extra" not in m
    m = {"lay": {"b_frames": 100}, "hit": 3.0}                   # the play ends too soon after the hit: left alone
    reel._big_meta(m, False, None, None, 4.0, "shock", None, ("s.mp4", "shock", impact.spec("shock", 0.5)))
    assert "fx" not in m


def test_a_play_with_a_hit_effect_keeps_its_tail_and_holds_up_the_next_play():
    lay = {"a_frames": 60, "f_frames": 0, "b_frames": 300}
    base = [{"lay": lay, "lead_nom": 1.0, "post_nom": 3.2, "hit": 3.0}, {"lay": lay, "lead_nom": 1.0, "post_nom": 4.0, "hit": 2.0}]
    plain = reel.plan_timing([dict(m) for m in base], None, 0.0, 5.0, 0.3)
    sp = impact.spec("replay", 0.5)
    fx = [dict(m) for m in base]
    reel._big_meta(fx[0], False, None, None, 4.0, "replay", None, ("r.mp4", "replay", sp))
    held = reel.plan_timing(fx, None, 0.0, 5.0, 0.3)
    assert held[0]["post"] >= 3.0 + impact.TAIL - 1e-6
    grew = held[0]["post"] - plain[0]["post"]
    assert abs(held[1]["hit_reel"] - plain[1]["hit_reel"] - grew - sp["extra"] / 30) < 1e-3


def test_sound_track_plays_a_hit_effects_own_sounds(tmp_path):
    sp = impact.spec("triple", 0.5)
    p = ghost.sound_track(tmp_path / "fx.wav", 8.0, [{"t": 2.0, "drop": False, "pops": None, "len": 1.0, "swing": False, "sounds": sp["sounds"]}])
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float32)
    sr = ghost.SR
    loud = lambda t: float(np.abs(x[int(t * sr):int((t + 0.1) * sr)]).max())    # noqa: E731
    assert loud(1.5) == 0 and loud(2.0) > 3000 and loud(2.5) > 3000 and loud(3.0) > loud(2.0)
