from hlv import segment

CLIP = {"id": "c", "source": "c.mov", "probe": {"duration": 20.0},
        "detection": {"snap": 5.0, "end": 12.0}, "proposed": {"in": 4.0, "out": 13.0}, "review": {}}


def test_no_freeze_no_slowmo_is_real_time():
    lay = segment.predict_layout(CLIP, {"freeze": True}, hit_t=9.0, freeze_t=None, banner=False)
    assert lay["f_frames"] == 0 and lay["split"] == 5.0
    assert lay["a_frames"] == round(segment.LEAD_MAX * segment.FPS)        # lead kept before the snap
    assert lay["b_frames"] == round((13.0 + segment.TAIL_EXTRA - 5.0) * segment.FPS)
    assert abs(lay["hit_b"] - 4.0) < 1 / segment.FPS                       # hit 4 s after the snap
    assert abs(lay["post_nom"] - 8.0) < 1 / segment.FPS


def test_freeze_and_banner_hold():
    s = {"freeze": True}
    assert segment.predict_layout(CLIP, s, 9.0, 5.0, False)["f_frames"] == round(segment.FREEZE_MAX * segment.FPS)
    held = segment.predict_layout(CLIP, s, 9.0, 5.0, True)
    assert held["banner"] and held["f_frames"] == round(segment.BANNER_FREEZE[2] * segment.FPS)
    off = segment.predict_layout(CLIP, {"freeze": False}, 9.0, 5.0, True)   # freeze off: no hold, no banner
    assert off["f_frames"] == 0 and not off["banner"]


def test_slowmo_puts_the_hit_later_in_the_play():
    plain = segment.predict_layout(CLIP, {}, 9.0, None, False)
    slow = segment.predict_layout(CLIP, {"slowmo": True, "slowmo_speed": 0.4}, 9.0, None, False)
    assert slow["hit_b"] > plain["hit_b"] + 0.3
    assert slow["b_frames"] > plain["b_frames"]
    assert slow["post_t"][0] == 5.0 and slow["post_t"] == sorted(slow["post_t"])
