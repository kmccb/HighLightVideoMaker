from hlv import reel


def _play(i, length, star=False, keep=None):
    return {"id": f"p{i}", "source": f"G/p{i}.mov", "group": "G", "kind": "iphone",
            "proposed": {"in": 0.0, "out": length},
            "review": {"in": 0.0, "out": length, "star": star, "keep": keep, "spotlight": []}}


def test_recruiting_cut_starred_first_then_fills_to_about_three_minutes():
    plays = [_play(i, 20.0) for i in range(12)]          # 12 x 20 s = 240 s, more than fits
    plays[7]["review"]["star"] = True
    plays[10]["review"]["star"] = True
    plays[3]["review"]["keep"] = False                    # left out of the reel: never in the cut
    proj = {"clips": plays, "settings": {"order": [p["id"] for p in plays]}}
    ids = [c["id"] for c in reel.recruit_plays(proj)]
    assert ids[:2] == ["p7", "p10"]                        # starred open it, in reel order
    assert "p3" not in ids
    assert len(ids) == 9                                   # 9 x 20 s = 180 s: stops at about 3 minutes
    assert ids[2:] == ["p0", "p1", "p2", "p4", "p5", "p6", "p8"]


def test_starred_plays_always_go_in_even_past_the_limit():
    plays = [_play(i, 100.0, star=True) for i in range(3)]
    proj = {"clips": plays, "settings": {}}
    assert len(reel.recruit_plays(proj)) == 3


def test_output_names():
    assert reel.output_name("caleb") == "caleb_reel.mp4"
    assert reel.output_name("caleb", vertical=True, recruiting=True) == "caleb_recruiting_vertical.mp4"
