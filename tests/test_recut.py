from hlv import recut, reel, segment


def test_pick_moves_forward_and_never_repeats():
    marks = [[10.0, 1.0], [30.0, 2.0], [60.0, 1.5], [90.0, 2.5], [120.0, 1.2]]
    used = []
    a = recut.pick(marks, 20.0, used)
    assert a > 24.0                                         # something ahead of where the song is
    used.append(a)
    b = recut.pick(marks, a + 20, used)
    assert b not in used and b > a
    used.append(b)
    # nothing left ahead: the best one not used yet, wherever it is
    assert recut.pick(marks, 119.0, used) in (10.0, 30.0, 60.0, 90.0) and recut.pick(marks, 119.0, used) not in used
    assert recut.pick([], 0.0, []) is None
    assert recut.pick(marks, 0.0, [10.0, 30.0, 60.0, 90.0, 120.0]) is None


def test_song_time_follows_the_cuts():
    cuts = [(0.0, 5.0), (20.0, 60.0), (40.0, 100.0)]
    assert recut.song_time(cuts, 3.0) == 8.0
    assert recut.song_time(cuts, 20.0) == 60.0
    assert recut.song_time(cuts, 25.0) == 65.0
    assert recut.song_time(cuts, 41.5) == 101.5


def test_a_drop_makes_the_song_jump_and_later_hits_follow_the_new_beats():
    lay = {"a_frames": 75, "f_frames": 0, "b_frames": 300, "banner": False}
    metas = [{"lay": dict(lay), "lead_nom": 1.0, "post_nom": 6.0, "hit": 3.0} for _ in range(3)]
    metas[0]["drop"] = True
    grid = {"beats": [0.5 * k for k in range(600)], "strength": [1.0] * 600}
    marks = [[100.25, 2.0], [200.0, 1.0]]                   # off the half-second grid on purpose
    plain = reel.plan_timing([dict(m) for m in metas], grid, 0.0, 4.0, 0.0)
    cut = reel.plan_timing(metas, grid, 0.0, 4.0, 0.0, marks)
    assert cut[0]["jump"] == 100.25 and "jump" not in cut[1]
    assert cut[0]["hit_reel"] == plain[0]["hit_reel"]       # its own hit stays where it was
    cuts = recut.cuts(0.0, cut)
    assert cuts == [(0.0, 0.0), (cut[0]["hit_reel"], 100.25)]
    for tm in cut[1:]:                                      # later hits sit on beats of the song as heard from the jump
        s = recut.song_time(cuts, tm["hit_reel"])
        assert abs(s / 0.5 - round(s / 0.5)) * 0.5 <= 1 / segment.FPS
        assert abs(tm["err"]) <= 1 / segment.FPS
    assert all("jump" not in tm for tm in plain)
