import numpy as np

from hlv import matchcut

SIZE = (1920, 1080)


def _lands(pos, scale, q):
    """Where the point pos ends up, and whether the enlarged picture still covers the whole frame."""
    M = matchcut._affine(pos, scale, q, SIZE)
    x, y = M @ np.array([pos[0], pos[1], 1.0])
    corners = [M @ np.array([cx, cy, 1.0]) for cx in (0, SIZE[0]) for cy in (0, SIZE[1])]
    covers = min(c[0] for c in corners) <= 1e-6 and max(c[0] for c in corners) >= SIZE[0] - 1e-6         and min(c[1] for c in corners) <= 1e-6 and max(c[1] for c in corners) >= SIZE[1] - 1e-6
    return (x, y), covers


def test_same_size_players_meet_on_one_spot():
    a, b = (850.0, 520.0, 150.0), (1000.0, 600.0, 150.0)
    sa, sb, q = matchcut.meet(a, b, SIZE)
    assert sa == sb == matchcut.ROOM                        # both enlarged a little, to have room to move
    (ax, ay), a_ok = _lands(a, sa, q)
    (bx, by), b_ok = _lands(b, sb, q)
    assert a_ok and b_ok                                    # never the frame's edge
    assert abs(ax - bx) < 1.0 and abs(ay - by) < 1.0        # the same spot
    assert abs(ax - q[0]) < 1.0 and abs(ay - q[1]) < 1.0


def test_the_smaller_one_is_enlarged_to_the_same_height():
    a, b = (900.0, 540.0, 100.0), (1000.0, 560.0, 130.0)
    sa, sb, q = matchcut.meet(a, b, SIZE)
    assert sa > sb >= 1.0 and abs(a[2] * sa - b[2] * sb) < 1e-6
    (ax, ay), _ = _lands(a, sa, q)
    (bx, by), _ = _lands(b, sb, q)
    assert abs(ax - bx) < 1.0 and abs(ay - by) < 1.0


def test_size_gap_beyond_the_limit_is_capped_and_far_apart_spots_get_as_close_as_they_can():
    a, b = (300.0, 500.0, 50.0), (1700.0, 500.0, 300.0)
    sa, sb, q = matchcut.meet(a, b, SIZE)
    assert sa == matchcut.MAX_SCALE and sb == matchcut.ROOM
    (ax, _), a_ok = _lands(a, sa, q)
    (bx, _), b_ok = _lands(b, sb, q)
    assert a_ok and b_ok and ax < bx                        # can't meet without showing an edge: each goes as far as it can
    assert bx - ax < b[0] - a[0]
