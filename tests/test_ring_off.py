"""Ring off / back on per clip: visibility over time and cleaning what the page sends."""
import pytest

from hlv.review_server import _clean_ring_off
from hlv.segment import RING_FADE, ring_visibility


def test_no_ranges_always_visible():
    assert ring_visibility([], 3.0) == 1.0


def test_gone_at_off_point_and_fades_before_it():
    r = [[5.0, None]]
    assert ring_visibility(r, 5.0 - RING_FADE - 0.01) == 1.0
    assert 0 < ring_visibility(r, 5.0 - RING_FADE / 2) < 1
    assert ring_visibility(r, 5.0) == 0.0
    assert ring_visibility(r, 20.0) == 0.0          # off to the end of the clip


def test_comes_back_after_on_point():
    r = [[5.0, 8.0]]
    assert ring_visibility(r, 7.99) == 0.0
    assert 0 < ring_visibility(r, 8.0 + RING_FADE / 2) < 1
    assert ring_visibility(r, 8.0 + RING_FADE) == pytest.approx(1.0)


def test_two_ranges():
    r = [[2.0, 3.0], [6.0, None]]
    assert ring_visibility(r, 2.5) == 0.0
    assert ring_visibility(r, 4.5) == 1.0
    assert ring_visibility(r, 7.0) == 0.0


def test_clean_sorts_merges_and_drops_junk():
    raw = [[6, None], [1, 2], ["x", 3], [4, 3.5], [1.5, 2.5], [-1, 0.5], {"a": 1}]
    assert _clean_ring_off(raw, 10.0) == [[0.0, 0.5], [1.0, 2.5], [6.0, None]]


def test_clean_clamps_to_duration():
    assert _clean_ring_off([[9.5, 30]], 10.0) == [[9.5, None]]
    assert _clean_ring_off([[12, None]], 10.0) == []
    assert _clean_ring_off("nope", 10.0) == []
