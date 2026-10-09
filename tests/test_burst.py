"""The intro blowing apart into the first play: brick physics stays finite at every frame size."""
import numpy as np
import pytest

from hlv.burst import _bricks


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920)])
def test_bricks_have_finite_motion(size):
    # The vertical reel used to give the bottom-corner bricks a negative "near",
    # and negative ** 1.5 is NaN: the build failed with "cannot convert float NaN to integer".
    w, h = size
    img = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    bricks = _bricks(img, np.random.default_rng(2))
    assert bricks
    for b in bricks:
        assert np.all(np.isfinite(b["v"]))
        assert np.isfinite(b["w"]) and np.isfinite(b["grow"]) and np.isfinite(b["delay"])
