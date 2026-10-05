"""The composed soundtrack: a moment of silence, then a big impact, on every hit."""
import wave

import numpy as np
import pytest

from hlv import composer


@pytest.mark.parametrize("style", sorted(composer.STYLES))
def test_impact_lands_on_every_hit(style, tmp_path):
    tl = composer.demo_timeline(style)
    out = composer.render(tmp_path / "s.wav", style, tl, seed=2)
    with wave.open(str(out)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)[:, 0] / 32768
    sr = composer.SR

    def level(t0, t1):
        return 20 * np.log10(np.sqrt(np.mean(x[int(t0 * sr):int(t1 * sr)] ** 2)) + 1e-9)

    for p in tl["plays"]:
        h = p["hit"]
        assert level(h, h + 0.15) - level(h - 0.15, h - 0.02) > 8      # silence, then the impact
    assert np.abs(x).max() <= 0.96                                        # no clipping


def test_grid_is_steady_with_strong_downbeats():
    g = composer.grid("rock", 10)
    d = np.diff(g["beats"])
    assert np.allclose(d, 60 / 150, atol=1e-3) and g["strength"][0] == 1.0 and g["strength"][1] < 1.0
