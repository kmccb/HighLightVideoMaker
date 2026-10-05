"""The hit model learns which signal marks your hits, and scores itself honestly."""
import numpy as np

from hlv import hitfx, hitmodel


def _item(rng, hit_t, side=None):
    n = 300
    f = {k: rng.normal(0, 1, n) for k in hitmodel.KEYS}
    i = int(hit_t * 30)
    f["near"][i - 3:i + 3] += 8                      # someone right on him exactly at the hit
    f["u"] = (np.arange(n) / 30 - 1.0) / 8.0          # snap at 1 s, end at 9 s
    return {"f": f, "snap": 1.0, "end": 9.0, "hit": hit_t, "side": side, "id": str(hit_t)}


def test_fit_finds_the_signal_and_scores_itself():
    rng = np.random.default_rng(1)
    items = [_item(rng, t) for t in rng.uniform(2.5, 8.0, 14)]
    m = hitmodel.fit(items)
    w = m["all"]["weights"]
    assert w.get("near", 0) > 0
    assert m["all"]["loo_within"] >= 0.8                 # it generalises to plays it didn't see
    # and the runtime scorer agrees with the fitted weights
    it = items[0]
    lo, hi = int((it["snap"] + 0.4) * 30), int((it["end"] + 0.3) * 30)
    t = (lo + int(np.argmax(hitfx.score(it["f"], w, lo, hi)))) / 30
    assert abs(t - it["hit"]) <= hitmodel.TOL


def test_per_side_models_need_enough_examples():
    rng = np.random.default_rng(2)
    items = [_item(rng, t, "O") for t in rng.uniform(2.5, 8.0, hitmodel.MIN_SIDE)] + \
            [_item(rng, t, "D") for t in rng.uniform(2.5, 8.0, 3)]
    m = hitmodel.fit(items)
    assert "O" in m and "D" not in m
