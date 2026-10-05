"""The hit suggestion's weights, learned from every hit you've marked (all projects).

Each marked play contributes its hitfx signals and your hit time. A few thousand weightings
are tried and the one that lands most suggestions within TOL of your marks wins. Plays you've
tagged Offense or Defense also train a model per side (your "hit" means contact/tackle on
defense, the catch or big move on offense), used once that side has MIN_SIDE examples.

Saved to <app folder>/hit_model.json with an honest score (leave-one-out: each play judged
by weights chosen without it). Refit in the background whenever you mark more hits.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import hitfx

FPS = 30.0
TOL = 0.4
MIN_SIDE = 8
MODEL_FILE = "hit_model.json"
FEAT_VERSION = "f1"
KEYS = ("speed", "decel", "accel", "near", "crowd", "loud", "onset")
U0S = (0.35, 0.45, 0.55, 0.65)
_lock = threading.Lock()
_pending = {"t": 0.0, "thread": None}


# ---------------------------------------------------------------- data
def _times(clip: dict) -> tuple[float, float]:
    r, d = clip.get("review") or {}, clip["detection"]
    snap = r["snap"] if r.get("snap") is not None else d["snap"]
    end = r["end"] if r.get("end") is not None else d["end"]
    return float(snap), float(end)


def clip_features(root: Path, clip: dict) -> dict[str, np.ndarray]:
    """hitfx signals for a clip, cached by everything they depend on."""
    from . import camera, detect
    root = Path(root)
    cache = root / ".hlv_cache"
    snap, end = _times(clip)
    dets_files = sorted((cache / "det").glob(f"{clip['id']}-{detect.DET_VERSION}-*.npz"))
    key = hashlib.sha1(json.dumps([FEAT_VERSION, camera.anchors_of(clip), snap, end,
                                   [f.name for f in dets_files]], default=str).encode()).hexdigest()[:12]
    f = cache / "hitfx" / f"{clip['id']}-{key}.npz"
    if f.exists():
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    n = int(round(clip["probe"]["duration"] * FPS)) + 1
    player = None
    if camera.anchors_of(clip):
        try:
            player = camera.player_track(cache, clip)
        except Exception:  # noqa: BLE001
            player = None
    if player is None:
        player = np.full((n, 5), np.nan, np.float32)
    camf = cache / "track" / f"{clip['id']}-cam.npy"
    cam = np.load(camf) if camf.exists() else None
    feats = hitfx.features(root, clip, player, cam, detect.cached(cache, clip), snap, end)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **feats)
    return feats


def training_set(log=lambda m: None) -> list[dict]:
    """Every marked hit in every project: signals, window, your hit, side."""
    from . import library, project
    items = []
    for pr in library.projects():
        root = Path(pr["folder"])
        if not (root / project.FILENAME).exists():
            continue
        for c in project.load(root)["clips"]:
            r = c.get("review") or {}
            if r.get("keep") is False or r.get("hit") is None:
                continue
            try:
                feats = clip_features(root, c)
            except Exception as e:  # noqa: BLE001 - skip a clip we can't read
                log(f"hit model: skipped {c['id']}: {e}")
                continue
            snap, end = _times(c)
            items.append({"f": feats, "snap": snap, "end": end, "hit": float(r["hit"]), "side": r.get("side"),
                          "id": c["id"]})
    return items


# ---------------------------------------------------------------- fitting
def _window(it) -> tuple[int, int]:
    n = len(it["f"]["u"])
    return int((it["snap"] + 0.4) * FPS), min(n, int((it["end"] + 0.3) * FPS))


def _matrix(it) -> np.ndarray:
    """Rows: the standardised signals, then -4(u-u0)^2 for each U0 - what hitfx.score sums."""
    lo, hi = _window(it)
    rows = []
    for k in KEYS:
        v = it["f"][k][lo:hi].astype(np.float64)
        sd = v.std()
        rows.append((v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v))
    u = it["f"]["u"][lo:hi]
    rows += [-4 * (u - u0) ** 2 for u0 in U0S]
    return np.array(rows)


# Each signal on or off with a fixed direction, plus the "mid-play" preference. A small, fixed
# set of weightings: with a few dozen marked plays, searching thousands of random weightings
# overfits (its leave-one-out score swung 17-43% with the random seed; this one is stable).
GRID_KEYS = (("near", 1), ("crowd", 1), ("onset", 1), ("decel", 1), ("accel", -1))


def _candidates() -> np.ndarray:
    import itertools
    rows = []
    for combo in itertools.product((0, 1), repeat=len(GRID_KEYS)):
        for ui, uv in itertools.product(range(len(U0S)), (0, 1, 2)):
            c = np.zeros(len(KEYS) + len(U0S))
            for (k, sign), on in zip(GRID_KEYS, combo):
                c[KEYS.index(k)] = on * sign
            c[len(KEYS) + ui] = uv
            rows.append(c)
    C = np.array(rows)
    mid = np.zeros(len(KEYS) + len(U0S))
    mid[len(KEYS) + 2] = 1                                      # "middle of the play" first, so ties keep it
    return np.vstack([mid, C])


def _errors(items, C) -> np.ndarray:
    """(candidates, items) seconds off."""
    E = np.zeros((len(C), len(items)))
    for j, it in enumerate(items):
        lo, hi = _window(it)
        if hi <= lo + 2:
            E[:, j] = 9
            continue
        S = C @ _matrix(it)
        E[:, j] = np.abs((lo + S.argmax(1)) / FPS - it["hit"])
    return E


def _best(E: np.ndarray, cols=None) -> int:
    M = E if cols is None else E[:, cols]
    return int(np.argmax((M <= TOL).sum(1) - 0.01 * np.median(M, 1)))


def _to_weights(c: np.ndarray) -> dict:
    w = {k: float(v) for k, v in zip(KEYS, c[:len(KEYS)]) if v}
    j = int(np.argmax(np.abs(c[len(KEYS):])))
    w["u"] = float(c[len(KEYS) + j])
    w["_u0"] = U0S[j]
    return w


def fit(items: list[dict]) -> dict:
    """Weights for all plays, and per side when there are enough; with leave-one-out scores."""
    C = _candidates()
    E = _errors(items, C)
    out = {"fitted": datetime.now().isoformat(timespec="seconds"), "n": len(items), "tol": TOL}

    def group(idx):
        b = _best(E, idx)
        loo = []
        for i in idx:
            rest = [k for k in idx if k != i]
            loo.append(E[_best(E, rest), i] if rest else E[b, i])
        loo = np.array(loo)
        return {"weights": _to_weights(C[b]), "n": len(idx),
                "loo_within": round(float(np.mean(loo <= TOL)), 3) if len(loo) else None,
                "loo_median_s": round(float(np.median(loo)), 2) if len(loo) else None}

    out["all"] = group(list(range(len(items))))
    for side in ("O", "D"):
        idx = [i for i, it in enumerate(items) if it["side"] == side]
        if len(idx) >= MIN_SIDE:
            out[side] = group(idx)
    return out


# ---------------------------------------------------------------- use
def path() -> Path:
    from . import library
    return library.home() / MODEL_FILE


def load() -> dict | None:
    p = path()
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def weights_for(side: str | None, default: dict) -> dict:
    m = load()
    if not m:
        return default
    g = m.get(side) if side in ("O", "D") else None
    return (g or m.get("all") or {}).get("weights") or default


def refit(log=print) -> dict | None:
    """Fit on everything marked so far and save. Returns the model."""
    with _lock:
        items = training_set(log)
        if len(items) < 5:
            return None
        t = time.time()
        m = fit(items)
        tmp = path().with_suffix(".tmp")
        tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
        tmp.replace(path())
        a = m["all"]
        log(f"hit model: refit on {m['n']} hits in {time.time() - t:.0f} s; "
            f"leave-one-out {a['loo_within']:.0%} within {TOL}s (median {a['loo_median_s']} s)")
        return m


def schedule(delay: float = 45.0, log=print) -> None:
    """Refit a little after the last change (so marking ten hits refits once)."""
    _pending["t"] = time.time() + delay

    def run():
        while time.time() < _pending["t"]:
            time.sleep(2)
        try:
            refit(log)
        except Exception as e:  # noqa: BLE001 - never break the app over this
            log(f"hit model: refit failed: {e}")
        _pending["thread"] = None

    if _pending["thread"] is None:
        _pending["thread"] = threading.Thread(target=run, daemon=True)
        _pending["thread"].start()
