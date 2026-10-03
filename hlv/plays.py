"""Turn cached per-frame signals into snap / end-of-play / in-out proposals.

Pure numpy and fast, so the scorer can re-run it across parameter grids.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import medfilt, savgol_filter

from .config import DetectConfig


def smooth(x: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    if len(x) < 7:
        return x.astype(np.float64)
    k = max(3, int(seconds * fps) | 1)
    y = medfilt(x.astype(np.float64), k)
    win = min(len(y) - (1 - len(y) % 2), max(5, (2 * k) | 1))
    return savgol_filter(y, win, 2) if win >= 5 else y


def activity(sig: dict, fps: float, cfg: DetectConfig) -> np.ndarray:
    """Combined, smoothed activity curve: player residual plus a little camera pan."""
    def norm(v):
        s = np.percentile(v, 95)
        return v / s if s > 1e-9 else v * 0
    a = norm(np.asarray(sig["residual"], np.float64))
    if cfg.pan_weight:
        a = a + cfg.pan_weight * norm(np.asarray(sig["pan"], np.float64))
    return smooth(a, fps, cfg.smooth_s)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def detect_play(act: np.ndarray, fps: float, duration: float, cfg: DetectConfig,
                whistles: list[tuple[float, float]] | None = None) -> dict:
    n = len(act)
    whistles = whistles or []
    if n < int(fps):
        return {"snap": 0.0, "end": duration, "whistle": None, "confidence": 0.0,
                "tight": True, "quiet_s": 0.0}

    lo, hi = np.percentile(act, 10), np.percentile(act, 95)
    rng = max(hi - lo, 1e-9)
    quiet_thr = lo + cfg.quiet_frac * rng
    onset_thr = lo + cfg.onset_frac * rng
    quiet = act <= quiet_thr
    min_quiet = int(cfg.min_quiet_s * fps)
    min_active = max(1, int(cfg.min_active_s * fps))
    horizon = int(5 * fps)

    # --- snap: choose the onset with the most activity after it, favouring a quiet lead-in.
    best = None
    for s, _ in _runs(act >= onset_thr):
        if act[s:s + min_active].mean() < onset_thr:
            continue
        # Walk back to the end of the preceding quiet stretch (the "set").
        k = s
        while k > 0 and not quiet[k - 1]:
            k -= 1
        q_end = k
        while k > 0 and quiet[k - 1]:
            k -= 1
        quiet_len = q_end - k
        gap = s - q_end
        has_quiet = quiet_len >= min_quiet and gap <= int(2.0 * fps)
        mass = float(np.clip(act[s:s + horizon] - lo, 0, None).sum())
        score = mass * (1.0 if has_quiet else 0.5)
        # Motion right after a whistle is players stopping/celebrating, not a new snap.
        t_s = s / fps
        if any(st >= cfg.whistle_min_strength and t_s - 2.5 <= t <= t_s + 0.2 for t, st in whistles):
            score *= cfg.post_whistle_penalty
        if best is None or score > best[0]:
            if has_quiet:
                # The snap is where activity first lifts off the "set" level, which is often
                # well before it clears the coarse quiet threshold.
                level = float(np.median(act[k:q_end]))
                snap_i = q_end
                while snap_i > k and act[snap_i - 1] > level + cfg.snap_eps * rng:
                    snap_i -= 1
            else:
                snap_i = _knee_back(act, s, onset_thr, lo)
            best = (score, snap_i, quiet_len if has_quiet else 0, s)
    if best is None:  # flat clip; nothing stood out
        return {"snap": 0.0, "end": duration, "whistle": None, "confidence": 0.0,
                "tight": True, "quiet_s": 0.0}
    _, snap_i, quiet_len, onset_i = best

    # The main burst often comes seconds after the real snap (play develops, swings toward
    # the camera). Walk back through any lead-up to the last time players were near-still.
    if cfg.snap_floor_eps > 0:
        lb = max(0, onset_i - int(cfg.snap_lookback_s * fps))
        floor = float(act[lb:snap_i + 1].min())
        burst = float(act[onset_i:min(n, onset_i + horizon)].max())
        thr = floor + cfg.snap_floor_eps * (burst - floor)
        j = snap_i
        while j > lb and act[j - 1] > thr:
            j -= 1
        if j < snap_i:
            snap_i = j

    # --- end: settle after the peak of this play.
    pk_hi = min(n, snap_i + int(10 * fps))
    peak_i = snap_i + int(np.argmax(act[snap_i:pk_hi]))
    decay_thr = lo + cfg.decay_frac * (act[peak_i] - lo)
    settle = int(cfg.min_settle_s * fps)
    end_i, settled = n - 1, False
    for s, e in _runs(act[peak_i:] < decay_thr):
        if e - s >= settle or peak_i + e >= n:
            end_i, settled = peak_i + s, (e - s >= settle)
            break
    end_t = end_i / fps

    whistle_t = None
    earliest = snap_i / fps + cfg.min_play_s
    cands = [(t, st) for t, st in whistles
             if st >= cfg.whistle_min_strength and earliest <= t <= end_t + cfg.whistle_window_s]
    if cands:
        whistle_t = min(cands)[0]
        end_t = whistle_t + cfg.whistle_offset_s

    snap_t = snap_i / fps
    end_t = min(end_t, duration)

    # --- confidence
    sharp = (act[onset_i:onset_i + min_active].mean() - act[max(0, snap_i - min_quiet):snap_i + 1].mean()) / rng
    conf = (0.35 * min(1.0, quiet_len / max(min_quiet, 1))
            + 0.35 * float(np.clip(sharp, 0, 1))
            + 0.15 * float(settled)
            + 0.15 * float(whistle_t is not None))
    tight = snap_t < cfg.tight_margin_s and end_t > duration - cfg.tight_margin_s
    return {"snap": round(snap_t, 3), "end": round(end_t, 3),
            "whistle": None if whistle_t is None else round(whistle_t, 3),
            "confidence": round(conf, 3), "tight": bool(tight),
            "quiet_s": round(quiet_len / fps, 3)}


def _knee_back(act: np.ndarray, s: int, onset_thr: float, lo: float) -> int:
    """No quiet lead-in: walk back from the crossing to where the rise began."""
    half = lo + 0.5 * (onset_thr - lo)
    k = s
    while k > 0 and act[k - 1] > half and act[k - 1] <= act[k] + 1e-9:
        k -= 1
    return k


def propose(det: dict, duration: float, kind: str, cfg: DetectConfig) -> tuple[dict, str]:
    """Proposed in/out plus status ('auto' | 'needs_review')."""
    if kind == "pre_edited":
        return {"in": 0.0, "out": round(duration, 3)}, "needs_review"
    pad_b, pad_a = cfg.pad_before, cfg.pad_after
    if kind == "screen_recording" or det["tight"]:
        # Already trimmed by someone (Hudl etc.): only shave obvious dead time, generously.
        pad_b, pad_a = 2 * pad_b, 2 * pad_a
    tin = max(0.0, det["snap"] - pad_b)
    tout = min(duration, det["end"] + pad_a)
    if tout - tin < 2.0:  # degenerate; fall back to whole clip
        tin, tout = 0.0, duration
    status = "auto" if det["confidence"] >= cfg.min_confidence else "needs_review"
    return {"in": round(tin, 3), "out": round(tout, 3)}, status
