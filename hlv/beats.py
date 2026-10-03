"""Find the beat grid of a music track, and which beats hit hardest.

Onset strength = rectified spectral flux on a log spectrogram (kick/bass band weighted
up, since that's what makes a hit "slam"). Tempo from autocorrelation, beats from
dynamic-programming beat tracking (Ellis 2007), then each beat gets a strength and the
bar's downbeat phase gets a bonus. Pure numpy/scipy - no extra installs.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import stft

SR = 22050
HOP = 256                       # ~11.6 ms
FPS = SR / HOP
NFFT = 1024
# An onset shows up as soon as it enters the analysis window, i.e. early by about the
# window length less a hop; shift beat times back by that.
LAG_S = (NFFT - HOP) / SR


def read_audio(path: Path, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def onset_envelope(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """(all-band onset strength, low-band onset strength), one value per hop."""
    f, _, z = stft(x, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
    S = np.log1p(100 * np.abs(z))
    flux = np.maximum(0, np.diff(S, axis=1, prepend=S[:, :1]))
    norm = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
    low = norm(flux[f < 200].sum(axis=0))
    allb = norm(flux.sum(axis=0))
    # Track on the kick/bass: hi-hats on the off-beats would otherwise steal the grid.
    return 0.35 * allb + low, low


def tempo(env: np.ndarray, lo_bpm: float = 70, hi_bpm: float = 180) -> float:
    """Beat period (in hops) from the onset autocorrelation, with a mild preference for ~120 BPM."""
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lags = np.arange(len(ac))
    bpm = 60 * FPS / np.maximum(lags, 1)
    ok = (bpm >= lo_bpm) & (bpm <= hi_bpm)
    weight = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 120) / 0.9) ** 2)
    score = np.where(ok, ac * weight, -np.inf)
    k = int(np.argmax(score))
    # Parabolic interpolation: integer lags are too coarse (120 vs 117 BPM drifts fast).
    if 0 < k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return float(k + 0.5 * (a - c) / den)
    return float(k)


def track(env: np.ndarray, period: float, tightness: float = 100.0) -> np.ndarray:
    """Dynamic-programming beat tracker; returns beat positions in hops."""
    n = len(env)
    score = env.copy()
    back = np.full(n, -1)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    prev = np.arange(-hi, -lo + 1)
    penalty = -tightness * np.log(-prev / period) ** 2
    for t in range(hi, n):
        cand = score[t + prev] + penalty
        k = int(np.argmax(cand))
        score[t] = env[t] + cand[k]
        back[t] = t + prev[k]
    # Best end within the last period, then follow the chain back.
    t = int(n - hi + np.argmax(score[n - hi:]))
    beats = [t]
    while back[t] >= 0:
        t = back[t]
        beats.append(t)
    return np.array(beats[::-1])


def analyze(path: Path) -> dict:
    x = read_audio(path)
    env, low = onset_envelope(x)
    period = tempo(env)
    b = track(env, period)
    # Strength of each beat: peak low-band + overall onset near it (kick drums slam).
    w = max(1, int(0.05 * FPS))
    pk_low, pk_all = maximum_filter1d(low, 2 * w + 1), maximum_filter1d(env, 2 * w + 1)
    raw = 0.65 * pk_low[b] + 0.35 * pk_all[b]
    # Downbeats: the phase (of 4) with the strongest average beat gets a bonus.
    phase = int(np.argmax([raw[p::4].mean() if len(raw[p::4]) else -9 for p in range(4)]))
    down = (np.arange(len(b)) % 4) == phase
    strength = (raw - raw.min()) / (np.ptp(raw) + 1e-9)
    strength = np.clip(strength + 0.25 * down, 0, 1.25) / 1.25
    return {"bpm": round(60 * FPS / period, 1), "beats": np.round(b / FPS + LAG_S, 3).tolist(),
            "strength": np.round(strength, 3).tolist(), "downbeat": down.tolist(),
            "duration": round(len(x) / SR, 2)}


def cached(path: Path, cache_dir: Path) -> dict:
    st = path.stat()
    f = cache_dir / "beats" / f"{path.stem}-{st.st_size}-{int(st.st_mtime)}.json"
    if f.exists():
        return json.loads(f.read_text())
    res = analyze(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res))
    return res
