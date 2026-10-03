"""Referee-whistle detection.

A whistle is a strong, narrow tone around 3–4 kHz. Crowd noise and cheering are
broadband, so we need both: a sharp spectral peak in the band (tonality) and a band
that carries a real share of the total energy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import medfilt, stft

SR = 16000


def read_audio(path: Path, sr: int = SR) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def whistle_score(x: np.ndarray, sr: int = SR, lo: float = 2800, hi: float = 4500,
                  hop_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Per-hop whistle score in [0, ~1]. Returns (times, score)."""
    if x.size < sr // 10:
        return np.zeros(0), np.zeros(0)
    nper = int(0.032 * sr)
    f, t, z = stft(x, fs=sr, nperseg=nper, noverlap=nper - int(hop_s * sr), boundary=None)
    p = np.abs(z) ** 2 + 1e-12
    band = (f >= lo) & (f <= hi)
    pb = p[band]
    peak = pb.max(axis=0)
    tonality = peak / np.median(pb, axis=0)                     # sharp line vs band floor
    band_share = pb.sum(axis=0) / p[f > 300].sum(axis=0)        # band vs everything above rumble
    # Map to soft 0..1 scores; whistles typically show tonality >> 30 and share > 0.2.
    s_ton = np.clip(np.log10(tonality) / 2.0, 0, 1)             # 100x -> 1.0
    s_share = np.clip(band_share / 0.4, 0, 1)
    score = medfilt(s_ton * s_share, 5)
    return t, score


def whistle_events(t: np.ndarray, score: np.ndarray, thresh: float = 0.25,
                   min_dur: float = 0.15) -> list[tuple[float, float]]:
    """[(onset_time, strength)] for sustained runs above thresh."""
    events = []
    above = score > thresh
    i, n = 0, len(score)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if t[j - 1] - t[i] >= min_dur:
                events.append((float(t[i]), float(score[i:j].mean())))
            i = j
        else:
            i += 1
    return events
