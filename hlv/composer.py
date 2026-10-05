"""A soundtrack composed around the reel: every hit lands on a big impact.

How it fits the video:
  1. grid(): a steady beat grid for the style's tempo. The reel's timing (reel.plan_timing)
     then moves each play so its hit lands on a strong beat - the same machinery as beat
     sync to a song, but the song is written afterwards, to the reel.
  2. render(): the arrangement follows the reel's timeline -
       intro      a pad and a riser into the title slams, impacts on the slams
       each play  groove from the snap; during the freeze the band drops out to a pad and a
                  reverse cymbal; it crashes back in when play resumes; a drum fill and half a
                  beat of silence before the hit; a huge impact on the hit; full energy after
       photos     a lighter breakdown
       ending     a last big chord that rings out under the end card and ScottForge card

Styles: rock (150 BPM, distorted guitars, live kit), trap (140 BPM half-time, 808s, hat
rolls, bells), cinematic (120 BPM, taiko, strings, brass). All instruments are synthesised
here (no samples), so it's free and works offline.
"""
from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
STYLES = {"rock": 150, "trap": 140, "cinematic": 120}
E2 = 82.41                                   # key: E minor
PROGRESSIONS = {"rock": [[0, 8, 10, 7], [0, 10, 8, 10], [0, 5, 8, 7]],
                "trap": [[0, 8, 5, 7], [0, 5, 8, 10], [0, 3, 8, 7]],
                "cinematic": [[0, 8, 3, 10], [0, 5, 8, 7], [0, 8, 10, 5]]}
KEYS = (0, 2, -2, 3, 5)                       # E, F#, D, G, A (minor) - chosen per take
MINOR = {0, 5, 7}                            # chords on these degrees are minor, others major
COMPOSER_VERSION = 2


def grid(style: str, seconds: float = 900.0) -> dict:
    """A steady beat grid (bar downbeats strongest) for reel.plan_timing and the intro."""
    bpm = STYLES.get(style, 150)
    p = 60.0 / bpm
    n = int(seconds / p)
    beats = [round(k * p, 4) for k in range(n)]
    strength = [1.0 if k % 4 == 0 else (0.75 if k % 2 == 0 else 0.5) for k in range(n)]
    return {"beats": beats, "strength": strength, "bpm": float(bpm), "composed": style}


# ================================================================ synthesis helpers
def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * SR)) / SR


def _noise(rng, n) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _filt(x, kind, f, order=2):
    if isinstance(f, (list, tuple)):
        f = [min(v, SR / 2 - 100) for v in f]
    else:
        f = min(f, SR / 2 - 100)
    return sosfilt(butter(order, f, kind, fs=SR, output="sos"), x).astype(np.float32)


def _env(n, attack=0.002, decay=0.3) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return (a * np.exp(-np.maximum(t - attack, 0) / decay)).astype(np.float32)


def _sweep(f0, f1, dur, tau) -> np.ndarray:
    """Sine whose pitch falls from f0 to f1 with time constant tau."""
    t = _t(dur)
    f = f1 + (f0 - f1) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32)


def _add(buf, x, t, gain=1.0):
    i = int(round(t * SR))
    if i >= len(buf) or i + len(x) <= 0:
        return
    j0 = max(0, -i)
    i0 = max(0, i)
    n = min(len(x) - j0, len(buf) - i0)
    buf[i0:i0 + n] += x[j0:j0 + n] * gain


def hz(semi: float) -> float:
    return E2 * 2 ** (semi / 12)


# ---------------------------------------------------------------- drums
def kick(rng, punch=1.0):
    d = 0.55
    body = _sweep(160 * punch, 46, d, 0.035) * _env(int(d * SR), 0.001, 0.32)
    click = _filt(_noise(rng, int(0.004 * SR)), "highpass", 2500) * 0.6
    body[:len(click)] += click
    return np.tanh(body * 1.6) * 0.9


def snare(rng):
    d = 0.4
    n = int(d * SR)
    tone = _sweep(240, 180, d, 0.02) * _env(n, 0.001, 0.07) * 0.6
    sn = _filt(_noise(rng, n), "bandpass", [1200, 9000]) * _env(n, 0.001, 0.13)
    return (tone + sn * 0.9) * 0.8


def clap(rng):
    n = int(0.45 * SR)
    out = np.zeros(n, np.float32)
    for k, off in enumerate((0, 0.011, 0.023, 0.034)):
        burst = _filt(_noise(rng, n), "bandpass", [900, 4500]) * _env(n, 0.0005, 0.012 if k < 3 else 0.16)
        _add(out, burst, off, 0.7)
    return out


def hat(rng, open_=False):
    d = 0.35 if open_ else 0.06
    n = int(d * SR)
    return _filt(_noise(rng, n), "highpass", 7000) * _env(n, 0.0005, 0.12 if open_ else 0.018) * 0.5


def crash(rng, d=2.4):
    n = int(d * SR)
    x = _filt(_noise(rng, n), "highpass", 3500)
    t = np.arange(n) / SR
    ring = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (3120, 4370, 5830, 7210)) * 0.08
    return ((x + ring) * _env(n, 0.001, d / 3.2) * 0.55).astype(np.float32)


def tom(rng, f=110, d=0.5):
    return (_sweep(f * 1.6, f, d, 0.04) * _env(int(d * SR), 0.001, 0.2) * 0.9 +
            _filt(_noise(rng, int(d * SR)), "lowpass", 1500) * _env(int(d * SR), 0.001, 0.03) * 0.3)


def taiko(rng, f=62, d=1.2):
    n = int(d * SR)
    body = _sweep(f * 1.8, f, d, 0.05) * _env(n, 0.002, 0.45)
    skin = _filt(_noise(rng, n), "bandpass", [80, 900]) * _env(n, 0.001, 0.06) * 0.6
    return np.tanh((body + skin) * 1.4)


# ---------------------------------------------------------------- impacts and transitions
def impact(rng, d=3.0):
    """The hit: sub drop, a distorted thud, a noise blast and a long tail."""
    n = int(d * SR)
    sub = _sweep(90, 32, d, 0.25) * _env(n, 0.002, 1.1)
    thud = np.tanh(_sweep(220, 55, d, 0.03) * _env(n, 0.001, 0.12) * 4) * 0.6
    blast = _filt(_noise(rng, n), "lowpass", 3000) * _env(n, 0.001, 0.25) * 0.5
    return (sub * 1.1 + thud + blast).astype(np.float32)


def riser(rng, d):
    """Filtered noise sweeping up plus a rising tone, getting louder into the next moment."""
    n = int(d * SR)
    if n < 100:
        return np.zeros(max(n, 1), np.float32)
    x = _noise(rng, n)
    out = np.zeros(n, np.float32)
    k = 16
    for i in range(k):                                     # piecewise filter sweep
        a, b = i * n // k, (i + 1) * n // k
        f = 300 * (12000 / 300) ** (i / (k - 1))
        out[a:b] = _filt(x[max(0, a - 2000):b], "bandpass", [f * 0.6, f * 1.4])[-(b - a):]
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * np.cumsum(200 * (8 ** (t / d))) / SR) * 0.15
    ramp = (t / d) ** 2
    return ((out * 0.5 + tone) * ramp).astype(np.float32)


def reverse_cymbal(rng, d):
    return crash(rng, max(d, 0.2))[::-1].copy() * 0.8


# ---------------------------------------------------------------- tonal
def ks(freq, dur, rng, damp=0.996, bright=0.5):
    """Karplus-Strong plucked string, computed a period at a time (vectorised)."""
    N = max(2, int(SR / freq))
    n = int(dur * SR)
    y = np.zeros(n + N + 1, np.float32)
    y[:N] = _filt(_noise(rng, N), "lowpass", 2000 + 8000 * bright)
    for s in range(N, n + N, N):
        e = min(s + N, n + N)
        prev = y[s - N:e - N + 1]
        y[s:e] = damp * 0.5 * (prev[:e - s] + prev[1:e - s + 1])
    return y[:n]


def guitar_chord(rng, semi, dur, mute=False):
    """Distorted power chord (root, fifth, octave), slightly strummed, through a 'cabinet'."""
    f = hz(semi + 12)
    damp = 0.97 if mute else 0.9985
    x = np.zeros(int(dur * SR) + 2000, np.float32)
    for k, mul in enumerate((1.0, 1.5, 2.0)):
        _add(x, ks(f * mul, dur, rng, damp, 0.7), 0.006 * k, 0.6)
    if mute:
        x *= _env(len(x), 0.001, 0.09)
    y = np.tanh(x * (14 if not mute else 10))
    y = _filt(_filt(y, "lowpass", 4200, 4), "highpass", 90)
    return y * 0.28


def bass(rng, semi, dur, kind="pick"):
    if kind == "808":
        n = int(dur * SR)
        t = np.arange(n) / SR
        f = hz(semi - 12) * (1 + 0.6 * np.exp(-t / 0.03))
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.002, max(dur * 0.8, 0.3))
        return np.tanh(x * 2.2).astype(np.float32) * 0.8
    x = ks(hz(semi), dur, rng, 0.997, 0.3)
    return np.tanh(_filt(x, "lowpass", 900) * 3) * 0.5


def saw(freq, dur, harmonics=24, detune=0.0):
    t = _t(dur)
    out = np.zeros_like(t, dtype=np.float32)
    for h in range(1, harmonics + 1):
        if freq * h > SR / 2 - 500:
            break
        out += (np.sin(2 * np.pi * freq * h * (1 + detune) * t) / h).astype(np.float32)
    return out


def pad(semis, dur, attack=0.6):
    n = int(dur * SR)
    out = np.zeros(n, np.float32)
    for s in semis:
        for dt in (-0.004, 0.004):
            out += saw(hz(s + 24), dur, 10, dt)[:n]
    out = _filt(out, "lowpass", 1400)
    t = np.arange(n) / SR
    env = np.clip(t / attack, 0, 1) * np.clip((dur - t) / 0.4, 0, 1)
    return out * env * 0.06


def bell(semi, dur=0.9):
    t = _t(dur)
    f = hz(semi + 36)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.15) + \
        0.2 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.06)
    return (x * np.exp(-t / 0.35) * 0.18).astype(np.float32)


def brass(semis, dur):
    """Stab: harmonics that fade from the top down (a closing filter), with a fast attack."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for s in semis:
        f = hz(s + 24)
        for h in range(1, 16):
            if f * h > SR / 2 - 500:
                break
            out += np.sin(2 * np.pi * f * h * t) / h * np.exp(-t * h / (dur * 3))
    return np.tanh(out * 0.6) * _env(n, 0.01, dur / 2) * 0.22


def string_note(semi, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t)
    f = hz(semi + 24) * vib
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(ph * h) / h ** 1.3 for h in range(1, 9))
    return (_filt(x.astype(np.float32), "lowpass", 3000) * _env(n, 0.008, dur * 0.5) * 0.09)


def reverb_ir(rng, d=1.4):
    n = int(d * SR)
    ir = _filt(_noise(rng, n), "lowpass", 5000) * np.exp(-np.arange(n) / SR / (d / 4))
    ir[0] = 0
    return ir / np.sqrt(np.sum(ir ** 2))


def chord_tones(deg, koff=0):
    third = 3 if (deg - koff) % 12 in MINOR else 4
    return [deg, deg + third, deg + 7]


# ================================================================ arrangement
def _energy_fn(tl, total):
    """Energy over time from the reel timeline: 0 pad only, 1 light groove, 2 full, 3 peak."""
    marks = [(0.0, 0)]
    marks.append((tl["intro"]["slams"][1] if len(tl["intro"]["slams"]) > 1 else 1.0, 1))
    for p in tl["plays"]:
        marks.append((p["start"], 1))
        if p.get("fs") is not None:
            marks += [(p["fs"], 0), (p["fe"], 2)]
        else:
            marks.append((p["start"] + 0.5, 2))
        if p.get("hit") is not None:
            marks.append((p["hit"], 3))
    for s, _ in tl.get("photos", []):
        marks.append((s, 1))
    if tl.get("end"):
        marks.append((tl["end"][0], 2))
    if tl.get("credit"):
        marks.append((tl["credit"][0], 0))
    marks.sort()
    ts = np.array([m[0] for m in marks])
    es = [m[1] for m in marks]

    def energy(t):
        i = int(np.searchsorted(ts, t, side="right")) - 1
        return es[max(0, i)]
    return energy


def render(out: Path, style: str, tl: dict, seed: int = 1) -> Path:
    """Write the soundtrack (stereo 16-bit WAV). tl: reel timeline - see reel.build."""
    style = style if style in STYLES else "rock"
    rng = np.random.default_rng(seed)
    bpm = STYLES[style]
    p = 60.0 / bpm
    total = float(tl["total"]) + 2.0
    n = int(total * SR)
    drums, music, low, fxb, verb = (np.zeros(n, np.float32) for _ in range(5))
    energy = _energy_fn(tl, total)
    pick = np.random.default_rng(seed * 7919)
    koff = int(KEYS[(seed - 1) % len(KEYS)])
    prog = [d + koff for d in PROGRESSIONS[style][int(pick.integers(len(PROGRESSIONS[style])))]]
    bars_per_chord = 2 if style == "trap" else 1
    hits = [h for h in ([pp.get("hit") for pp in tl["plays"]]) if h is not None]
    slams = list(tl["intro"]["slams"])
    stops = [(h - 0.5 * p, h) for h in hits]                 # half a beat of silence before each hit
    fills = [(h - 2 * p, h - 0.5 * p) for h in hits]
    freezes = [(pp["fs"], pp["fe"]) for pp in tl["plays"] if pp.get("fs") is not None]

    def in_any(t, spans):
        return any(a <= t < b for a, b in spans)

    # one-shot samples (a few variants so repeats don't sound copy-pasted)
    K = [kick(rng, 1 + 0.05 * i) for i in range(3)]
    S = [snare(rng) for _ in range(3)] if style != "trap" else [clap(rng) for _ in range(3)]
    HC = [hat(rng) for _ in range(4)]
    HO = hat(rng, True)
    CR = crash(rng)
    TK = [taiko(rng, 62 + 6 * i) for i in range(3)]
    TM = [tom(rng, f) for f in (160, 130, 100)]
    IMP = impact(rng)

    steps_total = int(total / (p / 4))
    for st in range(steps_total):
        t = st * p / 4
        beat, sub = divmod(st, 4)
        bar, bib = divmod(beat, 4)                       # beat in bar
        e = energy(t)
        if in_any(t, stops) or in_any(t, freezes):
            continue
        filling = in_any(t, fills)
        deg = prog[(bar // bars_per_chord) % len(prog)]
        step16 = bib * 4 + sub
        vel = 0.85 + 0.15 * rng.random()
        # -------------------------------------------- drums
        if style == "rock":
            if filling:
                if sub % 2 == 0:
                    _add(drums, TM[(st // 2) % 3] if bib >= 2 else S[st % 3], t, 0.7 + 0.3 * vel)
                continue
            if e >= 1 and step16 in (0, 8, 10) and (e >= 2 or step16 != 10):
                _add(drums, K[st % 3], t, vel)
            if e >= 2 and step16 in (4, 12):
                _add(drums, S[st % 3], t, vel)
                _add(verb, S[st % 3], t, 0.25)
            if e >= 1 and sub % 2 == 0:
                _add(drums, HO if (e >= 3 and sub == 0) else HC[st % 4], t, 0.55 * vel)
            if e >= 3 and step16 == 0:
                _add(drums, CR, t, 0.5)
        elif style == "trap":
            if filling:
                roll = 8 if bib >= 2 else 4
                for r in range(roll // 4):
                    _add(drums, HC[r % 4], t + r * p / roll, 0.45)
                if bib == 3 and sub in (0, 2):
                    _add(drums, S[st % 3], t, 0.8)
                continue
            if e >= 1:
                if step16 in (0, 11) or (e >= 2 and step16 == 14 and bar % 2):
                    _add(drums, K[st % 3], t, vel)
                if e >= 2 and step16 == 8:
                    _add(drums, S[st % 3], t, vel)
                    _add(verb, S[st % 3], t, 0.3)
                hat_div = 2 if e >= 2 else 4
                if sub % hat_div == 0 or (sub % 2 == 0 and e >= 1):
                    _add(drums, HC[st % 4], t, 0.4 * vel)
                if e >= 3 and step16 in (13, 15):                 # triplet-ish rolls
                    for r in range(3):
                        _add(drums, HC[r], t + r * p / 12, 0.3)
        else:                                                      # cinematic
            if filling:
                if sub % 2 == 0:
                    _add(drums, TK[st % 3], t, 0.5 + 0.5 * ((t - (min(h for h in hits if h > t) - 2 * p)) / (1.5 * p)))
                continue
            pat = {1: (0, 8), 2: (0, 6, 8, 12), 3: (0, 3, 6, 8, 11, 12, 14)}.get(e, ())
            if step16 in pat:
                _add(drums, TK[st % 3], t, vel * (1.0 if step16 in (0, 8) else 0.7))
                _add(verb, TK[st % 3], t, 0.35)
            if e >= 2 and step16 in (4, 12):
                _add(drums, S[st % 3], t, 0.45 * vel)
            if e >= 3 and step16 == 0:
                _add(drums, CR, t, 0.35)
        # -------------------------------------------- bass / guitars / tonal
        if style == "rock":
            if e >= 2 and sub % 2 == 0:
                mute = not (e >= 3 and step16 in (0, 6, 10))
                _add(music, guitar_chord(rng, deg, p / 2 if mute else p * 1.5, mute), t, 0.9)
                _add(low, bass(rng, deg, p / 2), t, 0.9)
            elif e == 1 and step16 == 0:
                _add(music, guitar_chord(rng, deg, p * 4), t, 0.5)
                _add(low, bass(rng, deg, p * 2), t, 0.7)
        elif style == "trap":
            if e >= 1 and step16 in (0, 11) and (bar % 2 == 0 or step16 == 11):
                _add(low, bass(rng, deg, p * (2.5 if step16 == 0 else 1.2), "808"), t, 0.9)
            if e >= 1 and sub == 0 and bib % 2 == 0:
                tones = chord_tones(deg, koff)
                _add(music, bell(tones[(beat // 2) % 3] + 12 * (bar % 2)), t, 0.8)
        else:
            if e >= 2 and sub % 2 == 0:
                tones = chord_tones(deg, koff)
                _add(music, string_note(tones[(st // 2) % 3] + 12, p / 2), t, 0.9)
            if e >= 3 and step16 in (0, 6):
                _add(music, brass(chord_tones(deg, koff), p * 1.2), t, 1.0)
            if e >= 1 and step16 == 0:
                _add(low, bass(rng, deg, p * 3), t, 0.6)

    # -------------------------------------------- pads (always, quieter when the band plays)
    bar_len = 4 * p
    for b in range(int(total / bar_len) + 1):
        t = b * bar_len
        if b % bars_per_chord:
            continue
        deg = prog[(b // bars_per_chord) % len(prog)]
        e = energy(t)
        _add(music, pad(chord_tones(deg, koff), bar_len * bars_per_chord + 0.4), t, 1.4 if e <= 1 else 0.8)

    # -------------------------------------------- moments: risers, impacts, freezes, ending
    for h in hits:
        _add(fxb, riser(rng, 2 * p), h - 2 * p, 0.7)
        _add(fxb, IMP, h, 1.0)
        _add(drums, CR, h, 0.8)
        _add(verb, IMP, h, 0.3)
        deg = koff
        if style == "rock":
            _add(music, guitar_chord(rng, deg, p * 4), h, 1.2)
        elif style == "trap":
            _add(low, bass(rng, deg, p * 4, "808"), h, 1.1)
        else:
            _add(music, brass(chord_tones(deg, koff) + [12 + koff], p * 3), h, 1.3)
    for k, s in enumerate(slams):                                  # intro title slams
        big = k < 2
        if k == 0:
            _add(fxb, riser(rng, min(s, 3.0)), s - min(s, 3.0), 0.8)
        _add(fxb, IMP, s, 0.9 if big else 0.45)
        _add(drums, CR, s, 0.6 if big else 0.3)
    for fs, fe in freezes:
        d = fe - fs
        _add(fxb, reverse_cymbal(rng, d), fe - d, 0.7)
        _add(drums, CR, fe, 0.6)
        _add(drums, K[0], fe, 1.0)
    if tl.get("credit"):
        t = tl["credit"][0]
        _add(fxb, IMP, t, 0.8)
        _add(drums, CR, t, 0.7)
        _add(music, pad(chord_tones(koff, koff) + [12 + koff], 7.0, 0.05), t, 2.2)
        if style == "rock":
            _add(music, guitar_chord(rng, koff, 6.0), t, 1.3)
        elif style == "cinematic":
            _add(music, brass(chord_tones(koff, koff) + [12 + koff], 3.0), t, 1.3)

    # -------------------------------------------- mix
    ir = reverb_ir(rng)
    wet = fftconvolve(verb + music * 0.25 + fxb * 0.15, ir)[:n].astype(np.float32) * 0.35
    lowb = _filt(low, "lowpass", 220 if style == "trap" else 400)
    # sidechain: the low end ducks under each kick a little
    # the stop before each hit: everything but the riser drops away (pads and echoes too)
    duck = np.ones(n, np.float32)
    ramp = int(0.03 * SR)
    for a_, b_ in stops:
        i0, i1 = int(a_ * SR), int(b_ * SR)
        if 0 <= i0 < i1 <= n:
            duck[i0:i1] = 0.08
            duck[i0:i0 + ramp] = np.linspace(1, 0.08, min(ramp, i1 - i0))
    mix = (drums + music + lowb + wet) * duck + fxb * 0.9
    mix = _filt(mix, "highpass", 28)
    rms = np.sqrt(np.mean(mix ** 2) + 1e-12)
    mix = np.tanh(mix / rms * 0.16 * 1.4) / np.tanh(1.4) * 0.9                    # glue + loudness
    # stereo: a touch of width from the reverb
    side = (wet - np.roll(wet, int(0.012 * SR))) * 0.3
    st = np.stack([mix + side, mix - side], 1)
    st /= max(1.0, float(np.abs(st).max()) / 0.95)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    return out


def build(cache: Path, style: str, tl: dict, seed: int = 1) -> Path:
    """Render (or reuse) the soundtrack for this timeline."""
    key = hashlib.sha1(json.dumps([COMPOSER_VERSION, style, seed, tl], sort_keys=True, default=lambda v: round(float(v), 3)).encode()).hexdigest()[:12]
    out = Path(cache) / "compose" / f"score-{style}-{key}.wav"
    return out if out.exists() else render(out, style, tl, seed)


def demo_timeline(style: str) -> dict:
    """A short made-up reel (intro, three plays, a photo, the ending) for previewing a style."""
    p = 60.0 / STYLES.get(style, 150)
    b = lambda k: round(k * p, 3)                                   # noqa: E731
    slams = [b(4), b(8), b(10), b(11)]
    plays, t = [], b(14)
    for k in range(3):
        fs, fe = t + b(2), t + b(4)
        hit = fe + b(6 + 2 * k)
        plays.append({"start": t, "end": hit + b(6), "fs": fs, "fe": fe, "hit": hit})
        t = hit + b(6)
        if k == 1:
            plays[-1]["end"] = t
            photo = (t, t + b(8))
            t += b(8)
    return {"intro": {"end": b(14), "slams": slams}, "plays": plays, "photos": [photo],
            "end": (t, t + 4.0), "credit": (t + 4.0, t + 11.0), "total": t + 11.0}
