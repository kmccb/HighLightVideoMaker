import cv2
import numpy as np
import pytest

from hlv import audio, crop, motion, plays, project
from hlv.config import DetectConfig

FPS = 30.0


# ---------- camera compensation ----------

def _texture(h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8), np.uint8)
    img = cv2.resize(base, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _pan_zoom_frames(n=30, blob_from=None):
    big = _texture()
    h, w = 360, 640
    frames = []
    for i in range(n):
        s = 1.0 + 0.004 * i            # slow zoom
        tx, ty = 6.0 * i, 1.5 * i      # pan
        m = np.array([[s, 0, -tx - w / 2], [0, s, -ty - h / 2]], np.float32)
        f = cv2.warpAffine(big, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if blob_from is not None and i >= blob_from:
            x = 200 + 12 * (i - blob_from)
            cv2.circle(f, (x, 200), 14, 255, -1)
        frames.append(f)
    return np.stack(frames)


def test_pan_zoom_alone_gives_little_residual():
    sig = motion.motion_signals(_pan_zoom_frames())
    assert sig["pan"][5:].mean() > 3        # camera motion was seen...
    assert sig["residual"][2:].max() < 0.01  # ...and subtracted


def test_moving_player_survives_compensation():
    sig = motion.motion_signals(_pan_zoom_frames(blob_from=15))
    before, after = sig["residual"][2:14].mean(), sig["residual"][17:].mean()
    assert after > 5 * max(before, 1e-4)
    assert after > 0.001


# ---------- whistle ----------

def _pink(n, rng):
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    return np.fft.irfft(f, n)


def test_whistle_found_in_crowd_noise():
    sr, rng = audio.SR, np.random.default_rng(1)
    x = _pink(sr * 6, rng)
    x /= np.abs(x).max()
    t = np.arange(int(0.6 * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 3500 * t)
    x[int(4.0 * sr): int(4.0 * sr) + len(tone)] += tone
    ev = audio.whistle_events(*audio.whistle_score(x.astype(np.float32)))
    assert ev, "whistle not detected"
    assert abs(ev[0][0] - 4.0) < 0.1


def test_no_whistle_in_noise_only():
    rng = np.random.default_rng(2)
    x = _pink(audio.SR * 6, rng)
    x = (x / np.abs(x).max()).astype(np.float32)
    assert audio.whistle_events(*audio.whistle_score(x)) == []


# ---------- segmentation ----------

def _curve(dur, snap, end, rng_seed=0, quiet=0.05, play=1.0):
    n = int(dur * FPS)
    t = np.arange(n) / FPS
    a = np.full(n, quiet)
    rise = np.clip((t - snap) / 0.4, 0, 1)
    fall = np.clip(1 - (t - end) / 0.8, 0, 1)
    a = a + (play - quiet) * rise * fall
    a += np.random.default_rng(rng_seed).normal(0, 0.02, n)
    return np.clip(a, 0, None)


def _sig(a):
    return {"residual": a, "pan": np.zeros_like(a)}


def test_snap_and_end_found():
    cfg = DetectConfig()
    a = plays.activity(_sig(_curve(15, snap=4.0, end=10.0)), FPS, cfg)
    d = plays.detect_play(a, FPS, 15.0, cfg)
    assert abs(d["snap"] - 4.0) < 0.35
    assert abs(d["end"] - 10.3) < 0.8
    assert not d["tight"]
    assert d["confidence"] > 0.5
    prop, status = plays.propose(d, 15.0, "iphone", cfg)
    assert prop["in"] == pytest.approx(d["snap"] - 1.0, abs=1e-3)
    assert status == "auto"


def test_whistle_sets_end():
    cfg = DetectConfig()
    a = plays.activity(_sig(_curve(15, snap=4.0, end=10.0)), FPS, cfg)
    d = plays.detect_play(a, FPS, 15.0, cfg, whistles=[(9.6, 0.9)])
    assert d["end"] == pytest.approx(9.6 + cfg.whistle_offset_s)


def test_tight_clip():
    cfg = DetectConfig()
    a = plays.activity(_sig(_curve(7, snap=0.2, end=6.9)), FPS, cfg)
    d = plays.detect_play(a, FPS, 7.0, cfg)
    assert d["tight"]
    prop, _ = plays.propose(d, 7.0, "screen_recording", cfg)
    assert prop["in"] == 0.0 and prop["out"] == 7.0


def test_biggest_burst_wins_over_presnap_shift():
    cfg = DetectConfig()
    a = _curve(16, snap=6.0, end=12.0)
    a[int(2.0 * FPS):int(2.6 * FPS)] += 0.45  # brief pre-snap motion
    d = plays.detect_play(plays.activity(_sig(a), FPS, cfg), FPS, 16.0, cfg)
    assert abs(d["snap"] - 6.0) < 0.4


def test_pre_edited_passthrough():
    cfg = DetectConfig()
    d = {"snap": 3.0, "end": 8.0, "confidence": 0.9, "tight": False}
    prop, status = plays.propose(d, 57.7, "pre_edited", cfg)
    assert prop == {"in": 0.0, "out": 57.7} and status == "needs_review"


# ---------- crop ----------

def test_crop_finds_live_region():
    rng = np.random.default_rng(3)
    frames = np.full((20, 200, 400), 30, np.uint8)
    frames[:, 0:20, :] = 200                               # static toolbar
    frames[:, 40:180, 50:350] = rng.integers(0, 255, (20, 140, 300))  # live video
    c = crop.detect_crop(frames, 800, 400)
    assert c is not None
    assert abs(c["x"] - 100) <= 4 and abs(c["w"] - 600) <= 6
    assert abs(c["y"] - 80) <= 4 and abs(c["h"] - 280) <= 6


def test_no_crop_for_full_frame_video():
    rng = np.random.default_rng(4)
    frames = rng.integers(0, 255, (20, 100, 200)).astype(np.uint8)
    assert crop.detect_crop(frames, 400, 200) is None


# ---------- project merge ----------

def test_merge_preserves_review():
    proj = {"clips": [{"id": "a", "source": "a.mov", "review": {"in": 1.0, "out": 5.0, "keep": True,
                                                                "spotlight": None, "players": ["caleb"]}}]}
    project.merge_clip(proj, {"id": "a", "source": "a.mov", "proposed": {"in": 2, "out": 6}})
    assert proj["clips"][0]["review"]["players"] == ["caleb"]
    assert proj["clips"][0]["proposed"]["in"] == 2
    assert project.final_window(proj["clips"][0]) == (1.0, 5.0)


def test_snap_found_at_small_liftoff_before_big_jump():
    cfg = DetectConfig()
    a = _curve(20, snap=9.0, end=15.0)
    a[int(7.6 * FPS):int(9.0 * FPS)] += np.linspace(0.0, 0.15, int(1.4 * FPS))  # creeping start
    d = plays.detect_play(plays.activity(_sig(a), FPS, cfg), FPS, 20.0, cfg)
    assert 7.4 < d["snap"] < 8.4


def test_clip_starting_mid_play_ignores_post_whistle_burst():
    cfg = DetectConfig()
    a = _curve(11, snap=0.0, end=4.6, quiet=0.3)
    a[int(5.2 * FPS):int(6.4 * FPS)] += 1.2   # celebration / pile-up after whistle
    d = plays.detect_play(plays.activity(_sig(a), FPS, cfg), FPS, 11.0, cfg, whistles=[(4.9, 0.6)])
    assert d["snap"] < 0.6
    assert d["end"] == pytest.approx(4.9 + cfg.whistle_offset_s)


# ---------- beats ----------

def _drum_track(bpm=120.0, seconds=20.0, sr=22050, seed=0):
    """Kick on every beat (louder on the 1), hats on eighths, plus noise."""
    rng = np.random.default_rng(seed)
    x = 0.02 * rng.standard_normal(int(seconds * sr))
    period = 60.0 / bpm
    t = np.arange(int(0.15 * sr)) / sr
    kick = np.sin(2 * np.pi * (60 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 18)
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr))
    beats = np.arange(0.5, seconds - 0.5, period)
    for k, b in enumerate(beats):
        i = int(b * sr)
        amp = 1.0 if k % 4 == 0 else 0.55
        x[i:i + len(kick)] += amp * kick[:len(x) - i]
        j = int((b + period / 2) * sr)
        x[j:j + len(hat)] += 0.15 * hat[:max(0, len(x) - j)]
    return x.astype(np.float32), beats


def test_beat_tracking_finds_tempo_beats_and_downbeats(monkeypatch, tmp_path):
    from hlv import beats
    x, truth = _drum_track(bpm=120)
    monkeypatch.setattr(beats, "read_audio", lambda p, sr=beats.SR: x)
    res = beats.analyze(tmp_path / "x.wav")
    assert abs(res["bpm"] - 120) < 3
    found = np.array(res["beats"])
    err = np.array([np.abs(found - t).min() for t in truth[2:-2]])
    assert np.median(err) < 0.03 and (err < 0.06).mean() > 0.9
    # The loud "1" of each bar should be among the strongest beats.
    strong = found[np.array(res["strength"]) > 0.8]
    ones = truth[::4]
    assert np.mean([np.abs(ones - s).min() < 0.06 for s in strong]) > 0.8


def test_plan_timing_lands_hits_on_beats():
    from hlv import reel, segment
    period = 0.5
    grid = {"beats": [round(0.25 + k * period, 3) for k in range(200)],
            "strength": [1.0 if k % 4 == 0 else 0.4 for k in range(200)]}
    metas = []
    for k in range(8):   # varied plays: hit 1.5-4 s after the snap, some without a freeze
        lay = {"a_frames": 75, "f_frames": 66 if k % 3 else 0, "b_frames": 300}
        metas.append({"lay": lay, "lead_nom": 1.0, "post_nom": 6.0, "hit": 1.5 + 0.37 * k})
    t = reel.plan_timing(metas, grid, music_start=0.0, t_first=3.65, d=0.35)
    errs = [abs(x["err"]) for x in t if x["err"] is not None]
    assert len(errs) == 8 and max(errs) <= 1 / 30 + 1e-6     # within a frame
    # Nearby beats win: no play is stretched/shrunk by more than about a beat.
    assert max(abs(x["shift"]) for x in t) <= period + 1e-6
    assert np.mean([abs(x["shift"]) for x in t]) <= period / 2
    for x, m in zip(t, metas):
        assert 0 <= x["lead"] <= 2.5
        if m["lay"]["f_frames"]:
            assert segment.FREEZE_MIN - 1e-6 <= x["fz"] <= segment.FREEZE_MAX + 1e-6
        else:
            assert x["fz"] == 0


# ---------- slow-mo ----------

def test_slowmo_timeline_slows_around_hit_and_returns_to_full_speed():
    from hlv import slowmo
    win = slowmo.window(hit=5.0, lo=2.0, hi=9.0)
    ts = np.array(slowmo.timeline(2.0, 9.0, 30, win, 0.4))
    steps = np.diff(ts) * 30                      # source frames advanced per output frame
    assert abs(steps[0] - 1) < 1e-6 and abs(steps[-1] - 1) < 1e-6
    assert abs(steps[np.argmin(np.abs(ts[:-1] - 5.0))] - 0.4) < 1e-3
    assert np.all(np.abs(np.diff(steps)) < 0.1)  # smooth ramp, no jolts
    extra = len(ts) / 30 - 7.0
    assert 0.9 < extra < 1.6                      # ~1.2 s longer than real time


def test_interpolated_frame_puts_moving_object_halfway():
    from hlv import slowmo
    a = np.zeros((216, 384, 3), np.uint8); b = a.copy()
    tex = np.random.default_rng(0).integers(0, 255, (216, 384, 3)).astype(np.uint8)
    a[:] = b[:] = cv2.GaussianBlur(tex, (0, 0), 3) // 3     # textured background
    cv2.rectangle(a, (100, 90), (130, 130), (255, 255, 255), -1)
    cv2.rectangle(b, (120, 90), (150, 130), (255, 255, 255), -1)
    mid = slowmo.Interpolator().between(a, b, 0.5)
    cols = np.where(mid[110, :, 0] > 200)[0]
    assert abs(cols.mean() - 125) < 4             # object centred between 115 and 135


def test_plan_timing_still_takes_a_much_stronger_beat_if_it_is_close():
    from hlv import reel
    # beats 0.5 s apart; the one right after nominal is weak, the next is the big "1"
    grid = {"beats": [0.25 + 0.5 * k for k in range(100)],
            "strength": [1.0 if k % 4 == 0 else 0.2 for k in range(100)]}
    lay = {"a_frames": 75, "f_frames": 66, "b_frames": 300}
    # nominal hit lands at 3.65 + 1.0 + 1.4 + 2.0 = 8.05: weak beats at 7.75 (k=15) and
    # 8.25 (k=16, a "1"), the 1 is only 0.2 s away -> take it
    t = reel.plan_timing([{"lay": lay, "lead_nom": 1.0, "post_nom": 6.0, "hit": 2.0}], grid, 0.0, 3.65, 0.35)
    assert abs(t[0]["beat"] - 8.25) < 1e-6


def test_plan_timing_uses_previous_tail_when_play_has_no_room():
    from hlv import reel
    grid = {"beats": [0.25 + 0.5 * k for k in range(200)], "strength": [0.5] * 200}
    roomy = {"lay": {"a_frames": 75, "f_frames": 0, "b_frames": 300}, "lead_nom": 1.0, "post_nom": 6.0, "hit": 2.0}
    tight = {"lay": {"a_frames": 1, "f_frames": 0, "b_frames": 300}, "lead_nom": 0.0, "post_nom": 5.0, "hit": 1.37}
    t = reel.plan_timing([roomy, tight], grid, 0.0, 3.67, 1 / 3)
    assert t[1]["beat"] is not None and abs(t[1]["err"]) <= 1 / 30 + 1e-6


def test_clicks_act_as_keyframes_when_tracker_strays():
    from hlv import track
    # Textureless frames: the tracker can't follow anything meaningful. Two clicks 1 s apart
    # must still give a position on the straight path between them, never far off it.
    n, H, W = 40, 120, 200
    frames = np.full((n, H, W, 3), 60, np.uint8)
    frames[..., 1] = 140                                   # "grass"
    a0 = {"t": 0.0, "x": 0.10, "y": 0.40, "w": 0.06, "h": 0.15}
    a1 = {"t": 1.0, "x": 0.70, "y": 0.40, "w": 0.06, "h": 0.15}
    res = track.track_player(frames, 30, [a0, a1])
    mid = res[15]
    assert not np.isnan(mid[0])
    assert abs(mid[0] - 0.43) < 0.08 and abs(mid[1] - 0.475) < 0.05


def test_adding_a_click_reuses_runs_and_matches_fresh_track():
    from hlv import track
    # A moving square on grass. Adding a click between two others must give exactly the
    # same track as tracking from scratch (runs are reused as prefixes, not approximated).
    n, H, W = 60, 120, 200
    frames = np.full((n, H, W, 3), 60, np.uint8)
    frames[..., 1] = 140
    for i in range(n):
        x = 20 + 2 * i
        frames[i, 50:70, x:x + 10] = (230, 230, 230)
    box = lambda t: {"t": t, "x": (20 + 60 * t) / W, "y": 48 / H, "w": 14 / W, "h": 24 / H}
    cache = {}
    track.track_player(frames, 30, [box(0.2), box(1.6)], run_cache=cache)
    three = [box(0.2), box(0.9), box(1.6)]
    assert np.array_equal(track.track_player(frames, 30, three, run_cache=cache),
                          track.track_player(frames, 30, three), equal_nan=True)


def test_no_hit_play_keeps_natural_timing():
    from hlv import reel, segment
    grid = {"beats": [0.25 + 0.5 * k for k in range(200)], "strength": [1.0] * 200}
    lay = {"a_frames": 75, "f_frames": 66, "b_frames": 300}
    t = reel.plan_timing([{"lay": lay, "lead_nom": 1.0, "post_nom": 6.0, "hit": -1.0}], grid, 0.0, 3.67, 1 / 3)
    assert t[0]["beat"] is None and t[0]["lead"] == 1.0
    assert abs(t[0]["fz"] - segment.FREEZE_S) < 1 / 30 and abs(t[0]["post"] - 6.0) < 1 / 30
    assert reel._hit_in_play(lay, None, 2.0) == -1.0
