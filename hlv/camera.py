"""Shared camera planning for the review page and renders.

Works on the 720p review proxy (fast to decode, same timeline as the source) and returns
crop rects normalised to the full display frame, so they apply to any resolution.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from . import autozoom, frames, motion, proxy, track
from .probe import probe

FPS = 30.0
TRACK_W = 960
_frames_cache: OrderedDict = OrderedDict()
_run_cache: dict[str, dict] = {}     # clip id -> tracker runs per click (see track._cached_run)
RUNS_VERSION = "r1"                  # bump when track._run changes, so saved runs are dropped
_clip_locks: dict[str, threading.Lock] = {}
_lock = threading.Lock()


def _clip_lock(cid: str) -> threading.Lock:
    with _lock:
        return _clip_locks.setdefault(cid, threading.Lock())


def _runs_path(cache: Path, cid: str) -> Path:
    return cache / "track" / f"{cid}-runs-{RUNS_VERSION}.pkl"


def _runs(cache: Path, cid: str) -> dict:
    """Tracker runs for this clip, kept on disk so a new click is quick even after a restart."""
    with _lock:
        if cid in _run_cache:
            return _run_cache[cid]
    runs = {}
    f = _runs_path(cache, cid)
    if f.exists():
        try:
            runs = pickle.loads(f.read_bytes())
        except Exception:  # noqa: BLE001 - a damaged cache just means tracking again
            runs = {}
    with _lock:
        return _run_cache.setdefault(cid, runs)


def _proxy_frames(cache: Path, cid: str) -> np.ndarray:
    with _lock:
        if cid in _frames_cache:
            _frames_cache.move_to_end(cid)
            return _frames_cache[cid]
    p = proxy.proxy_path(cache, cid)
    info = probe(p)
    f = frames.read_frames(p, info.width, info.height, fps=FPS, width=TRACK_W, color=True)
    with _lock:
        _frames_cache[cid] = f
        while len(_frames_cache) > 3:  # a few clips' worth of 960px frames in memory
            _frames_cache.popitem(last=False)
    return f


def prepare(cache: Path, cid: str) -> None:
    """Decode a clip's frames ahead of a first click (best effort)."""
    try:
        _proxy_frames(cache, cid)
    except Exception:  # noqa: BLE001 - the click will just load it itself
        pass


def _proxy_size(cache: Path, cid: str) -> tuple[int, int]:
    """Size of the analysis frames (proxy scaled to TRACK_W) without decoding them."""
    info = probe(proxy.proxy_path(cache, cid))
    return TRACK_W, int(round(info.height * TRACK_W / info.width / 2)) * 2


def content_rect(clip: dict, W: int, H: int) -> tuple:
    """Real video area in analysis px (screen recordings: inside the app chrome)."""
    c = clip.get("crop")
    if not c:
        return (0.0, 0.0, float(W), float(H))
    sx, sy = W / clip["probe"]["width"], H / clip["probe"]["height"]
    return (c["x"] * sx, c["y"] * sy, c["w"] * sx, c["h"] * sy)


def anchors_of(clip: dict) -> list[dict]:
    sp = (clip.get("review") or {}).get("spotlight")
    return [a for a in sp if isinstance(a, dict)] if isinstance(sp, list) else []


def _track_file(cache: Path, clip: dict) -> Path:
    from . import segment   # (segment imports this module)
    win = segment._window(clip)   # only the footage a render can use is worth tracking
    key = hashlib.sha1(("v13" + json.dumps([anchors_of(clip), win], sort_keys=True)).encode()).hexdigest()[:12]
    return cache / "track" / f"{clip['id']}-{key}.npy"


def freeze_found(cache: Path, clip: dict) -> str | None:
    """Where the render will freeze on the player, from the saved track only (never tracks):
    "snap", "late" (he's only found after the snap), "none", or None (no clicks / not tracked yet)."""
    if not anchors_of(clip):
        return None
    f = _track_file(cache, clip)
    if not f.exists():
        return None
    from . import segment
    player = np.nan_to_num(track.smooth_track(np.load(f), FPS), nan=-1)
    t_in, t_out, snap = segment._window(clip)
    t0, t1 = segment.extents(clip, clip["probe"]["duration"])
    t, at_snap = segment.freeze_time(player, snap, t0, t1, t_out, [a["t"] for a in anchors_of(clip)])
    return "none" if t is None else ("snap" if at_snap else "late")


_live: dict[str, dict] = {}          # clip id -> unfinished tracker runs (track.Tracking live)
_jobs: dict[str, "_Job"] = {}        # clip id -> latest tracking job
CHUNK = 6                            # frames tracked between looks at "stop" (~0.2 s)


class _Job(threading.Thread):
    """Tracks one clip's clicks in the background, a few frames at a time, so the page can
    show what's tracked so far and a newer click can stop it (finished runs are kept)."""

    def __init__(self, cache: Path, clip: dict):
        super().__init__(daemon=True)
        self.cache, self.clip, self.cid = cache, clip, clip["id"]
        self.file = _track_file(cache, clip)
        self.cancel = False
        self.finished = threading.Event()
        self.result: np.ndarray | None = None
        self.error: Exception | None = None
        self.tracking: track.Tracking | None = None
        self.snapshot: tuple[np.ndarray, float] | None = None   # (raw track so far, progress)
        self.want = threading.Event()        # the page asked what's tracked so far
        self.handed = threading.Event()      # ...and it's in self.snapshot

    def run(self):
        from . import segment   # (segment imports this module)
        try:
            with _clip_lock(self.cid):   # one job per clip at a time
                if self.cancel:
                    return
                fr = _proxy_frames(self.cache, self.cid)
                t0, t1 = segment.extents(self.clip, len(fr) / FPS)
                runs = _runs(self.cache, self.cid)
                with _lock:
                    live = _live.setdefault(self.cid, {})
                self.tracking = track.Tracking(fr, FPS, anchors_of(self.clip), run_cache=runs,
                                               span=(int(t0 * FPS), int(np.ceil(t1 * FPS))), live=live)
                done = self.tracking.finished
                while not done and not self.cancel:
                    if self.want.is_set():   # between chunks, so the runs aren't changing
                        self.want.clear()
                        self.snapshot = (self.tracking.result(), self.tracking.progress)
                        self.handed.set()
                    done = self.tracking.work(CHUNK)
                f = _runs_path(self.cache, self.cid)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(pickle.dumps(runs))   # so the next click is quick, even after a restart
                if done:
                    res = self.tracking.result()
                    np.save(self.file, res)   # raw track cached; smoothing is cheap and tweakable
                    self.result = res
        except Exception as e:  # noqa: BLE001 - reported to whoever waits on the job
            self.error = e
        finally:
            self.finished.set()
            self.handed.set()

    def peek(self) -> tuple[np.ndarray | None, float]:
        """What's tracked so far (raw track) and how far along (waits at most one chunk)."""
        if self.result is None and self.tracking is not None and not self.finished.is_set():
            self.handed.clear()
            self.want.set()
            self.handed.wait(2.0)
        if self.result is not None:
            return self.result, 1.0
        return self.snapshot if self.snapshot is not None else (None, 0.0)


def _job(cache: Path, clip: dict, preempt: bool) -> _Job:
    """The tracking job for this clip's current clicks, started if needed.
    preempt: stop a job tracking other clicks (the page, after a new click). Without it
    (renders, warm-up) a job for other clicks is left to finish first."""
    key = _track_file(cache, clip)
    while True:
        with _lock:
            j = _jobs.get(clip["id"])
            if j is not None and j.file == key and not (j.finished.is_set() and j.result is None):
                return j
            if j is None or j.finished.is_set() or preempt:
                if j is not None:
                    j.cancel = True
                j = _jobs[clip["id"]] = _Job(cache, clip)
                j.start()
                return j
        j.finished.wait()


def player_track(cache: Path, clip: dict, need_runs: bool = False) -> np.ndarray | None:
    """need_runs: track even if the result is cached, when the per-click runs aren't saved
    yet (so the next click on this clip is quick)."""
    if not anchors_of(clip):
        return None
    f = _track_file(cache, clip)
    while True:
        if f.exists() and not (need_runs and not _runs_path(cache, clip["id"]).exists()):
            return track.smooth_track(np.load(f), FPS)
        j = _job(cache, clip, preempt=False)
        j.finished.wait()
        if j.error is not None:
            raise j.error
        if j.result is not None:
            return track.smooth_track(j.result, FPS)
        # Stopped by a newer click on the page: wait our turn and track these clicks again.


def player_track_progressive(cache: Path, clip: dict, wait: float = 2.0) -> tuple[np.ndarray | None, float]:
    """For the review page: what's tracked so far, as soon as there's something around the
    clicks (or the whole track if it's done first; gives up waiting after `wait` s).
    Returns (smoothed track, progress 0..1); a new click stops tracking older clicks."""
    if not anchors_of(clip):
        return None, 1.0
    f = _track_file(cache, clip)
    if f.exists():
        return track.smooth_track(np.load(f), FPS), 1.0
    j = _job(cache, clip, preempt=True)
    end = time.time() + wait
    while not j.finished.is_set() and time.time() < end and (
            j.tracking is None or (j.tracking.done_frames < 2 * CHUNK and not j.tracking.finished)):
        time.sleep(0.03)
    if j.error is not None:
        raise j.error
    res, prog = j.peek()
    return (None if res is None else track.smooth_track(res, FPS)), prog


def warm(cache: Path, clips: list[dict], busy=lambda: False, log=print, stop=lambda: False) -> None:
    """Track every clip that has spotlight clicks ahead of time, so opening one is instant.
    Waits while `busy()` (a reel build is using the CPU and memory); ends when `stop()`
    (another project was opened)."""
    for clip in clips:
        if not anchors_of(clip):
            continue
        while busy() and not stop():
            time.sleep(5)
        if stop():
            return
        t = time.time()
        try:
            player_track(cache, clip, need_runs=True)
        except Exception as e:  # noqa: BLE001 - warming is best effort
            log(f"warm-up: {clip['id']} failed: {e}")
            continue
        if time.time() - t > 1:
            log(f"warm-up: tracked {clip['id']} ({time.time() - t:.0f} s)")


def _camera_motion(cache: Path, cid: str, fr: np.ndarray) -> np.ndarray:
    f = cache / "track" / f"{cid}-cam.npy"
    if f.exists():
        m = np.load(f)
        if len(m) == len(fr):
            return m
    m = track.camera_motion(fr)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, m)
    return m


def _motion_box(cache: Path, clip: dict) -> np.ndarray:
    f = cache / "track" / f"{clip['id']}-motion.npy"
    if f.exists():
        return np.load(f)
    fr = _proxy_frames(cache, clip["id"])
    H, W = fr.shape[1:3]
    x0, y0, cw, ch = (int(round(v)) for v in content_rect(clip, W, H))
    fr = fr[:, y0:y0 + ch, x0:x0 + cw]
    small = np.stack([cv2.resize(x, (640, int(640 * ch / cw) // 2 * 2)) for x in fr])
    gray = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in small])
    box = motion.motion_signals(gray, small, track=True)["box"]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, box)
    return box


def plan(cache: Path, clip: dict, progressive: bool = False) -> dict:
    """Everything the review page needs to preview the camera for one clip.
    progressive: don't wait for tracking to finish; "partial" says it's still going (ask
    again to get more) and "progress" how far along it is."""
    r = clip.get("review") or {}
    d = clip["detection"]
    snap = r.get("snap") if r.get("snap") is not None else d["snap"]
    end = r.get("end") if r.get("end") is not None else d["end"]
    level = float(r.get("zoom") or 0)
    W, H = _proxy_size(cache, clip["id"])
    content = content_rect(clip, W, H)
    if level <= 1 and not anchors_of(clip):
        # Nothing to plan: don't touch the frames at all (they're big).
        return {"fps": FPS, "mode": "off", "level": level, "rects": None, "player": None}
    progress = 1.0
    if progressive:
        player, progress = player_track_progressive(cache, clip)
    else:
        player = player_track(cache, clip)
    rects, mode = None, "off"
    if level > 1:
        if player is not None:
            rects, mode = autozoom.plan_follow(player, W, H, content, snap, end, level), "player"
        elif progress < 1.0:
            pass   # still loading the clip for tracking: no zoom yet
        else:
            # No spotlight: frame the busiest part of the play, gently.
            x0, y0, cw, ch = content
            zc = autozoom.ZoomConfig(max_zoom=min(level, 1.4), min_src_px=0,
                                     out_w=16, out_h=9)
            rr = autozoom.plan(_motion_box(cache, clip), cw, ch, snap, end, zc)
            rects, mode = rr + np.array([x0, y0, 0, 0]), "play"
    norm = np.array([W, H, W, H], np.float64)
    return {
        "fps": FPS, "mode": mode, "level": level,
        "partial": progress < 1.0, "progress": round(progress, 3),
        "rects": None if rects is None else np.round(rects / norm, 4).tolist(),
        "player": None if player is None else np.round(np.nan_to_num(player, nan=-1), 4).tolist(),
    }
