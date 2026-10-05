"""People in every frame of a play: a person detector (YOLO11s) run once per clip over the
part a render can use, at 15 frames a second, on the 960-px analysis frames.

Saved to .hlv_cache/det/<clip>-<key>.npz so it never runs twice. Boxes are normalised to the
full frame as (cx, cy, w, h, conf), like the player track.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MODEL = Path(__file__).parent / "models" / "yolo11s.pt"
STEP = 2                 # detect on every 2nd analysis frame (30 fps -> 15 fps)
IMGSZ = 960
CONF = 0.15
DET_VERSION = "d1"
_model = None
_model_lock = threading.Lock()


def available() -> bool:
    if not MODEL.exists():
        return False
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(str(MODEL))
        return _model


def _path(cache: Path, clip: dict, span: tuple[int, int]) -> Path:
    return Path(cache) / "det" / f"{clip['id']}-{DET_VERSION}-{span[0]}-{span[1]}.npz"


def _load(f: Path) -> dict[int, np.ndarray]:
    with np.load(f) as z:
        idx, cnt, boxes = z["idx"], z["cnt"], z["boxes"]
    out, o = {}, 0
    for i, k in zip(idx, cnt):
        out[int(i)] = boxes[o:o + k]
        o += k
    return out


def cached(cache: Path, clip: dict) -> dict[int, np.ndarray] | None:
    """Saved detections for this clip (any span), or None - never runs the detector."""
    fs = sorted((Path(cache) / "det").glob(f"{clip['id']}-{DET_VERSION}-*.npz"))
    return _load(fs[-1]) if fs else None


def detections(cache: Path, clip: dict, frames: np.ndarray, span: tuple[int, int]) -> dict[int, np.ndarray]:
    """{frame index: (k, 5) array of cx, cy, w, h, conf} for every STEP-th frame in span."""
    f = _path(cache, clip, span)
    if f.exists():
        return _load(f)
    model = _get_model()
    n, H, W = frames.shape[:3]
    lo, hi = max(0, span[0]), min(n, span[1])
    idx = list(range(lo - lo % STEP, hi, STEP))
    out = {}
    for b0 in range(0, len(idx), 8):                       # small batches keep memory flat
        batch = [frames[i] for i in idx[b0:b0 + 8]]
        res = model.predict(batch, imgsz=IMGSZ, conf=CONF, classes=[0], verbose=False)
        for i, r in zip(idx[b0:b0 + 8], res):
            b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            c = r.boxes.conf.cpu().numpy().astype(np.float32)
            out[i] = np.c_[(b[:, 0] + b[:, 2]) / 2 / W, (b[:, 1] + b[:, 3]) / 2 / H,
                           (b[:, 2] - b[:, 0]) / W, (b[:, 3] - b[:, 1]) / H, c] if len(b) else np.zeros((0, 5), np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(out)
    np.savez_compressed(f, idx=np.array(keys, np.int32), cnt=np.array([len(out[i]) for i in keys], np.int32),
                        boxes=np.concatenate([out[i] for i in keys]) if keys else np.zeros((0, 5), np.float32))
    return out


def warm(cache: Path, clips: list[dict], busy=lambda: False, log=print, stop=lambda: False) -> None:
    """Detect people in every play ahead of time (for hit suggestions). Waits while busy()."""
    import time
    from . import camera, segment
    if not available():
        return
    for clip in clips:
        if (clip.get("review") or {}).get("keep") is False or cached(cache, clip) is not None:
            continue
        while busy() and not stop():
            time.sleep(5)
        if stop():
            return
        t = time.time()
        try:
            fr = camera._proxy_frames(cache, clip["id"])
            t0, t1 = segment.extents(clip, len(fr) / camera.FPS)
            detections(cache, clip, fr, (int(t0 * camera.FPS), int(np.ceil(t1 * camera.FPS))))
        except Exception as e:  # noqa: BLE001 - best effort
            log(f"detect warm-up: {clip['id']} failed: {e}")
            continue
        log(f"detect warm-up: {clip['id']} ({time.time() - t:.0f} s)")
