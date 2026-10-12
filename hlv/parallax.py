"""Big-play effect: the 3D freeze. On the hit the picture stops, moves in on your player and the
camera makes one slow pass around him with real depth, his name on a line from his shoulder.

No depth model: a football field is a flat plane seen at an angle, so how far away something is
follows from how high its feet are in the picture. The freeze frame is split into layers - the
field (sheared about the row he stands on), everything beyond the field's far edge (moves as one),
and every person as an upright cut-out standing at his feet - and a sideways camera move slides
each layer by its own depth. Outlines come from a person-segmentation model (models/yolo11m-seg.pt);
behind the people the empty field comes from a clean plate (the median of the seconds before, with
anyone still standing in it painted out).

`render` makes the freeze as its own silent clip, cut into the play at the hit by reel.build
exactly like the ghost trail (ghost.py), whose helpers it shares.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import camera, detect, encoder, ghost, graphics
from .pipeline import CACHE_DIRNAME

VERSION = 1
FPS = 30
SEG = detect.MODEL.with_name("yolo11m-seg.pt")     # person outlines (Ultralytics; Tom OK'd the download 2026-10-11)
BACK = 4.0            # seconds before the hit used for the clean plate
SWING = 0.085         # how far the camera goes: the nearest row slides this fraction of the picture width
IN, OUT = 0.45, 0.4   # moving in, and back out to the live picture
_seg = None


def available() -> bool:
    return SEG.exists() and detect.available()


def _key(root: Path, clip: dict, pl: dict | None, hit_t: float, secs: float, size=None) -> str:
    from . import segment
    r = clip.get("review") or {}
    src = root / clip["source"]
    pl = pl or {}
    payload = {"v": VERSION, "src": [src.stat().st_size, src.stat().st_mtime], "win": segment._window(clip),
               "hit": round(float(hit_t), 3), "spot": r.get("spotlight"), "zoom": r.get("zoom"), "crop": clip.get("crop"),
               "secs": secs, "pl": [pl.get("color"), pl.get("name"), pl.get("number")]}
    if size and tuple(size) != ghost.WIDE:
        payload["size"] = list(size)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _paths(root: Path, clip: dict, pl: dict | None, hit_t: float, secs: float, size=None) -> tuple[Path, Path]:
    stem = root / CACHE_DIRNAME / "freeze3d" / f"{clip['id']}-{_key(root, clip, pl, hit_t, secs, size)}"
    return stem.with_suffix(".mp4"), stem.with_suffix(".json")


def status(root: Path, clip: dict, pl: dict | None, hit_t: float | None, secs: float, size=None) -> dict | None:
    """The saved result ({"ok": True} or {"none": reason}); None if it hasn't been made yet."""
    if hit_t is None:
        return {"none": "no hit"}
    if not available():
        return {"none": "the cut-out model isn't installed"}
    mp4, meta = _paths(root, clip, pl, hit_t, secs, size)
    if not meta.exists():
        return None
    d = json.loads(meta.read_text())
    return d if d.get("none") or mp4.exists() else None


def _outlines(img: np.ndarray, imgsz: int) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Everyone in the picture: (box x0 y0 x1 y1, confidence, mask the size of img)."""
    global _seg
    if _seg is None:
        _seg = detect._import_ultralytics().YOLO(str(SEG))
    r = _seg.predict([img], imgsz=imgsz, conf=0.2, classes=[0], verbose=False, retina_masks=True)[0]
    if r.masks is None:
        return []
    b, c, m = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.masks.data.cpu().numpy() > 0.5
    return [(b[i], float(c[i]), m[i]) for i in range(len(b))]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)


def people(img: np.ndarray, focus: tuple[float, float, float]) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Everyone in the frame with an outline. focus (x, y, his height): the area round your player is
    looked at again enlarged - a far-off player is only a few dozen pixels tall, and the outlines of
    him and whoever is on him come out much finer; those replace the first-pass ones."""
    H, W = img.shape[:2]
    out = _outlines(img, 1600)
    fx, fy, fh = focus
    R = int(max(6 * fh, 360))
    X0, Y0 = int(np.clip(fx - R / 2, 0, max(0, W - R))), int(np.clip(fy - R / 2, 0, max(0, H - R)))
    X1, Y1 = min(W, X0 + R), min(H, Y0 + R)
    k = min(3.0, 1280 / max(X1 - X0, Y1 - Y0))
    big = cv2.resize(img[Y0:Y1, X0:X1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    for b, c, m in _outlines(big, 1280):
        bb = b / k + [X0, Y0, X0, Y0]
        if min(bb[0] - X0, bb[1] - Y0, X1 - bb[2], Y1 - bb[3]) < 3:      # cut off by the edge of the close look
            continue
        mm = np.zeros((H, W), bool)
        mm[Y0:Y1, X0:X1] = cv2.resize(m.astype(np.uint8), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR) > 0
        out = [o for o in out if _iou(bb, o[0]) < 0.35] + [(bb, c, mm)]
    return out


def which(boxes: np.ndarray, hx: float, hy: float, hw: float, hh: float) -> int | None:
    """The person box that is your player: the one holding the tracker's box, centred on it."""
    best, best_s = None, 0.0
    for k, b in enumerate(boxes):
        ix = max(0.0, min(hx + hw / 2, b[2]) - max(hx - hw / 2, b[0]))
        iy = max(0.0, min(hy + hh / 2, b[3]) - max(hy - hh / 2, b[1]))
        held = ix * iy / max(1.0, hw * hh)
        score = held - 0.5 * abs((b[0] + b[2]) / 2 - hx) / max(1.0, b[2] - b[0])
        if held >= 0.5 and (b[3] - b[1]) <= 5 * hh and score > best_s:
            best, best_s = k, score
    return best


def swing(t: float, secs: float) -> tuple[float, float, float, float]:
    """The camera move at t seconds into a freeze `secs` long: (zoom 0..1, orbit -1..1, lift 0..1, back 0..1).
    In fast, one slow pass around him left to right, back out fast; everything is 0 at both ends
    so the clip starts and ends on the live picture."""
    e = ghost._ease
    back = e((t - (secs - OUT)) / OUT)
    zoom = e(t / IN) * (1 - back)
    orbit = (-1 + 2 * e((t - IN * 0.4) / (secs - OUT - IN * 0.4))) * e(t / IN) * (1 - back)
    return zoom, orbit, e(t / 0.5) * (1 - back), back


def render(root: Path, clip: dict, pl: dict | None, hit_t: float | None, secs: float, log=print,
           size: tuple[int, int] | None = None) -> Path | None:
    """Make (or reuse) the 3D freeze for this play: a silent 16:9 clip `secs` long that starts and ends
    on the reel's own framing of the hit frame. None when it can't be made (no hit, he isn't
    marked, the model is missing)."""
    from . import segment
    if hit_t is None or not available():
        return None
    root = Path(root)
    size = tuple(size) if size else ghost.WIDE
    mp4, meta = _paths(root, clip, pl, hit_t, secs, size)
    known = status(root, clip, pl, hit_t, secs, size)
    if known is not None:
        return None if known.get("none") else mp4
    mp4.parent.mkdir(parents=True, exist_ok=True)

    def none(why: str):
        meta.write_text(json.dumps({"none": why}))
        log(f"  no 3D freeze for {clip['source']}: {why}")
        return None

    pl = pl or {}
    cache = root / CACHE_DIRNAME
    plan = camera.plan(cache, clip)
    if not plan.get("player"):
        return none("your player isn't marked on this play")
    info, crop = clip["probe"], clip.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    Wc, Hc = (crop["w"], crop["h"]) if crop else (info["width"], info["height"])
    OW, OH = size
    tF = round(float(hit_t) * FPS) / FPS
    ta = round(max(0.0, tF - BACK) * FPS) / FPS
    iF = int(round((tF - ta) * FPS))
    if iF < 20:
        return none("the hit comes too soon in the video")
    ws = min(1.0, 1920 / Wc)
    W, H = int(round(Wc * ws / 2)) * 2, int(round(Hc * ws / 2)) * 2
    frames = ghost._read(root / clip["source"], ta, iF + 1, crop, (W, H))
    if len(frames) < iF + 1:
        return none("the video ends before the hit")
    Fw = frames[iF]
    if ws < 1:
        full = ghost._read(root / clip["source"], ta, iF + 1, crop, (Wc, Hc), keep={iF}).get(iF)
        full = ghost._like(full, Fw) if full is not None else cv2.resize(Fw, (Wc, Hc), interpolation=cv2.INTER_CUBIC)
    else:
        full = Fw
    fs = 1 / ws                                                  # working px -> full px

    # ---- the empty field behind the players ----
    G = ghost._homographies(frames, iF)
    idx = list(range(0, iF + 1, max(2, (iF + 1) // 40)))
    stack = np.stack([cv2.warpPerspective(frames[k], G[k], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for k in idx])
    del frames
    plate = np.empty((H, W, 3), np.uint8)
    for y in range(0, H, 120):
        plate[y:y + 120] = np.median(stack[:, y:y + 120], axis=0).astype(np.uint8)
    del stack
    # Match the plate's light to the freeze frame's: the exposure drifts over a few seconds, and an
    # unmatched plate shows as a blob of a different colour round every player. A rough pass over
    # everything first (so "didn't change" can be told at all), then one measured only on the pixels
    # that didn't change (a white jersey mustn't brighten the grass around it).
    Ff, Pf = Fw.astype(np.float32), plate.astype(np.float32)
    Pf *= np.clip(cv2.GaussianBlur(Ff, (0, 0), 35) / np.maximum(1.0, cv2.GaussianBlur(Pf, (0, 0), 35)), 0.6, 1.6)
    same = (np.abs(Ff - Pf).max(axis=2) < 22).astype(np.float32)[..., None]
    wsum = cv2.GaussianBlur(same, (0, 0), 30)[..., None]
    ratio = cv2.GaussianBlur(Ff * same, (0, 0), 30) / np.maximum(1e-3, cv2.GaussianBlur(Pf * same, (0, 0), 30))
    plate = np.clip(Pf * np.where(wsum > 0.15, np.clip(ratio, 0.75, 1.35), 1.0), 0, 255).astype(np.uint8)
    del Ff, Pf, same, wsum, ratio
    moved = cv2.GaussianBlur(cv2.absdiff(Fw, plate).max(axis=2), (0, 0), 1.2) > 46

    # ---- everyone in the picture, and which one is him ----
    p = np.array(plan["player"])[min(len(plan["player"]) - 1, int(round(tF * FPS)))]
    if p[0] < 0 or np.isnan(p[0]):
        return none("he isn't found at the hit")
    hx, hy = (p[0] * info["width"] - ox) * ws, (p[1] * info["height"] - oy) * ws
    hw, hh = p[2] * info["width"] * ws, p[3] * info["height"] * ws
    found = people(Fw, (hx, hy, hh))
    boxes = np.array([[*b, c_] for b, c_, _ in found], np.float32).reshape(-1, 5)
    masks = [m for _, _, m in found]
    best = which(boxes, hx, hy, hw, hh)
    if best is None:
        return none("couldn't make him out from the players around him")
    owner = np.full((H, W), -1, np.int32)                        # whose outline each pixel is (him last, so he wins)
    for k in [k for k in range(len(masks)) if k != best] + [best]:
        owner[masks[k]] = k

    # geometry: the far edge of the field (everything above it is "beyond"), and the row he stands on
    green = cv2.inRange(cv2.cvtColor(Fw, cv2.COLOR_BGR2HSV), (30, 40, 30), (90, 255, 255)) > 0
    rows = np.where(green.mean(axis=1) > 0.35)[0]
    y_piv = float(boxes[best][3])                               # his feet
    y_top = min(float(rows.min()) if len(rows) else H * 0.3, y_piv - 0.03 * H)

    # anyone still standing in the "empty" field (he didn't move in those seconds) is painted out of it
    still = np.zeros((H, W), np.uint8)
    for _, _, m in _outlines(plate, 1600):
        still |= m.astype(np.uint8)
    if still.any():
        still = cv2.dilate(still, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        plate = cv2.inpaint(plate, still * 255, 5, cv2.INPAINT_TELEA)
    plate_full = cv2.resize(plate, (Wc, Hc), interpolation=cv2.INTER_CUBIC) if ws < 1 else plate
    k5, k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    layers = []                                                  # per person: rect (full px), rgb, mask, feet row, is him
    bg = full.copy()
    for k, b in enumerate(boxes):
        if (b[3] - b[1]) < 14 or (b[2] - b[0]) < 6:
            continue
        # his outline, plus anything touching it that moved and isn't someone else (a hand, a foot the outline missed)
        px_, py_ = 0.22 * (b[2] - b[0]) + 6, 0.12 * (b[3] - b[1]) + 6
        X0, Y0, X1, Y1 = int(max(0, b[0] - px_)), int(max(0, b[1] - py_)), int(min(W, b[2] + px_)), int(min(H, b[3] + py_))
        m = masks[k][Y0:Y1, X0:X1].astype(np.uint8)
        own = owner[Y0:Y1, X0:X1]
        extra = moved[Y0:Y1, X0:X1] & (cv2.dilate(m, k11) > 0) & ((own < 0) | (own == k))
        m = cv2.morphologyEx(m | extra.astype(np.uint8), cv2.MORPH_CLOSE, k5)
        m[(own >= 0) & (own != k)] = 0
        if m.sum() < 30:
            continue
        grown = cv2.dilate(m, k9)
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
        FX0, FY0, FX1, FY1 = [int(round(v * fs)) for v in (X0, Y0, X1, Y1)]
        if ws < 1:
            m = cv2.resize(m, (FX1 - FX0, FY1 - FY0), interpolation=cv2.INTER_LINEAR)
            grown = cv2.resize(grown, (FX1 - FX0, FY1 - FY0), interpolation=cv2.INTER_NEAREST)
        layers.append({"rect": (FX0, FY0, FX1, FY1), "rgb": full[FY0:FY1, FX0:FX1].astype(np.float32), "m": m,
                       "feet": float(b[3]) * fs, "hero": k == best})
        gf = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), 2 * fs)[..., None]      # the field behind him, from the plate
        bg[FY0:FY1, FX0:FX1] = (bg[FY0:FY1, FX0:FX1] * (1 - gf) + plate_full[FY0:FY1, FX0:FX1] * gf).astype(np.uint8)
    del plate_full, masks, owner
    layers.sort(key=lambda L: L["feet"])                         # far ones first
    hero_L = next((L for L in layers if L["hero"]), None)
    if hero_L is None:
        return none("couldn't cut him out")
    y_top_f, y_piv_f = y_top * fs, y_piv * fs

    # ---- focus: sharp at his depth, softer with distance (in front of him and behind) ----
    def blur_at(feet_row: float) -> float:
        d = (feet_row - y_piv_f) / max(1.0, Hc - y_top_f)
        return (9.0 * d if d > 0 else 6.0 * min(1.0, -d * 1.6)) * fs

    b1, b2 = cv2.GaussianBlur(bg, (0, 0), 2.5 * fs), cv2.GaussianBlur(bg, (0, 0), 6.5 * fs)
    sig = np.array([blur_at(max(y, y_top_f)) / fs for y in range(Hc)], np.float32)
    w1, w2 = np.clip(sig / 2.5, 0, 1)[:, None, None], np.clip((sig - 2.5) / 4.0, 0, 1)[:, None, None]
    bgd = bg * (1 - w1) + b1 * w1
    bgd = (bgd * (1 - w2) + b2 * w2).astype(np.uint8)
    del b1, b2, bg
    for L in layers:
        s_ = 0.0 if L["hero"] else blur_at(L["feet"])
        if s_ > 0.6:
            L["rgb"] = cv2.GaussianBlur(L["rgb"], (0, 0), s_)
            L["m"] = cv2.GaussianBlur(L["m"], (0, 0), s_ * 0.6)
        L["m"] = L["m"][..., None]
    tc = pl.get("color") or "#1f4fd1"
    team = np.array([int(tc[j:j + 2], 16) for j in (5, 3, 1)], np.float32)
    pad = int(16 * fs)
    hm_p = cv2.copyMakeBorder(hero_L["m"][..., 0], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    hero_rim = (np.clip(cv2.GaussianBlur(hm_p, (0, 0), 3.5 * fs) * 1.5 - hm_p, 0, 1) * 0.4)[..., None] * team
    x0h, y0h, x1h, y1h = hero_L["rect"]
    rim_rect = (x0h - pad, y0h - pad, x1h + pad, y1h + pad)

    # ---- the camera: from the reel's own framing of this moment to a close view of him ----
    A = OW / OH
    wide = ghost.reel_view(clip, plan, tF, size)
    hh_full = (boxes[best][3] - boxes[best][1]) * fs
    hcx, hcy = (boxes[best][0] + boxes[best][2]) / 2 * fs, (boxes[best][1] + boxes[best][3]) / 2 * fs
    tw = max(hh_full / 0.34 * A, wide[2] / 2.4, OW / 1.8)        # he fills a third of the height, without going mushy
    tw = min(tw, wide[2], Wc, Hc * A)
    tight = np.array([np.clip(hcx - tw / 2, 0, Wc - tw), np.clip(hcy - tw / A * 0.52, 0, Hc - tw / A), tw, tw / A])
    s_max = SWING * tw / max(1.0, tight[1] + tight[3] - y_piv_f, y_piv_f - y_top_f)
    uu, vv = np.meshgrid(np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32))
    rng = np.random.default_rng(5)
    # dust in the air: specks at their own depths (in front of him and behind) that slide with the camera like everything else
    ND = 90
    dust = np.c_[rng.uniform(-0.1, 1.1, ND), rng.uniform(0, 1, ND), rng.uniform(-0.9, 1.6, ND), rng.uniform(0.25, 0.8, ND),
                 rng.uniform(-0.012, 0.012, ND), rng.uniform(-0.03, -0.008, ND)]            # x, y, depth, brightness, drift
    sprites = {}
    for r in range(2, 13):
        kk = np.arange(-3 * r, 3 * r + 1, dtype=np.float32)
        g1 = np.exp(-kk ** 2 / (2 * (r * 0.9) ** 2))
        sprites[r] = np.outer(g1, g1)[..., None] * np.array([205, 228, 255], np.float32)    # warm white (BGR)
    # his name, pinned to him
    tag_txt = ((f"#{pl['number']}  " if pl.get("number") else "") + (pl.get("name") or "").upper()).strip()
    tag_bgr = tag_a = None
    if tag_txt:
        tf = graphics.font(54)
        tgw, tgh = int(tf.getlength(tag_txt)) + 44, 82
        ti = Image.new("RGBA", (tgw, tgh), (0, 0, 0, 0))
        td = ImageDraw.Draw(ti)
        td.rectangle((0, 0, tgw - 1, tgh - 1), fill=(10, 12, 15, 200))
        td.rectangle((0, 0, 7, tgh - 1), fill=tuple(int(tc[j:j + 2], 16) for j in (1, 3, 5)) + (255,))
        td.text((26, tgh // 2 - 2), tag_txt, font=tf, fill=(245, 244, 240, 255), anchor="lm")
        ta_ = np.asarray(ti, np.float32)
        tag_bgr, tag_a = ta_[..., 2::-1], ta_[..., 3:] / 255
    hero_top_f, hx0_f, hx1_f = boxes[best][1] * fs, boxes[best][0] * fs, boxes[best][2] * fs
    tag_right = hcx < tight[0] + tight[2] * 0.6                  # the tag goes on the side with more room

    def frame_at(t: float) -> np.ndarray:
        zoom, orbit, lift, back = swing(t, secs)
        s = s_max * orbit
        creep = 1 - 0.07 * zoom * min(1.0, t / (secs - OUT))      # still drifting in while it swings
        vx, vy, vw, vh = wide + (tight - wide) * zoom
        vx, vy, vw, vh = vx + vw * (1 - creep) * 0.5, vy + vh * (1 - creep) * 0.52, vw * creep, vh * creep
        zs = OW / vw
        ys = vy + vv / zs
        shift = s * (np.maximum(ys, y_top_f) - y_piv_f)           # the field shears; beyond it everything moves as one
        img = cv2.remap(bgd, (vx + uu / zs - shift).astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        hxo, hyo = (hcx - vx) * zs, (hcy - vy) * zs
        near_a = np.zeros((OH, OW, 1), np.float32)                # how much of each pixel is someone standing in front of him
        sh = np.zeros((OH, OW), np.float32)                       # his shadow stays on the ground
        cv2.ellipse(sh, (int(hxo), int((y_piv_f - vy) * zs)), (max(1, int(hh_full * 0.42 * zs)), max(1, int(hh_full * 0.09 * zs))), 0, 0, 360, 1.0, -1)
        img *= 1 - 0.45 * lift * cv2.GaussianBlur(sh, (0, 0), 6)[..., None]

        def put(rgb, m, rect, dx, dy, scale, gain, near=False, hero_pass=False):
            X0, Y0, X1, Y1 = rect
            k = zs * scale
            ax, ay = (X0 + X1) / 2, Y1                            # scaled about his feet, then moved by the layer's slide
            ox_ = (ax + dx - vx) * zs - (ax - X0) * k
            oy_ = (ay + dy - vy) * zs - (ay - Y0) * k
            dw, dh = int(np.ceil((X1 - X0) * k)) + 2, int(np.ceil((Y1 - Y0) * k)) + 2
            ix, iy = int(np.floor(ox_)), int(np.floor(oy_))
            if ix >= OW or iy >= OH or ix + dw <= 0 or iy + dh <= 0:
                return
            M = np.array([[k, 0, ox_ - ix], [0, k, oy_ - iy]])
            interp = cv2.INTER_AREA if k < 0.7 else cv2.INTER_LINEAR
            pr = cv2.warpAffine(rgb, M, (dw, dh), flags=interp)
            pm = cv2.warpAffine(m, M, (dw, dh), flags=interp)
            if pm.ndim == 2:
                pm = pm[..., None]
            a0, b0, a1, b1_ = max(0, iy), max(0, ix), min(OH, iy + dh), min(OW, ix + dw)
            pr, pm = pr[a0 - iy:a1 - iy, b0 - ix:b1_ - ix], pm[a0 - iy:a1 - iy, b0 - ix:b1_ - ix]
            if hero_pass:                                         # people in front of him still cover him
                pm = pm * (1 - near_a[a0:a1, b0:b1_])
            elif near:
                near_a[a0:a1, b0:b1_] = 1 - (1 - near_a[a0:a1, b0:b1_]) * (1 - pm)
            if gain is None:                                      # an added glow
                img[a0:a1, b0:b1_] += pr * (1 - near_a[a0:a1, b0:b1_])
            else:
                img[a0:a1, b0:b1_] = img[a0:a1, b0:b1_] * (1 - pm) + pr * gain * pm

        for L in layers:                                          # everyone else first, at full light
            if not L["hero"]:
                put(L["rgb"], L["m"], L["rect"], s * (L["feet"] - y_piv_f), 0.0, 1.0, 1.0, near=L["feet"] > y_piv_f)
        # Then the whole world dims and the light falls off away from him - people and field together,
        # so no cut-out edge is ever lit differently from the grass behind it ...
        img *= 1 - 0.26 * lift
        img *= 1 - 0.42 * lift * np.clip(((uu - hxo) / (OW * 0.75)) ** 2 + ((vv - hyo) / (OH * 0.75)) ** 2, 0, 1)[..., None]
        sc = 1 + 0.06 * lift                                      # ... and he goes on top, lit, a touch off the field
        put(hero_rim * lift, hero_rim[..., :1], rim_rect, 0.0, -0.012 * hh_full * lift, sc, None, hero_pass=True)
        put(hero_L["rgb"], hero_L["m"], hero_L["rect"], 0.0, -0.012 * hh_full * lift, sc, 1.0 + 0.08 * lift, hero_pass=True)
        for dx_, dy_, dz, db, dvx, dvy in dust:                   # nearer specks are bigger, softer and slide more
            r = int(np.clip(2 + 6 * abs(dz), 2, 12))
            x0_, y0_ = int((dx_ + dvx * t) * OW + s * dz * 0.55 * OH) - 3 * r, int((dy_ + dvy * t) * OH) - 3 * r
            sp = sprites[r]
            a0, b0, a1, b1_ = max(0, y0_), max(0, x0_), min(OH, y0_ + sp.shape[0]), min(OW, x0_ + sp.shape[1])
            if a1 > a0 and b1_ > b0:
                img[a0:a1, b0:b1_] += sp[a0 - y0_:a1 - y0_, b0 - x0_:b1_ - x0_] * (db * lift * (0.5 if r > 6 else 0.9) * 2.4 / r)
        tu = ghost._ease((t - 0.55) / 0.3) * (1 - ghost._ease((t - (secs - OUT - 0.2)) / 0.2))
        if tu > 0 and tag_bgr is not None:                        # his name tag, on a line from his shoulder
            sx_ = ((hx1_f if tag_right else hx0_f) - vx) * zs
            sy_ = (hero_top_f + 0.22 * hh_full - vy) * zs
            ex_, ey_ = sx_ + (90 if tag_right else -90) + s * 0.1 * OH, sy_ - 0.34 * hh_full * zs - 30
            tx_, ty_ = int(ex_ if tag_right else ex_ - tag_bgr.shape[1]), int(ey_ - tag_bgr.shape[0] / 2)
            tx_ = int(np.clip(tx_, 12, max(12, OW - tag_bgr.shape[1] - 12)))     # all of it on screen (a tall frame is narrow)
            ty_ = int(np.clip(ty_, 12, OH - tag_bgr.shape[0] - 12))
            ex_, ey_ = (tx_ if tag_right else tx_ + tag_bgr.shape[1]), ty_ + tag_bgr.shape[0] / 2
            ln = np.zeros((OH, OW), np.uint8)
            cv2.line(ln, (int(sx_), int(sy_)), (int(ex_), int(ey_)), 255, 2, cv2.LINE_AA)
            cv2.circle(ln, (int(sx_), int(sy_)), 5, 255, -1, cv2.LINE_AA)
            la = (ln.astype(np.float32) / 255 * tu)[..., None]
            img = img * (1 - la) + 245 * la
            wshow = int(tag_bgr.shape[1] * tu)                    # wipes open from the line
            a0, a1 = max(0, ty_), min(OH, ty_ + tag_bgr.shape[0])
            b0, b1_ = (max(0, tx_), min(OW, tx_ + wshow)) if tag_right else \
                (max(0, tx_ + tag_bgr.shape[1] - wshow), min(OW, tx_ + tag_bgr.shape[1]))
            if a1 > a0 and b1_ > b0:
                sl = (slice(a0 - ty_, a1 - ty_), slice(b0 - tx_, b1_ - tx_))
                img[a0:a1, b0:b1_] = img[a0:a1, b0:b1_] * (1 - tag_a[sl]) + tag_bgr[sl] * tag_a[sl]
        # crisp it up (moving in softens it), with a touch of grain so the enlargement doesn't look plastic
        img = img + 0.55 * zoom * (img - cv2.GaussianBlur(img, (0, 0), 1.4))
        img += rng.normal(0, 2.2, (OH, OW, 1)).astype(np.float32) * zoom
        roll = 1.1 * orbit                                        # a slight roll with the pass, like a camera on an arm
        if abs(roll) > 0.01:
            img = cv2.warpAffine(img, cv2.getRotationMatrix2D((OW / 2, OH / 2), roll, 1 + 0.03 * abs(roll) / 1.1), (OW, OH),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fl = max(0.0, 1 - t / 0.3)                                # the hit: a white flash with a streak of light across him
        if fl > 0:
            img += (np.exp(-((vv - hyo) / 26) ** 2) * 230 * fl ** 2)[..., None]
        flash = max(0.0, 1 - t / 0.16) * 0.5
        return np.clip(img * (1 - flash) + 255 * flash, 0, 255).astype(np.uint8)

    tmp = mp4.with_name(mp4.stem + ".tmp.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i", f"anullsrc=r={ghost.SR}:cl=stereo",
                            *encoder.h264(17), "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(tmp)],
                           stdin=subprocess.PIPE)
    try:
        for j in range(int(round(secs * FPS))):
            enc.stdin.write(frame_at(j / FPS).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"3D freeze render failed for {clip['source']}")
    tmp.replace(mp4)
    meta.write_text(json.dumps({"ok": True, "seconds": secs, "people": len(layers)}))
    log(f"  3D freeze {clip['source']}: {len(layers)} people in layers")
    return mp4
