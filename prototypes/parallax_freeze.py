"""Prototype: the 3D freeze. On the hit the picture stops, your player lifts off the field and the
camera swings around him with real depth.

parallax_freeze.py <project> <clip id> <out.mp4> [--freeze T] [--debug DIR]

How the depth is made (no depth model): a football field is a flat plane seen at an angle, so how
far something is from the camera follows from how high its feet are in the picture. The freeze
frame is split into layers - the field, everything beyond it, and every person as an upright
cut-out standing at his feet - and a sideways camera move slides each layer by its own depth
(the field shears, people stay upright). Behind moving players the empty field comes from a clean
plate (the median over the seconds before); behind people standing still it is painted in.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

from PIL import Image, ImageDraw

from hlv import beats, camera, detect, ghost, graphics, hits, library, profiles, project, segment

FPS, OW, OH, SR = 30, 1920, 1080, 48000
BACK = 4.0            # seconds before the hit used for the clean plate (and shown live in the demo)
AFTER = 2.5           # live seconds after the freeze in the demo
SECS = 3.9            # the freeze
SWING = 0.085         # how far the camera swings: the nearest row slides this fraction of the picture width


def ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


SEG = detect.MODEL.with_name("yolo11m-seg.pt")     # person outlines (Ultralytics, downloaded 2026-10-11)
_seg = None


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


def people(img: np.ndarray, focus: tuple[float, float, float] | None = None) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Everyone in the frame with an outline. focus (x, y, his height): the area round your player is
    looked at again enlarged, where a far-off player is only a few dozen pixels tall and the outlines
    of him and whoever is on him come out much finer; those replace the first-pass ones."""
    H, W = img.shape[:2]
    out = _outlines(img, 1600)
    if focus is None:
        return out
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

        def iou(o):
            ix = max(0.0, min(bb[2], o[2]) - max(bb[0], o[0]))
            iy = max(0.0, min(bb[3], o[3]) - max(bb[1], o[1]))
            return ix * iy / max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1]) + (o[2] - o[0]) * (o[3] - o[1]) - ix * iy)
        out = [o for o in out if iou(o[0]) < 0.35] + [(bb, c, mm)]
    return out


def cutout(img: np.ndarray, box, moving: np.ndarray | None, side: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """A soft mask of the person in `box` (GrabCut, helped by what moved): ((x0, y0, x1, y1), mask)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box[:4]
    bw, bh = x1 - x0, y1 - y0
    px, py = 0.22 * bw + 4, 0.12 * bh + 4
    X0, Y0, X1, Y1 = int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py))
    crop = img[Y0:Y1, X0:X1]
    if moving is not None:                                      # someone who moved: what differs from the empty field is him
        mv = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        cv2.ellipse(mv, (int((x0 + x1) / 2 - X0), int((y0 + y1) / 2 - Y0)), (int(bw * 0.62), int(bh * 0.58)), 0, 0, 360, 1, -1)
        mv &= (moving[Y0:Y1, X0:X1] > 0.5).astype(np.uint8)
        mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mv, 8)
        core = lab[int((y0 + y1) / 2 - Y0 - bh * 0.3):int((y0 + y1) / 2 - Y0 + bh * 0.3) + 1,
                   int((x0 + x1) / 2 - X0 - bw * 0.25):int((x0 + x1) / 2 - X0 + bw * 0.25) + 1]
        keep = np.isin(lab, [v for v in np.unique(core) if v and stats[v, cv2.CC_STAT_AREA] >= 0.03 * bw * bh]).astype(np.uint8)
        if keep.sum() >= 0.16 * bw * bh:
            keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            return (X0, Y0, X1, Y1), cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.2)
    k = min(1.0, side / max(crop.shape[:2]))
    small = cv2.resize(crop, (max(8, int(crop.shape[1] * k)), max(8, int(crop.shape[0] * k))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    m = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    bx0, by0, bx1, by1 = [int(round(v)) for v in ((x0 - X0) * k, (y0 - Y0) * k, (x1 - X0) * k, (y1 - Y0) * k)]
    m[max(0, by0):by1, max(0, bx0):bx1] = cv2.GC_PR_FGD
    core_w, core_h = max(1, int((bx1 - bx0) * 0.16)), max(1, int((by1 - by0) * 0.3))      # his middle is surely him
    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
    m[cy - core_h:cy + core_h, cx - core_w:cx + core_w] = cv2.GC_FGD
    if moving is not None:
        mv = cv2.resize(moving[Y0:Y1, X0:X1], (sw, sh), interpolation=cv2.INTER_AREA) > 0.5
        m[mv & (m == cv2.GC_PR_FGD)] = cv2.GC_FGD
    try:
        cv2.grabCut(small, m, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
        fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        fg = (m != cv2.GC_BGD).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 2:                                                   # keep the piece he's in
        keep = lab[min(sh - 1, cy), min(sw - 1, cx)] or 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (lab == keep).astype(np.uint8)
    mask = cv2.resize(fg.astype(np.float32), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
    return (X0, Y0, X1, Y1), cv2.GaussianBlur(mask, (0, 0), max(0.8, 0.9 / k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project"), ap.add_argument("clip"), ap.add_argument("out")
    ap.add_argument("--freeze", type=float, default=None)
    ap.add_argument("--debug", default=None)
    a = ap.parse_args()
    root = Path(a.project).resolve()
    proj = project.load(root)
    settings = project.settings(proj)
    c = next(c for c in proj["clips"] if c["id"] == a.clip)
    cache = root / ".hlv_cache"
    src = root / c["source"]
    info, crop = c["probe"], c.get("crop")
    ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
    Wc, Hc = (crop["w"], crop["h"]) if crop else (info["width"], info["height"])
    pl = profiles.get(library.home(), settings.get("player")) or {}
    hit = a.freeze if a.freeze is not None else hits.hit_of(root, c)[0]
    tF = round(float(hit) * FPS) / FPS
    ta = round(max(0.0, tF - BACK) * FPS) / FPS
    iF = int(round((tF - ta) * FPS))
    ws = min(1.0, 1920 / Wc)
    W, H = int(round(Wc * ws / 2)) * 2, int(round(Hc * ws / 2)) * 2
    n_after = int(min(AFTER, info["duration"] - tF - 0.2) * FPS)
    frames = ghost._read(src, ta, iF + 1 + n_after, crop, (W, H))
    after = frames[iF + 1:]
    frames = frames[:iF + 1]
    print(f"{c['source']}: hit {tF:.2f}s, {len(frames)} frames before, {len(after)} after, working at {W}x{H}", flush=True)
    Fw = frames[iF]
    full = ghost._like(ghost._read(src, ta, iF + 1, crop, (Wc, Hc), keep={iF})[iF], Fw) if ws < 1 else Fw
    fs = 1 / ws                                                  # working px -> full px

    # ---- the empty field behind the players ----
    G = ghost._homographies(frames, iF)
    idx = list(range(0, iF + 1, max(2, (iF + 1) // 40)))
    stack = np.stack([cv2.warpPerspective(frames[k], G[k], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for k in idx])
    plate = np.empty((H, W, 3), np.uint8)
    for y in range(0, H, 120):
        plate[y:y + 120] = np.median(stack[:, y:y + 120], axis=0).astype(np.uint8)
    del stack
    # match the plate's light to the freeze frame's (exposure drifts over a few seconds), or every
    # filled-in patch shows as a lighter blob and plain grass counts as "moved"
    # ... measured on the pixels that didn't change only: a white jersey must not brighten the grass around it
    Ff, Pf = Fw.astype(np.float32), plate.astype(np.float32)
    # (a first rough pass over everything, so "didn't change" can be told at all when the exposure moved a lot)
    Pf *= np.clip(cv2.GaussianBlur(Ff, (0, 0), 35) / np.maximum(1.0, cv2.GaussianBlur(Pf, (0, 0), 35)), 0.6, 1.6)
    same = (np.abs(Ff - Pf).max(axis=2) < 22).astype(np.float32)[..., None]
    wsum = cv2.GaussianBlur(same, (0, 0), 30)[..., None]
    ratio = cv2.GaussianBlur(Ff * same, (0, 0), 30) / np.maximum(1e-3, cv2.GaussianBlur(Pf * same, (0, 0), 30))
    ratio = np.where(wsum > 0.15, np.clip(ratio, 0.75, 1.35), 1.0)
    plate = np.clip(Pf * ratio, 0, 255).astype(np.uint8)
    moved =(cv2.GaussianBlur(cv2.absdiff(Fw, plate).max(axis=2), (0, 0), 1.2) > 46).astype(np.float32)

    # ---- everyone in the picture, and which one is him ----
    plan = camera.plan(cache, c)
    p = np.array(plan["player"])[min(len(plan["player"]) - 1, int(round(tF * FPS)))]
    hx, hy, hw, hh = (p[0] * info["width"] - ox) * ws, (p[1] * info["height"] - oy) * ws, p[2] * info["width"] * ws, p[3] * info["height"] * ws
    found = people(Fw, (hx, hy, hh))
    boxes = np.array([[*b, c_] for b, c_, _ in found], np.float32).reshape(-1, 5)
    masks = [m for _, _, m in found]
    best, best_s = None, 0.0
    for k, b in enumerate(boxes):
        ix = max(0.0, min(hx + hw / 2, b[2]) - max(hx - hw / 2, b[0]))
        iy = max(0.0, min(hy + hh / 2, b[3]) - max(hy - hh / 2, b[1]))
        held = ix * iy / max(1.0, hw * hh)
        score = held - 0.5 * abs((b[0] + b[2]) / 2 - hx) / max(1.0, b[2] - b[0])
        if held >= 0.5 and (b[3] - b[1]) <= 5 * hh and score > best_s:
            best, best_s = k, score
    if best is None:
        boxes = np.vstack([boxes, [hx - 0.7 * hw, hy - 0.85 * hh, hx + 0.7 * hw, hy + 0.85 * hh, 1.0]])
        masks.append(None)                                       # no outline for him: cut him out the old way
        best = len(boxes) - 1
    owner = np.full((H, W), -1, np.int32)                        # whose outline each pixel belongs to (him last, so he wins)
    for k in [k for k in range(len(masks)) if k != best] + [best]:
        if masks[k] is not None:
            owner[masks[k]] = k
    print(f"{len(boxes)} people found; your player's box {boxes[best][:4].round()}", flush=True)

    # geometry: the far edge of the field (everything above it is "beyond"), and the row he stands on
    gm = cv2.inRange(cv2.cvtColor(Fw, cv2.COLOR_BGR2HSV), (30, 40, 30), (90, 255, 255)) > 0
    rows = gm.mean(axis=1)
    green_rows = np.where(rows > 0.35)[0]
    y_top = float(green_rows.min()) if len(green_rows) else H * 0.3
    y_piv = float(boxes[best][3])                               # his feet
    y_top = min(y_top, y_piv - 0.03 * H)
    print(f"field's far edge at row {y_top:.0f}, his feet at {y_piv:.0f} (of {H})", flush=True)

    layers = []                                                 # per person: rect (working px), rgb, mask, feet row, is hero
    bg = full.copy()
    paint = np.zeros((H, W), np.uint8)
    # anyone still standing in the "empty" field (he didn't move in those seconds) is painted out of it
    still = np.zeros((H, W), np.uint8)
    for _, _, m in _outlines(plate, 1600):
        still |= m.astype(np.uint8)
    if still.any():
        still = cv2.dilate(still, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        plate = cv2.inpaint(plate, still * 255, 5, cv2.INPAINT_TELEA)
    plate_full = cv2.resize(plate, (Wc, Hc), interpolation=cv2.INTER_CUBIC) if ws < 1 else plate
    k5, k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for k, b in enumerate(boxes):
        if (b[3] - b[1]) < 14 or (b[2] - b[0]) < 6:
            continue
        hero = k == best
        if masks[k] is None:
            (X0, Y0, X1, Y1), m = cutout(Fw, b, moved, 420 if hero else 200)
        else:
            # his outline, plus anything touching it that moved and isn't someone else (a hand, a foot the outline missed)
            px_, py_ = 0.22 * (b[2] - b[0]) + 6, 0.12 * (b[3] - b[1]) + 6
            X0, Y0, X1, Y1 = int(max(0, b[0] - px_)), int(max(0, b[1] - py_)), int(min(W, b[2] + px_)), int(min(H, b[3] + py_))
            m = masks[k][Y0:Y1, X0:X1].astype(np.uint8)
            own = owner[Y0:Y1, X0:X1]
            extra = (moved[Y0:Y1, X0:X1] > 0.5) & (cv2.dilate(m, k11) > 0) & ((own < 0) | (own == k))
            m = cv2.morphologyEx(m | extra.astype(np.uint8), cv2.MORPH_CLOSE, k5)
            m[(own >= 0) & (own != k)] = 0
            m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
        if m.sum() < 30:
            continue
        # the same cut-out at full size
        FX0, FY0, FX1, FY1 = [int(round(v * fs)) for v in (X0, Y0, X1, Y1)]
        mf = cv2.resize(m, (FX1 - FX0, FY1 - FY0), interpolation=cv2.INTER_LINEAR) if ws < 1 else m
        layers.append({"rect": (FX0, FY0, FX1, FY1), "rgb": full[FY0:FY1, FX0:FX1].astype(np.float32), "m": mf,
                       "feet": float(b[3]) * fs, "hero": hero})
        grown = cv2.dilate((m > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        was_moving = True                                        # (the plate is clean everywhere now)
        if was_moving:                                           # the plate shows the field behind him
            gf = cv2.resize(grown, (FX1 - FX0, FY1 - FY0), interpolation=cv2.INTER_NEAREST) if ws < 1 else grown
            gf = cv2.GaussianBlur(gf.astype(np.float32), (0, 0), 2 * fs)[..., None]
            reg = bg[FY0:FY1, FX0:FX1]
            bg[FY0:FY1, FX0:FX1] = (reg * (1 - gf) + plate_full[FY0:FY1, FX0:FX1] * gf).astype(np.uint8)
        else:                                                    # standing still: paint the hole in
            paint[Y0:Y1, X0:X1] |= grown
    if paint.any():
        small_bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA) if ws < 1 else bg
        filled = cv2.inpaint(small_bg, paint * 255, 6, cv2.INPAINT_TELEA)
        pm = cv2.GaussianBlur(paint.astype(np.float32), (0, 0), 2)
        if ws < 1:
            filled = cv2.resize(filled, (Wc, Hc), interpolation=cv2.INTER_CUBIC)
            pm = cv2.resize(pm, (Wc, Hc), interpolation=cv2.INTER_LINEAR)
        bg = (bg * (1 - pm[..., None]) + filled * pm[..., None]).astype(np.uint8)
    layers.sort(key=lambda L: L["feet"])                         # far ones first
    y_top_f, y_piv_f = y_top * fs, y_piv * fs

    # ---- focus: sharp at his depth, softer with distance (in front of him and behind) ----
    def blur_at(feet_row: float) -> float:
        d = (feet_row - y_piv_f) / max(1.0, Hc - y_top_f)
        return (9.0 * d if d > 0 else 6.0 * min(1.0, -d * 1.6)) * fs
    b1, b2 = cv2.GaussianBlur(bg, (0, 0), 2.5 * fs), cv2.GaussianBlur(bg, (0, 0), 6.5 * fs)
    sig = np.array([blur_at(max(y, y_top_f)) / fs for y in range(Hc)], np.float32)
    w2 = np.clip((sig - 2.5) / 4.0, 0, 1)[:, None, None]
    w1 = np.clip(sig / 2.5, 0, 1)[:, None, None]
    bgd = (bg * (1 - w1) + b1 * w1)
    bgd = (bgd * (1 - w2) + b2 * w2).astype(np.uint8)
    del b1, b2
    for L in layers:
        s_ = 0.0 if L["hero"] else blur_at(L["feet"])
        if s_ > 0.6:
            L["rgb"] = cv2.GaussianBlur(L["rgb"], (0, 0), s_)
            L["m"] = cv2.GaussianBlur(L["m"], (0, 0), s_ * 0.6)
        L["m"] = L["m"][..., None]
    tc = pl.get("color") or "#1f4fd1"
    team = np.array([int(tc[j:j + 2], 16) for j in (5, 3, 1)], np.float32)
    hero_L = next(L for L in layers if L["hero"])
    hm = hero_L["m"][..., 0]
    pad = int(16 * fs)
    hm_p = cv2.copyMakeBorder(hm, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    hero_rim = (np.clip(cv2.GaussianBlur(hm_p, (0, 0), 3.5 * fs) * 1.5 - hm_p, 0, 1) * 0.4)[..., None] * team
    x0h, y0h, x1h, y1h = hero_L["rect"]
    rim_rect = (x0h - pad, y0h - pad, x1h + pad, y1h + pad)

    # ---- the camera ----
    A = OW / OH
    fw0 = min(Wc, Hc * A)
    wide = np.array([(Wc - fw0) / 2, (Hc - fw0 / A) / 2, fw0, fw0 / A])
    hh_full = (boxes[best][3] - boxes[best][1]) * fs
    tw = max(hh_full / 0.34 * A, fw0 / 2.4, OW / 1.8)             # he fills a third of the height, without going mushy
    tw = min(tw, fw0)
    hcx = (boxes[best][0] + boxes[best][2]) / 2 * fs
    hcy = (boxes[best][1] + boxes[best][3]) / 2 * fs
    tight = np.array([np.clip(hcx - tw / 2, 0, Wc - tw), np.clip(hcy - tw / A * 0.52, 0, Hc - tw / A), tw, tw / A])
    s_max = SWING * tw / max(1.0, tight[1] + tight[3] - y_piv_f)  # shear so the bottom of the close view slides SWING of its width
    print(f"close view {tw:.0f} px wide ({OW / tw:.2f}x of the source), swing {s_max:.3f}", flush=True)
    uu, vv = np.meshgrid(np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32))
    rng = np.random.default_rng(5)
    # dust in the air: specks at their own depths (in front of him and behind), which slide with the camera like everything else
    ND = 90
    dust = np.c_[rng.uniform(-0.1, 1.1, ND), rng.uniform(0, 1, ND), rng.uniform(-0.9, 1.6, ND), rng.uniform(0.25, 0.8, ND),
                 rng.uniform(-0.012, 0.012, ND), rng.uniform(-0.03, -0.008, ND)]            # x, y, depth, brightness, drift
    sprites = {}
    for r in range(2, 13):
        k = np.arange(-3 * r, 3 * r + 1, dtype=np.float32)
        g1 = np.exp(-k ** 2 / (2 * (r * 0.9) ** 2))
        sprites[r] = np.outer(g1, g1)[..., None] * np.array([205, 228, 255], np.float32)    # warm white (BGR)
    # his name, pinned to him
    tag_txt = (f"#{pl['number']}  " if pl.get("number") else "") + (pl.get("name") or "").upper()
    tf = graphics.font(54)
    tw_, th_ = int(tf.getlength(tag_txt)) + 44, 82
    tag_img = Image.new("RGBA", (tw_, th_), (0, 0, 0, 0))
    td = ImageDraw.Draw(tag_img)
    td.rectangle((0, 0, tw_ - 1, th_ - 1), fill=(10, 12, 15, 200))
    td.rectangle((0, 0, 7, th_ - 1), fill=tuple(int(tc[j:j + 2], 16) for j in (1, 3, 5)) + (255,))
    td.text((26, th_ // 2 - 2), tag_txt, font=tf, fill=(245, 244, 240, 255), anchor="lm")
    tag = np.asarray(tag_img, np.float32)
    tag_bgr, tag_a = tag[..., 2::-1], tag[..., 3:] / 255
    hero_top_f = boxes[best][1] * fs
    hx0_f, hx1_f = boxes[best][0] * fs, boxes[best][2] * fs
    tag_right = hcx < Wc * 0.62                                  # the tag goes on the side with more room
    cy_o, cx_o = OH / 2, OW / 2

    def freeze_frame(t: float) -> np.ndarray:
        u = t / SECS
        IN, OUT = 0.45, 0.4
        back = ease((t - (SECS - OUT)) / OUT)
        zoom = ease(t / IN) * (1 - back)
        orbit = -1 + 2 * ease((t - IN * 0.4) / (SECS - OUT - IN * 0.4))    # one slow pass around him, left to right
        s = s_max * orbit * ease(t / IN) * (1 - back)
        lift = ease(t / 0.5) * (1 - back)
        creep = 1 - 0.07 * zoom * min(1.0, t / (SECS - OUT))      # still drifting in while it swings
        vx, vy, vw, vh = wide + (tight - wide) * zoom
        vx, vy, vw, vh = vx + vw * (1 - creep) * 0.5, vy + vh * (1 - creep) * 0.52, vw * creep, vh * creep
        zs = OW / vw
        ys = vy + vv / zs
        shift = s * (np.maximum(ys, y_top_f) - y_piv_f)           # the field shears; beyond it everything moves as one
        img = cv2.remap(bgd, (vx + uu / zs - shift).astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        hxo, hyo = (hcx - vx) * zs, (hcy - vy) * zs
        near_a = np.zeros((OH, OW, 1), np.float32)                # how much of each pixel is someone standing in front of him
        # his shadow stays on the ground
        fxo, fyo = (hcx - vx) * zs, (y_piv_f - vy) * zs
        sh = np.zeros((OH, OW), np.float32)
        cv2.ellipse(sh, (int(fxo), int(fyo)), (int(hh_full * 0.42 * zs), int(hh_full * 0.09 * zs)), 0, 0, 360, 1.0, -1)
        img *= 1 - 0.45 * lift * cv2.GaussianBlur(sh, (0, 0), 6)[..., None]

        def put(rgb, m, rect, dx, dy, scale, gain, near=False, hero_pass=False):
            X0, Y0, X1, Y1 = rect
            k = zs * scale
            # scaled about the bottom-centre of the rect (his feet), then moved by the layer's slide
            ax, ay = (X0 + X1) / 2, Y1
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
            a0, b0 = max(0, iy), max(0, ix)
            a1, b1_ = min(OH, iy + dh), min(OW, ix + dw)
            pr, pm = pr[a0 - iy:a1 - iy, b0 - ix:b1_ - ix], pm[a0 - iy:a1 - iy, b0 - ix:b1_ - ix]
            if hero_pass:                                        # people in front of him still cover him
                pm = pm * (1 - near_a[a0:a1, b0:b1_])
            elif near:
                near_a[a0:a1, b0:b1_] = 1 - (1 - near_a[a0:a1, b0:b1_]) * (1 - pm)
            if gain is None:                                     # an added glow
                img[a0:a1, b0:b1_] += pr * (1 - near_a[a0:a1, b0:b1_]) if hero_pass else pr
            else:
                img[a0:a1, b0:b1_] = img[a0:a1, b0:b1_] * (1 - pm) + pr * gain * pm

        for L in layers:                                          # everyone else first, at full light
            if not L["hero"]:
                put(L["rgb"], L["m"], L["rect"], s * (L["feet"] - y_piv_f), 0.0, 1.0, 1.0, near=L["feet"] > y_piv_f)
        # then the whole world dims and the light falls off away from him (people and field together, so
        # no cut-out edge is ever lit differently from the grass behind it) ...
        img *= 1 - 0.26 * lift
        img *= 1 - 0.42 * lift * np.clip((((uu - hxo) / (OW * 0.75)) ** 2 + ((vv - hyo) / (OH * 0.75)) ** 2), 0, 1)[..., None]
        sc = 1 + 0.06 * lift                                      # ... and he goes on top, lit, a touch off the field
        put(hero_rim * lift, hero_rim[..., :1], rim_rect, 0.0, -0.012 * hh_full * lift, sc, None, hero_pass=True)
        put(hero_L["rgb"], hero_L["m"], hero_L["rect"], 0.0, -0.012 * hh_full * lift, sc, 1.0 + 0.08 * lift, hero_pass=True)
        # dust: nearer specks are bigger, softer and slide more
        for dx_, dy_, dz, db, vx_, vy_ in dust:
            r = int(np.clip(2 + 6 * abs(dz), 2, 12))
            px_ = (dx_ + vx_ * t) * OW + s * dz * 0.55 * OH
            py_ = (dy_ + vy_ * t) * OH
            x0_, y0_ = int(px_) - 3 * r, int(py_) - 3 * r
            sp = sprites[r]
            a0, b0, a1, b1_ = max(0, y0_), max(0, x0_), min(OH, y0_ + sp.shape[0]), min(OW, x0_ + sp.shape[1])
            if a1 > a0 and b1_ > b0:
                img[a0:a1, b0:b1_] += sp[a0 - y0_:a1 - y0_, b0 - x0_:b1_ - x0_] * (db * lift * (0.5 if r > 6 else 0.9) * 2.4 / r)
        # his name tag, on a leader line from his shoulder
        tu = ease((t - 0.55) / 0.3) * (1 - ease((t - (SECS - OUT - 0.2)) / 0.2))
        if tu > 0 and tag_txt.strip():
            sx_ = ((hx1_f if tag_right else hx0_f) - vx) * zs
            sy_ = (hero_top_f + 0.22 * hh_full - vy) * zs
            ex_ = sx_ + (90 if tag_right else -90) + s * 0.1 * OH
            ey_ = sy_ - 0.34 * hh_full * zs - 30
            tx_ = int(ex_ if tag_right else ex_ - tag_bgr.shape[1])
            ty_ = int(ey_ - tag_bgr.shape[0] / 2)
            ln = np.zeros((OH, OW), np.uint8)
            cv2.line(ln, (int(sx_), int(sy_)), (int(ex_), int(ey_)), 255, 2, cv2.LINE_AA)
            cv2.circle(ln, (int(sx_), int(sy_)), 5, 255, -1, cv2.LINE_AA)
            la = (ln.astype(np.float32) / 255 * tu)[..., None]
            img = img * (1 - la) + 245 * la
            wshow = int(tag_bgr.shape[1] * tu)                    # wipes open from the line
            a0, a1 = max(0, ty_), min(OH, ty_ + tag_bgr.shape[0])
            if tag_right:
                b0, b1_ = max(0, tx_), min(OW, tx_ + wshow)
                sl = (slice(a0 - ty_, a1 - ty_), slice(b0 - tx_, b1_ - tx_))
            else:
                b0, b1_ = max(0, tx_ + tag_bgr.shape[1] - wshow), min(OW, tx_ + tag_bgr.shape[1])
                sl = (slice(a0 - ty_, a1 - ty_), slice(b0 - tx_, b1_ - tx_))
            if a1 > a0 and b1_ > b0:
                img[a0:a1, b0:b1_] = img[a0:a1, b0:b1_] * (1 - tag_a[sl]) + tag_bgr[sl] * tag_a[sl]
        # crisp it up (the push-in softens it), a touch of grain so the enlargement doesn't look plastic
        img = img + 0.55 * zoom * (img - cv2.GaussianBlur(img, (0, 0), 1.4))
        img += rng.normal(0, 2.2, (OH, OW, 1)).astype(np.float32) * zoom
        # a slight roll with the swing, like a camera on an arm
        roll = 1.1 * orbit * ease(t / IN) * (1 - back)
        if abs(roll) > 0.01:
            img = cv2.warpAffine(img, cv2.getRotationMatrix2D((cx_o, cy_o), roll, 1 + 0.03 * abs(roll) / 1.1), (OW, OH),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # the hit: a white flash with a streak of light across him
        fl = max(0.0, 1 - t / 0.3)
        if fl > 0:
            img += (np.exp(-((vv - hyo) / 26) ** 2) * 230 * fl ** 2)[..., None]
        flash = max(0.0, 1 - t / 0.16) * 0.5
        img = img * (1 - flash) + 255 * flash
        return np.clip(img, 0, 255).astype(np.uint8)

    # ---- the demo: live -> the drop -> the 3D freeze -> live ----
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_v = out.with_suffix(".video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", str(tmp_v)],
                           stdin=subprocess.PIPE)

    def live(f):                                                 # the working frame, in the wide view
        x, y, w_, h_ = wide * ws
        return cv2.resize(f[int(y):int(y + h_), int(x):int(x + w_)], (OW, OH), interpolation=cv2.INTER_AREA if w_ > OW else cv2.INTER_CUBIC)

    for i in range(iF):
        f = live(frames[i])
        enc.stdin.write((f * (i / 8)).astype(np.uint8).tobytes() if i < 8 else f.tobytes())
    nfz = int(round(SECS * FPS))
    dbg = Path(a.debug) if a.debug else None
    for j in range(nfz):
        f = freeze_frame(j / FPS)
        if dbg and j in (nfz // 4, nfz // 2, 3 * nfz // 4):
            cv2.imwrite(str(dbg / f"3d-{a.clip[-12:]}-{j:03d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
        enc.stdin.write(f.tobytes())
    for j, f in enumerate(after):
        f = live(f)
        left = len(after) - j
        enc.stdin.write((f * (left / 15)).astype(np.uint8).tobytes() if left <= 15 else f.tobytes())
    enc.stdin.close()
    enc.wait()
    if dbg:
        cv2.imwrite(str(dbg / f"3d-{a.clip[-12:]}-bg.jpg"), cv2.resize(bg, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---- sound: the song lands a heavy hit on the freeze, silence before it, the drop on it, air moving with the swing ----
    total = (iF + nfz + len(after)) / FPS
    N = int(round(total * SR))
    tt = np.arange(N) / SR
    th = iF / FPS
    mix = np.zeros((N, 2), np.float32)

    def read_audio(path, t0, dur):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t0):.4f}", "-t", f"{dur:.4f}", "-i", str(path), "-vn", "-ac", "2",
                              "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
        x = np.frombuffer(raw, np.float32).reshape(-1, 2)
        o = np.zeros((int(round(dur * SR)), 2), np.float32)
        o[:min(len(x), len(o))] = x[:len(o)]
        return o

    game = read_audio(src, ta, (iF + 1 + len(after)) / FPS)
    cut = int(round(th * SR))
    gs = np.zeros((N, 2), np.float32)
    gs[:cut] = game[:cut]
    aft = int(round((th + SECS) * SR))
    rest = game[cut:cut + (N - aft)]
    gs[aft:aft + len(rest)] = rest
    genv = ((tt < th - ghost.GAP) | (tt >= th + SECS)).astype(np.float32)
    mix += gs * genv[:, None] * 0.55
    music = project.music_path(root, settings)
    if music and settings.get("music_source") != "compose":
        heavy = beats.heavy_hits(music, cache)
        mh = max((h_ for h_ in heavy if h_[0] > th + 20), key=lambda h_: h_[1])[0]
        song = read_audio(music, mh - th, total)
        menv = np.full(N, 0.8, np.float32) * np.clip(tt / 0.5, 0, 1)
        menv[(tt >= th - ghost.GAP) & (tt < th)] = 0.0
        menv[tt >= th] = 1.0
        menv *= np.clip((total - tt) / 1.2, 0, 1)
        mix += song * menv[:, None]
    fx = out.with_suffix(".fx.wav")
    ghost.sound_track(fx, total, [{"t": th, "drop": True, "pops": None}])
    with wave.open(str(fx)) as wf:
        mix += np.frombuffer(wf.readframes(wf.getnframes()), np.int16).reshape(-1, 2)[:N] / 32768
    fx.unlink()
    # air moving past as the camera swings: filtered noise that pans with the move
    rng = np.random.default_rng(3)
    zn = int(SECS * SR)
    zu = np.arange(zn) / zn
    air = ghost._bandpass(rng.standard_normal(zn), 250, 1800)
    air = air / np.abs(air).max() * (0.55 + 0.45 * np.sin(np.pi * zu)) * np.clip(zu / 0.1, 0, 1) * np.clip((1 - zu) / 0.15, 0, 1)
    pan = 0.08 + 0.84 * zu
    s0 = int(round(th * SR))
    mix[s0:s0 + zn, 0] += (air * (1 - pan) * 0.22)[:N - s0]
    mix[s0:s0 + zn, 1] += (air * pan * 0.22)[:N - s0]
    mix = np.tanh(mix * 0.75) * 0.8
    tmp_a = out.with_suffix(".wav")
    with wave.open(str(tmp_a), "wb") as wf:
        wf.setnchannels(2), wf.setsampwidth(2), wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_v), "-i", str(tmp_a), "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-movflags", "+faststart", "-shortest", str(out)], check=True)
    tmp_v.unlink(), tmp_a.unlink()
    print(f"wrote {out}  {total:.1f}s  freeze at {th:.2f}s for {SECS}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
