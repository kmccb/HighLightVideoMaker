"""Camera-compensated motion signals.

For each consecutive frame pair we fit a similarity transform (pan + zoom + roll) to
tracked background features, warp the previous frame onto the current one, and measure
what still moves. With the camera's motion removed, that residual is mostly players.
"""
from __future__ import annotations

import cv2
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def grid_features(gray: np.ndarray, grid: int = 8, per_cell: int = 12) -> np.ndarray | None:
    """Shi-Tomasi corners spread over a grid so background texture dominates the fit."""
    h, w = gray.shape
    pts = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * h // grid, (gy + 1) * h // grid
            x0, x1 = gx * w // grid, (gx + 1) * w // grid
            c = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], per_cell, 0.01, 5)
            if c is not None:
                pts.append(c.reshape(-1, 2) + (x0, y0))
    if not pts:
        return None
    return np.concatenate(pts).astype(np.float32).reshape(-1, 1, 2)


def estimate_global(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """2x3 similarity transform mapping prev -> cur (identity on failure)."""
    ident = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    p0 = grid_features(prev)
    if p0 is None or len(p0) < 12:
        return ident
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **LK)
    ok = st.reshape(-1) == 1
    if ok.sum() < 12:
        return ident
    m, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                       ransacReprojThreshold=1.5, maxIters=2000, confidence=0.99)
    return m.astype(np.float32) if m is not None else ident


def field_mask(bgr: np.ndarray, min_frac: float = 0.15) -> np.ndarray | None:
    """Rough grass/turf mask, dilated to include players standing on it. None if no field."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    if green.mean() / 255 < min_frac:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    # Fill holes (players, yard lines) via closing, then grow upward a bit for torsos/helmets.
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    green = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 41)))
    return green > 0


def motion_signals(gray: np.ndarray, color: np.ndarray | None = None,
                   diff_thresh: int = 18, track: bool = False) -> dict[str, np.ndarray]:
    """Per-frame signals (length N; frame 0 copies frame 1).

    residual: fraction of (field) pixels still moving after camera compensation
    residual_mag: mean abs diff over the top 2% moving pixels
    pan: camera translation speed, px/frame at analysis width
    zoom: |log scale| per frame
    box (only if track): (N, 6) where the movers are, normalised 0..1:
         centre x, centre y, x5, x95, y5, y95, cluster mass - NaN when too little moves
    """
    n = len(gray)
    out = {k: np.zeros(n, np.float32) for k in ("residual", "residual_mag", "pan", "zoom")}
    if track:
        out["box"] = np.full((n, 7), np.nan, np.float32)
    if n < 2:
        return out
    h, w = gray.shape[1:]
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ones = np.full((h, w), 255, np.uint8)
    mask = acc = prev_c = None
    for i in range(1, n):
        prev, cur = gray[i - 1], gray[i]
        m = estimate_global(prev, cur)
        warped = cv2.warpAffine(prev, m, (w, h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpAffine(ones, m, (w, h), flags=cv2.INTER_NEAREST) == 255
        valid = cv2.erode(valid.astype(np.uint8), kern, iterations=3).astype(bool)
        if color is not None and (i % 15 == 1 or mask is None):
            mask = field_mask(color[i])
        region = valid & mask if mask is not None else valid
        # Blur first so sub-pixel misregistration of fine texture doesn't register as motion.
        diff = cv2.absdiff(cv2.GaussianBlur(warped, (5, 5), 0), cv2.GaussianBlur(cur, (5, 5), 0))
        moving = (diff > diff_thresh).astype(np.uint8)
        moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, kern).astype(bool) & region
        denom = max(region.sum(), 1)
        out["residual"][i] = moving.sum() / denom
        vals = diff[region]
        if vals.size:
            top = np.partition(vals, int(vals.size * 0.98))[int(vals.size * 0.98):]
            out["residual_mag"][i] = top.mean()
        if track:
            # Densest cluster of motion, accumulated over a few frames. A play is many
            # bodies running together; sideline fidgeting is sparse, so it drops out.
            dens = cv2.GaussianBlur(moving.astype(np.float32), (0, 0), w / 50)
            # Re-align the accumulator to this frame (camera moved), then blend.
            acc = dens if acc is None else 0.7 * cv2.warpAffine(acc, m, (w, h)) + 0.3 * dens
            peak = float(acc.max())
            if moving.sum() >= 40 and peak > 1e-3:
                blob = (acc > 0.3 * peak).astype(np.uint8)
                nlab, labels = cv2.connectedComponents(blob)
                # Pick a cluster: strong, and near where the play was last frame. The play
                # moves continuously; a jump to the sideline needs a much bigger cluster.
                best_lab, best_score = 0, -1.0
                for lab in range(1, nlab):
                    ys, xs = np.nonzero(labels == lab)
                    mass = float(acc[ys, xs].sum())
                    score = mass
                    if prev_c is not None:
                        d2 = ((xs.mean() / w - prev_c[0]) ** 2 + (ys.mean() / h - prev_c[1]) ** 2)
                        score *= np.exp(-d2 / (2 * 0.25 ** 2))
                    if score > best_score:
                        best_lab, best_score = lab, score
                ys, xs = np.nonzero(labels == best_lab)
                wts = acc[ys, xs]
                cx, cy = np.average(xs, weights=wts) / w, np.average(ys, weights=wts) / h
                bx = np.percentile(xs, [5, 95]) / w
                by = np.percentile(ys, [5, 95]) / h
                out["box"][i] = (cx, cy, bx[0], bx[1], by[0], by[1], float(wts.sum()))
                prev_c = (cx, cy)
        scale = float(np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2))
        out["pan"][i] = float(np.hypot(m[0, 2], m[1, 2]))
        out["zoom"][i] = abs(np.log(max(scale, 1e-6)))
    for k in out:
        out[k][0] = out[k][1]
    return out


def frame_diff(gray: np.ndarray) -> np.ndarray:
    """Raw (uncompensated) mean abs diff between consecutive frames; d[i] is frame i vs i-1."""
    d = np.zeros(len(gray), np.float32)
    for i in range(1, len(gray)):
        d[i] = cv2.absdiff(gray[i], gray[i - 1]).mean()
    return d


def scene_cuts(diff: np.ndarray, fps: float, ratio: float = 5.0) -> list[float]:
    """Hard cuts = single-frame spikes in raw frame difference.

    Cuts in football broadcasts keep the same green palette, so colour histograms barely
    move; structure changes completely though. Whip pans raise the diff for many frames,
    so we demand a spike well above both the local median and its neighbours.
    """
    if len(diff) < 3:
        return []
    from scipy.ndimage import median_filter
    local = median_filter(diff, 31) + 0.5
    nb = np.maximum(np.r_[diff[1:], 0], np.r_[0, diff[:-1]])
    spikes = np.where((diff / local > ratio) & (diff > 2 * nb))[0]
    return [round(float(i) / fps, 3) for i in spikes]
