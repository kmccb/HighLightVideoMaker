"""Match cuts between plays: one play ends with your player at a spot of the screen and the next
one opens with him on the same spot at the same size, so the cut reads as one continuous shot.

Nothing is re-timed. Over the last moments of play A the picture eases in (a push and a pan on
the play's own rendered frames) until he sits at a spot both plays can reach; it holds him there
through the transition, play B opens holding him at that same spot and eases back out to its own
framing. With the reel's crossfade this is a match dissolve - he turns into himself - and with
hard cuts a match cut. Where he is in each frame comes from the tracker's path and the reel's
framing (ghost.reel_view); a pair is left alone when he isn't found at the end of A or the start of B.

`apply` rewrites the reel's piece lists (reel.build's `items`) in place.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import camera, encoder, ghost, slowmo
from .pipeline import CACHE_DIRNAME

VERSION = 1
FPS = 30
EASE = 12             # frames of moving in (A) / back out (B)
MAX_SCALE = 1.7       # never enlarge the rendered play more than this
ROOM = 1.15           # ... and always at least this much: a picture that isn't enlarged can't be shifted at all
AIM = (0.5, 0.55)     # where he's put when both plays can reach it
MIN_LEFT = 6          # frames of a piece left untouched


class _Play:
    """Where the player is in each frame of a rendered segment, in output px."""

    def __init__(self, root: Path, clip: dict, seg: Path, size: tuple[int, int]):
        from . import segment
        self.clip, self.seg, self.size = clip, Path(seg), size
        self.lay = segment.layout(self.seg)
        self.plan = camera.plan(Path(root) / CACHE_DIRNAME, clip)
        self.player = np.array(self.plan["player"]) if self.plan.get("player") else None
        info = clip["probe"]
        _, t1 = segment.extents(clip, info["duration"])
        smo = self.lay.get("slowmo") or {}
        self.post_t = slowmo.timeline(self.lay["split"], t1, FPS, smo.get("window"), float(smo.get("speed") or 1.0))

    def source_time(self, f: int) -> float:
        a, fz = self.lay["a_frames"], self.lay["f_frames"]
        if f < a:
            return self.lay["t0"] + f / FPS
        if f < a + fz or not self.post_t:
            return self.lay["split"]
        return self.post_t[min(len(self.post_t) - 1, f - a - fz)]

    def where(self, f: int) -> tuple[float, float, float] | None:
        """(x, y, height) of him in segment frame f, or None where the tracker doesn't have him."""
        if self.player is None:
            return None
        t = self.source_time(f)
        p = self.player[min(len(self.player) - 1, max(0, int(round(t * FPS))))]
        if p[0] < 0 or np.isnan(p[0]) or p[4] < 0.3 or p[3] <= 0:
            return None
        info, crop = self.clip["probe"], self.clip.get("crop")
        ox, oy = (crop["x"], crop["y"]) if crop else (0, 0)
        rx, ry, rw, rh = ghost.reel_view(self.clip, self.plan, t, self.size)
        OW, OH = self.size
        return ((p[0] * info["width"] - ox - rx) * OW / rw, (p[1] * info["height"] - oy - ry) * OH / rh, p[3] * info["height"] * OH / rh)


    def near(self, f: int, step: int, reach: int) -> tuple[float, float, float] | None:
        """Where he is at frame f, or failing that at the nearest frame up to `reach` frames on (step 1)
        or back (step -1). At the end of a play the tracker has often lost him in the pile a moment
        earlier, and before the snap he stands still, so a spot from a second away is still about right."""
        for k in range(reach + 1):
            w = self.where(f + step * k)
            if w is not None:
                return w
        return None


def meet(a: tuple[float, float, float], b: tuple[float, float, float], size: tuple[int, int]) -> tuple[float, float, tuple[float, float]]:
    """How much each play is enlarged and the screen spot he's put on, given where he is (x, y, height)
    at the end of A and the start of B: (scale A, scale B, (x, y)). Both end up the same height where
    MAX_SCALE allows (enlarged, never shrunk: that would show the frame's edge), and the spot is the one nearest AIM that both can reach without showing an edge."""
    OW, OH = size
    h = max(a[2], b[2]) * ROOM                               # both enlarged a little, so both have room to be moved
    sa, sb = float(np.clip(h / a[2], 1.0, MAX_SCALE)), float(np.clip(h / b[2], 1.0, MAX_SCALE))
    q = []
    for axis, full, aim in ((0, OW, AIM[0] * OW), (1, OH, AIM[1] * OH)):
        # enlarged s times about some point, a pixel at p can sit anywhere in [full - s (full - p), s p] with no edge showing
        lo = max(full - sa * (full - a[axis]), full - sb * (full - b[axis]))
        hi = min(sa * a[axis], sb * b[axis])
        q.append(float(np.clip(aim, lo, hi)) if lo <= hi else (lo + hi) / 2)
    return sa, sb, (q[0], q[1])


def _affine(pos: tuple[float, float, float], scale: float, q: tuple[float, float], size: tuple[int, int]) -> np.ndarray:
    """Enlarge `scale` times and put the point pos on q, without ever showing the frame's edge."""
    OW, OH = size
    tx = float(np.clip(q[0] - scale * pos[0], OW - scale * OW, 0.0))
    ty = float(np.clip(q[1] - scale * pos[1], OH - scale * OH, 0.0))
    return np.array([[scale, 0, tx], [0, scale, ty]], np.float64)


def _clip(root: Path, play: _Play, f0: int, n: int, hold: int, scale: float, q: tuple[float, float], tail: bool,
          anchor: tuple[float, float, float]) -> Path | None:
    """n frames of the segment from f0, eased to (tail) or from (head) holding him on q for `hold` frames."""
    OW, OH = play.size
    key = hashlib.sha1(f"{VERSION}|{play.seg.stem}|{f0}|{n}|{hold}|{scale:.3f}|{q[0]:.1f}|{q[1]:.1f}|{tail}|{anchor[0]:.0f}|{anchor[1]:.0f}".encode()).hexdigest()[:12]
    out = Path(root) / CACHE_DIRNAME / "matchcut" / f"{play.clip['id']}-{key}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(play.seg), "-vf", f"trim=start_frame={f0}:end_frame={f0 + n},setpts=PTS-STARTPTS",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    if len(raw) < n * OW * OH * 3:
        return None
    fr = np.frombuffer(raw, np.uint8)[:n * OW * OH * 3].reshape(n, OH, OW, 3)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-ss", f"{f0 / FPS:.5f}", "-t", f"{n / FPS:.5f}", "-i", str(play.seg), "-map", "0:v", "-map", "1:a",
                            *encoder.h264(17), "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
    eye = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    # where he is in each frame; frames the tracker doesn't have him in take the nearest one that does
    # (the spot the match was worked out from, when none of these do), so the picture never jumps
    found = [play.where(f0 + k) for k in range(n)]
    known = [k for k, w in enumerate(found) if w is not None]
    spots = [found[min(known, key=lambda j: abs(j - k))] if known else anchor for k in range(n)]
    try:
        for k in range(n):
            pos = spots[k]
            moving = n - hold
            w = (k + 1) / moving if tail else (n - k) / moving   # 0 = the play's own framing, 1 = holding him on the spot
            w = ghost._ease(min(1.0, w))
            M = eye if pos is None else eye + (_affine(pos, scale, q, play.size) - eye) * (w * w if tail else w)
            enc.stdin.write(cv2.warpAffine(fr[k], M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0 or not tmp.exists():
        return None
    tmp.replace(out)
    return out


def apply(root: Path, plays: list[tuple[int, dict, Path]], items: list[list[tuple]], d: float, size: tuple[int, int] | None = None,
          link=None, log=print) -> int:
    """plays: (index in items, clip, rendered segment) for every play, in reel order; items: the reel's
    piece lists [[(file, start, duration), ...], ...] (edited in place); d: the transition's length.
    link(clip path, name) -> the path the reel should use (reel.build puts it next to the segments).
    Returns how many cuts were matched."""
    size = tuple(size) if size else ghost.WIDE
    hold = int(round(d * FPS))
    n = EASE + hold
    done = 0
    cache: dict[int, _Play] = {}
    for (ia, ca, sa_), (ib, cb, sb_) in zip(plays, plays[1:]):
        if ib != ia + 1:                                        # a photo between them
            continue
        try:
            pa_file, pa_start, pa_dur = items[ia][-1]
            pb_file, pb_start, pb_dur = items[ib][0]
            fa_end, fb0 = int(round((pa_start + pa_dur) * FPS)), int(round(pb_start * FPS))
            na = min(n, int(round(pa_dur * FPS)) - MIN_LEFT)
            nb = min(n, int(round(pb_dur * FPS)) - MIN_LEFT)
            if min(na, nb) < hold + 5:
                continue
            A = cache.setdefault(ia, _Play(root, ca, sa_, size))
            B = cache.setdefault(ib, _Play(root, cb, sb_, size))
            # B: up to the snap (he's standing still until then); A: up to a second back
            wa = A.near(fa_end - 1, -1, min(FPS, int(round(pa_dur * FPS)) - 1))
            wb = B.near(fb0, 1, max(0, min(int(1.5 * FPS), B.lay["a_frames"] - fb0)))
            if wa is None or wb is None:
                continue
            s_a, s_b, q = meet(wa, wb, size)
            ta = _clip(root, A, fa_end - na, na, hold, s_a, q, True, wa)
            tb = _clip(root, B, fb0, nb, hold, s_b, q, False, wb)
            if ta is None or tb is None:
                continue
            if link:
                ta, tb = link(ta, f"{ia:02d}_{ca['id']}_matchout.mp4"), link(tb, f"{ib:02d}_{cb['id']}_matchin.mp4")
            items[ia][-1:] = [(pa_file, pa_start, pa_dur - na / FPS), (ta, 0.0, na / FPS)]
            items[ib][:1] = [(tb, 0.0, nb / FPS), (pb_file, pb_start + nb / FPS, pb_dur - nb / FPS)]
            done += 1
        except Exception as e:  # noqa: BLE001 - that cut just stays as it was
            log(f"  match cut skipped ({ca['id']} -> {cb['id']}): {e}")
    if done:
        log(f"  match cuts: {done}")
    return done
