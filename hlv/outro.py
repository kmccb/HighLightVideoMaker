"""Animated ScottForge ending, in the look of ScottForge.ai.

Background: the forge clip from the website (glowing bar on the anvil), darkened like the
site's hero, slowly pushing in, with embers rising. Timeline (5 s):
  0.0-0.6  fade up from black
  0.5      the anvil mark strikes in (scale + flash + sparks), its three sparks pop
  0.9/1.2  "Scott" then the italic ember "Forge" rise into place; "Forge" glows and flickers
           like hot metal, with a spark burst as it lands
  1.9-2.9  "— CREATED BY" label, the website, then the invite line
  4.3-5.0  fade out with the music
Silent: the song plays on underneath. Cached by everything that changes it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import graphics
from .graphics import _rgb
from .intro import Sparks, _clamp, _glow, _paste, ease_out, ease_out_back, smooth

W, H = graphics.W, graphics.H
FPS = 30
OUTRO_S = 5.0
OUTRO_VERSION = 1
FONT_DIR = Path(__file__).parent / "fonts"
PAPER = (221, 235, 242)          # BGR of #F2EBDD (site --paper)
PAPER_DIM = (185, 203, 212)      # #D4CBB9
SPARK_YEL = (87, 200, 255)       # #FFC857


# ---------------------------------------------------------------- fonts + layers
def fraunces(size: int, italic: bool = False, weight: int = 600) -> ImageFont.FreeTypeFont:
    """The logo's face (Fraunces, SOFT 100, opsz 72 - as on the website); Georgia if missing."""
    try:
        f = ImageFont.truetype(str(FONT_DIR / ("Fraunces-Italic-VF.ttf" if italic else "Fraunces-VF.ttf")), size)
        f.set_variation_by_axes([72, weight, 100, 1])
        return f
    except (OSError, ValueError):
        return ImageFont.truetype(str(graphics.FONT_DIR / ("georgiaz.ttf" if italic else "georgiab.ttf")), size)


def _mono(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(graphics.FONT_DIR / "consola.ttf"), size)
    except OSError:
        return graphics.font(size, "SemiBold")


def _layer(text: str, f, fill_bgr, spacing: float = 0.0) -> np.ndarray:
    """Tightly cropped BGRA layer of one line of text (optional letter spacing)."""
    widths = [f.getlength(ch) for ch in text]
    tw = int(sum(widths) + spacing * max(0, len(text) - 1)) + 40
    asc, desc = f.getmetrics()
    img = Image.new("RGBA", (tw, asc + desc + 40), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    fill = tuple(int(v) for v in fill_bgr[::-1]) + (255,)
    if spacing:
        x = 20
        for ch, wch in zip(text, widths):
            dr.text((x, 20), ch, font=f, fill=fill)
            x += wch + spacing
    else:
        dr.text((20, 20), text, font=f, fill=fill)
    return np.array(img.crop(img.getbbox() or (0, 0, 1, 1)))[:, :, [2, 1, 0, 3]].copy()


def _mark(size: int, ember_bgr) -> tuple[np.ndarray, np.ndarray]:
    """The site's anvil mark (40x40 SVG: outlined anvil, solid base, three yellow sparks),
    drawn at 4x and scaled down. Returns (anvil, sparks) BGRA layers of the same size."""
    k = size * 4 / 40
    s = lambda *pts: [(x * k, y * k) for x, y in pts]          # noqa: E731
    ember = tuple(int(v) for v in ember_bgr[::-1]) + (255,)
    yel = tuple(int(v) for v in SPARK_YEL[::-1])
    lw = max(2, int(round(1.5 * k)))
    anvil = Image.new("RGBA", (size * 4, size * 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(anvil)
    d.line(s((8, 14), (32, 14), (28, 22), (12, 22), (8, 14)), fill=ember, width=lw, joint="curve")
    d.rectangle(s((16, 22), (24, 28)), outline=ember, width=lw)
    d.rectangle(s((12, 28), (28, 32)), fill=ember)
    sparks = Image.new("RGBA", (size * 4, size * 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(sparks)
    for cx, cy, r, a in ((20, 10, 2, 255), (14, 8, 1, 153), (26, 8, 1, 153)):
        d.ellipse(s((cx - r, cy - r), (cx + r, cy + r)), fill=yel + (a,))
    out = []
    for im in (anvil, sparks):
        out.append(np.array(im.resize((size, size), Image.LANCZOS))[:, :, [2, 1, 0, 3]].copy())
    return out[0], out[1]


# ---------------------------------------------------------------- background
def _forge_frames(video: Path | None, n: int) -> list[np.ndarray] | None:
    """n frames of the forge clip at 30 fps, cover-scaled to 1920x1080 (looped if short)."""
    if not video or not Path(video).exists():
        return None
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-stream_loop", "-1", "-ss", "1.0", "-i", str(video), "-t", f"{n / FPS + 0.2:.3f}",
         "-an", "-vf", f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},format=bgr24",
         "-f", "rawvideo", "pipe:1"], capture_output=True).stdout
    fb = W * H * 3
    frames = [np.frombuffer(raw[i * fb:(i + 1) * fb], np.uint8).reshape(H, W, 3) for i in range(len(raw) // fb)]
    return frames[:n] if frames else None


def _grade() -> np.ndarray:
    """Darkening like the site's hero: dim overall, darker at the edges and top, so the text reads."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.55) / (H * 0.7)) ** 2)
    g = np.clip(0.62 - 0.38 * r ** 1.5, 0.1, 0.62)
    g *= np.clip(0.7 + 0.3 * yy / H, 0, 1)                       # a touch darker at the top
    # a soft dark band behind the website + invite lines: the hot bar sits right behind them
    band = np.exp(-(((xx - W / 2) / (W * 0.42)) ** 2 + ((yy - 705) / 95) ** 2))
    g *= 1 - 0.72 * band
    return g[..., None]


class Embers:
    """Glowing specks drifting up and swaying, like the site's hero."""

    def __init__(self, rng, n=110):
        self.x = rng.uniform(0, W, n)
        self.y = rng.uniform(0, H * 1.1, n)
        self.v = rng.uniform(35, 140, n)
        self.r = rng.uniform(1.2, 3.4, n)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.hot = rng.uniform(0, 1, n)

    def draw(self, layer, t, ember):
        for i in range(len(self.x)):
            y = (self.y[i] - self.v[i] * t) % (H * 1.1)
            x = self.x[i] + 24 * np.sin(t * 1.3 + self.ph[i])
            fl = 0.55 + 0.45 * np.sin(t * 7 + self.ph[i] * 3)
            c = np.array(SPARK_YEL) * self.hot[i] + np.array(ember) * (1 - self.hot[i])
            cv2.circle(layer, (int(x), int(y)), int(round(self.r[i])), tuple(float(v) * fl for v in c), -1, cv2.LINE_AA)


# ---------------------------------------------------------------- render
def _texts(b: dict) -> tuple[str, str, str]:
    """(label, website, invite) from the branding text. 'Created by ScottForge' -> 'CREATED BY'
    above the logo, since the logo itself says ScottForge."""
    title = (b.get("title") or "").strip()
    label = title.replace("ScottForge", "").strip(" -·:").upper()
    url = (b.get("url") or "").replace("https://", "").replace("http://", "").strip("/")
    return label, url, (b.get("invite") or "").strip()


def render(out: Path, b: dict, video: Path | None, seconds: float = OUTRO_S, seed: int = 11,
           only_frame: float | None = None) -> Path | np.ndarray:
    """Render the ending to out (mp4, silent stereo track). only_frame: return that one frame
    (BGR uint8) instead, for the still preview."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * FPS))
    ember = _rgb(b.get("accent"), (232, 93, 47))[::-1]              # BGR
    label, url, invite = _texts(b)

    # ---- layers
    L_scott = _layer("Scott", fraunces(170), PAPER)
    L_forge = _layer("Forge", fraunces(170, italic=True), ember)
    L_forge_glow = _glow(L_forge, ember, 26)
    mark_px = 190
    L_anvil, L_dots = _mark(mark_px, ember)
    L_anvil_glow, L_dots_glow = _glow(L_anvil, ember, 18), _glow(L_dots, SPARK_YEL, 10)
    L_label = _layer(label, _mono(30), ember, spacing=9) if label else None
    L_url = _layer(url.upper(), graphics.font(64, "SemiBold"), PAPER, spacing=6) if url else None
    L_url_glow = _glow(L_url, ember, 14) if L_url is not None else None
    L_inv = _layer(invite, graphics.font(42, "SemiLight"), PAPER_DIM) if invite else None

    # ---- layout: one centred lockup  [mark] Scott Forge
    gap = 26
    word_w = L_scott.shape[1] + 8 + L_forge.shape[1]
    total_w = mark_px + gap + word_w
    x0 = (W - total_w) / 2
    y_logo = 470
    mark_c = (x0 + mark_px / 2, y_logo - 6)
    scott_c = (x0 + mark_px + gap + L_scott.shape[1] / 2, y_logo)
    forge_c = (x0 + mark_px + gap + L_scott.shape[1] + 8 + L_forge.shape[1] / 2, y_logo + 6)
    y_label, y_url, y_inv = 300, 660, 750

    frames = _forge_frames(video, n) if only_frame is None else _forge_frames(video, int(only_frame * FPS) + 1)
    grade = _grade()
    embers = Embers(rng)
    T_MARK, T_SCOTT, T_FORGE = 0.5, 0.9, 1.2
    T_LABEL, T_URL, T_INV = 1.9, 2.25, 2.6
    bursts = [Sparks(rng, mark_c[0], mark_c[1] - 40, T_MARK, SPARK_YEL, n=60, spread=(300, 1100)),
              Sparks(rng, forge_c[0], forge_c[1] + 50, T_FORGE + 0.25, SPARK_YEL, n=70, spread=(350, 1300),
                     cone=(np.pi * 1.05, np.pi * 1.95))]
    grain = [rng.normal(0, 5, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
    plain = np.zeros((H, W, 3), np.float32) + np.array([9, 11, 12], np.float32)

    def frame_at(k: int) -> np.ndarray:
        t = k / FPS
        if frames is not None:
            src = frames[min(k, len(frames) - 1)].astype(np.float32)
            z = 1.0 + 0.06 * (t / seconds)                            # slow push-in
            m = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.6]])
            bg = cv2.warpAffine(src, m, (W, H), borderMode=cv2.BORDER_REFLECT)
        else:
            bg = plain.copy()
        frame = bg * grade
        fx = np.zeros((H, W, 3), np.float32)
        embers.draw(fx, t, ember)
        for s in bursts:
            s.draw(fx, t)

        # mark: strikes in, glow breathes (like the site's logo), sparks pop one by one
        u = (t - T_MARK) / 0.35
        if u > 0:
            sc = 1.35 - 0.35 * ease_out_back(u, 2.2)
            a = _clamp(u * 2.5)
            breathe = 0.55 + 0.35 * np.sin(2 * np.pi * (t - T_MARK) / 3.5)
            _paste(frame, L_anvil_glow, *mark_c, sc, a * (breathe + (1.2 if u < 1 else 0) * (1 - _clamp(u - 0.5))), add=True)
            _paste(frame, L_anvil, *mark_c, sc, a)
            dots_a = _clamp((t - T_MARK - 0.25) / 0.2)
            _paste(frame, L_dots_glow, *mark_c, sc, dots_a * (0.8 + 0.4 * np.sin(t * 9)), add=True)
            _paste(frame, L_dots, *mark_c, sc, dots_a)
        # wordmark
        for L, c, t0 in ((L_scott, scott_c, T_SCOTT), (L_forge, forge_c, T_FORGE)):
            u = (t - t0) / 0.55
            if u > 0:
                _paste(frame, L, c[0], c[1] + 46 * (1 - ease_out(u)), 1.0, smooth(u))
        u = (t - T_FORGE) / 0.55
        if u > 0:                                                     # hot-metal glow + flicker
            heat = 0.75 + 0.18 * np.sin(t * 5.3) + 0.08 * np.sin(t * 13.1)
            land = 0.9 * max(0.0, 1 - abs(t - T_FORGE - 0.3) / 0.3)
            _paste(frame, L_forge_glow, forge_c[0], forge_c[1] + 46 * (1 - ease_out(u)), 1.0,
                   smooth(u) * (heat + land), add=True)
        # label / website / invite
        if L_label is not None and t > T_LABEL:
            u = (t - T_LABEL) / 0.5
            _paste(frame, L_label, W / 2 + 34, y_label, 1.0, smooth(u))
            lw = int(52 * ease_out(u))
            lx = int(W / 2 + 34 - L_label.shape[1] / 2 - 20)
            cv2.line(frame, (lx - lw, y_label), (lx, y_label), tuple(float(v) * smooth(u) for v in ember), 2, cv2.LINE_AA)
        if L_url is not None and t > T_URL:
            u = (t - T_URL) / 0.5
            _paste(frame, L_url_glow, W / 2, y_url, 1.0, 0.35 * smooth(u), add=True)
            _paste(frame, L_url, W / 2, y_url + 24 * (1 - ease_out(u)), 1.0, smooth(u))
        if L_inv is not None and t > T_INV:
            u = (t - T_INV) / 0.6
            _paste(frame, L_inv, W / 2, y_inv + 20 * (1 - ease_out(u)), 1.0, smooth(u))
        # glowing ember line along the bottom, sweeping across (like the site's progress bar)
        sweep = int(W * ease_out(_clamp(t / 2.4)))
        cv2.line(fx, (0, H - 6), (sweep, H - 6), ember, 3, cv2.LINE_AA)

        glow = cv2.resize(cv2.GaussianBlur(cv2.resize(fx, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3),
                          (W, H))
        frame += fx + glow * 1.4
        frame += cv2.resize(grain[k % len(grain)], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        fade = smooth(t / 0.6) * (1 - smooth((t - (seconds - 0.7)) / 0.7))
        return np.clip(frame * fade, 0, 255).astype(np.uint8)

    if only_frame is not None:
        return frame_at(int(only_frame * FPS))

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "pipe:0", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{n / FPS:.4f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
    try:
        for k in range(n):
            enc.stdin.write(frame_at(k).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ScottForge ending render failed")
    tmp.replace(out)
    return out


def video_path(lib_root: Path, b: dict) -> Path | None:
    from . import profiles
    return profiles.asset(lib_root, b.get("video", ""))


def cache_key(b: dict, video: Path | None, seconds: float) -> str:
    payload = {"v": OUTRO_VERSION, "b": {k: b.get(k) for k in ("title", "url", "invite", "accent")},
               "video": [str(video), video.stat().st_mtime] if video else None, "s": seconds}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def build(cache: Path, lib_root: Path, b: dict, seconds: float = OUTRO_S) -> Path:
    """Render (or reuse) the ending. cache: a project's .hlv_cache folder."""
    video = video_path(lib_root, b)
    out = Path(cache) / "outro" / f"outro-{cache_key(b, video, seconds)}.mp4"
    return out if out.exists() else render(out, b, video, seconds)


def still(lib_root: Path, b: dict) -> np.ndarray:
    """The finished look (one frame near the end) for the branding editor's preview."""
    return render(Path("unused.mp4"), b, video_path(lib_root, b), only_frame=3.6)
