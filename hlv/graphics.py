"""Reel graphics drawn with Pillow: title card, end card, and the freeze-frame name banner.

All functions return images at the reel size (1920x1080). Cards are BGR arrays; the
banner is a premultiplied BGRA overlay composited per frame with a fade.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .profiles import asset

W, H = 1920, 1080
FONT_DIR = Path("C:/Windows/Fonts")


@lru_cache(maxsize=64)
def font(size: int, style: str = "Bold Condensed") -> ImageFont.FreeTypeFont:
    """Bahnschrift (condensed, sporty) with fallbacks."""
    try:
        f = ImageFont.truetype(str(FONT_DIR / "bahnschrift.ttf"), size)
        f.set_variation_by_name(style)
        return f
    except (OSError, ValueError):
        for name in ("seguibl.ttf", "arialbd.ttf"):
            try:
                return ImageFont.truetype(str(FONT_DIR / name), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _rgb(hex_: str, default=(31, 79, 209)) -> tuple:
    h = (hex_ or "").lstrip("#")
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def _bgr(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))[:, :, ::-1].copy()


def _background(pl: dict, size: tuple[int, int] = (W, H)) -> Image.Image:
    """Dark field with a team-colour sweep and an accent stripe."""
    W, H = size
    team = _rgb(pl.get("color"))
    accent = _rgb(pl.get("accent"), (255, 210, 63))
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (11, 13, 18)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # diagonal glow of team colour from the left
    d = np.clip(1.0 - (xx * 0.8 + (H - yy) * 0.35) / (W * 0.95), 0, 1) ** 1.6
    base += d[..., None] * np.array(team, np.float32) * 0.55
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    dr.polygon([(0, H - 40), (W, H - 120), (W, H - 108), (0, H - 28)], fill=accent)
    return img


def _logo(root: Path, pl: dict, box: int) -> Image.Image | None:
    p = asset(root, pl.get("logo", ""))
    if not p:
        return None
    try:
        lg = Image.open(p).convert("RGBA")
    except OSError:
        return None
    lg.thumbnail((box, box), Image.LANCZOS)
    return lg


def _paste_logo(img: Image.Image, lg: Image.Image, cx: int, cy: int) -> None:
    # soft shadow so a logo reads on any background
    sh = Image.new("RGBA", lg.size, (0, 0, 0, 0))
    sh.putalpha(lg.getchannel("A").point(lambda a: int(a * 0.55)))
    sh = sh.filter(ImageFilter.GaussianBlur(12))
    x, y = cx - lg.width // 2, cy - lg.height // 2
    img.paste(sh, (x + 8, y + 12), sh)
    img.paste(lg, (x, y), lg)


def _detail_line(pl: dict, custom: bool = True) -> str:
    """The small line under the name. A profile's own "tagline" (e.g. "2026 HIGHLIGHTS") replaces
    the automatic position · class of · size line; custom=False always gives the automatic one."""
    if custom and (pl.get("tagline") or "").strip():
        return pl["tagline"].strip().upper()
    bits = [pl.get("position", ""), f"CLASS OF {pl['class_year']}" if pl.get("class_year") else "",
            pl.get("height_weight", "")]
    return "  ·  ".join(b for b in bits if b).upper()


def title_card(root: Path, pl: dict) -> np.ndarray:
    img = _background(pl)
    dr = ImageDraw.Draw(img)
    accent = _rgb(pl.get("accent"), (255, 210, 63))
    lg = _logo(root, pl, 440)
    left = 880 if lg else 260
    if lg:
        _paste_logo(img, lg, 470, 500)
    y = 300
    if pl.get("number"):
        dr.text((left, y), f"#{pl['number']}", font=font(130), fill=accent)
        y += 150
    name = (pl.get("name") or "").upper()
    size = 170 if len(name) <= 12 else 130
    dr.text((left, y), name, font=font(size), fill=(255, 255, 255))
    y += size + 30
    det = _detail_line(pl)
    if det:
        dr.text((left, y), det, font=font(52, "SemiBold Condensed"), fill=(220, 224, 232))
        y += 70
    if pl.get("school"):
        dr.text((left, y), pl["school"].upper(), font=font(52, "SemiBold Condensed"), fill=accent)
    return _bgr(img)


def recruit_card(root: Path, pl: dict, size: tuple[int, int] = (W, H)) -> np.ndarray:
    """The recruiting cut's opening and closing card: everything a coach needs at a glance,
    centred in one column (so it works wide and vertical): logo, #, name, position / class /
    size, school, GPA, Hudl link."""
    W, H = size
    img = _background(pl, size)
    dr = ImageDraw.Draw(img)
    accent = _rgb(pl.get("accent"), (255, 210, 63))
    tall = H > W
    k = W / 1080 if tall else H / 1080          # text scale
    lg = _logo(root, pl, int((420 if tall else 250) * (W / 1080 if tall else 1)))
    name = (pl.get("name") or "").upper()
    spec = []                                   # (text, size, style, fill, gap after)
    if pl.get("number"):
        spec.append((f"#{pl['number']}", 120, "Bold Condensed", accent, 10))
    spec.append((name, 150 if len(name) <= 14 else 115, "Bold Condensed", (255, 255, 255), 40))
    det = _detail_line(pl, custom=False)
    if det:
        spec.append((det, 54, "SemiBold Condensed", (225, 230, 238), 18))
    if pl.get("school"):
        spec.append((pl["school"].upper(), 54, "SemiBold Condensed", accent, 46))
    if pl.get("gpa"):
        g = pl["gpa"].strip()
        spec.append((g if "gpa" in g.lower() else f"{g} GPA", 50, "SemiBold", (235, 238, 245), 16))
    if pl.get("hudl"):
        link = pl["hudl"].strip()
        for pre in ("https://", "http://", "www."):
            link = link[len(pre):] if link.lower().startswith(pre) else link
        spec.append((link, 50, "SemiBold", (235, 238, 245), 16))
    max_w = W * (0.88 if tall else 0.8)
    rows = []
    for t, size_, style, c, g in spec:
        f = font(int(size_ * k), style)
        while dr.textlength(t, font=f) > max_w and f.size > 20:     # long names / links shrink to fit
            f = font(int(f.size * 0.92), style)
        rows.append((t, f, c, int(g * k)))
    heights = [f.getbbox(t)[3] for t, f, _, _ in rows]
    logo_h = (lg.height + int(50 * k)) if lg else 0
    block = logo_h + sum(heights) + sum(g for *_, g in rows)
    y = (H - block) // 2 - int(30 * k)
    if lg:
        _paste_logo(img, lg, W // 2, y + lg.height // 2)
        y += logo_h
    for (t, f, c, g), h in zip(rows, heights):
        w = dr.textlength(t, font=f)
        dr.text(((W - w) / 2, y), t, font=f, fill=c)
        y += h + g
    return _bgr(img)


def end_card(root: Path, pl: dict) -> np.ndarray:
    img = _background(pl)
    dr = ImageDraw.Draw(img)
    accent = _rgb(pl.get("accent"), (255, 210, 63))
    lg = _logo(root, pl, 260)
    lines = [l for l in (_detail_line(pl), pl.get("school"), pl.get("contact")) if l]
    block = (300 if lg else 0) + 150 + 70 * len(lines)
    y = (H - block) // 2 - 20
    if lg:
        _paste_logo(img, lg, W // 2, y + 130)
        y += 300
    name = (pl.get("name") or "").upper()
    num = f"  #{pl['number']}" if pl.get("number") else ""

    def centred(text, f, fill, yy):
        w = dr.textlength(text, font=f)
        dr.text(((W - w) / 2, yy), text, font=f, fill=fill)

    centred(name + num, font(120), (255, 255, 255), y)
    y += 150
    for line, f, fill in ((_detail_line(pl), font(50, "SemiBold Condensed"), (220, 224, 232)),
                          ((pl.get("school") or "").upper(), font(50, "SemiBold Condensed"), accent),
                          (pl.get("contact") or "", font(44, "SemiBold"), (200, 205, 215))):
        if line:
            centred(line, f, fill, y)
            y += 70
    return _bgr(img)


def name_banner(pl: dict, size: tuple[int, int] = (W, H)) -> tuple[np.ndarray, np.ndarray]:
    """Lower-third banner for the freeze frame (any frame size; tall frames get it lower and
    tighter to the left edge). Returns (bgr, alpha 0..1)."""
    W, H = size
    team = _rgb(pl.get("color"))
    accent = _rgb(pl.get("accent"), (255, 210, 63))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    x0, y0, h = (90, 860, 120) if W >= H else (40, H - 420, 120)
    name = (pl.get("name") or "").upper()
    det = _detail_line(pl)
    nf, df = font(78), font(36, "SemiBold Condensed")
    num = f"#{pl['number']}" if pl.get("number") else ""
    num_w = int(dr.textlength(num, font=nf)) + 50 if num else 0
    body_w = int(max(dr.textlength(name, font=nf), dr.textlength(det, font=df) if det else 0)) + 70
    # slanted panels: accent block with number, team-colour block with name
    if num:
        dr.polygon([(x0, y0), (x0 + num_w + 20, y0), (x0 + num_w, y0 + h), (x0 - 20, y0 + h)],
                   fill=accent + (255,))
        dr.text((x0 + 14, y0 + 14), num, font=nf, fill=(15, 15, 20, 255))
    bx = x0 + num_w + 6
    dr.polygon([(bx + 20, y0), (bx + body_w + 20, y0), (bx + body_w, y0 + h), (bx, y0 + h)],
               fill=team + (235,))
    dr.text((bx + 34, y0 + (6 if det else 18)), name, font=nf, fill=(255, 255, 255, 255))
    if det:
        dr.text((bx + 36, y0 + 82), det, font=df, fill=(225, 230, 240, 255))
    # drop shadow
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(img.getchannel("A").point(lambda a: int(a * 0.5)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(shadow, (6, 8), shadow)
    out = Image.alpha_composite(out, img)
    arr = np.array(out).astype(np.float32)
    return arr[:, :, 2::-1].copy(), arr[:, :, 3] / 255.0


def composite(frame: np.ndarray, overlay: tuple[np.ndarray, np.ndarray], alpha: float,
              slide_px: int = 0) -> None:
    """Blend a (bgr, a) overlay onto frame in place, optionally slid in from the left."""
    if alpha <= 0:
        return
    bgr, a = overlay
    if slide_px:
        bgr = np.roll(bgr, -slide_px, axis=1)
        a = np.roll(a, -slide_px, axis=1)
        a[:, frame.shape[1] - slide_px:] = 0
    a = (a * alpha)[..., None]
    frame[:] = (frame.astype(np.float32) * (1 - a) + bgr * a).astype(np.uint8)


def scottforge_card(root: Path, b: dict) -> np.ndarray:
    """Closing credit: logo, 'Created by ScottForge', the URL in the accent colour, and an
    invite line. root holds the logo (the app folder)."""
    bg, accent = _rgb(b.get("bg"), (10, 10, 12)), _rgb(b.get("accent"), (255, 90, 31))
    base = np.zeros((H, W, 3), np.float32)
    base[:] = bg
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow = np.clip(1.0 - np.hypot((xx - W / 2) / (W * 0.55), (yy - H * 0.42) / (H * 0.6)), 0, 1) ** 2.2
    base += glow[..., None] * np.array(accent, np.float32) * 0.22
    base += np.random.default_rng(0).normal(0, 1.4, (H, W, 1))   # dither: a dark glow bands in 8 bit
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    lg = _logo(root, {"logo": b.get("logo", "")}, 300)
    title, url, invite = b.get("title", ""), b.get("url", ""), b.get("invite", "")
    tf, uf, inf = font(84), font(72), font(44, "SemiBold")
    block = (lg.height + 60 if lg else 0) + (110 if title else 0) + (100 if url else 0) + (80 if invite else 0)
    y = (H - block) // 2

    def centred(text, f, fill, yy):
        w = dr.textlength(text, font=f)
        dr.text(((W - w) / 2, yy), text, font=f, fill=fill)

    if lg:
        _paste_logo(img, lg, W // 2, y + lg.height // 2)
        y += lg.height + 60
    if title:
        centred(title, tf, (255, 255, 255), y)
        y += 110
    if url:
        centred(url.replace("https://", "").replace("http://", ""), uf, accent, y)
        y += 100
    if invite:
        centred(invite, inf, (200, 196, 190), y)
    dr.rectangle([0, H - 14, W, H], fill=accent)
    return _bgr(img)
