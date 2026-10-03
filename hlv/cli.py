"""Command line: python -m hlv detect|gt-template|score|normalize <clips_dir>"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import project, score
from .config import DetectConfig
from .pipeline import CACHE_DIRNAME, analyze, load_clip
from .probe import find_clips, probe

CONFIG_FILENAME = "hlv_config.json"


def _cfg(root: Path, path: str | None) -> DetectConfig:
    return DetectConfig.load(Path(path) if path else root / CONFIG_FILENAME)


def _extract_one(root: Path, path: Path, cfg: DetectConfig, force: bool):
    t0 = time.time()
    load_clip(root, path, cfg, force)
    return path, time.time() - t0


def _analyze_all(root: Path, cfg: DetectConfig, force: bool = False, jobs: int = 1,
                 quiet: bool = False) -> list[tuple[dict, dict]]:
    paths = find_clips(root)
    if not paths:
        sys.exit(f"No video clips found under {root}")
    # Slow step (decode + signals) in parallel; results land in the on-disk cache.
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            futs = [ex.submit(_extract_one, root, p, cfg, force) for p in paths]
            for f in as_completed(futs):
                p, dt = f.result()
                if not quiet:
                    print(f"  signals {p.relative_to(root)}  ({dt:.1f}s)", flush=True)
    out = []
    for p in paths:
        t0 = time.time()
        info, cid, sig = load_clip(root, p, cfg, force and jobs <= 1)
        clip = analyze(root, p, info, cid, sig, cfg)
        if not quiet:
            d = clip["detection"]
            print(f"  {clip['source'][:48]:48} {clip['kind']:16} snap {d['snap']:6.2f}  end {d['end']:6.2f}"
                  f"  -> [{clip['proposed']['in']:.2f}, {clip['proposed']['out']:.2f}]"
                  f"  conf {d['confidence']:.2f} {clip['status']}"
                  + (f"  ({time.time() - t0:.1f}s)" if jobs <= 1 else ""), flush=True)
        out.append((clip, sig))
    return out


def cmd_detect(a) -> None:
    root = Path(a.dir).resolve()
    cfg = _cfg(root, a.config)
    results = _analyze_all(root, cfg, a.force, a.jobs)
    proj = project.load(root)
    proj["config"] = cfg.to_dict()
    for clip, _ in results:
        project.merge_clip(proj, clip)
    print(f"\nwrote {project.save(root, proj)}")
    if a.plots:
        from .debug_plot import plot_clip
        gt = score.load_gt(root)
        for clip, sig in results:
            plot_clip(root / "debug" / f"{clip['id']}.png", clip, sig, cfg, gt.get(clip["source"]))
        print(f"plots in {root / 'debug'}")


def cmd_gt_template(a) -> None:
    root = Path(a.dir).resolve()
    sources = [p.relative_to(root).as_posix() for p in find_clips(root)]
    print(f"wrote {score.write_template(root, sources)}  -- fill in snap_s / end_s (seconds or m:ss.s)")


def cmd_score(a) -> None:
    root = Path(a.dir).resolve()
    gt = score.load_gt(root)
    if not gt:
        sys.exit(f"No labelled rows in {root / score.GT_FILENAME} (run gt-template, then fill it in)")
    cfg = _cfg(root, a.config)
    labelled = [p for p in find_clips(root) if p.relative_to(root).as_posix() in gt]
    loaded = [(p, *load_clip(root, p, cfg)) for p in labelled]

    def run(c: DetectConfig):
        clips = [analyze(root, p, info, cid, sig, c) for p, info, cid, sig in loaded]
        return score.evaluate(clips, gt)

    rows, summary = run(cfg)
    score.print_report(rows, summary)
    skipped = len(gt) - len(rows)
    if skipped:
        print(f"({skipped} labelled clip(s) not scored: edited/TV footage is trimmed manually, or the file is missing)")
    if a.sweep and len(rows) < 8:
        print(f"\nnote: only {len(rows)} scored clips - sweep results will overfit; label 10+ for tuning you can trust")
    if a.sweep:
        print(f"\nsweeping {len(loaded)} clips ...")
        best, _ = score.sweep(cfg, lambda c: run(c)[1])
        rows, summary = run(best)
        print("\n=== best config ===")
        score.print_report(rows, summary)
        changed = {k: v for k, v in best.to_dict().items() if cfg.to_dict()[k] != v}
        print(f"changed: {changed or 'nothing'}")
        if changed:
            best.save(root / CONFIG_FILENAME)
            print(f"saved {root / CONFIG_FILENAME}  (detect will use it automatically)")


def cmd_normalize(a) -> None:
    """Full-res constant-frame-rate mezzanines for Resolve / FCPXML (Phase 4)."""
    root = Path(a.dir).resolve()
    out_dir = root / "normalized"
    for p in find_clips(root):
        dst = out_dir / p.relative_to(root).with_suffix(".mp4")
        if dst.exists() and not a.force:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        info = probe(p)
        fps = a.fps or (60 if info.fps >= 50 else 30)
        print(f"  {p.relative_to(root)} -> {dst.relative_to(root)} @ {fps}fps", flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(p), "-vf", f"fps={fps}",
                        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
                        str(dst)], check=True)


def cmd_zoom_preview(a) -> None:
    """Render auto-zoomed versions (and side-by-side comparisons) of selected clips."""
    from . import autozoom
    root = Path(a.dir).resolve()
    proj = project.load(root)
    by_src = {c["source"].lower(): c for c in proj["clips"]}
    zc = autozoom.ZoomConfig()
    for name in a.clips:
        c = by_src.get(name.replace("\\", "/").lower()) or next(
            (v for k, v in by_src.items() if k.endswith("/" + name.lower())), None)
        if c is None:
            print(f"  ? {name} not in highlights.json")
            continue
        path = root / c["source"]
        info = probe(path)
        r, d = c.get("review") or {}, c["detection"]
        snap = r.get("snap") if r.get("snap") is not None else d["snap"]
        end = r.get("end") if r.get("end") is not None else d["end"]
        t0, t1 = project.final_window(c)
        t0 = max(0.0, t0)
        print(f"  {c['source']}: play {snap:.2f}-{end:.2f}s, rendering {t0:.2f}-{t1:.2f}s", flush=True)
        t = time.time()
        # Same camera as the review page: follows your spotlight clicks if any, else the play.
        from . import camera
        if not r.get("zoom"):
            c.setdefault("review", {})["zoom"] = a.level
        pl = camera.plan(root / CACHE_DIRNAME, c)
        if pl["rects"] is None:
            if pl["player"] is None:
                print("    zoom is off and no spotlight for this clip; skipping")
                continue
            pl["rects"] = [[0.0, 0.0, 1.0, 1.0]] * len(pl["player"])   # highlight only
        look = project.settings(proj)
        print(f"    camera mode: {pl['mode']} at {pl['level']}x", flush=True)
        cx, cy = (c["crop"]["x"], c["crop"]["y"]) if c.get("crop") else (0, 0)
        W = c["crop"]["w"] if c.get("crop") else info.width
        H = c["crop"]["h"] if c.get("crop") else info.height
        # Normalised full-frame rects -> source px relative to the decoded (cropped) frame.
        rects = np.array(pl["rects"]) * [info.width, info.height, info.width, info.height] - [cx, cy, 0, 0]
        rects[:, 2] = np.minimum(rects[:, 2], W)
        rects[:, 3] = np.minimum(rects[:, 3], H)
        rects[:, 0] = np.clip(rects[:, 0], 0, W - rects[:, 2])
        rects[:, 1] = np.clip(rects[:, 1], 0, H - rects[:, 3])
        out = root / "previews" / c["id"]
        autozoom.render(path, info, c.get("crop"), rects, t0, t1,
                        out.with_name(c["id"] + "_zoom.mp4"), out.with_name(c["id"] + "_compare.mp4"), zc,
                        player=None if pl["player"] is None else np.array(pl["player"]),
                        style=a.style or look["highlight"], color=look["highlight_color"])
        zoom = W / rects[int(t0 * 30):int(t1 * 30) + 1, 2]
        print(f"    zoom {zoom.min():.2f}x-{zoom.max():.2f}x  ({time.time() - t:.0f}s)", flush=True)
    print(f"previews in {root / 'previews'}")


def cmd_reel(a) -> None:
    """Build the finished reel (same as the Build button in the review page)."""
    from . import profiles, reel
    root = Path(a.dir).resolve()
    if not profiles.load(root):
        p = profiles.ensure_template(root)
        sys.exit(f"Fill in your player profile first: {p}  (logo/music go in {root / profiles.ASSETS})")
    res = reel.build(root, a.player, a.transition, jobs=a.jobs)
    print(f"\nreel: {res['output']}  ({res['plays']} plays, {res['duration']}s, built in {res['seconds']}s)")
    print(f"Resolve timeline: {res['fcpxml']}")


def cmd_review(a) -> None:
    from .review_server import serve
    root = Path(a.dir).resolve()
    serve(root, _cfg(root, a.config), port=a.port, open_browser=not a.no_browser)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="hlv", description="Football highlight reel tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect snap/end and propose in/out points")
    d.add_argument("dir")
    d.add_argument("--config")
    d.add_argument("--plots", action="store_true", help="write debug PNGs to <dir>/debug")
    d.add_argument("--force", action="store_true", help="ignore signal cache")
    d.add_argument("-j", "--jobs", type=int, default=4)
    d.set_defaults(fn=cmd_detect)

    g = sub.add_parser("gt-template", help="write ground_truth.csv to fill in")
    g.add_argument("dir")
    g.set_defaults(fn=cmd_gt_template)

    s = sub.add_parser("score", help="score against ground_truth.csv")
    s.add_argument("dir")
    s.add_argument("--config")
    s.add_argument("--sweep", action="store_true", help="grid-search parameters, save best")
    s.set_defaults(fn=cmd_score)

    r = sub.add_parser("review", help="open the review page to confirm/correct snap & end")
    r.add_argument("dir")
    r.add_argument("--config")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--no-browser", action="store_true")
    r.set_defaults(fn=cmd_review)

    z = sub.add_parser("zoom-preview", help="render auto-zoom previews for some clips")
    z.add_argument("dir")
    z.add_argument("clips", nargs="+", help="clip file names, e.g. IMG_0761.MOV")
    z.add_argument("--style", choices=["ring", "arrow", "shadow", "off"],
                   help="highlight style (default: the one chosen in the review page)")
    z.add_argument("--level", type=float, default=1.5,
                   help="zoom for clips that don't have one set in the review page")
    z.set_defaults(fn=cmd_zoom_preview)

    b = sub.add_parser("reel", help="build the finished highlight reel")
    b.add_argument("dir")
    b.add_argument("--player", help="player id from players.json (default: the one chosen in the review page)")
    b.add_argument("--transition", choices=["crossfade", "cut", "dip"])
    b.add_argument("-j", "--jobs", type=int, default=3)
    b.set_defaults(fn=cmd_reel)

    n = sub.add_parser("normalize", help="write CFR H.264 mezzanines to <dir>/normalized")
    n.add_argument("dir")
    n.add_argument("--fps", type=int)
    n.add_argument("--force", action="store_true")
    n.set_defaults(fn=cmd_normalize)

    a = ap.parse_args(argv)
    a.fn(a)
