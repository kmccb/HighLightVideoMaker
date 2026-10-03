"""Assemble the finished highlight reel.

title card -> plays (in your order, each with freeze intro) -> end card, joined with the
chosen transition, game audio ducked under a music bed that fades in and out.
Also writes an FCPXML timeline (cuts + music lane) that free DaVinci Resolve imports,
pointing at the rendered segments, so you can retime or restyle in Resolve.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from . import graphics, library, profiles, project, segment

# Beat choice: per second away from the play's natural timing, and extra per second of
# stretching. With beats ~0.5 s apart, a beat one step further must be ~0.5 stronger.
BEAT_NEAR_COST = 1.0
BEAT_STRETCH_COST = 0.6
TRANSITIONS = {"crossfade": ("fade", 10 / 30), "dip": ("fadeblack", 18 / 30), "cut": (None, 0.0)}  # whole frames
TITLE_S, END_S = 4.0, 4.0
OUT_DIRNAME = "reel"


def included(clip: dict) -> bool:
    r = clip.get("review") or {}
    if r.get("keep") is False:
        return False
    # Edited/TV footage only once you've trimmed it yourself.
    return clip["kind"] != "pre_edited" or bool(r.get("labeled"))


def ordered_clips(proj: dict) -> list[dict]:
    """Your saved order first, then anything new in game/file order."""
    by_id = {c["id"]: c for c in proj["clips"]}
    order = [i for i in project.settings(proj).get("order", []) if i in by_id]
    rest = sorted((c for c in proj["clips"] if c["id"] not in order),
                  key=lambda c: (c.get("group", "").lower(), c["source"].lower()))
    return [by_id[i] for i in order] + rest


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard link (no extra disk space) when possible, else copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


BANNER_MODES = ("first", "every", "off")


def banner_ids(clips: list[dict], settings: dict) -> set[str]:
    """Which plays show the name banner. It rides on the freeze, which needs a spotlight,
    so "first" means the first play (in reel order) that has one."""
    from . import camera
    mode = settings.get("name_banner", "first")
    if mode == "off":
        return set()
    spot = [c["id"] for c in clips if camera.anchors_of(c)]
    return set(spot) if mode == "every" else set(spot[:1])


def _render_one(root: str, clip: dict, pl: dict, settings: dict) -> str:
    return str(segment.render(Path(root), clip, pl, settings))


def build(root: Path, player_id: str | None = None, transition: str | None = None,
          jobs: int = 3, progress=None) -> dict:
    """Render every included play and assemble the reel. progress(step, done, total)."""
    from . import beats as beatmod, hits
    t_start = time.time()
    say = progress or (lambda step, done, total: print(f"  {step} ({done}/{total})", flush=True))
    proj = project.load(root)
    settings = project.settings(proj)
    transition = transition or settings.get("transition", "crossfade")
    if transition not in TRANSITIONS:
        raise ValueError(f"transition must be one of {list(TRANSITIONS)}")
    lib = library.home()
    pl = profiles.get(lib, player_id or settings.get("player"))
    if pl is None:
        raise ValueError("no player profile yet - add one on the Player profiles page")
    clips = [c for c in ordered_clips(proj) if included(c)]
    if not clips:
        raise ValueError("no plays to include (all skipped?)")

    # 1) play segments, in parallel (decoding 4K is the slow part). Rendered with extra
    #    lead/tail and a long freeze; the timing below decides how much of each is used.
    total = len(clips) + 3
    seg_paths: list[Path | None] = [None] * len(clips)
    done = 0
    say("rendering plays", done, total)
    with ProcessPoolExecutor(jobs) as ex:
        with_banner = banner_ids(clips, settings)
        futs = {ex.submit(_render_one, str(root), c, pl, {**settings, "_banner": c["id"] in with_banner}): i
                for i, c in enumerate(clips)}
        for f in as_completed(futs):
            seg_paths[futs[f]] = Path(f.result())
            done += 1
            say(f"rendered {clips[futs[f]]['source'].split('/')[-1]}", done, total)

    # 2) title + end cards
    out_dir = root / OUT_DIRNAME
    seg_dir = out_dir / f"{pl['id']}_segments"
    if seg_dir.exists():
        shutil.rmtree(seg_dir)
    seg_dir.mkdir(parents=True)
    title = segment.card(seg_dir / "00_title.mp4", graphics.title_card(lib, pl), TITLE_S, fade_in=True)
    endc = segment.card(seg_dir / f"{len(clips) + 1:02d}_end.mp4", graphics.end_card(lib, pl), END_S, fade_in=False)
    done += 1
    say("cards", done, total)
    plays = []
    for i, (c, p) in enumerate(zip(clips, seg_paths), 1):
        dst = seg_dir / f"{i:02d}_{c['id']}.mp4"      # readable names for Resolve
        _link_or_copy(p, dst)
        _link_or_copy(p.with_suffix(".json"), dst.with_suffix(".json"))
        plays.append(dst)

    # 3) timing: nominal, or fitted so each hit lands on a strong beat
    music = project.music_path(root, settings)
    music_start = float(settings.get("music_start") or 0)
    _, d = TRANSITIONS[transition]
    grid = None
    if music and settings.get("beat_sync", True):
        grid = beatmod.cached(music, root / ".hlv_cache")
    metas = []
    for c, p in zip(clips, plays):
        lay = segment.layout(p)
        t_in, t_out, snap = segment._window(c)
        # The layout knows where the hit and out-point land in output time (slow-mo stretches it).
        metas.append({"lay": lay, "lead_nom": snap - t_in,
                      "post_nom": lay.get("post_nom", t_out - snap),
                      "hit": _hit_in_play(lay, hits.hit_of(root, c)[0], snap)})
    timing = plan_timing(metas, grid, music_start, TITLE_S - d, d)

    items = [[(title, 0.0, TITLE_S)]]
    for p, m, tm in zip(plays, metas, timing):
        a = m["lay"]["a_frames"] / segment.FPS
        f = m["lay"]["f_frames"] / segment.FPS
        pieces = [(p, a - tm["lead"], tm["lead"])]
        if tm["fz"] > 0:
            pieces.append((p, a, tm["fz"]))
        pieces.append((p, a + f, tm["post"]))
        items.append(pieces)
    items.append([(endc, 0.0, END_S)])

    # 4) assemble
    out = out_dir / f"{pl['id']}_reel.mp4"
    _assemble(items, transition, music, music_start, settings, out)
    done += 1
    say("assembled", done, total)

    # 5) Resolve timeline
    fcp = out_dir / f"{pl['id']}_reel.fcpxml"
    write_fcpxml(fcp, items, music, music_start, f"{pl.get('name') or pl['id']} highlights")
    done += 1
    say("done", done, total)
    synced = [t for t in timing if t.get("beat") is not None]
    return {"output": str(out), "fcpxml": str(fcp), "plays": len(clips),
            "duration": round(_duration(out), 1), "seconds": round(time.time() - t_start),
            "beat_sync": None if grid is None else {
                "bpm": grid["bpm"], "on_beat": sum(1 for t in synced if abs(t["err"]) <= 0.034),
                "avg_shift": round(float(np.mean([abs(t["shift"]) for t in timing])), 2) if timing else 0.0,
                "plays": len(timing), "timing": timing}}


def _hit_in_play(lay: dict, hit: float | None, snap: float) -> float:
    """Seconds from the start of the play part to the hit; -1 = no hit (timing left alone)."""
    if hit is None:
        return -1.0
    return lay["hit_b"] if lay.get("hit_b") is not None else hit - snap


def _fr(s: float) -> float:
    """Snap to whole frames."""
    return round(s * segment.FPS) / segment.FPS


def plan_timing(metas: list[dict], grid: dict | None, music_start: float, t_first: float,
                d: float) -> list[dict]:
    """Choose lead-in, freeze and tail per play.

    Without a beat grid: nominal values. With one: walk through the reel; for each play
    pick the strongest reachable beat for its hit (preferring small changes from nominal),
    then split the needed shift between lead-in and freeze within their limits. If a play
    has no room of its own (clip starts right at the snap, no freeze), the previous play's
    tail is lengthened or shortened instead.
    """
    beats_t = beats_s = None
    if grid:
        beats_t = np.array(grid["beats"]) - music_start      # reel time of each beat
        beats_s = np.array(grid["strength"])
    out, t = [], t_first
    prev_grow = prev_shrink = 0.0            # how much the previous play's tail can move
    for m in metas:
        lay = m["lay"]
        a_max = lay["a_frames"] / segment.FPS
        has_fz = lay["f_frames"] > 0
        lead_nom = float(np.clip(m["lead_nom"], 0, a_max))
        fz_nom = segment.FREEZE_S if has_fz else 0.0
        lead_lo, lead_hi = min(0.4, lead_nom), a_max
        fz_lo, fz_hi = (segment.FREEZE_MIN, segment.FREEZE_MAX) if has_fz else (0.0, 0.0)
        h = m["hit"]
        lead, fz, beat, err = lead_nom, fz_nom, None, None
        if beats_t is not None and h > 0:
            base = t + h                               # hit time if lead = fz = 0
            lo = base + lead_lo + fz_lo - prev_shrink
            hi = base + lead_hi + fz_hi + prev_grow
            nom = base + lead_nom + fz_nom
            ok = (beats_t >= lo) & (beats_t <= hi)
            if ok.any():
                cand = np.where(ok)[0]
                # Prefer the nearest beat: moving further from the play's natural timing
                # costs a lot, stretching it (lead-in/freeze longer) costs extra. A stronger
                # beat only wins if it's barely further away.
                delta = beats_t[cand] - nom
                score = (beats_s[cand] - BEAT_NEAR_COST * np.abs(delta)
                         - BEAT_STRETCH_COST * np.maximum(delta, 0))
                k = cand[int(np.argmax(score))]
                beat = float(beats_t[k])
                shift = beat - base                      # lead + fz must equal this
                own = float(np.clip(shift, lead_lo + fz_lo, lead_hi + fz_hi))
                spill = _fr(shift - own)
                if spill and out:                         # move this play via the last one's tail
                    out[-1]["post"] = _fr(out[-1]["post"] + spill)
                    t += spill
                    base += spill
                    shift = beat - base
                fz = float(np.clip(shift - lead_nom, fz_lo, fz_hi)) if has_fz else 0.0
                lead = float(np.clip(shift - fz, lead_lo, lead_hi))
                fz = float(np.clip(shift - lead, fz_lo, fz_hi)) if has_fz else 0.0
        lead, fz = _fr(lead), _fr(fz)
        b_max = lay["b_frames"] / segment.FPS
        post = _fr(float(np.clip(max(m["post_nom"], h + 0.8) if h > 0 else m["post_nom"], 0.5, b_max)))
        hit_reel = t + lead + fz + h
        if beat is not None:
            err = round(hit_reel - beat, 3)
        nominal = lead_nom + fz_nom
        min_post = max(h + 0.8, 0.5) if h > 0 else 0.5
        prev_grow, prev_shrink = max(0.0, b_max - post), max(0.0, post - min_post)
        out.append({"lead": lead, "fz": fz, "post": post, "hit_reel": round(hit_reel, 3),
                    "shift": round(lead + fz - nominal, 3),
                    "beat": None if beat is None else round(beat + music_start, 3), "err": err})
        t += lead + fz + post - d
    return out


def _assemble(items: list[list[tuple]], transition: str, music: Path | None, music_start: float,
              settings: dict, out: Path) -> None:
    """items: per timeline item, a list of (file, start, duration) pieces played in order."""
    kind, d = TRANSITIONS[transition]
    files = []
    for it in items:
        for p, _, _ in it:
            if p not in files:
                files.append(p)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for p in files:
        cmd += ["-i", str(p)]
    if music:
        cmd += ["-ss", f"{music_start:.3f}", "-stream_loop", "-1", "-i", str(music)]
    f, durs = [], []
    for n_it, it in enumerate(items):
        parts = []
        n_frames = 0
        for j, (p, st, du) in enumerate(it):
            k = files.index(p)
            # Cut on exact frame / sample numbers: rounded seconds can gain or drop a frame
            # at each piece boundary, which nudges hits off the beat.
            f0, nf = int(round(st * segment.FPS)), int(round(du * segment.FPS))
            spf = 48000 // segment.FPS
            f.append(f"[{k}:v]trim=start_frame={f0}:end_frame={f0 + nf},setpts=PTS-STARTPTS,"
                     f"settb=AVTB,fps={segment.FPS},format=yuv420p,setsar=1[i{n_it}v{j}]")
            f.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,"
                     f"atrim=start_sample={f0 * spf}:end_sample={(f0 + nf) * spf},asetpts=PTS-STARTPTS[i{n_it}a{j}]")
            parts.append(f"[i{n_it}v{j}][i{n_it}a{j}]")
            n_frames += nf
        f.append("".join(parts) + f"concat=n={len(it)}:v=1:a=1[v{n_it}][a{n_it}]")
        durs.append(n_frames / segment.FPS)
    n = len(items)
    if kind is None:
        f.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
        total = sum(durs)
    else:
        v, a, t = "v0", "a0", durs[0]
        for i in range(1, n):
            # Offset just *under* the exact frame time: a rounded-up offset (10.467 for
            # 10.46667) starts the transition a frame late and every hit after it slips.
            f.append(f"[{v}][v{i}]xfade=transition={kind}:duration={d:.6f}:offset={t - d - 1e-4:.6f}[vx{i}]")
            f.append(f"[{a}][a{i}]acrossfade=d={d:.6f}[ax{i}]")
            v, a, t = f"vx{i}", f"ax{i}", t + durs[i] - d
        f.append(f"[{v}]null[vc]")
        f.append(f"[{a}]anull[ac]")
        total = t
    f.append(f"[vc]fade=t=out:st={max(0.0, total - 0.8):.3f}:d=0.8[vout]")
    if music:
        game = float(settings.get("game_audio", 0.3))
        mvol = float(settings.get("music_volume", 0.9))
        f.append(f"[ac]volume={game}[g]")
        f.append(f"[{len(files)}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{total:.3f},"
                 f"afade=t=in:d=0.5,afade=t=out:st={max(0.0, total - 3):.3f}:d=3,volume={mvol}[m]")
        f.append("[g][m]amix=inputs=2:normalize=0:duration=first[aout]")
    else:
        f.append(f"[ac]afade=t=out:st={max(0.0, total - 1.5):.3f}:d=1.5[aout]")
    out.parent.mkdir(parents=True, exist_ok=True)
    script = out.with_suffix(".filter.txt")
    script.write_text(";\n".join(f), encoding="utf-8")   # long graphs exceed the command-line limit
    cmd += ["-filter_complex_script", str(script), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}", str(out)]
    try:
        subprocess.run(cmd, check=True)
    finally:
        script.unlink(missing_ok=True)


def write_fcpxml(path: Path, items: list[list[tuple]], music: Path | None, music_start: float,
                 name: str) -> None:
    """FCPXML 1.9 timeline: each play's pieces (lead / freeze / play) as cuts on the spine,
    music on a lane underneath. Add transitions in Resolve: select the clips, Ctrl+T."""
    fps = segment.FPS
    fr = lambda s: f"{int(round(s * fps)) * 100}/{fps * 100}s"  # noqa: E731
    files = []
    for it in items:
        for p, _, _ in it:
            if p not in files:
                files.append(p)
    res = [f'<format id="r0" name="FFVideoFormat1080p{fps}" frameDuration="100/{fps * 100}s" '
           f'width="1920" height="1080"/>']
    for i, p in enumerate(files, 1):
        res.append(f'<asset id="r{i}" name="{escape(p.stem)}" start="0s" duration="{fr(_duration(p))}" '
                   f'hasVideo="1" hasAudio="1" format="r0" audioSources="1" audioChannels="2" audioRate="48000">'
                   f'<media-rep kind="original-media" src="{escape(p.resolve().as_uri())}"/></asset>')
    total = sum(du for it in items for _, _, du in it)
    mid = None
    if music:
        mid = f"r{len(files) + 1}"
        res.append(f'<asset id="{mid}" name="{escape(music.stem)}" start="0s" duration="{fr(_duration(music))}" '
                   f'hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">'
                   f'<media-rep kind="original-media" src="{escape(music.resolve().as_uri())}"/></asset>')
    spine, t, first = [], 0.0, True
    for it in items:
        for p, st, du in it:
            inner = ""
            if first and mid:
                md = min(_duration(music) - music_start, total)
                inner = (f'<asset-clip ref="{mid}" lane="-1" offset="{fr(st)}" name="{escape(music.stem)}" '
                         f'duration="{fr(md)}" start="{fr(music_start)}"/>')
            first = False
            spine.append(f'<asset-clip ref="r{files.index(p) + 1}" offset="{fr(t)}" name="{escape(p.stem)}" '
                         f'duration="{fr(du)}" start="{fr(st)}">{inner}</asset-clip>')
            t += du
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.9">\n'
           f'<resources>\n{chr(10).join(res)}\n</resources>\n'
           f'<library><event name="{escape(name)}"><project name="{escape(name)}">'
           f'<sequence format="r0" duration="{fr(t)}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">'
           f'<spine>\n{chr(10).join(spine)}\n</spine></sequence></project></event></library>\n</fcpxml>\n')
    path.write_text(xml, encoding="utf-8")
