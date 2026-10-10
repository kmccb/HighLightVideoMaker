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

from . import encoder, graphics, library, outro, photo_slide, photos, profiles, project, segment

# Beat choice: per second away from the play's natural timing, and extra per second of
# stretching. With beats ~0.5 s apart, a beat one step further must be ~0.5 stronger.
BEAT_NEAR_COST = 1.0
BEAT_STRETCH_COST = 0.6
TRANSITIONS = {"crossfade": ("fade", 10 / 30), "dip": ("fadeblack", 18 / 30), "cut": (None, 0.0)}  # whole frames
TITLE_S, END_S = 4.0, 4.0
CREDIT_S = outro.OUTRO_S  # animated ScottForge ending
OUT_DIRNAME = "reel"
PHOTO_FLEX = (0.75, 1.25)  # beat-fitted photo length, as a fraction of the "Photo length" setting
RECRUIT_MAX_S = 180.0      # recruiting cut: plays stop being added at about 3 minutes
CUTS = ("reel", "recruit", "both")


def included(clip: dict) -> bool:
    if photos.is_photo(clip):
        return clip.get("keep") is not False
    r = clip.get("review") or {}
    if r.get("keep") is False:
        return False
    # Edited/TV footage only once you've trimmed it yourself.
    return clip["kind"] != "pre_edited" or bool(r.get("labeled"))


def ordered_items(proj: dict) -> list[dict]:
    """Plays and photos in reel order: your saved order first, then new plays in game/file
    order. Each new photo goes just after the last item taken on or before it (first if it's
    older than everything), so photos sit with their game without moving anything you placed."""
    plays, pics = proj["clips"], proj.get("photos") or []
    by_id = {c["id"]: c for c in plays + pics}
    order = [i for i in project.settings(proj).get("order", []) if i in by_id]
    placed = set(order)
    seq = [by_id[i] for i in order] + sorted((c for c in plays if c["id"] not in placed),
                                             key=lambda c: (c.get("group", "").lower(), c["source"].lower()))
    for ph in sorted((p for p in pics if p["id"] not in placed), key=lambda p: (p.get("taken") or "", p["source"])):
        dated = [k for k, it in enumerate(seq) if it.get("taken")]
        if not dated or not ph.get("taken"):
            seq.append(ph)
            continue
        before = [k for k in dated if seq[k]["taken"] <= ph["taken"]]
        seq.insert(before[-1] + 1 if before else 0, ph)
    return seq


def ordered_clips(proj: dict) -> list[dict]:
    """Just the plays, in reel order."""
    return [c for c in ordered_items(proj) if not photos.is_photo(c)]


def starred(clip: dict) -> bool:
    return bool((clip.get("review") or {}).get("star"))


def play_seconds(clip: dict, settings: dict) -> float:
    """Roughly how long a play runs in the reel (its trim plus the freeze)."""
    from . import camera
    t0, t1 = project.final_window(clip)
    spot = bool(camera.anchors_of(clip))
    return max(0.0, t1 - max(0.0, t0)) + (segment.FREEZE_S if spot and settings.get("freeze", True) else 0.0)


def recruit_plays(proj: dict) -> list[dict]:
    """The recruiting cut: your starred plays first (in reel order), then the reel's other plays,
    until it's about RECRUIT_MAX_S long. Starred plays always go in; no photos."""
    s = project.settings(proj)
    plays = [c for c in ordered_clips(proj) if included(c)]
    out = [c for c in plays if starred(c)]
    total = sum(play_seconds(c, s) for c in out)
    for c in plays:
        if starred(c):
            continue
        n = play_seconds(c, s)
        if total + n > RECRUIT_MAX_S:
            break
        out.append(c)
        total += n
    return out


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


def _to_vertical(src: Path, dst: Path) -> Path:
    """A 16:9 card/intro on a 9:16 frame: full width in the middle over a blurred, darkened fill."""
    W_, H_ = segment.VERTICAL
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
                    f"[0:v]split[a][b];[a]scale=-2:{H_},crop={W_}:{H_},boxblur=28:2,eq=brightness=-0.22[bg];"
                    f"[b]scale={W_}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]",
                    "-map", "[v]", "-map", "0:a?", *encoder.h264(17),
                    "-c:a", "copy", "-movflags", "+faststart", str(dst)], check=True)
    return dst


def build(root: Path, player_id: str | None = None, transition: str | None = None,
          jobs: int = 3, progress=None, vertical: bool = False, recruiting: bool = False) -> dict:
    """Render every included play and assemble the reel. progress(step, done, total).
    vertical: a 9:16 version (1080x1920) for phones, framed on the player.
    recruiting: the recruiting cut instead (recruit_plays, info card at both ends, no photos,
    no name banner)."""
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
    if photos.sync(root, proj):             # photos added to / removed from the folder
        project.save(root, proj)
    entries = recruit_plays(proj) if recruiting else [c for c in ordered_items(proj) if included(c)]
    if recruiting:
        settings = {**settings, "name_banner": "off"}   # the info card says who he is
    clips = [c for c in entries if not photos.is_photo(c)]
    pics = [c for c in entries if photos.is_photo(c)]
    if not clips:
        raise ValueError("no plays to include (all skipped?)")
    photo_s = float(settings.get("photo_s", 4))

    # 1) play segments and photo slides, in parallel (decoding 4K is the slow part). Plays are
    #    rendered with extra lead/tail and a long freeze, photos a bit long; the timing below
    #    decides how much of each is used.
    total = len(clips) + len(pics) + 3
    seg_paths: list[Path | None] = [None] * len(clips)
    pic_paths: list[Path | None] = [None] * len(pics)
    skipped_photos: list[str] = []
    done = 0
    say("rendering plays", done, total)
    with ProcessPoolExecutor(jobs) as ex:
        with_banner = banner_ids(clips, settings)
        aspect = {"_aspect": "9:16"} if vertical else {}
        futs = {ex.submit(_render_one, str(root), c, pl, {**settings, **aspect, "_banner": c["id"] in with_banner}): ("play", i)
                for i, c in enumerate(clips)}
        for k, ph in enumerate(pics):         # alternate slides zoom in / out
            futs[ex.submit(photo_slide.render, root, ph, photo_slide.slide_length(photo_s), k % 2 == 1,
                           segment.VERTICAL if vertical else None)] = ("photo", k)
        for f in as_completed(futs):
            kind, i = futs[f]
            if kind == "play":
                seg_paths[i] = Path(f.result())
                name = clips[i]["source"]
            else:
                name = pics[i]["source"]
                try:
                    pic_paths[i] = Path(f.result())
                except Exception as e:  # noqa: BLE001 - a bad photo shouldn't stop the reel
                    skipped_photos.append(f"{name}: {e}")
            done += 1
            say(f"rendered {name.split('/')[-1]}", done, total)
    rendered = {c["id"]: p for c, p in zip(clips, seg_paths)} | {c["id"]: p for c, p in zip(pics, pic_paths) if p}
    seq = [(c, rendered[c["id"]]) for c in entries if c["id"] in rendered]

    # 2) title + end cards
    out_dir = root / OUT_DIRNAME
    suffix = ("_recruiting" if recruiting else "") + ("_vertical" if vertical else "")
    seg_dir = out_dir / f"{pl['id']}_segments{suffix}"
    size = segment.VERTICAL if vertical else (graphics.W, graphics.H)
    if seg_dir.exists():
        shutil.rmtree(seg_dir)
    seg_dir.mkdir(parents=True)
    # The song's beats are needed for the intro's slams as well as the play timing.
    music = project.music_path(root, settings)
    music_start = float(settings.get("music_start") or 0)
    grid = None
    composing = settings.get("music_source") == "compose"
    if composing:
        # A soundtrack written for this reel: a steady grid now (hits are timed onto it),
        # the music itself once the timeline is known.
        from . import composer
        grid, music, music_start = composer.grid(settings.get("compose_style", "rock")), None, 0.0
    elif music and settings.get("beat_sync", True):
        grid = beatmod.cached(music, root / ".hlv_cache")
    animated = settings.get("intro", "animated") != "classic" and not recruiting
    burst_mov = None                                                 # the intro blowing apart into the first play
    if animated:
        from . import intro
        say("animating the intro", done, total)
        intro_grid = grid or (beatmod.cached(music, root / ".hlv_cache") if music else None)
        intro_grid = intro.timing_grid(intro_grid, music, root / ".hlv_cache", settings)   # heavy hits / burst
        intro_slams = intro.beat_times(intro_grid, music_start)[0] if intro_grid else []
        src, title_s = intro.build(root, lib, pl, clips, intro_grid, music_start,
                                   sfx=bool(settings.get("intro_sfx", False)), style=settings.get("intro", "animated"),
                                   size=size)                       # rendered natively for the vertical reel
        title = seg_dir / "00_title.mp4"
        _link_or_copy(src, title)
        if intro_grid and intro_grid.get("burst"):
            from . import burst
            burst_mov = burst.render(root / ".hlv_cache", src, settings.get("intro_out"), title_s, pl.get("color"))
    elif recruiting:
        intro_slams = []
        title_s = TITLE_S
        title = segment.card(seg_dir / "00_title.mp4", graphics.recruit_card(lib, pl, size), TITLE_S, fade_in=True)
    else:
        intro_slams = []
        title_s = TITLE_S
        title = segment.card(seg_dir / "00_title.mp4", graphics.title_card(lib, pl), TITLE_S, fade_in=True)
    endc = segment.card(seg_dir / f"{len(seq) + 1:02d}_end.mp4",
                        graphics.recruit_card(lib, pl, size) if recruiting else graphics.end_card(lib, pl), END_S, fade_in=False)
    credit = None
    brand = library.branding()
    if settings.get("scottforge_card") and library.branding_ready(brand):
        credit = seg_dir / f"{len(seq) + 2:02d}_scottforge.mp4"
        _link_or_copy(outro.build(lib / "cache", lib, brand, CREDIT_S, size=size), credit)   # native 9:16 when vertical
    if vertical:                                    # 16:9 cards: place them on a tall frame
        if not animated and not recruiting:
            title = _to_vertical(title, seg_dir / "00_title_v.mp4")
        if not recruiting:
            endc = _to_vertical(endc, endc.with_name(endc.stem + "_v.mp4"))
    done += 1
    say("cards", done, total)
    linked = []
    for i, (c, p) in enumerate(seq, 1):
        dst = seg_dir / f"{i:02d}_{c['id']}.mp4"      # readable names for Resolve
        _link_or_copy(p, dst)
        if not photos.is_photo(c):
            _link_or_copy(p.with_suffix(".json"), dst.with_suffix(".json"))
        linked.append(dst)

    # 3) timing: nominal, or fitted so each hit lands on a strong beat
    _, d = TRANSITIONS[transition]
    metas = []
    for (c, _), p in zip(seq, linked):
        if photos.is_photo(c):
            metas.append({"photo": True, "len": photo_s})
            continue
        lay = segment.layout(p)
        t_in, t_out, snap = segment._window(c)
        # The layout knows where the hit and out-point land in output time (slow-mo stretches it).
        # lead-in runs up to the freeze (the snap, or later if he was only found later)
        metas.append({"lay": lay, "lead_nom": lay.get("split", snap) - t_in,
                      "post_nom": lay.get("post_nom", t_out - lay.get("split", snap)),
                      "hit": _hit_in_play(lay, hits.hit_of(root, c)[0], snap)})
    d0 = 0.0 if burst_mov else d                     # a burst is a hard cut on the hit, not a crossfade
    timing = plan_timing(metas, grid, music_start, title_s - d0, d)

    items = [[(title, 0.0, title_s)]]
    for p, m, tm in zip(linked, metas, timing):
        if m.get("photo"):
            items.append([(p, 0.0, tm["post"])])
            continue
        a = m["lay"]["a_frames"] / segment.FPS
        f = m["lay"]["f_frames"] / segment.FPS
        pieces = [(p, a - tm["lead"], tm["lead"])]
        if tm["fz"] > 0:
            pieces.append((p, a, tm["fz"]))
        pieces.append((p, a + f, tm["post"]))
        items.append(pieces)
    items.append([(endc, 0.0, END_S)])
    if credit is not None:
        items.append([(credit, 0.0, CREDIT_S)])     # music keeps playing and fades out over it

    if composing:
        say("composing the soundtrack", done, total)
        music = composer.build(root / ".hlv_cache", settings.get("compose_style", "rock"),
                               _timeline(items, metas, timing, d, intro_slams if animated else [],
                                         credit is not None, first_d=d0),
                               int(settings.get("compose_seed", 1)))

    # 4) assemble
    out = out_dir / output_name(pl["id"], vertical, recruiting)
    # Intro impact sounds play at full level; game audio sits under the music.
    gains = [1.0] + [None] * (len(items) - 1)
    duck = None
    if credit is not None and outro.has_sound(brand):
        gains[-1] = 1.0                             # the ending's own sound (hammer / quench) at full level
        duck = CREDIT_S                             # ... with the music pulled down under it
    _assemble(items, transition, music, music_start, settings, out, gains, duck_tail=duck, burst=burst_mov)
    done += 1
    say("assembled", done, total)

    # 5) Resolve timeline
    fcp = out.with_suffix(".fcpxml")
    write_fcpxml(fcp, items, music, music_start, f"{pl.get('name') or pl['id']} {'recruiting' if recruiting else 'highlights'}"
                 f"{' (vertical)' if vertical else ''}",
                 size=segment.VERTICAL if vertical else (1920, 1080),
                 music_end=None if composing or settings.get("music_end") is None else float(settings["music_end"]))
    done += 1
    say("done", done, total)
    play_timing = [t for t in timing if not t.get("photo")]
    synced = [t for t in play_timing if t.get("beat") is not None]
    return {"output": str(out), "fcpxml": str(fcp), "plays": len(clips), "vertical": vertical, "recruiting": recruiting,
            "photos": len(seq) - len(clips), "skipped_photos": skipped_photos,
            "duration": round(_duration(out), 1), "seconds": round(time.time() - t_start),
            "beat_sync": None if grid is None else {
                "bpm": grid["bpm"], "on_beat": sum(1 for t in synced if abs(t["err"]) <= 0.034),
                "avg_shift": round(float(np.mean([abs(t["shift"]) for t in play_timing])), 2) if play_timing else 0.0,
                "plays": len(play_timing), "timing": timing}}


def output_name(player_id: str, vertical: bool = False, recruiting: bool = False) -> str:
    return f"{player_id}_{'recruiting' if recruiting else 'reel'}{'_vertical' if vertical else ''}.mp4"


def _hit_in_play(lay: dict, hit: float | None, snap: float) -> float:
    """Seconds from the start of the play part to the hit; -1 = no hit (timing left alone)."""
    if hit is None:
        return -1.0
    return lay["hit_b"] if lay.get("hit_b") is not None else hit - lay.get("split", snap)


def _fr(s: float) -> float:
    """Snap to whole frames."""
    return round(s * segment.FPS) / segment.FPS


def _timeline(items, metas, timing, d, slams, has_credit, first_d=None) -> dict:
    """Where everything happens in the finished reel (seconds), for the composer.
    first_d: the intro-to-first-item overlap if it differs from d (0 for a burst cut)."""
    durs = [sum(du for _, _, du in it) for it in items]
    first_d = d if first_d is None else first_d
    starts = [sum(durs[:i]) - (first_d + (i - 1) * d if i else 0.0) for i in range(len(items))]
    plays, pics = [], []
    for k, (m, tm) in enumerate(zip(metas, timing), 1):
        s0, s1 = starts[k], starts[k] + durs[k]
        if m.get("photo"):
            pics.append((round(s0, 3), round(s1, 3)))
            continue
        fs = s0 + tm["lead"] if tm["fz"] > 0 else None
        plays.append({"start": round(s0, 3), "end": round(s1, 3),
                      "fs": None if fs is None else round(fs, 3), "fe": None if fs is None else round(fs + tm["fz"], 3),
                      "hit": round(tm["hit_reel"], 3) if m["hit"] > 0 else None})
    n_end = len(metas) + 1
    total = starts[-1] + durs[-1]
    return {"intro": {"end": round(durs[0], 3), "slams": [round(t, 3) for t in slams]}, "plays": plays, "photos": pics,
            "end": (round(starts[n_end], 3), round(starts[n_end] + durs[n_end], 3)),
            "credit": (round(starts[-1], 3), round(total, 3)) if has_credit else None, "total": round(total, 3)}


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
        if m.get("photo"):
            # A photo slide: no hit. Its length (within 0.75-1.25x of nominal) is chosen so the
            # cut out of it - the middle of the transition - lands on a beat.
            nom_len = float(m["len"])
            lo, hi = PHOTO_FLEX[0] * nom_len, PHOTO_FLEX[1] * nom_len
            length, beat, err = nom_len, None, None
            if beats_t is not None:
                ok = (beats_t >= t + lo - d / 2) & (beats_t <= t + hi - d / 2)
                if ok.any():
                    cand = np.where(ok)[0]
                    delta = beats_t[cand] - (t + nom_len - d / 2)
                    k = cand[int(np.argmax(beats_s[cand] - BEAT_NEAR_COST * np.abs(delta)))]
                    beat = float(beats_t[k])
                    length = beat - t + d / 2
            length = _fr(length)
            if beat is not None:
                err = round(t + length - d / 2 - beat, 3)
            out.append({"photo": True, "lead": 0.0, "fz": 0.0, "post": length, "shift": round(length - nom_len, 3),
                        "beat": None if beat is None else round(beat + music_start, 3), "err": err})
            prev_grow, prev_shrink = max(0.0, hi - length), max(0.0, length - lo)
            t += length - d
            continue
        lay = m["lay"]
        a_max = lay["a_frames"] / segment.FPS
        has_fz = lay["f_frames"] > 0
        lead_nom = float(np.clip(m["lead_nom"], 0, a_max))
        long_fz = has_fz and lay.get("banner")                 # the name-banner play holds longer
        fz_nom = (segment.BANNER_FREEZE[1] if long_fz else segment.FREEZE_S) if has_fz else 0.0
        lead_lo, lead_hi = min(0.4, lead_nom), a_max
        fz_lo, fz_hi = ((segment.BANNER_FREEZE[0], segment.BANNER_FREEZE[2]) if long_fz else
                        (segment.FREEZE_MIN, segment.FREEZE_MAX)) if has_fz else (0.0, 0.0)
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


def music_window(total: float, settings: dict, music_start: float | None = None) -> tuple[float, float, float]:
    """When the song stops in the reel and its fades: (stop, fade_in, fade_out), all in reel seconds.
    music_end (seconds into the song) trims it; a soundtrack written for the reel is never trimmed."""
    stop = total
    end = settings.get("music_end")
    if end is not None and settings.get("music_source") != "compose":
        ms = float(settings.get("music_start") or 0) if music_start is None else music_start
        stop = max(0.0, min(total, float(end) - ms))
    fade_in = max(0.0, float(settings.get("music_fade_in", 0.5) or 0))
    fade_out = max(0.0, float(settings.get("music_fade_out", 3.0) or 0))
    fade_in = min(fade_in, stop / 2)
    fade_out = min(fade_out, max(0.0, stop - fade_in))
    return stop, fade_in, fade_out


def _assemble(items: list[list[tuple]], transition: str, music: Path | None, music_start: float,
              settings: dict, out: Path, gains: list | None = None, duck_tail: float | None = None,
              burst: Path | None = None) -> None:
    """items: per timeline item, a list of (file, start, duration) pieces played in order.
    gains: per-item audio level (None = game-audio level under the music).
    duck_tail: seconds at the end where the music drops to a third (an ending with its own sound).
    burst: an overlay (.mov with alpha) of the intro blowing apart: the intro hard-cuts to item 1
    and this plays on top of its first moments."""
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
    if burst is not None:
        cmd += ["-i", str(burst)]
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
        if burst is not None and n_it == 1:
            bi = len(files) + (1 if music else 0)
            f.append("".join(parts) + f"concat=n={len(it)}:v=1:a=1[v{n_it}pre][r{n_it}]")
            f.append(f"[{bi}:v]setpts=PTS-STARTPTS,settb=AVTB,fps={segment.FPS},format=yuva420p[bst]")
            f.append(f"[v{n_it}pre][bst]overlay=0:0:eof_action=pass,format=yuv420p[v{n_it}]")
        else:
            f.append("".join(parts) + f"concat=n={len(it)}:v=1:a=1[v{n_it}][r{n_it}]")
        g = (gains[n_it] if gains and gains[n_it] is not None
             else (float(settings.get("game_audio", 0.3)) if music else 1.0))
        f.append(f"[r{n_it}]volume={g}[a{n_it}]")
        durs.append(n_frames / segment.FPS)
    n = len(items)
    if kind is None:
        f.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
        total = sum(durs)
    else:
        v, a, t = "v0", "a0", durs[0]
        for i in range(1, n):
            if i == 1 and burst is not None:            # hard cut on the hit; the burst covers it
                f.append(f"[{v}][{a}][v1][a1]concat=n=2:v=1:a=1[vx1][ax1]")
                v, a, t = "vx1", "ax1", t + durs[1]
                continue
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
        mvol = float(settings.get("music_volume", 0.9))
        stop, fin, fout = music_window(total, settings, music_start)
        f.append("[ac]anull[g]")                      # per-item levels already applied
        f.append(f"[{len(files)}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{max(stop, 0.05):.3f},"
                 + (f"afade=t=in:d={fin:.3f}," if fin > 0 else "")
                 + (f"afade=t=out:st={max(0.0, stop - fout):.3f}:d={fout:.3f}," if fout > 0 else "")
                 + f"volume={mvol}"
                 + (f",volume=eval=frame:volume='1-0.67*clip((t-{total - duck_tail:.3f})/0.4,0,1)'" if duck_tail else "")
                 + "[m]")
        f.append("[g][m]amix=inputs=2:normalize=0:duration=first[aout]")
    else:
        f.append(f"[ac]afade=t=out:st={max(0.0, total - 1.5):.3f}:d=1.5[aout]")
    out.parent.mkdir(parents=True, exist_ok=True)
    script = out.with_suffix(".filter.txt")
    script.write_text(";\n".join(f), encoding="utf-8")   # long graphs exceed the command-line limit
    cmd += ["-filter_complex_script", str(script), "-map", "[vout]", "-map", "[aout]",
            *encoder.h264(18, "medium"),
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}", str(out)]
    try:
        subprocess.run(cmd, check=True)
    finally:
        script.unlink(missing_ok=True)


def write_fcpxml(path: Path, items: list[list[tuple]], music: Path | None, music_start: float,
                 name: str, size: tuple[int, int] = (1920, 1080), music_end: float | None = None) -> None:
    """FCPXML 1.9 timeline: each play's pieces (lead / freeze / play) as cuts on the spine,
    music on a lane underneath. Add transitions in Resolve: select the clips, Ctrl+T."""
    fps = segment.FPS
    fr = lambda s: f"{int(round(s * fps)) * 100}/{fps * 100}s"  # noqa: E731
    files = []
    for it in items:
        for p, _, _ in it:
            if p not in files:
                files.append(p)
    fmt = f'name="FFVideoFormat1080p{fps}" ' if size == (1920, 1080) else ""
    res = [f'<format id="r0" {fmt}frameDuration="100/{fps * 100}s" width="{size[0]}" height="{size[1]}"/>']
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
                md = min(_duration(music) - music_start, total,
                         (music_end - music_start) if music_end is not None else total)
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
