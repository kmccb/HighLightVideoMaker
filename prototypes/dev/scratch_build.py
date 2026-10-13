"""A 4-play scratch copy of Caleb's project, one play per hit effect; build it and compare with the plan."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

SRC = Path(r"C:\Users\tom\Claude\HighLightVideoMaker\TestClips\Caleb_Footbal")
DST = Path(sys.argv[1])
KINDS = dict(zip(sys.argv[3].split(","), ("triple", "replay", "shock", "frames")))
vertical = sys.argv[2] == "tall"


def link(a: Path, b: Path):
    b.parent.mkdir(parents=True, exist_ok=True)
    if not b.exists():
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)


def main():
    if not (DST / "highlights.json").exists():
        h = json.loads((SRC / "highlights.json").read_text(encoding="utf-8"))
        h["clips"] = [c for c in h["clips"] if c["id"] in KINDS]
        h["photos"] = []
        for c in h["clips"]:
            c.setdefault("review", {})["big"] = KINDS[c["id"]]
            c["review"]["keep"] = True
            link(SRC / c["source"], DST / c["source"])
        h.pop("order", None)
        if "source_dir" in h:
            h["source_dir"] = str(DST)
        s = h.setdefault("settings", {})
        s.update({"journey": False, "match_cuts": False, "colour": False, "scottforge_card": False, "intro": "classic",
                  "format": "wide", "cut": "reel", "coaches": [], "ghost_trail": True, "drop": True, "recut": True})
        DST.mkdir(parents=True, exist_ok=True)
        (DST / "highlights.json").write_text(json.dumps(h, indent=1), encoding="utf-8")
        for f in ("hlv_config.json", "players.json"):
            if (SRC / f).exists():
                shutil.copy2(SRC / f, DST / f)
        for d in ("music", "assets"):
            if (SRC / d).exists():
                for f in (SRC / d).rglob("*"):
                    if f.is_file():
                        link(f, DST / d / f.relative_to(SRC / d))
        for f in (SRC / ".hlv_cache").rglob("*"):
            rel = f.relative_to(SRC / ".hlv_cache")
            if f.is_file() and rel.parts[0] not in ("impact", "ghost", "freeze3d", "colour", "journey", "matchcut", "photos") \
                    and (any(f.name.startswith(i) for i in KINDS) or rel.parts[0] in ("beats", "music")):
                link(f, DST / ".hlv_cache" / rel)

    from hlv import project, reel

    proj = project.load(DST)
    print("clips:", [(c["id"], c["review"].get("big")) for c in proj["clips"]], flush=True)
    plan = reel.preview_plan(DST)
    for i in plan["items"]:
        if i["kind"] == "play":
            print("  plan", i["name"], "hit", i["hit"], "on beat" if i["on_beat"] else "OFF", "fx", i.get("fx"), "block", i.get("ghost"))
    t = time.time()
    res = reel.build(DST, vertical=vertical, log=lambda *a, **k: None) if "log" in reel.build.__code__.co_varnames else reel.build(DST, vertical=vertical)
    out = Path(res["output"] if isinstance(res, dict) and "output" in res else res if isinstance(res, (str, Path)) else list(DST.glob("reel/*.mp4"))[0])
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
    nf = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip()
    print(f"built {out.name} in {time.time() - t:.0f}s: file {dur:.3f}s ({nf} frames), plan {plan['total']:.3f}s ({round(plan['total'] * 30)} frames)")
    print("result:", {k: v for k, v in res.items() if k in ("beat_sync", "plays", "duration")} if isinstance(res, dict) else res)


if __name__ == '__main__':
    main()
