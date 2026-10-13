"""A one-play scratch copy of Caleb's project with changed marks, and that play's big-play preview."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

SRC = Path(r"C:\Users\tom\Claude\HighLightVideoMaker\TestClips\Caleb_Footbal")


def link(a: Path, b: Path):
    b.parent.mkdir(parents=True, exist_ok=True)
    if not b.exists():
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)


def main():
    dst, cid, changes = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
    h = json.loads((SRC / "highlights.json").read_text(encoding="utf-8"))
    h["clips"] = [c for c in h["clips"] if c["id"] == cid]
    h["photos"] = []
    c = h["clips"][0]
    c["review"].update(changes.get("review", {}))
    c["review"]["keep"] = True
    link(SRC / c["source"], dst / c["source"])
    if "source_dir" in h:
        h["source_dir"] = str(dst)
    h.setdefault("settings", {}).update({"journey": False, "match_cuts": False, **changes.get("settings", {})})
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "highlights.json").write_text(json.dumps(h, indent=1), encoding="utf-8")
    for f in ("hlv_config.json", "players.json"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, dst / f)
    for d in ("music", "assets"):
        for f in (SRC / d).rglob("*") if (SRC / d).exists() else []:
            if f.is_file():
                link(f, dst / d / f.relative_to(SRC / d))
    for f in (SRC / ".hlv_cache").rglob("*"):
        rel = f.relative_to(SRC / ".hlv_cache")
        if f.is_file() and rel.parts[0] not in ("impact", "ghost", "freeze3d", "colour", "journey", "matchcut", "photos", "previews", "segments") \
                and (f.name.startswith(cid) or rel.parts[0] in ("beats", "music")):
            link(f, dst / ".hlv_cache" / rel)
    from hlv import reel
    t = time.time()
    p = reel.preview_big(dst, cid)
    d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    print(p, d, f"{time.time() - t:.0f}s")
    if len(sys.argv) > 4:
        shutil.copy2(p, sys.argv[4])


if __name__ == "__main__":
    main()
