import sys, subprocess, time
from pathlib import Path
from hlv import reel
if __name__ == "__main__":
    root = Path(sys.argv[1]); plan = reel.preview_plan(root)
    it = [i for i in plan["items"] if i["kind"] == "play"][0]
    print("plan: play", it["start"], "-", it["end"], "big moment at", it["hit"], "on beat" if it["on_beat"] else "OFF", "trail block", it["ghost"], "colour", it["colour"])
    res = reel.build(root)
    out = Path(res["output"])
    nf = subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(out)],capture_output=True,text=True).stdout.strip()
    print("file frames", nf, "plan frames", round(plan["total"]*30), "timing", res["beat_sync"]["timing"][0])
