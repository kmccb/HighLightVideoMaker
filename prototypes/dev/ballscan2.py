"""Scan game videos for stretches where the ball is in the air above everyone (a shot)."""
import sys, time, subprocess, json
from pathlib import Path
import numpy as np
from hlv import detect
fps, W = 6.0, 1920; H = 1080
model = detect._import_ultralytics().YOLO(str(detect.MODEL))
out = {}
for src in sys.argv[2:]:
    src = Path(src); t0 = time.time(); k = 0; rows = []
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        b = model.predict(f, imgsz=W, classes=[0, 32], conf=0.1, verbose=False)[0].boxes
        ppl = [x.xyxy[0].tolist() for x in b if int(x.cls) == 0 and float(x.conf) > 0.3]
        for x in b:
            if int(x.cls) == 32:
                bb = x.xyxy[0].tolist(); cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                # how far above the nearest person's head (in that person's heights); >0 = clear of everyone
                near = [q for q in ppl if abs((q[0] + q[2]) / 2 - cx) < 3 * (q[2] - q[0])]
                clear = min(((q[1] - cy) / max(1, q[3] - q[1]) for q in near), default=9.0)
                rows.append([round(k / fps, 2), round(cx), round(cy), round(float(x.conf), 2), round(clear, 2), round(bb[2] - bb[0])])
        k += 1
    out[src.name] = rows
    air = [r for r in rows if r[4] > 0.25 and r[4] < 9]
    print(f"{src.name}: {k} frames {time.time()-t0:.0f}s, {len(rows)} ball hits, {len(air)} in the air", flush=True)
Path(sys.argv[1]).write_text(json.dumps(out))
