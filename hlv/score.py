"""Score detections against hand-labelled ground truth, and sweep parameters."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import DetectConfig

GT_FILENAME = "ground_truth.csv"
GT_FIELDS = ["source", "snap_s", "end_s", "notes"]

# Parameters worth sweeping, with candidate values.
SWEEP_GRID = {
    "quiet_frac": [0.1, 0.15, 0.2, 0.3],
    "onset_frac": [0.25, 0.35, 0.5],
    "decay_frac": [0.1, 0.15, 0.2, 0.35, 0.5],
    "min_settle_s": [1.0, 1.5, 2.0, 3.0],
    "pan_weight": [0.0, 0.3, 0.6],
    "snap_eps": [0.02, 0.04, 0.08],
    "snap_floor_eps": [0.0, 0.05, 0.1, 0.15, 0.25],
    "whistle_offset_s": [0.0, 0.25, 0.5, 0.75],
    "whistle_window_s": [2.0, 4.0, 8.0],
    "whistle_min_strength": [0.35, 0.5, 0.7],
}


def _num(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    if ":" in s:  # allow m:ss.s
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def write_template(root: Path, sources: list[str]) -> Path:
    p = root / GT_FILENAME
    if p.exists():
        raise FileExistsError(f"{p} already exists; not overwriting")
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, GT_FIELDS)
        w.writeheader()
        for s in sources:
            w.writerow({"source": s, "snap_s": "", "end_s": "", "notes": ""})
    return p


def upsert_gt(root: Path, source: str, snap: float | None, end: float | None, notes: str = "",
              all_sources: list[str] | None = None) -> None:
    """Set one clip's row in ground_truth.csv, keeping every other row as-is."""
    p = root / GT_FILENAME
    rows: list[dict] = []
    if p.exists():
        with p.open(newline="", encoding="utf-8-sig") as f:
            rows = [{k: r.get(k, "") or "" for k in GT_FIELDS} for r in csv.DictReader(f)]
    have = {r["source"].replace("\\", "/") for r in rows}
    for s in all_sources or []:
        if s not in have:
            rows.append({"source": s, "snap_s": "", "end_s": "", "notes": ""})
            have.add(s)
    if source not in have:
        rows.append({"source": source, "snap_s": "", "end_s": "", "notes": ""})
    fmt = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
    for r in rows:
        if r["source"].replace("\\", "/") == source:
            r.update(snap_s=fmt(snap), end_s=fmt(end), notes=notes or "")
    tmp = p.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, GT_FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["source"].lower()))
    tmp.replace(p)


def load_gt(root: Path) -> dict[str, dict]:
    """Hand-typed CSV rows, overridden by clips confirmed on the review page."""
    out = _load_gt_csv(root)
    from . import project
    for c in project.load(root).get("clips", []):
        r = c.get("review") or {}
        if r.get("labeled") and r.get("snap") is not None and r.get("end") is not None:
            out[c["source"]] = {"snap": r["snap"], "end": r["end"], "notes": r.get("notes", "")}
    return out


def _load_gt_csv(root: Path) -> dict[str, dict]:
    p = root / GT_FILENAME
    if not p.exists():
        return {}
    out = {}
    with p.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            snap, end = _num(row.get("snap_s", "")), _num(row.get("end_s", ""))
            if snap is not None or end is not None:
                out[row["source"].replace("\\", "/")] = {"snap": snap, "end": end,
                                                         "notes": row.get("notes", "")}
    return out


def evaluate(clips: list[dict], gt: dict[str, dict]) -> tuple[list[dict], dict]:
    rows = []
    for c in clips:
        g = gt.get(c["source"])
        # Edited/TV footage is trimmed by hand by design; its auto "play" isn't a prediction.
        if not g or c["kind"] == "pre_edited":
            continue
        d, p = c["detection"], c["proposed"]
        r = {"source": c["source"], "kind": c["kind"], "conf": d["confidence"],
             "snap": d["snap"], "gt_snap": g["snap"], "end": d["end"], "gt_end": g["end"],
             "snap_err": None if g["snap"] is None else d["snap"] - g["snap"],
             "end_err": None if g["end"] is None else d["end"] - g["end"]}
        if g["snap"] is not None and g["end"] is not None:
            r["covered"] = p["in"] <= g["snap"] and p["out"] >= g["end"]
            # Dead time beyond the ideal padded window.
            ideal_in, ideal_out = max(0, g["snap"] - 1.0), g["end"] + 1.0
            r["excess"] = max(0, ideal_in - p["in"]) + max(0, p["out"] - ideal_out)
        rows.append(r)

    def stats(key):
        e = np.array([abs(r[key]) for r in rows if r[key] is not None])
        if not e.size:
            return {}
        return {"mae": float(e.mean()), "within_0.5": float((e <= 0.5).mean()),
                "within_1.0": float((e <= 1.0).mean()), "n": int(e.size)}

    cov = [r["covered"] for r in rows if "covered" in r]
    exc = [r["excess"] for r in rows if "excess" in r]
    summary = {"snap": stats("snap_err"), "end": stats("end_err"),
               "coverage": float(np.mean(cov)) if cov else None,
               "excess_s": float(np.mean(exc)) if exc else None, "n": len(rows)}
    return rows, summary


def objective(summary: dict) -> float:
    """Lower is better. Cutting off part of a play is the worst error."""
    s, e = summary.get("snap", {}), summary.get("end", {})
    cov = summary.get("coverage") or 0.0
    return s.get("mae", 99) + e.get("mae", 99) + 3.0 * (1 - cov) + 0.2 * (summary.get("excess_s") or 0)


def print_report(rows: list[dict], summary: dict) -> None:
    f = lambda v: "   -  " if v is None else f"{v:6.2f}"  # noqa: E731
    print(f"{'clip':45} {'kind':16} {'conf':>5} {'snap':>6} {'gt':>6} {'err':>6}  {'end':>6} {'gt':>6} {'err':>6} cov")
    for r in rows:
        print(f"{r['source'][:45]:45} {r['kind']:16} {r['conf']:5.2f} {f(r['snap'])} {f(r['gt_snap'])} "
              f"{f(r['snap_err'])}  {f(r['end'])} {f(r['gt_end'])} {f(r['end_err'])} "
              f"{'' if 'covered' not in r else ('yes' if r['covered'] else 'NO')}")
    print()
    for k in ("snap", "end"):
        s = summary[k]
        if s:
            print(f"{k:5}: MAE {s['mae']:.2f}s | within 0.5s {s['within_0.5']:.0%} | "
                  f"within 1.0s {s['within_1.0']:.0%} | n={s['n']}")
    if summary["coverage"] is not None:
        print(f"coverage (play fully inside proposed window): {summary['coverage']:.0%}")
        print(f"mean excess dead time: {summary['excess_s']:.2f}s")
    print(f"objective (lower=better): {objective(summary):.3f}")


def sweep(base: DetectConfig, run, passes: int = 3) -> tuple[DetectConfig, dict]:
    """run(cfg) -> summary. Coordinate descent over SWEEP_GRID: tune one setting at a time,
    holding the rest, for a few passes. Far cheaper than every combination."""
    best_cfg, best_sum = base, run(base)
    best_obj = objective(best_sum)
    for _ in range(passes):
        improved = False
        for key, values in SWEEP_GRID.items():
            for v in values:
                d = best_cfg.to_dict()
                if d[key] == v:
                    continue
                d[key] = v
                if d["quiet_frac"] >= d["onset_frac"]:
                    continue
                cfg = DetectConfig.from_dict(d)
                s = run(cfg)
                o = objective(s)
                if o < best_obj - 1e-9:
                    best_cfg, best_sum, best_obj, improved = cfg, s, o, True
        if not improved:
            break
    return best_cfg, best_sum
