"""Tunable detection parameters. Everything the scorer may sweep lives here."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class DetectConfig:
    # --- analysis proxy ---
    analysis_fps: float = 30.0
    analysis_width: int = 640

    # --- padding around the play ---
    pad_before: float = 1.0
    pad_after: float = 1.0

    # --- signal smoothing ---
    smooth_s: float = 0.3

    # --- activity signal ---
    pan_weight: float = 0.3        # activity = residual/p95 + pan_weight * pan/p95

    # --- snap detection (fractions of the clip's activity range p10..p95) ---
    quiet_frac: float = 0.15       # "set" / pre-snap if activity below this
    onset_frac: float = 0.35       # play underway once activity above this
    min_quiet_s: float = 0.5       # need this much quiet before the onset
    min_active_s: float = 1.0      # onset must stay active this long
    snap_eps: float = 0.08         # snap = where activity first rises this far above the set level
    post_whistle_penalty: float = 0.3  # down-weight bursts that start just after a whistle
    snap_floor_eps: float = 0.1    # walk snap back to where activity is within this of the pre-play floor
    snap_lookback_s: float = 6.0   # ... looking at most this far before the main burst
    tight_margin_s: float = 1.5    # snap/end this close to clip edges => clip is "tight"

    # --- end-of-play detection ---
    decay_frac: float = 0.35       # activity below baseline + decay_frac*(peak-baseline) ...
    min_settle_s: float = 1.5      # ... for this long => play over
    whistle_window_s: float = 2.0  # accept whistle up to this long after motion end
    whistle_min_strength: float = 0.5
    whistle_offset_s: float = 0.5  # play is "over" a beat after the whistle
    min_play_s: float = 2.0        # ignore whistles sooner than this after the snap

    # --- review flagging ---
    min_confidence: float = 0.35

    # --- clip classification ---
    scene_cut_ratio: float = 5.0   # frame-diff spike vs local median that counts as a hard cut
    min_scene_cuts: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def load(cls, path: Path | None) -> "DetectConfig":
        if path and Path(path).exists():
            return cls.from_dict(json.loads(Path(path).read_text()))
        return cls()

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
