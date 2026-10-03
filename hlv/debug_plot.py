"""Per-clip diagnostic plot: the signals detection saw, what it chose, and ground truth."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import plays  # noqa: E402
from .config import DetectConfig  # noqa: E402


def plot_clip(out: Path, clip: dict, sig: dict, cfg: DetectConfig, gt: dict | None = None) -> None:
    fps = sig["meta"]["fps"]
    t = np.arange(len(sig["residual"])) / fps
    act = plays.activity(sig, fps, cfg)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    ax[0].plot(t, sig["residual"], lw=0.8, color="tab:blue", label="residual (players)")
    ax[0].set_ylabel("residual")
    ax0b = ax[0].twinx()
    ax0b.plot(t, sig["pan"], lw=0.8, color="tab:orange", alpha=0.7, label="camera pan")
    ax0b.set_ylabel("pan px/frame")
    ax[0].legend(loc="upper left")
    ax0b.legend(loc="upper right")

    ax[1].plot(t, act, color="k", lw=1, label="activity")
    ax[1].legend(loc="upper left")

    if len(sig["whistle_t"]):
        ax[2].plot(sig["whistle_t"], sig["whistle_s"], color="tab:purple", lw=0.8, label="whistle score")
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("seconds")
    ax[2].legend(loc="upper left")

    d, p = clip["detection"], clip["proposed"]
    for a in ax:
        a.axvspan(p["in"], p["out"], color="tab:green", alpha=0.08)
        a.axvline(d["snap"], color="tab:green", lw=1.5)
        a.axvline(d["end"], color="tab:red", lw=1.5)
        for c in d.get("scenes", []):
            a.axvline(c, color="gray", ls=":", lw=0.8)
        if gt:
            if gt.get("snap") is not None:
                a.axvline(gt["snap"], color="tab:green", ls="--", lw=1.5)
            if gt.get("end") is not None:
                a.axvline(gt["end"], color="tab:red", ls="--", lw=1.5)
    fig.suptitle(f"{clip['source']}  [{clip['kind']}]  snap={d['snap']} end={d['end']} "
                 f"conf={d['confidence']}  (solid=detected, dashed=truth)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
