"""'Digital twin lite': calibrate the tracked skeleton with the player's own body.

Without hardware we can't build a true digital twin, but known segment lengths
let us (1) put the skeleton on the player's real metric scale, and (2) check how
consistently the camera tracked each bone - a direct, honest quality metric.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C

PROFILES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "twin_profiles.json")
TWIN_FIELDS = ["height_cm", "upper_arm_cm", "forearm_cm", "thigh_cm", "shank_cm"]

# (segment name, (landmark pairs for left and right side))
BONES = {
    "upper_arm": [(11, 13), (12, 14)],
    "forearm": [(13, 15), (14, 16)],
    "thigh": [(23, 25), (24, 26)],
    "shank": [(25, 27), (26, 28)],
}


def load_profiles() -> dict:
    try:
        with open(PROFILES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def bone_lengths(P: np.ndarray, frames=None) -> dict:
    """Per-frame lengths of each bone (both sides pooled): dict name -> (N,2)."""
    out = {}
    for name, pairs in BONES.items():
        L = [np.linalg.norm(P[:, a] - P[:, b], axis=-1) for a, b in pairs]
        arr = np.stack(L, axis=1)
        out[name] = arr if frames is None else arr[frames]
    return out


def calibrate(world: np.ndarray, valid_frames: np.ndarray, twin: dict | None) -> dict:
    """Scale factor from MediaPipe world metres to the player's metres.

    twin: {"height_cm": .., "upper_arm_cm": .., ...} (any subset; None = no twin)
    """
    lens = bone_lengths(world)
    measured = {k: float(np.nanmedian(v[valid_frames])) for k, v in lens.items()}
    est_height = float(np.nanmedian([measured["thigh"] / C.ANTHRO["thigh"],
                                     measured["shank"] / C.ANTHRO["shank"]]))
    info = {"used": False, "scale": 1.0, "height_m": est_height,
            "measured_cm": {k: round(v * 100, 1) for k, v in measured.items()},
            "expected_cm": {}, "source": "MediaPipe default body scale"}
    if not twin:
        return info
    h = twin.get("height_cm")
    expected = {}
    for k in BONES:
        val = twin.get(f"{k}_cm")
        if val:
            expected[k] = float(val) / 100.0
        elif h:
            expected[k] = float(h) / 100.0 * C.ANTHRO[k]
    if not expected:
        return info
    ratios = [expected[k] / measured[k] for k in expected if measured.get(k, 0) > 1e-3]
    if not ratios:
        return info
    s = float(np.median(ratios))
    info.update(used=True, scale=s,
                height_m=float(h) / 100.0 if h else est_height * s,
                expected_cm={k: round(v * 100, 1) for k, v in expected.items()},
                source="your measurements" if any(twin.get(f"{k}_cm") for k in BONES)
                else "your height + standard body proportions")
    return info


def tracking_consistency(P: np.ndarray, frames: np.ndarray) -> dict:
    """Coefficient of variation (%) of each bone length over the swing.

    Real bones don't change length, so any variation is tracking / depth error.
    """
    lens = bone_lengths(P, frames)
    cv = {}
    for k, arr in lens.items():
        vals = []
        for side in range(arr.shape[1]):
            x = arr[:, side]
            x = x[np.isfinite(x)]
            if len(x) > 3 and np.mean(x) > 1e-6:
                vals.append(np.std(x) / np.mean(x) * 100)
        cv[k] = float(np.mean(vals)) if vals else float("nan")
    overall = float(np.nanmean(list(cv.values())))
    return {"bone_cv_pct": {k: round(v, 1) for k, v in cv.items()}, "overall_cv_pct": round(overall, 1)}
