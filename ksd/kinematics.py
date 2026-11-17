"""From raw pose landmarks to segment rotation speeds, swing phases and the
kinematic sequence.

Coordinate approach ("hybrid 2.5D")
-----------------------------------
* x, y (across and up/down the image) come from MediaPipe's precise 2D image
  landmarks, converted to metres with a per-clip pixels-per-metre factor.
* z (depth, towards/away from the camera) comes from MediaPipe's learned 3D
  world landmarks, because a single camera cannot see depth directly.
Front-on, a side-on batter's hip and shoulder lines point mostly along the
depth axis, so their rotation genuinely needs that depth estimate - this is the
main reason every rotation number here is labelled an estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import signals as S
from .pose import AnalysisError, PoseTrack
from .twin import calibrate, tracking_consistency

KEY_LANDMARKS = [C.NOSE, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


@dataclass
class Kinematics:
    fs: float                      # true capture rate (fps x slow-motion factor)
    t: np.ndarray                  # (N,) seconds of real time
    P: np.ndarray                  # (N,33,3) hybrid coordinates, metres, hip-centred
    I: np.ndarray                  # (N,33,2) filtered image pixels
    vis: np.ndarray                # (N,33)
    ppm: float                     # image pixels per real metre
    lead_is_left: bool
    hand_source: str               # "auto-detected" / "selected"
    side: dict
    yaw: dict                      # pelvis/torso heading (deg)
    omega: dict                    # angular speeds (deg/s) for each segment
    hand_speed: np.ndarray         # m/s, relative to the pelvis
    foot_speed: np.ndarray         # m/s, lead foot
    x_factor: np.ndarray           # deg, torso minus pelvis rotation, relative to stance
    twin: dict
    consistency: dict
    phases: dict = field(default_factory=dict)
    seq: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.t)

    @property
    def handed(self):
        return "Right-handed" if self.lead_is_left else "Left-handed"


# ------------------------------------------------------------------ build
def build(track: PoseTrack, handed: str = "auto", slowmo: float = 1.0, twin: dict | None = None) -> Kinematics:
    N = track.n
    fs = track.fps * max(1.0, float(slowmo))
    warnings = list(track.notes)

    valid = track.present[:, None] & (track.vis >= C.VIS_MIN) & np.isfinite(track.img[..., 0])
    for k in KEY_LANDMARKS:
        if valid[:, k].mean() < 0.3:
            raise AnalysisError("Key body points (shoulders, hips, arms or legs) were hidden for most "
                                "of the clip. Use a clip where the whole batter is visible, front-on.")
    low_vis = [k for k in KEY_LANDMARKS if valid[:, k].mean() < 0.7]
    if low_vis:
        warnings.append("Some joints were hidden in parts of the clip (often the hands behind the bat); "
                        "those moments were interpolated.")

    I = S.lowpass(S.fill_gaps(track.img, valid), fs)
    W = S.lowpass(S.fill_gaps(track.world, valid), fs)
    for arr in (I, W):  # landmarks never seen at all (e.g. feet off-frame): copy nearest joint
        for a, b in ((31, 27), (32, 28), (29, 27), (30, 28), (19, 15), (20, 16), (17, 15), (18, 16)):
            if not np.isfinite(arr[:, a]).all():
                arr[:, a] = arr[:, b]

    # ---- pixels per (MediaPipe) metre: robust, least-foreshortened moments
    pairs = [(11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
    ratios = np.stack([np.linalg.norm(I[:, a] - I[:, b], axis=-1) /
                       np.maximum(np.linalg.norm(W[:, a] - W[:, b], axis=-1), 1e-3)
                       for a, b in pairs], axis=1)
    per_frame = np.nanmedian(ratios, axis=1)
    ppm_world = float(np.nanpercentile(per_frame[track.present], 90))

    hip_px = (I[:, 23] + I[:, 24]) / 2
    hip_w = (W[:, 23] + W[:, 24]) / 2
    H = np.empty((N, 33, 3))
    H[..., 0] = (I[..., 0] - hip_px[:, None, 0]) / ppm_world
    H[..., 1] = (I[..., 1] - hip_px[:, None, 1]) / ppm_world
    H[..., 2] = W[..., 2] - hip_w[:, None, 2]

    # ---- digital twin lite: metric scale + tracking quality
    tw = calibrate(W, track.present, twin)
    P = H * tw["scale"]
    ppm = ppm_world / tw["scale"]

    # ---- handedness: the lead shoulder/hip is the one closer to the camera
    if handed in ("Right-handed", "Left-handed"):
        lead_is_left, hand_source = handed == "Right-handed", "selected"
    else:
        early = slice(0, max(5, int(0.4 * N)))
        dz = np.nanmedian((W[early, 11, 2] + W[early, 23, 2]) - (W[early, 12, 2] + W[early, 24, 2]))
        lead_is_left, hand_source = bool(dz < 0), "auto-detected"
    side = C.side_indices(lead_is_left)
    ld, tr = side["lead"], side["trail"]

    # ---- segment rotation (depth refined with the rigid-segment constraint)
    pelvis_vec = rigid_depth(P[:, ld["hip"]] - P[:, tr["hip"]], track.present)
    torso_vec = rigid_depth(P[:, ld["shoulder"]] - P[:, tr["shoulder"]], track.present)
    yaw = {"pelvis": S.yaw_deg(pelvis_vec), "torso": S.yaw_deg(torso_vec)}
    hands_c = P[:, [C.L_WRIST, C.R_WRIST, C.L_INDEX, C.R_INDEX]].mean(axis=1)
    # Bat proxy: the hands are rigidly gripped to the handle, so the direction
    # wrist -> knuckles (both hands averaged) turns at the bat's angular speed.
    knuckles = (P[:, [C.L_INDEX, C.R_INDEX, C.L_PINKY, C.R_PINKY]].mean(axis=1))
    wrists = P[:, [C.L_WRIST, C.R_WRIST]].mean(axis=1)
    omega = {
        "pelvis": np.abs(S.rate(yaw["pelvis"], fs)),
        "torso": np.abs(S.rate(yaw["torso"], fs)),
        "arm": S.angular_speed_3d(rigid_depth(P[:, ld["elbow"]] - P[:, ld["shoulder"]], track.present), fs),
        "hands": S.angular_speed_3d(knuckles - wrists, fs),
    }
    # Segment-specific smoothing of the speed curves: the big, heavy proximal
    # segments move slowly (low frequency); the hands are fast.
    omega = {k: np.clip(S.lowpass(v, fs, min(C.SEG_CUTOFF_HZ[k], 0.4 * fs)), 0, None)
             for k, v in omega.items()}
    hand_speed = S.norm(S.rate(hands_c, fs))
    foot_img = (I[:, ld["ankle"]] + I[:, ld["foot"]]) / 2 / ppm
    foot_z = (P[:, ld["ankle"], 2] + P[:, ld["foot"], 2]) / 2
    foot_speed = np.sqrt(S.norm(S.rate(foot_img, fs)) ** 2 + S.rate(foot_z, fs) ** 2)

    K = Kinematics(fs=fs, t=np.arange(N) / fs, P=P, I=I, vis=track.vis, ppm=ppm,
                   lead_is_left=lead_is_left, hand_source=hand_source, side=side, yaw=yaw,
                   omega=omega, hand_speed=hand_speed, foot_speed=foot_speed,
                   x_factor=np.zeros(N), twin=tw, consistency={}, warnings=warnings)
    K.phases = detect_phases(K)
    s0, s1 = K.phases["setup"]
    sep = yaw["torso"] - yaw["pelvis"]
    K.x_factor = sep - np.median(sep[s0:s1 + 1])
    a, b = K.phases["top"], K.phases["follow_end"]
    K.consistency = tracking_consistency(P, np.arange(max(0, a - 3), b + 1))
    K.seq = analyse_sequence(K)
    if slowmo > 1 and track.fps >= 100:
        warnings.append(f"The file already runs at {track.fps:.0f} fps, so it is probably real high-speed footage; "
                        "if so, choose 'Normal speed' - the slow-motion option multiplies the frame rate.")
    if fs < 50:
        warnings.append(f"Recorded at about {fs:.0f} frames per second: each frame is {1000 / fs:.0f} ms, "
                        "which is coarse for timing a swing. Slow-motion (120-240 fps) gives far better timing.")
    return K


def rigid_depth(vec: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Refine the depth (z) of a segment using the fact that bones keep their length.

    The image gives the segment's x/y precisely; if its true length L is known,
    depth magnitude follows from |z| = sqrt(L^2 - x^2 - y^2). The learned 3D
    estimate only supplies the sign (towards or away from the camera), plus the
    magnitude when the segment faces the camera square-on (where the
    constraint becomes ill-conditioned). This is the 'digital twin' idea in
    miniature: a known body constrains what one camera cannot see.
    """
    xy = vec[..., :2]
    l2d = S.norm(xy)
    l3d = S.norm(vec)
    ok = present & np.isfinite(l3d)
    L = max(float(np.nanpercentile(l2d[ok], 95)), float(np.nanmedian(l3d[ok])))
    zc = np.sqrt(np.clip(L ** 2 - l2d ** 2, 0, None))
    w = np.clip(l2d / L, 0, 1) ** 4          # trust the learned depth only near square-on
    out = vec.copy()
    out[..., 2] = np.sign(vec[..., 2]) * ((1 - w) * zc + w * np.abs(vec[..., 2]))
    return out


# ------------------------------------------------------------------ phases
def detect_phases(K: Kinematics) -> dict:
    """Find setup, top of backlift, front-foot plant, impact and follow-through
    from the hands' speed (relative to the pelvis, so camera pans don't matter)."""
    s, fs, N = K.hand_speed, K.fs, K.n
    lo, hi = 2, N - 3
    if hi - lo < 5:
        raise AnalysisError("The clip is too short to find a swing.")
    p = lo + int(np.argmax(s[lo:hi + 1]))
    peak = float(s[p])
    if peak < 1.0:
        raise AnalysisError("No clear bat swing was found (the hands barely moved). "
                            "Make sure the clip contains a full shot.")

    # top of backlift = the slow point just before the downswing
    i = p
    while i > 0 and s[i] > 0.25 * peak:
        i -= 1
    while i > 0 and s[i - 1] < s[i]:
        i -= 1
    top = i

    # setup = the quiet period before the backlift starts
    lo_bl = max(0, top - int(1.2 * fs))
    if top - lo_bl >= 2:
        j = lo_bl + int(np.argmax(s[lo_bl:top]))
        while j > 0 and s[j] > 0.15 * peak:
            j -= 1
        while j > 0 and s[j - 1] < s[j]:
            j -= 1
        setup_end = j
    else:
        setup_end = top
    setup_start = max(0, setup_end - int(0.4 * fs))
    if setup_end <= setup_start:
        setup_end = min(top, setup_start + 1)

    # follow-through
    k = p
    while k < N - 1 and s[k] > 0.2 * peak:
        k += 1
    follow_end = min(N - 1, k + int(0.25 * fs))

    # front-foot plant: lead foot moves, then settles before impact
    plant = None
    fsp = K.foot_speed
    a, b = setup_end, p
    if b - a >= 3:
        m = a + int(np.argmax(fsp[a:b + 1]))
        fmax = float(fsp[m])
        if fmax > 0.5:
            for q in range(m + 1, b):
                if fsp[q] < 0.3 * fmax and fsp[q + 1] < 0.3 * fmax:
                    plant = q
                    break
    return {"setup": (int(setup_start), int(setup_end)), "top": int(top), "plant": plant,
            "impact": int(p), "follow_end": int(follow_end), "peak_hand_speed": peak}


# ------------------------------------------------------------------ sequence
def analyse_sequence(K: Kinematics) -> dict:
    fs, N = K.fs, K.n
    top, p = K.phases["top"], K.phases["impact"]
    a = max(0, top - int(0.08 * fs))
    b = min(N - 1, p + int(0.15 * fs))
    frame_ms = 1000.0 / fs
    peaks = {}
    for seg in C.SEGMENTS:
        w = K.omega[seg]
        bs = min(b, p + int(round(C.SEG_PEAK_AFTER_IMPACT_S[seg] * fs)))
        i = a + int(np.argmax(w[a:bs + 1]))
        fi, val = S.refine_peak(w, i)
        peaks[seg] = {"frame": fi, "t_ms": (fi - p) * frame_ms, "deg_s": val}

    order = sorted(C.SEGMENTS, key=lambda s: peaks[s]["frame"])
    links = []
    for s1, s2 in zip(C.SEGMENTS[:-1], C.SEGMENTS[1:]):
        dt = peaks[s2]["t_ms"] - peaks[s1]["t_ms"]
        if abs(dt) < 0.5 * frame_ms:
            status = "simultaneous"
        elif dt > 0:
            status = "in order"
        else:
            status = "reversed"
        gain = peaks[s2]["deg_s"] / max(peaks[s1]["deg_s"], 1e-6)
        brake = S.interp_at(K.omega[s1], peaks[s2]["frame"]) / max(peaks[s1]["deg_s"], 1e-6)
        links.append({"from": s1, "to": s2, "dt_ms": dt, "status": status, "gain": gain,
                      "brake_ratio": brake, "braked": brake <= C.T["braking_ratio"]})

    order_pts = sum(1.0 if l["status"] == "in order" else 0.5 if l["status"] == "simultaneous" else 0.0
                    for l in links) / len(links)
    gain_pts = sum(1.0 for l in links if l["gain"] > 1.0) / len(links)
    brake_pts = sum(1.0 for l in links if l["braked"]) / len(links)
    score = round(40 * order_pts + 30 * gain_pts + 30 * brake_pts)
    ideal = order == C.SEGMENTS and all(l["status"] == "in order" for l in links)
    return {"peaks": peaks, "order": order, "links": links, "score": int(score),
            "ideal_order": ideal, "window": (a, b), "frame_ms": frame_ms,
            "parts": {"order": order_pts, "gain": gain_pts, "brake": brake_pts}}
