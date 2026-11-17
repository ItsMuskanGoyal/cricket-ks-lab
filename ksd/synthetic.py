"""A synthetic, physically-shaped batter for testing the pipeline without a camera.

We script each segment's rotation with a smooth S-curve whose peak speed time
we choose, project the 3D body through a pinhole camera placed front-on, add
noise, and check the analysis recovers the order we put in.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config as C
from .pose import PoseTrack


def _sig(t, t0, tau):
    return 1.0 / (1.0 + np.exp(-(t - t0) / tau))


def make_swing(fps=60.0, duration=2.4, order=("pelvis", "torso", "arm", "hands"),
               spacing=0.035, noise_px=1.2, noise_z=0.015, seed=0, right_handed=True,
               width=720, height=1280, collapse_knee=False):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fps)) / fps
    t_imp = 1.6
    # peak-time for each segment, in the requested order, ending ~10 ms before impact
    base = t_imp - 0.01 - spacing * (len(order) - 1)
    tpk = {seg: base + i * spacing for i, seg in enumerate(order)}

    # rotations ("opening" angles, deg) with amplitude rising up the chain
    alpha_p = 70 * _sig(t, tpk["pelvis"], 0.05) - 8 * _sig(t, 1.05, 0.08)
    alpha_t = 95 * _sig(t, tpk["torso"], 0.045) - 22 * _sig(t, 1.10, 0.08)
    # lead upper arm: lifted back in the backlift, swung down/through in the downswing
    beta = 25 + 110 * _sig(t, 0.95, 0.07) - 220 * _sig(t, tpk["arm"], 0.04)
    hinge = 80 * _sig(t, 1.0, 0.07) - 80 * _sig(t, tpk["hands"], 0.02)
    # stride: front foot lifts and moves towards the bowler before the downswing
    stride = 0.35 * _sig(t, 1.18, 0.05)
    lift = 0.08 * np.exp(-((t - 1.18) / 0.06) ** 2)

    N = len(t)
    W = np.zeros((N, 33, 3))
    sgn = 1.0 if right_handed else -1.0    # mirror for left-handers
    for k in range(N):
        ap, at = np.radians(alpha_p[k]), np.radians(alpha_t[k])
        hip_dir = np.array([sgn * np.sin(ap), 0, -np.cos(ap)])      # trail->lead hip
        sh_dir = np.array([sgn * np.sin(at), 0, -np.cos(at)])
        lead_hip, trail_hip = 0.1 * hip_dir, -0.1 * hip_dir
        sh_c = np.array([0.0, -0.50, 0.0])
        lead_sh, trail_sh = sh_c + 0.19 * sh_dir, sh_c - 0.19 * sh_dir
        # arm swings in a plane containing the vertical and the shoulder line
        b = np.radians(beta[k])
        down = np.array([0, 1.0, 0])
        swing = np.array([0.15 * sgn, 0, 1.0]); swing /= np.linalg.norm(swing)
        ua = np.cos(b) * down + np.sin(b) * swing
        lead_el = lead_sh + 0.30 * ua
        fa_ang = b + np.radians(15)
        fa = np.cos(fa_ang) * down + np.sin(fa_ang) * swing
        lead_wr = lead_el + 0.26 * fa
        h_ang = fa_ang + np.radians(hinge[k])
        hv = np.cos(h_ang) * down + np.sin(h_ang) * swing
        lead_idx = lead_wr + 0.08 * hv + np.array([0.01, 0, 0])
        lead_pk = lead_wr + 0.07 * hv - np.array([0.01, 0, 0])
        trail_wr = lead_wr + 0.06 * hv
        trail_el = (trail_sh + trail_wr) / 2 + np.array([0, 0.05, 0])
        trail_idx = trail_wr + 0.08 * hv
        # legs
        knee_bend = 0.05 + (0.06 * _sig(t[k], 1.45, 0.03) if collapse_knee else -0.03 * _sig(t[k], 1.45, 0.03))
        lead_an = np.array([0.05 * sgn, 0.92 - lift[k], -0.25 - stride[k]])
        trail_an = np.array([-0.05 * sgn, 0.92, 0.25])
        lead_kn = (lead_hip + lead_an) / 2 + np.array([0, 0, -knee_bend * 2])
        trail_kn = (trail_hip + trail_an) / 2 + np.array([0, 0, -0.05])
        nose = sh_c + np.array([0, -0.24, -0.06])
        L, R = (dict(sh=lead_sh, el=lead_el, wr=lead_wr, idx=lead_idx, pk=lead_pk, hip=lead_hip,
                     kn=lead_kn, an=lead_an),
                dict(sh=trail_sh, el=trail_el, wr=trail_wr, idx=trail_idx, pk=trail_idx, hip=trail_hip,
                     kn=trail_kn, an=trail_an))
        if not right_handed:
            L, R = R, L
        W[k, C.NOSE] = nose
        for i in range(1, 11):
            W[k, i] = nose + np.array([0, -0.02, 0.03])
        W[k, C.L_EAR] = nose + np.array([0.07, 0, 0.08]); W[k, C.R_EAR] = nose + np.array([-0.07, 0, 0.08])
        for side, d in ((("L"), L), (("R"), R)):
            o = 0 if side == "L" else 1
            W[k, 11 + o], W[k, 13 + o], W[k, 15 + o] = d["sh"], d["el"], d["wr"]
            W[k, 17 + o], W[k, 19 + o], W[k, 21 + o] = d["pk"], d["idx"], d["idx"]
            W[k, 23 + o], W[k, 25 + o], W[k, 27 + o] = d["hip"], d["kn"], d["an"]
            W[k, 29 + o] = d["an"] + np.array([0, 0.02, 0.06])
            W[k, 31 + o] = d["an"] + np.array([0, 0.04, -0.15])

    # project through a front-on pinhole camera 8 m away
    D, f = 8.0, 1.6 * height
    cx, cy = width / 2, height * 0.45
    hip_world = np.array([0.0, 0.0, 0.0])
    Z = W[..., 2] + D
    img = np.stack([cx + f * (W[..., 0] + hip_world[0]) / Z, cy + f * W[..., 1] / Z], axis=-1)
    img += rng.normal(0, noise_px, img.shape)
    world = W + rng.normal(0, noise_z, W.shape) * np.array([0.3, 0.3, 1.0])
    vis = np.full((N, 33), 0.95)
    track = PoseTrack(img=img, world=world, vis=vis, present=np.ones(N, bool), fps=fps,
                      width=width, height=height)
    truth = {"t_impact": t_imp, "peak_times": tpk, "order": list(order)}
    return track, truth


def write_video(track: PoseTrack, path: str):
    """Draw the synthetic batter as a stick figure so the renderer has a real video."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), track.fps, (track.width, track.height))
    for k in range(track.n):
        fr = np.full((track.height, track.width, 3), (60, 110, 70), np.uint8)
        cv2.rectangle(fr, (0, int(track.height * 0.62)), (track.width, track.height), (90, 150, 190), -1)
        for a, b in C.BODY_CONNECTIONS:
            pa, pb = track.img[k, a].astype(int), track.img[k, b].astype(int)
            cv2.line(fr, tuple(pa), tuple(pb), (235, 235, 235), 10, cv2.LINE_AA)
        cv2.circle(fr, tuple(track.img[k, 0].astype(int)), 34, (235, 235, 235), -1, cv2.LINE_AA)
        vw.write(fr)
    vw.release()
    track.video_path = path
    return path
