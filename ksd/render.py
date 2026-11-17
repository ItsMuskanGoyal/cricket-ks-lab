"""Visual outputs: skeleton replay video, kinematic-sequence chart, evidence frames."""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .kinematics import Kinematics  # noqa: E402
from .pose import PoseTrack, iter_frames  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (245, 245, 245)
PHASE_NAMES = {"setup": "Setup", "top": "Top of backlift", "plant": "Front-foot plant",
               "impact": "Impact (est.)", "follow": "Follow-through"}


# ------------------------------------------------------------------ helpers
def phase_at(K: Kinematics, k: int) -> str:
    ph = K.phases
    if k <= ph["setup"][1]:
        return "SETUP"
    if k < ph["top"]:
        return "BACKLIFT"
    if k < ph["impact"]:
        return "DOWNSWING"
    if k <= ph["impact"] + max(1, int(0.02 * K.fs)):
        return "IMPACT (EST.)"
    return "FOLLOW-THROUGH"


def _text(img, txt, org, scale, color=WHITE, thick=1, shadow=True):
    if shadow:
        cv2.putText(img, txt, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def _pt(K, k, i):
    return tuple(int(round(v)) for v in K.I[k, i])


def _scale(img):
    return min(img.shape[0], img.shape[1]) / 720.0


def draw_skeleton(img, K: Kinematics, k: int):
    s = _scale(img)
    lw = max(2, int(round(3 * s)))
    ov = img.copy()
    for a, b in C.BODY_CONNECTIONS:
        cv2.line(ov, _pt(K, k, a), _pt(K, k, b), (235, 235, 235), lw, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
    ld, tr = K.side["lead"], K.side["trail"]
    thick = max(3, int(round(6 * s)))
    cv2.line(img, _pt(K, k, ld["hip"]), _pt(K, k, tr["hip"]), C.SEGMENT_BGR["pelvis"], thick, cv2.LINE_AA)
    cv2.line(img, _pt(K, k, ld["shoulder"]), _pt(K, k, tr["shoulder"]), C.SEGMENT_BGR["torso"], thick, cv2.LINE_AA)
    cv2.line(img, _pt(K, k, ld["shoulder"]), _pt(K, k, ld["elbow"]), C.SEGMENT_BGR["arm"], thick, cv2.LINE_AA)
    # hands: wrist -> knuckles direction, drawn extended so it reads like a bat handle
    w = K.I[k, [C.L_WRIST, C.R_WRIST]].mean(0)
    kn = K.I[k, [C.L_INDEX, C.R_INDEX, C.L_PINKY, C.R_PINKY]].mean(0)
    d = kn - w
    n = np.linalg.norm(d)
    if n > 1e-3:
        end = w + d / n * max(n * 3.0, 40 * s)
        cv2.line(img, tuple(int(v) for v in w), tuple(int(v) for v in end), C.SEGMENT_BGR["hands"], thick, cv2.LINE_AA)
    for i in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        cv2.circle(img, _pt(K, k, i), max(3, int(4 * s)), (255, 255, 255), -1, cv2.LINE_AA)


def draw_hud(img, K: Kinematics, k: int, replay_label: str = ""):
    h, w = img.shape[:2]
    s = _scale(img)
    fs1, fs2 = 0.55 * s, 0.48 * s
    th = max(1, int(round(1.2 * s)))
    title = "KINEMATIC SEQUENCE LAB"
    t_ms = (k - K.phases["impact"]) / K.fs * 1000
    label = f"{phase_at(K, k)}  {t_ms:+.0f} ms"
    (tw0, _), _ = cv2.getTextSize(title, FONT, fs1, th + 1)
    (tw, _), _ = cv2.getTextSize(label, FONT, fs1, th + 1)
    narrow = tw0 + tw + int(40 * s) > w
    band_h = int((66 if narrow else 46) * s)
    band = img.copy()
    cv2.rectangle(band, (0, 0), (w, band_h), (20, 20, 20), -1)
    cv2.addWeighted(band, 0.55, img, 0.45, 0, img)
    _text(img, title, (int(12 * s), int(20 * s)), fs1, WHITE, th + 1)
    _text(img, "single-camera estimate", (int(12 * s), int(38 * s)), fs2 * 0.9, (200, 200, 200), th)
    lx = int(12 * s) if narrow else w - tw - int(12 * s)
    ly = int(58 * s) if narrow else int(20 * s)
    _text(img, label, (lx, ly), fs1, (120, 230, 255), th + 1)
    if replay_label:
        (rw, _), _ = cv2.getTextSize(replay_label, FONT, fs2, th)
        _text(img, replay_label, (w - rw - int(12 * s), int(38 * s)), fs2, (180, 255, 180), th)
    # speed bars
    pad, bh, gap = int(12 * s), int(14 * s), int(8 * s)
    bw = int(min(w * 0.42, 260 * s))
    y0 = h - pad - 4 * (bh + gap) - int(8 * s)
    panel = img.copy()
    cv2.rectangle(panel, (0, y0 - int(10 * s)), (pad + bw + int(150 * s), h), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.55, img, 0.45, 0, img)
    for j, seg in enumerate(C.SEGMENTS):
        y = y0 + j * (bh + gap)
        pk = K.seq["peaks"][seg]
        frac = float(np.clip(K.omega[seg][k] / max(pk["deg_s"], 1e-6), 0, 1))
        cv2.rectangle(img, (pad, y), (pad + bw, y + bh), (70, 70, 70), -1)
        cv2.rectangle(img, (pad, y), (pad + int(bw * frac), y + bh), C.SEGMENT_BGR[seg], -1)
        txt = C.SHORT_LABELS[seg]
        if abs(k - pk["frame"]) <= max(1.0, 0.012 * K.fs):
            txt += "  PEAK"
        _text(img, txt, (pad + bw + int(8 * s), y + bh - int(2 * s)), fs2, WHITE, th)


# ------------------------------------------------------------------ video
def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def render_video(track: PoseTrack, K: Kinematics, out_dir: str, progress=None) -> tuple[str, dict]:
    """Write the annotated replay (+ slow-motion replay of the swing) as H.264 mp4.

    Returns (video_path, {phase_name: annotated_frame}) for the evidence strip.
    """
    w, h = track.width, track.height
    out_fps = min(track.fps, 60.0)
    raw = os.path.join(out_dir, "replay_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    key_idx = {"setup": K.phases["setup"][1], "top": K.phases["top"], "impact": K.phases["impact"],
               "follow": K.phases["follow_end"]}
    if K.phases["plant"] is not None:
        key_idx["plant"] = K.phases["plant"]
    a = max(0, K.phases["top"] - int(0.25 * K.fs))
    b = K.phases["follow_end"]
    # slow the swing down so it lasts ~2.5 s on screen
    rep = int(np.clip(round(2.5 * out_fps / max(b - a + 1, 1)), 1, 8))
    keep, snaps = [], {}
    for k, frame in iter_frames(track.video_path, track.frame_step, track.n, (w, h)):
        clean = frame.copy() if a <= k <= b and rep > 1 else None
        draw_skeleton(frame, K, k)
        draw_hud(frame, K, k)
        vw.write(frame)
        for name, idx in key_idx.items():
            if idx == k:
                snaps[name] = frame.copy()
        if clean is not None:
            keep.append((k, clean))
        if progress is not None and k % 15 == 0:
            progress(0.7 + 0.2 * k / max(track.n, 1), desc="Drawing the skeleton replay")
    if keep:
        card = np.zeros((h, w, 3), np.uint8)
        s = min(h, w) / 720.0
        msg = f"SLOW-MOTION REPLAY  x{rep}"
        (tw, _), _ = cv2.getTextSize(msg, FONT, 0.9 * s, 2)
        _text(card, msg, ((w - tw) // 2, h // 2), 0.9 * s, WHITE, 2)
        for _ in range(int(out_fps * 0.8)):
            vw.write(card)
        for k, fr in keep:
            draw_skeleton(fr, K, k)
            draw_hud(fr, K, k, replay_label=f"slow-mo x{rep}")
            for _ in range(rep):
                vw.write(fr)
    vw.release()

    final = os.path.join(out_dir, "replay.mp4")
    ff = find_ffmpeg()
    if ff:
        cmd = [ff, "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", final]
        try:
            subprocess.run(cmd, check=True, timeout=240)
            os.remove(raw)
            return final, snaps
        except Exception:
            pass
    os.replace(raw, final)
    return final, snaps


# ------------------------------------------------------------------ evidence
def evidence_strip(K: Kinematics, snaps: dict, out_dir: str) -> str:
    order = [p for p in ("setup", "top", "plant", "impact", "follow") if p in snaps]
    tiles = []
    H = 420
    for name in order:
        img = snaps[name]
        k = {"setup": K.phases["setup"][1], "top": K.phases["top"], "plant": K.phases["plant"],
             "impact": K.phases["impact"], "follow": K.phases["follow_end"]}[name]
        pts = K.I[k][np.isfinite(K.I[k]).all(1)]
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hh = max(y1 - y0, x1 - x0) * 1.45
        ww = max(hh * 0.75, (x1 - x0) * 1.3)
        X0, X1 = int(max(0, cx - ww / 2)), int(min(img.shape[1], cx + ww / 2))
        Y0, Y1 = int(max(0, cy - hh / 2)), int(min(img.shape[0], cy + hh / 2))
        crop = img[Y0:Y1, X0:X1] if (Y1 - Y0 > 20 and X1 - X0 > 20) else img
        crop = cv2.resize(crop, (int(crop.shape[1] * H / crop.shape[0]), H), interpolation=cv2.INTER_AREA)
        bar = np.full((44, crop.shape[1], 3), 24, np.uint8)
        t_ms = (k - K.phases["impact"]) / K.fs * 1000
        _text(bar, PHASE_NAMES[name], (10, 20), 0.55, WHITE, 1, shadow=False)
        _text(bar, f"{t_ms:+.0f} ms from impact", (10, 38), 0.45, (190, 190, 190), 1, shadow=False)
        tiles.append(np.vstack([bar, crop]))
    gap = np.full((H + 44, 8, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])
    path = os.path.join(out_dir, "evidence_frames.png")
    cv2.imwrite(path, row)
    return path


# ------------------------------------------------------------------ chart
def sequence_chart(K: Kinematics, out_dir: str) -> str:
    seq, ph, fs = K.seq, K.phases, K.fs
    a = max(0, ph["top"] - int(0.35 * fs))
    b = min(K.n - 1, ph["follow_end"])
    idx = np.arange(a, b + 1)
    t = (idx - ph["impact"]) / fs * 1000

    fig = Figure(figsize=(11, 6.4), dpi=150, facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    def phase_bands(axis):
        t_top = (ph["top"] - ph["impact"]) / fs * 1000
        axis.axvspan(t[0], t_top, color="#f1f3f5", zorder=0)
        axis.axvspan(t_top, 0, color="#fff4e0", zorder=0)
        axis.axvline(0, color="#444", lw=1.2, ls="--", zorder=1)
        axis.axvline(t_top, color="#999", lw=1, ls=":", zorder=1)
        if ph["plant"] is not None:
            axis.axvline((ph["plant"] - ph["impact"]) / fs * 1000, color="#2e7d32", lw=1, ls=":", zorder=1)
        return t_top

    t_top = phase_bands(ax)
    ymax = max(seq["peaks"][s]["deg_s"] for s in C.SEGMENTS)
    rank_of = {seg: i + 1 for i, seg in enumerate(seq["order"])}
    for seg in C.SEGMENTS:
        ax.plot(t, K.omega[seg][idx], color=C.SEGMENT_HEX[seg], lw=2.4, label=C.SEGMENT_LABELS[seg], zorder=3)
        pk = seq["peaks"][seg]
        ax.scatter([pk["t_ms"]], [pk["deg_s"]], s=70, color=C.SEGMENT_HEX[seg], edgecolor="white",
                   linewidth=1.5, zorder=5)
        ax.annotate(f"{rank_of[seg]}", (pk["t_ms"], pk["deg_s"]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=C.SEGMENT_HEX[seg], zorder=6)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("Rotation speed (°/s)", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.075, 0.955),
               frameon=False, fontsize=9.5, ncol=4)
    for txt, x in (("backlift", (t[0] + t_top) / 2), ("downswing", t_top / 2)):
        ax.text(x, ymax * 1.12, txt, ha="center", va="center", fontsize=8.5, color="#777")
    ax.text(0, ymax * 1.12, " impact (est.)", ha="left", va="center", fontsize=8.5, color="#444")
    verdict = ("Ideal order: pelvis → torso → arm → hands" if seq["ideal_order"] else
               "Firing order: " + " → ".join(C.SHORT_LABELS[s] for s in seq["order"]))
    fig.text(0.08, 0.975, f"Kinematic sequence (estimated from a single 2D video)   ·   {verdict}",
             ha="left", va="top", fontsize=11.5, fontweight="bold", color="#222")

    phase_bands(ax2)
    ax2.plot(t, K.x_factor[idx], color="#6a4c93", lw=2)
    ax2.axhline(0, color="#bbb", lw=0.8)
    ax2.set_ylabel("Hip–shoulder\nseparation (°)", fontsize=9)
    ax2.set_xlabel("Time relative to estimated impact (ms)", fontsize=10)
    for axis in (ax, ax2):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e6e6e6", lw=0.8)
        axis.tick_params(labelsize=9)
        axis.set_xlim(t[0], t[-1])
    fig.text(0.99, 0.005, f"{fs:.0f} fps effective · {1000 / fs:.1f} ms per frame · peaks sub-frame interpolated",
             ha="right", fontsize=7.5, color="#888")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.1)
    path = os.path.join(out_dir, "kinematic_sequence.png")
    fig.savefig(path, facecolor="white")
    return path
