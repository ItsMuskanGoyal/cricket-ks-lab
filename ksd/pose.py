"""Pose extraction with Google MediaPipe Pose Landmarker (Tasks API).

Produces a PoseTrack: per-frame 2D image landmarks, MediaPipe's metric 3D
"world" landmarks, and visibility, for ONE tracked person (the batter).
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C

MODEL_URLS = {
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class AnalysisError(Exception):
    """A problem we can explain to the user in plain words."""


def ensure_model(variant: str = None) -> str:
    """Return a local path to the pose model, downloading it once if needed."""
    variant = variant or os.environ.get("POSE_MODEL", "full")
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"pose_landmarker_{variant}.task")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        os.replace(tmp, path)
    except Exception as e:  # pragma: no cover - network dependent
        raise AnalysisError(
            "Could not download the MediaPipe pose model. Check internet access, "
            f"or place pose_landmarker_{variant}.task in the 'models' folder. ({e})")
    return path


@dataclass
class PoseTrack:
    img: np.ndarray        # (N,33,2) pixel coordinates (x, y) in the resized frame
    world: np.ndarray      # (N,33,3) MediaPipe world coords, metres, hip-centred
    vis: np.ndarray        # (N,33) visibility 0..1
    present: np.ndarray    # (N,) person found in frame
    fps: float             # container frame rate
    width: int
    height: int
    video_path: str = ""
    frame_step: int = 1    # every n-th source frame was used
    scale: float = 1.0     # resize factor applied to source frames
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.img.shape[0]


def resized_dims(w: int, h: int) -> tuple[int, int, float]:
    s = min(1.0, C.MAX_SIDE_PX / max(w, h))
    nw, nh = int(round(w * s)) // 2 * 2, int(round(h * s)) // 2 * 2
    return nw, nh, s


def iter_frames(video_path: str, frame_step: int, max_frames: int, size: tuple[int, int]):
    """Yield (index, BGR frame) of the resized, sub-sampled clip."""
    cap = cv2.VideoCapture(video_path)
    k = src = 0
    while k < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if src % frame_step == 0:
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield k, frame
            k += 1
        src += 1
    cap.release()


def probe(video_path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise AnalysisError("Could not open this video. Try an .mp4 or .mov file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not (1.0 < fps < 1000.0):
        fps = 30.0
    if w <= 0 or h <= 0:
        raise AnalysisError("This video has no readable frames.")
    return fps, w, h, n


def _bbox_area(pts, vis):
    ok = vis > 0.5
    if ok.sum() < 6:
        return 0.0, None
    p = pts[ok]
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return float((x1 - x0) * (y1 - y0)), np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def extract_pose(video_path: str, progress=None) -> PoseTrack:
    """Run MediaPipe Pose Landmarker over the clip and track the batter."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    fps, w, h, n_src = probe(video_path)
    nw, nh, s = resized_dims(w, h)
    max_src = int(C.MAX_SECONDS * fps)
    notes = []
    if n_src > max_src:
        notes.append(f"Clip is longer than {C.MAX_SECONDS:.0f} s; only the first "
                     f"{C.MAX_SECONDS:.0f} s were analysed. Trim the clip to the shot for best results.")
    usable = min(n_src if n_src > 0 else max_src, max_src)
    step = max(1, int(np.ceil(usable / C.MAX_FRAMES)))
    if step > 1:
        notes.append(f"Very high frame count: every {step}th frame was used.")
    max_frames = int(np.ceil(usable / step)) if usable > 0 else C.MAX_FRAMES

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=3,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    imgs, worlds, viss, pres = [], [], [], []
    prev_c = None
    eff_fps = fps / step
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for k, frame in iter_frames(video_path, step, max_frames, (nw, nh)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                      int(round(k * 1000.0 / eff_fps)))
            cand = []
            for pi, pl in enumerate(res.pose_landmarks or []):
                pts = np.array([[p.x * nw, p.y * nh] for p in pl])
                vis = np.array([getattr(p, "visibility", 1.0) or 0.0 for p in pl])
                wl = res.pose_world_landmarks[pi]
                wpts = np.array([[p.x, p.y, p.z] for p in wl])
                area, c = _bbox_area(pts, vis)
                if c is not None:
                    cand.append((pts, wpts, vis, area, c))
            if cand:
                if prev_c is None:
                    # first sighting: biggest person, nudged towards frame centre
                    centre = np.array([nw / 2, nh / 2])
                    score = [a / (1.0 + np.linalg.norm(c - centre) / max(nw, nh)) for *_, a, c in cand]
                else:
                    score = [-np.linalg.norm(c - prev_c) for *_, a, c in cand]
                pts, wpts, vis, _, c = cand[int(np.argmax(score))]
                prev_c = c
                imgs.append(pts); worlds.append(wpts); viss.append(vis); pres.append(True)
            else:
                imgs.append(np.full((33, 2), np.nan)); worlds.append(np.full((33, 3), np.nan))
                viss.append(np.zeros(33)); pres.append(False)
            if progress is not None and k % 10 == 0:
                progress(min(0.6, 0.05 + 0.55 * k / max(1, max_frames)), desc="Tracking the batter's pose")

    if len(imgs) < 10:
        raise AnalysisError("The clip is too short to analyse (need at least ~0.5 s).")
    track = PoseTrack(np.array(imgs), np.array(worlds), np.array(viss), np.array(pres),
                      fps=eff_fps, width=nw, height=nh, video_path=video_path,
                      frame_step=step, scale=s, notes=notes)
    if track.present.mean() < 0.5:
        raise AnalysisError("A batter was found in fewer than half of the frames. Use a clearer, "
                            "front-on clip where the batter is large in the frame.")
    return track
