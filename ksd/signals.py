"""Signal-processing helpers: gap filling, zero-lag filtering, angles, peaks."""
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from . import config as C


def fill_gaps(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate samples where valid is False (along axis 0).

    arr:   (N, ...) values
    valid: (N,) bool, or same leading shape as arr without the last axis
    Edges are held at the nearest valid value. Columns with no valid sample
    are returned as NaN.
    """
    arr = np.asarray(arr, dtype=float).copy()
    N = arr.shape[0]
    flat = arr.reshape(N, -1)
    if valid.ndim == 1:
        vmask = np.repeat(valid[:, None], flat.shape[1], axis=1)
    else:
        v = valid.reshape(N, -1)
        reps = flat.shape[1] // v.shape[1]
        vmask = np.repeat(v, reps, axis=1)
    vmask = vmask & np.isfinite(flat)
    t = np.arange(N)
    for c in range(flat.shape[1]):
        ok = vmask[:, c]
        if ok.sum() == 0:
            flat[:, c] = np.nan
        elif ok.sum() < N:
            flat[:, c] = np.interp(t, t[ok], flat[ok, c])
    return flat.reshape(arr.shape)


def cutoff_hz(fs: float) -> float:
    return float(min(C.FILTER_MAX_HZ, C.FILTER_FRAC_FS * fs))


def lowpass(x: np.ndarray, fs: float, fc: float | None = None) -> np.ndarray:
    """4th-order zero-lag Butterworth along axis 0 (falls back to Savitzky-Golay)."""
    x = np.asarray(x, dtype=float)
    fc = cutoff_hz(fs) if fc is None else fc
    nyq = fs / 2.0
    N = x.shape[0]
    if fc >= nyq * 0.95:
        return x.copy()
    b, a = butter(C.FILTER_ORDER // 2, fc / nyq)  # filtfilt doubles the order
    padlen = 3 * max(len(a), len(b))
    if N > padlen + 1:
        return filtfilt(b, a, x, axis=0, padlen=padlen)
    win = N if N % 2 == 1 else N - 1
    if win >= 5:
        return savgol_filter(x, win, 2, axis=0)
    return x.copy()


def norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=-1)


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)[..., None]
    return v / np.where(n < 1e-9, 1e-9, n)


def yaw_deg(vec: np.ndarray) -> np.ndarray:
    """Heading of a vector in the horizontal (x-z) plane, unwrapped, degrees.

    Coordinates follow MediaPipe: x right, y down, z away from camera.
    """
    ang = np.arctan2(vec[..., 2], vec[..., 0])
    return np.degrees(np.unwrap(ang))


def rate(sig: np.ndarray, fs: float) -> np.ndarray:
    """Time derivative (central differences) in units per second."""
    return np.gradient(sig, axis=0) * fs


def angular_speed_3d(vec: np.ndarray, fs: float) -> np.ndarray:
    """Angular speed (deg/s) of a direction vector: |u x du/dt|."""
    u = unit(vec)
    du = np.gradient(u, axis=0) * fs
    return np.degrees(norm(np.cross(u, du)))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC at joint b in degrees (180 = straight)."""
    u, v = unit(a - b), unit(c - b)
    cosang = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def refine_peak(y: np.ndarray, i: int) -> tuple[float, float]:
    """Sub-frame peak location via parabolic interpolation through 3 samples."""
    if i <= 0 or i >= len(y) - 1:
        return float(i), float(y[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    if abs(den) < 1e-12:
        return float(i), float(y1)
    d = 0.5 * (y0 - y2) / den
    d = float(np.clip(d, -0.5, 0.5))
    return i + d, float(y1 - 0.25 * (y0 - y2) * d)


def interp_at(y: np.ndarray, fidx: float) -> float:
    """Value of y at a fractional frame index."""
    return float(np.interp(fidx, np.arange(len(y)), y))
