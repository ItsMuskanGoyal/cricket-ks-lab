"""Constants, colours and the (heuristic, editable) thresholds used by the analysis.

Every threshold lives here on purpose: an engineer reviewing the demo can see
exactly which numbers drive each verdict, and they can be re-calibrated later
against real elite data (which is what the full system would do).
"""

# ---------------------------------------------------------------- landmarks
# MediaPipe Pose landmark indices (33-point BlazePose topology)
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
    (15, 19), (15, 17), (17, 19), (16, 20), (16, 18), (18, 20),
]


def side_indices(lead_is_left: bool) -> dict:
    """Map 'lead'/'trail' body parts to landmark indices.

    Right-handed batter -> lead (front) side is the LEFT side.
    """
    L = dict(shoulder=L_SHOULDER, elbow=L_ELBOW, wrist=L_WRIST, index=L_INDEX,
             pinky=L_PINKY, hip=L_HIP, knee=L_KNEE, ankle=L_ANKLE, foot=L_FOOT,
             heel=L_HEEL)
    R = dict(shoulder=R_SHOULDER, elbow=R_ELBOW, wrist=R_WRIST, index=R_INDEX,
             pinky=R_PINKY, hip=R_HIP, knee=R_KNEE, ankle=R_ANKLE, foot=R_FOOT,
             heel=R_HEEL)
    return {"lead": L if lead_is_left else R, "trail": R if lead_is_left else L}


# ---------------------------------------------------------------- segments
SEGMENTS = ["pelvis", "torso", "arm", "hands"]
SEGMENT_LABELS = {
    "pelvis": "Pelvis",
    "torso": "Torso",
    "arm": "Lead arm",
    "hands": "Hands (bat proxy)",
}
SHORT_LABELS = {"pelvis": "Pelvis", "torso": "Torso", "arm": "Arm", "hands": "Hands"}
# Hex for matplotlib / HTML, BGR for OpenCV
SEGMENT_HEX = {
    "pelvis": "#2B59C3",   # blue
    "torso": "#119DA4",    # teal
    "arm": "#E8A33D",      # amber
    "hands": "#D7263D",    # crimson
}


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


SEGMENT_BGR = {k: hex_to_bgr(v) for k, v in SEGMENT_HEX.items()}

# ---------------------------------------------------------------- processing
MAX_SECONDS = 15.0          # longest clip we analyse (trim to the shot)
MAX_FRAMES = 900            # hard cap on frames sent to pose estimation
MAX_SIDE_PX = 960           # frames are resized so the longer side <= this
VIS_MIN = 0.35              # landmark visibility below this = treated as missing
FILTER_ORDER = 4            # zero-lag Butterworth order (biomechanics standard)
FILTER_MAX_HZ = 12.0        # cutoff ceiling for joint coordinates
FILTER_FRAC_FS = 0.33       # cutoff = min(FILTER_MAX_HZ, FILTER_FRAC_FS * fs)

# Extra smoothing applied to each segment's angular-speed curve (Hz), and how
# long after the estimated impact we still look for that segment's peak (s).
SEG_CUTOFF_HZ = {"pelvis": 7.0, "torso": 8.0, "arm": 10.0, "hands": 14.0}
SEG_PEAK_AFTER_IMPACT_S = {"pelvis": 0.05, "torso": 0.06, "arm": 0.10, "hands": 0.15}

# Winter (2009) segment-length ratios (fraction of standing height)
ANTHRO = {"upper_arm": 0.186, "forearm": 0.146, "thigh": 0.245, "shank": 0.246}

# ---------------------------------------------------------------- thresholds
# All heuristic.  "good" if better than the first number, "watch" if between,
# "work" beyond the second.  Calibrate against real players before trusting.
T = {
    "balance_head_offset": (0.25, 0.50),      # |head - base centre| / shoulder width
    "trigger_head_shift_m": (0.08, 0.15),     # lateral head move during trigger
    "stride_over_height": (0.12, 0.50),       # ok range for front-foot stride
    "brace_knee_change_deg": (-5.0, -15.0),   # knee angle change plant->impact
    "head_lateral_m": (0.06, 0.12),           # lateral head drift top->impact
    "head_drop_m": (0.10, 0.18),              # vertical head drop top->impact
    "hip_rotation_deg": (45.0, 25.0),         # pelvis range top->follow-through
    "x_factor_deg": (15.0, 8.0),              # max hip-shoulder separation
    "alignment_deg": (15.0, 30.0),            # |torso yaw - pelvis yaw| at stance
    "release_frac": (0.60, 0.40),             # wrist release as fraction of downswing
    "elbow_ext_deg": (150.0, 130.0),          # max lead-elbow angle around impact
    "braking_ratio": 0.85,                    # proximal speed at next peak / own peak
}
