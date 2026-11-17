"""Batting factors estimated from one front-on camera.

Each factor returns the same 'coach insight' shape used by the full vision:
what happened, why it matters, evidence (a moment in the clip), what to try -
plus a status, a confidence level, and the raw numbers.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import signals as S
from .kinematics import Kinematics

T = C.T
CONF = ["low", "medium", "high"]


def grade(v: float, good: float, bad: float) -> str:
    """good/watch/work for a metric; direction inferred from threshold order."""
    if not np.isfinite(v):
        return "na"
    if good < bad:  # lower is better
        return "good" if v <= good else "watch" if v <= bad else "work"
    return "good" if v >= good else "watch" if v >= bad else "work"


def _conf(base: str, K: Kinematics) -> str:
    i = CONF.index(base)
    if K.consistency.get("overall_cv_pct", 0) > 12:
        i -= 1
    return CONF[max(0, i)]


def _f(key, name, value, status, conf, what, why, evidence, try_, metrics=None, camera_note=""):
    return {"key": key, "name": name, "value": value, "status": status, "confidence": conf,
            "what": what, "why": why, "evidence": evidence, "try": try_,
            "metrics": metrics or {}, "camera_note": camera_note}


def compute_factors(K: Kinematics) -> list[dict]:
    P, I, ph, sd = K.P, K.I, K.phases, K.side
    ld, tr = sd["lead"], sd["trail"]
    s0, s1 = ph["setup"]
    top, imp, fe = ph["top"], ph["impact"], ph["follow_end"]
    plant = ph["plant"]
    fs = K.fs
    setup = slice(s0, s1 + 1)
    off = -1.0 if K.lead_is_left else 1.0   # image-x direction of the off side
    sw = float(np.median(S.norm(P[:, 11] - P[:, 12])))  # shoulder width (m)
    height = K.twin["height_m"]
    Iw = I / K.ppm                           # image coords in metres (camera plane)
    out = []

    # 1. stance & balance ----------------------------------------------------
    base_x = np.median((P[setup, ld["ankle"], 0] + P[setup, tr["ankle"], 0]) / 2)
    head_x = np.median(P[setup, C.NOSE, 0])
    rel = (head_x - base_x) / max(sw, 1e-3)
    st = grade(abs(rel), *T["balance_head_offset"])
    side_word = "towards the off side" if rel * off > 0 else "towards the leg side"
    stance_w = float(np.median(S.norm(P[setup, ld["ankle"]] - P[setup, tr["ankle"]])))
    out.append(_f(
        "stance_balance", "Stance & balance",
        f"head {abs(rel) * 100:.0f}% of shoulder width {side_word} of base",
        st, _conf("high", K),
        ("In your stance your head sat almost directly over the middle of your feet."
         if st == "good" else
         f"In your stance your head sat {abs(rel) * sw * 100:.0f} cm {side_word} of the centre of your feet."),
        "A head centred over the base lets you move forward or back equally well and keeps your eyes level.",
        {"phase": "setup", "label": "Setup"},
        ("Keep it - this is a strength to protect." if st == "good" else
         "Shadow-bat in front of a mirror: settle with your nose over the gap between your feet before every ball."),
        {"head_offset_ratio": rel, "stance_width_m": stance_w},
        "Side-to-side balance is clearly visible front-on; stance width runs along the pitch (depth), so it is only estimated."))

    # 2. trigger movement -----------------------------------------------------
    trig_end = plant if plant is not None and plant < top else top
    back_dx = float(Iw[trig_end, tr["ankle"], 0] - np.median(Iw[setup, tr["ankle"], 0]))
    head_dx = float(Iw[trig_end, C.NOSE, 0] - np.median(Iw[setup, C.NOSE, 0]))
    st = grade(abs(head_dx), *T["trigger_head_shift_m"])
    moved = abs(back_dx) > 0.04
    dirw = "across to the off side" if back_dx * off > 0 else "towards the leg side"
    out.append(_f(
        "trigger", "Trigger movement",
        (f"back foot {abs(back_dx) * 100:.0f} cm {dirw}" if moved else "minimal / still trigger") +
        f"; head moved {abs(head_dx) * 100:.0f} cm sideways",
        st, _conf("medium", K),
        (f"Your trigger moved the back foot about {abs(back_dx) * 100:.0f} cm {dirw}, and your head shifted "
         f"{abs(head_dx) * 100:.0f} cm sideways before the downswing." if moved else
         f"You kept a still, minimal trigger; your head shifted {abs(head_dx) * 100:.0f} cm sideways."),
        "A trigger should get you moving without throwing the head off line before the ball arrives.",
        {"phase": "top", "label": "Top of backlift"},
        ("Good - the trigger isn't costing you head position." if st == "good" else
         "Keep the trigger small and finished before the bowler releases; film from front-on and check the head stays on the same line."),
        {"back_foot_dx_m": back_dx, "head_dx_m": head_dx},
        "Front-on we see the 'across' part of a trigger well; the 'back' part (towards the stumps) is depth, so it isn't measured."))

    # 3. stride length ----------------------------------------------------------
    sep0 = float(np.median(S.norm(P[setup, ld["ankle"]] - P[setup, tr["ankle"]])))
    if plant is not None:
        sep1 = float(S.norm(P[plant, ld["ankle"]] - P[plant, tr["ankle"]]))
        stride = sep1 - sep0
        ratio = stride / max(height, 1e-3)
        lo, hi = T["stride_over_height"]
        st = "good" if lo <= ratio <= hi else "watch"
        what = (f"Your front foot strode about {stride * 100:.0f} cm "
                f"({ratio * 100:.0f}% of your height) and landed {(imp - plant) / fs * 1000:.0f} ms before impact.")
        val = f"~{stride * 100:.0f} cm ({ratio * 100:.0f}% of height)"
        try_ = ("Stride is in a comfortable range - stride to the pitch of the ball, not a fixed distance."
                if st == "good" else
                ("A very long stride can lock the hips; try a shorter, controlled step." if ratio > hi else
                 "Little stride detected - fine for back-foot shots; for drives, get the front foot closer to the pitch of the ball."))
        ev = {"phase": "plant", "label": "Front-foot plant"}
    else:
        stride, ratio, st = float("nan"), float("nan"), "info"
        what = "No clear front-foot stride was found - this looks like a back-foot shot or a very small step."
        val = "no clear stride"
        try_ = "If this was meant to be a front-foot shot, make the stride decisive towards the line of the ball."
        ev = {"phase": "impact", "label": "Impact (est.)"}
    out.append(_f("stride", "Stride length", val, st, "low", what,
                  "The stride sets your base and gets the head close to the ball's line.",
                  ev, try_, {"stride_m": stride, "stride_over_height": ratio},
                  "A front-on stride travels towards the camera (depth), so length is a low-confidence estimate."))

    # 4. front-leg bracing ------------------------------------------------------
    ref = plant if plant is not None else top
    knee = S.joint_angle(P[:, ld["hip"]], P[:, ld["knee"]], P[:, ld["ankle"]])
    k_ref, k_imp = float(knee[ref]), float(knee[imp])
    dk = k_imp - k_ref
    st = grade(dk, *T["brace_knee_change_deg"])
    desc = ("stayed firm / straightened (braced)" if dk >= T["brace_knee_change_deg"][0] else
            "gave a little" if dk >= T["brace_knee_change_deg"][1] else "kept collapsing")
    out.append(_f(
        "front_brace", "Front-leg bracing", f"front knee {k_ref:.0f}° → {k_imp:.0f}° ({dk:+.0f}°)",
        st, _conf("medium", K),
        f"From {'front-foot plant' if plant is not None else 'top of backlift'} to impact your front knee {desc} "
        f"({k_ref:.0f}° to {k_imp:.0f}°).",
        "A front leg that firms up acts as a brake: the hips stop and pass their speed up the chain. "
        "A knee that keeps collapsing soaks energy up (a power leak). A softly bent knee is still normal on classic drives.",
        {"phase": "impact", "label": "Impact (est.)"},
        ("Good - the front side is giving you a solid post to hit against." if st == "good" else
         "Step-and-hit drill: plant, feel the front leg push back up into the ground as the bat comes down."),
        {"knee_at_ref_deg": k_ref, "knee_at_impact_deg": k_imp, "change_deg": dk},
        "Knee angle uses estimated depth; trends are more reliable than exact degrees."))

    # 5. head stability ---------------------------------------------------------
    hx = Iw[top:imp + 1, C.NOSE, 0]
    hy = Iw[top:imp + 1, C.NOSE, 1]
    lat = float(np.ptp(hx)) if len(hx) else 0.0
    drop = float(hy[-1] - hy[0]) if len(hy) else 0.0
    st_lat, st_drop = grade(lat, *T["head_lateral_m"]), grade(max(drop, 0), *T["head_drop_m"])
    st = max([st_lat, st_drop], key=["good", "watch", "work"].index)
    over = float(P[imp, C.NOSE, 0] - P[imp, ld["ankle"], 0]) / max(sw, 1e-3)
    over_txt = ("over the front foot" if abs(over) < 0.5 else
                ("outside the front foot towards the off side" if over * off > 0 else "inside the line of the front foot"))
    out.append(_f(
        "head", "Head stability", f"{lat * 100:.0f} cm sideways, {drop * 100:+.0f} cm vertical; head {over_txt} at impact",
        st, _conf("high", K),
        f"From top of backlift to impact your head moved {lat * 100:.0f} cm sideways and "
        f"{abs(drop) * 100:.0f} cm {'down' if drop > 0 else 'up'}; at impact it was {over_txt}.",
        "The eyes are the measuring instrument: a still, level head means the ball's line is judged from a stable platform.",
        {"phase": "impact", "label": "Impact (est.)"},
        ("Strength to protect - your head is quiet through the shot." if st == "good" else
         "Throwdowns with a cue: 'nose over front knee' at contact. Hold the finish for two seconds and check it on video."),
        {"lateral_m": lat, "drop_m": drop, "head_over_front_foot_ratio": over},
        "Sideways and up-down head movement is seen well front-on; forward-back movement is depth and not measured."))

    # 6. hip rotation -----------------------------------------------------------
    py = K.yaw["pelvis"]
    rng = float(np.ptp(py[top:fe + 1]))
    st = grade(rng, *T["hip_rotation_deg"])
    pk = K.seq["peaks"]["pelvis"]["deg_s"]
    out.append(_f(
        "hip_rotation", "Hip rotation", f"{rng:.0f}° range, peak ≈ {pk:.0f}°/s",
        st, _conf("medium", K),
        f"Your hips rotated about {rng:.0f}° from top of backlift to the follow-through, peaking near {pk:.0f}°/s.",
        "The hips are the engine at the base of the chain; limited hip turn forces the arms to make the power.",
        {"phase": "follow", "label": "Follow-through"},
        ("Good hip drive - protect it." if st == "good" else
         "Rotational medicine-ball throws from a batting stance: let the hips start the throw before the chest."),
        {"pelvis_range_deg": rng, "pelvis_peak_deg_s": pk},
        "Rotation of a side-on body needs depth, which one camera only estimates."))

    # 7. hip-shoulder separation (X-factor) -------------------------------------
    xf = K.x_factor
    win = slice(s1, imp + 1)
    i_max = s1 + int(np.argmax(np.abs(xf[win])))
    xmax = float(abs(xf[i_max]))
    st = grade(xmax, *T["x_factor_deg"])
    out.append(_f(
        "x_factor", "Hip-shoulder separation (X-factor)", f"max ≈ {xmax:.0f}° at {(i_max - imp) / fs * 1000:+.0f} ms",
        st, _conf("medium", K) if xmax > 3 else "low",
        f"The largest gap between hip and shoulder rotation was about {xmax:.0f}°, "
        f"{abs(i_max - imp) / fs * 1000:.0f} ms {'before' if i_max <= imp else 'after'} impact.",
        "When the hips open while the shoulders stay closed, the torso is 'stretched' like a spring - "
        "that stored stretch is released into the bat. Little separation means hips and shoulders turn as one block.",
        {"phase": "top", "label": "Top of backlift"},
        ("Good stretch between hips and shoulders." if st == "good" else
         "Separation drill: stride and start the hips while keeping the front shoulder pointing at the bowler for a beat."),
        {"x_factor_max_deg": xmax, "frame": i_max},
        "Separation is the difference of two depth-based rotations, so it carries the most uncertainty here."))

    # 8. hip-shoulder alignment -------------------------------------------------
    t_y = float(np.median(K.yaw["torso"][setup]))
    p_y = float(np.median(py[setup]))
    diff = abs(((t_y - p_y) + 180) % 360 - 180)
    open_sh = float(abs(90 - abs(((t_y + 180) % 360) - 180)))  # 0 = fully side-on
    tilt = float(np.degrees(np.arctan2(I[imp, tr["shoulder"], 1] - I[imp, ld["shoulder"], 1],
                                       abs(I[imp, tr["shoulder"], 0] - I[imp, ld["shoulder"], 0]) + 1e-6)))
    st = grade(diff, *T["alignment_deg"])
    out.append(_f(
        "alignment", "Hip-shoulder alignment",
        f"stance mismatch {diff:.0f}°, shoulders {open_sh:.0f}° open; front shoulder {'lower' if tilt < 0 else 'higher'} at impact",
        st, _conf("medium", K),
        f"In your stance, hips and shoulders pointed {diff:.0f}° apart, with the shoulders about {open_sh:.0f}° open "
        f"from fully side-on. At impact your front shoulder was {abs(tilt):.0f}° {'below' if tilt < 0 else 'above'} the back shoulder.",
        "Hips and shoulders lined up in the stance give a consistent starting point for the swing path.",
        {"phase": "setup", "label": "Setup"},
        ("Well aligned - keep it." if st == "good" else
         "Mirror check: in your stance, both hips and shoulders should point down the pitch at the bowler."),
        {"stance_mismatch_deg": diff, "shoulder_open_deg": open_sh, "shoulder_tilt_impact_deg": tilt},
        "Shoulder tilt is seen well front-on; how 'open' the stance is relies on depth."))

    # 9. driving from the shoulder, not the wrist ---------------------------------
    hand_vec = (P[:, ld["index"]] + P[:, ld["pinky"]]) / 2 - P[:, ld["wrist"]]
    forearm = P[:, ld["wrist"]] - P[:, ld["elbow"]]
    hinge = np.degrees(np.arccos(np.clip(np.sum(S.unit(hand_vec) * S.unit(forearm), -1), -1, 1)))  # 0 = straight wrist
    seg = hinge[top:imp + 1]
    if len(seg) >= 3 and (seg.max() - seg[-1]) > 5:
        cocked = top + int(np.argmax(seg))
        half = (hinge[cocked] + hinge[imp]) / 2
        rel_i = next((i for i in range(cocked, imp + 1) if hinge[i] <= half), imp)
        frac = (rel_i - top) / max(imp - top, 1)
    else:
        frac = float("nan")
    seq_pk = K.seq["peaks"]
    hands_first = seq_pk["hands"]["frame"] < seq_pk["arm"]["frame"] - 0.5
    st = "work" if hands_first else grade(frac, *T["release_frac"])
    if not np.isfinite(frac) and not hands_first:
        st = "info"
    what = []
    if np.isfinite(frac):
        what.append(f"Your wrists released (uncocked) about {frac * 100:.0f}% of the way through the downswing.")
    what.append("The hands reached peak speed before the lead arm - the wrists took over early." if hands_first
                else "The lead arm reached peak speed before the hands, so the shoulder led the swing.")
    out.append(_f(
        "shoulder_drive", "Driving from the shoulder, not the wrist",
        (f"release at {frac * 100:.0f}% of downswing" if np.isfinite(frac) else "release not measurable") +
        ("; hands peaked before arm" if hands_first else "; arm before hands"),
        st, "low", " ".join(what),
        "Leading with the shoulder and releasing the wrists late keeps the bat face stable and adds speed at the right moment; "
        "early 'flicking' wrists lose control and power.",
        {"phase": "impact", "label": "Impact (est.)"},
        ("Good - the shoulder drives and the wrists finish." if st == "good" else
         "Top-hand-only drives (bottom hand off) to feel the front shoulder pull the bat through."),
        {"release_fraction": frac, "hands_peak_before_arm": bool(hands_first)},
        "Hands gripping a bat are often hidden or blurred, so wrist measurements are low confidence."))

    # 10. elbow & arm straightness -----------------------------------------------
    elbow = S.joint_angle(P[:, ld["shoulder"]], P[:, ld["elbow"]], P[:, ld["wrist"]])
    j1 = min(K.n - 1, imp + int(0.15 * fs))
    e_imp = float(elbow[imp])
    e_max = float(np.max(elbow[imp:j1 + 1]))
    high = I[imp, ld["elbow"], 1] < I[imp, ld["wrist"], 1]
    st = grade(e_max, *T["elbow_ext_deg"])
    out.append(_f(
        "elbow", "Elbow & arm extension", f"lead elbow {e_imp:.0f}° at impact, {e_max:.0f}° max",
        st, _conf("medium", K),
        f"Your lead elbow was {e_imp:.0f}° at impact and extended to {e_max:.0f}° just after "
        f"(180° = fully straight). The front elbow was {'above' if high else 'below'} the hands at impact.",
        "Full extension through the ball lengthens the lever, so the bat travels faster for the same body speed.",
        {"phase": "impact", "label": "Impact (est.)"},
        ("Good extension through the ball." if st == "good" else
         "Drive off a tee and 'push the bat face down the target line' after contact, arms long."),
        {"elbow_impact_deg": e_imp, "elbow_max_deg": e_max, "front_elbow_high": bool(high)},
        "Elbow angle relies partly on estimated depth."))

    # 11. wrist cock --------------------------------------------------------------
    wc = float(np.max(hinge[max(0, top - int(0.1 * fs)):top + 1])) if top >= 0 else float("nan")
    out.append(_f(
        "wrist_cock", "Wrist cock", f"≈ {wc:.0f}° at top of backlift", "info", "low",
        f"At the top of your backlift your lead wrist was hinged about {wc:.0f}° from straight.",
        "The wrist hinge stores the last link of energy, released late into the bat.",
        {"phase": "top", "label": "Top of backlift"},
        "Use this as a personal baseline; compare it across your best and worst shots.",
        {"wrist_cock_deg": wc},
        "Low confidence: the hand landmarks are small and often hidden by the bat handle."))

    # 12. swing timing ------------------------------------------------------------
    ds = (imp - top) / fs * 1000
    gaps = ", ".join(f"{C.SHORT_LABELS[l['from']]}→{C.SHORT_LABELS[l['to']]} "
                     f"{l['dt_ms']:+.0f} ms" for l in K.seq["links"])
    tconf = "high" if fs >= 120 else "medium" if fs >= 55 else "low"
    st = "good" if K.seq["ideal_order"] else "watch"
    out.append(_f(
        "timing", "Swing timing", f"downswing {ds:.0f} ms; {gaps}", st, tconf,
        f"Your downswing (top of backlift to impact) took about {ds:.0f} ms. Gaps between segment peaks: {gaps}.",
        "Power comes from the segments peaking one after another, each slightly later than the one below it.",
        {"phase": "impact", "label": "Impact (est.)"},
        ("Timing chain looks right - protect it." if st == "good" else
         "Rhythm drill: count 'hips - chest - arms - bat' in slow motion, then build speed while keeping the order."),
        {"downswing_ms": ds, "frame_ms": 1000 / fs},
        f"Each frame is {1000 / fs:.1f} ms at this frame rate; gaps smaller than that are interpolated."))
    return out


def phase_frame(K: Kinematics, phase: str) -> int:
    ph = K.phases
    return {"setup": ph["setup"][1], "top": ph["top"],
            "plant": ph["plant"] if ph["plant"] is not None else ph["top"],
            "impact": ph["impact"], "follow": ph["follow_end"]}[phase]
