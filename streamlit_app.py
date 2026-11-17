"""Cricket Kinematic Sequence Lab - Streamlit version (free hosting on Streamlit Community Cloud).

Same analysis engine as app.py (the Gradio version); only the web page differs.
Run locally:  streamlit run streamlit_app.py
"""
import hashlib
import os
import tempfile

import streamlit as st

from ksd.pipeline import AnalysisError, analyze
from ksd.report import CSS as REPORT_CSS
from ksd.twin import load_profiles

st.set_page_config(page_title="Kinematic Sequence Lab - cricket batting", page_icon="🏏", layout="wide")
st.markdown(f"<style>{REPORT_CSS}</style>", unsafe_allow_html=True)

PROFILES = load_profiles()
NO_TWIN = "None - type my own numbers below (or leave empty)"
SLOWMO = [("Normal speed (as filmed)", 1), ("2× slow-motion (filmed at ~60 fps)", 2),
          ("4× slow-motion (filmed at ~120 fps)", 4), ("8× slow-motion (filmed at ~240 fps)", 8)]
FIELDS = [("height_cm", "Height (cm)"), ("upper_arm_cm", "Upper arm: shoulder → elbow (cm)"),
          ("forearm_cm", "Forearm: elbow → wrist (cm)"), ("thigh_cm", "Thigh: hip → knee (cm)"),
          ("shank_cm", "Shin: knee → ankle (cm)")]

st.markdown("""
<div class="ksd" style="padding:4px 2px 2px">
  <div style="font-size:30px;font-weight:800;letter-spacing:-.3px">🏏 Kinematic Sequence Lab <span style="font-weight:500;opacity:.6">· cricket batting</span></div>
  <div style="font-size:15.5px;opacity:.85;max-width:900px;margin-top:4px">
  Upload one <b>front-on</b> batting clip (filmed from the bowler's end). The app tracks the batter with Google MediaPipe,
  estimates how fast the <b>pelvis, torso, lead arm and hands</b> rotate, and checks whether they peak in the efficient
  inside-out order - the <i>kinematic sequence</i> - then writes a coach-style report.</div>
  <div class="warn" style="max-width:900px">⚠️ <b>Single-camera proof of concept.</b> Every number is an <b>estimate from one 2D video</b>.
  The full system adds a lab-built digital twin, multi-angle high-speed cameras, force plates, eye tracking and bat sensors.</div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([4, 7], gap="large")

with left:
    up = st.file_uploader("Batting clip - front-on, from the bowler's end",
                          type=["mp4", "mov", "m4v", "avi", "webm"])
    hand = st.radio("Batter", ["Auto-detect", "Right-handed", "Left-handed"], horizontal=True)
    slow = st.selectbox("Was it filmed in slow motion?", SLOWMO, format_func=lambda o: o[0])
    with st.expander("Digital twin lite - the player's body measurements (optional)"):
        st.caption("Height alone is enough. Limb lengths make it better. Measure the front (lead) side, "
                   "in centimetres. Leave a box empty if you don't know it.")
        prof = st.selectbox("Start from a profile", [NO_TWIN] + list(PROFILES.keys()))
        preset = PROFILES.get(prof, {})
        twin_vals = {}
        for key, label in FIELDS:
            twin_vals[key] = st.number_input(label, min_value=0.0, max_value=250.0, step=0.5,
                                             value=float(preset[key]) if preset.get(key) else None,
                                             key=f"{prof}-{key}")
    go = st.button("Analyse the swing", type="primary", use_container_width=True)
    st.markdown("""
**For the best result**
- Camera **behind the bowler's arm, facing the batter**, whole body in frame, steady.
- Trim the clip to **one shot** (2-6 s, setup to follow-through).
- **Slow-motion** (120-240 fps) gives much finer timing than normal 30 fps video.
""")

with right:
    if go:
        if up is None:
            st.error("Please upload a batting clip first.")
        else:
            data = up.getvalue()
            twin = {k: float(v) for k, v in twin_vals.items() if v} or None
            handed = hand if hand in ("Right-handed", "Left-handed") else "auto"
            key = hashlib.md5(data + repr((handed, slow[1], twin)).encode()).hexdigest()
            if st.session_state.get("key") != key:
                work = tempfile.mkdtemp(prefix="ksd_")
                path = os.path.join(work, "input" + os.path.splitext(up.name)[1].lower())
                with open(path, "wb") as f:
                    f.write(data)
                bar = st.progress(0, text="Starting")

                def progress(frac, desc=""):
                    bar.progress(min(100, max(0, int(frac * 100))), text=desc)

                try:
                    r = analyze(path, handed=handed, slowmo=float(slow[1]), twin=twin,
                                out_dir=os.path.join(work, "out"), progress=progress)
                    st.session_state["key"], st.session_state["result"] = key, r
                except AnalysisError as e:
                    st.session_state.pop("result", None)
                    st.error(str(e))
                bar.empty()

    r = st.session_state.get("result")
    if r is None:
        st.markdown('<div class="ksd"><div class="honest" style="text-align:center;padding:36px 16px;opacity:.75">'
                    "Results appear here: sequence score, skeleton replay, the kinematic sequence chart, "
                    "evidence frames and the coach report.</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(r.summary_html, unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["Skeleton replay", "Kinematic sequence", "Evidence frames"])
        with t1:
            with open(r.video, "rb") as f:
                st.video(f.read())
        with t2:
            st.image(r.chart, use_container_width=True)
        with t3:
            st.image(r.evidence, use_container_width=True)
        st.markdown(r.report_html, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with open(r.report_file, "rb") as f:
            c1.download_button("Download shareable report (HTML)", f.read(), "batting_report.html",
                               "text/html", use_container_width=True)
        with open(r.metrics_json, "rb") as f:
            c2.download_button("Download metrics (JSON)", f.read(), "metrics.json",
                               "application/json", use_container_width=True)
