"""Cricket Kinematic Sequence Lab - Gradio web app (runs on Hugging Face Spaces).

Run locally:  python app.py   then open http://127.0.0.1:7860
"""
import glob
import os

import gradio as gr

from ksd.pipeline import AnalysisError, analyze
from ksd.report import CSS as REPORT_CSS
from ksd.twin import load_profiles

PROFILES = load_profiles()
NO_TWIN = "None - use MediaPipe's default body scale"
SLOWMO = [("Normal speed (as filmed)", 1), ("2× slow-motion (filmed at ~60 fps)", 2),
          ("4× slow-motion (filmed at ~120 fps)", 4), ("8× slow-motion (filmed at ~240 fps)", 8)]

HEADER = """
<div class="ksd" style="padding:6px 2px 2px">
  <div style="font-size:28px;font-weight:800;letter-spacing:-.3px">🏏 Kinematic Sequence Lab <span style="font-weight:500;opacity:.6">· cricket batting</span></div>
  <div style="font-size:15.5px;opacity:.85;max-width:900px;margin-top:4px">
  Upload one <b>front-on</b> batting clip (filmed from the bowler's end). The app tracks the batter with Google MediaPipe,
  estimates how fast the <b>pelvis, torso, lead arm and hands</b> rotate, and checks whether they peak in the efficient
  inside-out order - the <i>kinematic sequence</i> - then writes a coach-style report.</div>
  <div class="warn" style="max-width:900px">⚠️ <b>Single-camera proof of concept.</b> Every number is an <b>estimate from one 2D video</b>.
  The full system adds a lab-built digital twin, multi-angle high-speed cameras, force plates, eye tracking and bat sensors.</div>
</div>"""

TIPS = """
**For the best result**
- Camera **behind the bowler's arm, facing the batter**, whole body in frame, steady.
- Trim the clip to **one shot** (2-6 s, setup to follow-through).
- **Slow-motion** (120-240 fps) gives much finer timing than normal 30 fps video.
- If your phone exported slow-motion as a normal-looking 30 fps file, pick the matching slow-motion option.
"""

PLACEHOLDER = """<div class="ksd"><div class="honest" style="text-align:center;padding:36px 16px;opacity:.75">
Results appear here: sequence score, skeleton replay, the kinematic sequence chart, evidence frames and the coach report.</div></div>"""


def fill_profile(name):
    p = PROFILES.get(name, {})
    return [p.get(k) for k in ("height_cm", "upper_arm_cm", "forearm_cm", "thigh_cm", "shank_cm")]


def run_analysis(video, hand, slowmo, height, upper_arm, forearm, thigh, shank, progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a batting clip first.")
    vals = {"height_cm": height, "upper_arm_cm": upper_arm, "forearm_cm": forearm,
            "thigh_cm": thigh, "shank_cm": shank}
    twin = {k: float(v) for k, v in vals.items() if v not in (None, "") and float(v) > 0} or None
    handed = {"Right-handed": "Right-handed", "Left-handed": "Left-handed"}.get(hand, "auto")
    try:
        r = analyze(video, handed=handed, slowmo=float(slowmo or 1), twin=twin, progress=progress)
    except AnalysisError as e:
        raise gr.Error(str(e))
    return r.summary_html, r.video, r.chart, r.evidence, r.report_html, [r.report_file, r.metrics_json]


examples = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "*.mp4")))

with gr.Blocks(title="Kinematic Sequence Lab - cricket batting", theme=gr.themes.Soft(primary_hue="indigo"),
               css=REPORT_CSS + ".gradio-container{max-width:1280px !important}") as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            video_in = gr.Video(label="Batting clip - front-on, from the bowler's end", sources=["upload"])
            hand = gr.Radio(["Auto-detect", "Right-handed", "Left-handed"], value="Auto-detect", label="Batter")
            slowmo = gr.Dropdown(SLOWMO, value=1, label="Was it filmed in slow motion?")
            with gr.Accordion("Digital twin lite - calibrate to the player's body (optional)", open=False):
                gr.Markdown("Known body measurements put the skeleton on the player's real scale (distances, "
                            "hand speed) and show how well each bone was tracked. Height alone is enough; "
                            "limb lengths make it better.")
                profile = gr.Dropdown([NO_TWIN] + list(PROFILES.keys()), value=NO_TWIN, label="Profile")
                height = gr.Number(label="Height (cm)", precision=1)
                with gr.Row():
                    upper_arm = gr.Number(label="Upper arm (shoulder→elbow, cm)", precision=1)
                    forearm = gr.Number(label="Forearm (elbow→wrist, cm)", precision=1)
                with gr.Row():
                    thigh = gr.Number(label="Thigh (hip→knee, cm)", precision=1)
                    shank = gr.Number(label="Shin (knee→ankle, cm)", precision=1)
                profile.change(fill_profile, profile, [height, upper_arm, forearm, thigh, shank])
            run = gr.Button("Analyse the swing", variant="primary", size="lg")
            gr.Markdown(TIPS)
            if examples:
                gr.Examples(examples=[[p] for p in examples], inputs=[video_in], label="Example clips")
        with gr.Column(scale=7, min_width=420):
            summary = gr.HTML(PLACEHOLDER)
            with gr.Tabs():
                with gr.Tab("Skeleton replay"):
                    video_out = gr.Video(label="Pose skeleton + live segment speeds", interactive=False)
                with gr.Tab("Kinematic sequence"):
                    chart = gr.Image(label="Rotation speed of each segment", interactive=False, type="filepath")
                with gr.Tab("Evidence frames"):
                    evidence = gr.Image(label="Key moments of the swing", interactive=False, type="filepath")
            report = gr.HTML()
            files = gr.File(label="Download: shareable HTML report + metrics (JSON)", file_count="multiple")

    run.click(run_analysis, inputs=[video_in, hand, slowmo, height, upper_arm, forearm, thigh, shank],
              outputs=[summary, video_out, chart, evidence, report, files])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=10).launch(ssr_mode=False)
