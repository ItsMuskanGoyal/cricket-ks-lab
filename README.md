# 🏏 Kinematic Sequence Lab — cricket batting

A zero-hardware proof of concept: upload **one front-on batting clip** and get an **estimated kinematic sequence** —
the order and speed in which the pelvis, torso, lead arm and hands (bat proxy) reach peak rotation —
plus a coach-style report on twelve batting factors.

**Every value is an estimate from a single 2D video.** The app says so on every screen, rates each factor's
confidence, and lists what the full system (lab digital twin, multi-angle high-speed cameras, force plates,
eye tracking, bat sensors) would add.

## What you get
1. **Skeleton replay** — the clip with a pose skeleton, colour-coded segments, live speed bars, and an automatic slow-motion replay of the swing.
2. **Kinematic sequence chart** — rotation speed vs time for each segment, peaks numbered in firing order, swing phases shaded, plus hip–shoulder separation.
3. **Evidence frames** — setup, top of backlift, front-foot plant, impact (est.), follow-through.
4. **Coach report** — every insight in the same shape: *what happened → why it matters → evidence → what to try*, with strengths to protect and work-ons.
5. **Downloads** — a single shareable HTML report and a JSON file of all metrics.

## How it works (short)
Google **MediaPipe Pose Landmarker** → zero-lag Butterworth filtering → *hybrid 2.5D* coordinates (precise 2D image
x/y + learned depth) → depth refined with the **rigid-segment constraint** (bones don't change length) → segment
angular speeds → automatic swing-phase detection → sub-frame peak timing → order, speed gain and braking checks →
rules-based report. Details: [`docs/METHOD.md`](docs/METHOD.md).

## Project layout
```
streamlit_app.py     the web app (free on Streamlit Community Cloud)
app.py               same app built with Gradio (local use / paid Hugging Face)
analyze_cli.py       same analysis from the command line
ksd/                 the analysis package
  pose.py            MediaPipe pose extraction + batter tracking
  kinematics.py      coordinates, rotation speeds, swing phases, kinematic sequence
  factors.py         the 12 batting factors
  twin.py            digital-twin-lite calibration + tracking-quality check
  render.py          replay video, chart, evidence frames
  report.py          coach report, summary cards, JSON + HTML exports
  config.py          every threshold and colour in one place
  synthetic.py       synthetic batter used for testing
tests/               synthetic validation tests
twin_profiles.json   body-measurement presets (add your own)
docs/                deploy guide, method, testing plan, demo script
```

## Run it
- **Online (free):** follow [`docs/HOW_TO_DEPLOY.md`](docs/HOW_TO_DEPLOY.md) (GitHub + Streamlit Community Cloud).
- **On a computer:** `pip install -r requirements.txt` then `streamlit run streamlit_app.py`.
- **Body measurements:** [`docs/HOW_TO_MEASURE.md`](docs/HOW_TO_MEASURE.md).
- **Self-test, no video needed:** `python analyze_cli.py --synthetic` and `python tests/test_synthetic.py`.

Built with Google MediaPipe. A single-camera demo of a larger biomechanics vision.
