"""Analyse a clip from the command line (no web page needed) - handy for testing.

Examples
    python analyze_cli.py my_cover_drive.mp4
    python analyze_cli.py clip.mp4 --hand right --slowmo 8 --height 172 --out results
    python analyze_cli.py --synthetic            # no video needed: built-in test batter
"""
import argparse
import json
import os

from ksd.pipeline import AnalysisError, analyze


def main():
    ap = argparse.ArgumentParser(description="Cricket batting kinematic sequence (single-camera estimate)")
    ap.add_argument("video", nargs="?", help="front-on batting clip (.mp4/.mov)")
    ap.add_argument("--hand", choices=["auto", "right", "left"], default="auto")
    ap.add_argument("--slowmo", type=float, default=1.0, help="slow-motion factor (8 = 240 fps shown at 30)")
    ap.add_argument("--height", type=float, help="batter height in cm (digital twin lite)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true", help="run on the built-in synthetic batter")
    a = ap.parse_args()

    track = None
    video = a.video
    os.makedirs(a.out, exist_ok=True)
    if a.synthetic:
        from ksd.synthetic import make_swing, write_video
        track, _ = make_swing(fps=60)
        video = write_video(track, os.path.join(a.out, "synthetic_input.mp4"))
    if not video:
        ap.error("give a video file, or use --synthetic")
    handed = {"right": "Right-handed", "left": "Left-handed"}.get(a.hand, "auto")
    twin = {"height_cm": a.height} if a.height else None
    try:
        r = analyze(video, handed=handed, slowmo=a.slowmo, twin=twin, out_dir=a.out, track=track)
    except AnalysisError as e:
        print("Could not analyse:", e)
        raise SystemExit(1)
    seq = r.kin.seq
    print(f"\nBatter: {r.kin.handed} ({r.kin.hand_source}),  effective {r.kin.fs:.0f} fps")
    print(f"Firing order: {' -> '.join(seq['order'])}   score {seq['score']}/100")
    for s, p in seq["peaks"].items():
        print(f"  {s:7s} peak {p['deg_s']:7.0f} deg/s at {p['t_ms']:+6.0f} ms")
    for f in r.factors:
        print(f"  [{f['status']:5s}|{f['confidence']:6s}] {f['name']}: {f['value']}")
    print("\nOutputs:")
    for p in (r.video, r.chart, r.evidence, r.report_file, r.metrics_json):
        print("  ", os.path.abspath(p))
    with open(r.metrics_json) as fh:
        json.load(fh)


if __name__ == "__main__":
    main()
