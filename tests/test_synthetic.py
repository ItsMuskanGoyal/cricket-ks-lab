"""Synthetic validation: does the maths recover a sequence we know?

Run:  python -m pytest -q      (or simply: python tests/test_synthetic.py)
No camera, no model download needed.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ksd import kinematics as KM  # noqa: E402
from ksd.pipeline import analyze  # noqa: E402
from ksd.synthetic import make_swing, write_video  # noqa: E402

IDEAL = ["pelvis", "torso", "arm", "hands"]


def test_correct_order_recovered_at_common_frame_rates():
    hits = total = 0
    for fps in (60, 120, 240):
        for seed in range(4):
            track, _ = make_swing(fps=fps, seed=seed)
            hits += KM.build(track).seq["order"] == IDEAL
            total += 1
    assert hits / total >= 0.9, f"order recovered {hits}/{total}"


def test_reversed_pelvis_torso_is_flagged():
    for fps in (30, 60, 240):
        for seed in range(3):
            track, _ = make_swing(fps=fps, seed=seed, order=("torso", "pelvis", "arm", "hands"))
            seq = KM.build(track).seq
            assert seq["links"][0]["status"] != "in order"
            assert not seq["ideal_order"]


def test_handedness_auto_detect():
    for rh in (True, False):
        track, _ = make_swing(fps=60, right_handed=rh)
        assert KM.build(track).lead_is_left == rh


def test_collapsing_front_knee_flagged():
    from ksd.factors import compute_factors
    track, _ = make_swing(fps=120, collapse_knee=True)
    f = {x["key"]: x for x in compute_factors(KM.build(track))}
    assert f["front_brace"]["status"] == "work"


def test_full_pipeline_writes_outputs():
    out = tempfile.mkdtemp()
    track, _ = make_swing(fps=60, width=480, height=854)
    write_video(track, os.path.join(out, "in.mp4"))
    r = analyze(track.video_path, track=track, out_dir=out, twin={"height_cm": 172})
    for p in (r.video, r.chart, r.evidence, r.metrics_json, r.report_file):
        assert os.path.getsize(p) > 1000, p
    assert r.kin.twin["used"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
