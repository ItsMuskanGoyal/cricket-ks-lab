"""One call from video file to every output the app shows."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from . import factors as F
from . import kinematics as KM
from . import render as R
from . import report as RP
from .pose import AnalysisError, PoseTrack, extract_pose  # noqa: F401 (re-export)


@dataclass
class Result:
    video: str
    chart: str
    evidence: str
    summary_html: str
    report_html: str
    metrics_json: str
    report_file: str
    kin: KM.Kinematics
    factors: list


def analyze(video_path: str, handed: str = "auto", slowmo: float = 1.0, twin: dict | None = None,
            out_dir: str | None = None, progress=None, track: PoseTrack | None = None) -> Result:
    out_dir = out_dir or tempfile.mkdtemp(prefix="ksd_")
    os.makedirs(out_dir, exist_ok=True)
    if track is None:
        if progress is not None:
            progress(0.02, desc="Loading the pose model")
        track = extract_pose(video_path, progress=progress)
    if progress is not None:
        progress(0.62, desc="Computing rotation speeds and swing phases")
    K = KM.build(track, handed=handed, slowmo=slowmo, twin=twin)
    factors = F.compute_factors(K)
    seqi = RP.sequence_insight(K)
    head = RP.headline(K, factors, seqi)
    head = RP.maybe_polish(head, {"sequence": seqi, "factors": [{k: f[k] for k in ("name", "value", "status")}
                                                                 for f in factors]})
    if progress is not None:
        progress(0.68, desc="Drawing the chart")
    chart = R.sequence_chart(K, out_dir)
    video, snaps = R.render_video(track, K, out_dir, progress=progress)
    evidence = R.evidence_strip(K, snaps, out_dir)
    summary = RP.summary_html(K, factors)
    report = RP.report_html(K, factors, seqi, head)
    mj = RP.metrics_json(K, factors, seqi, out_dir)
    rf = RP.standalone_html(summary, report, chart, evidence, out_dir)
    if progress is not None:
        progress(1.0, desc="Done")
    return Result(video, chart, evidence, summary, report, mj, rf, K, factors)
