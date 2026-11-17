"""Coach-style report: summary cards, insights, factor table, honesty section.

Rules-based by design, so every sentence traces back to a measured number.
An optional Gemini step can re-phrase the headline summary (never add facts).
"""
from __future__ import annotations

import base64
import html
import json
import os

import numpy as np

from . import config as C
from .kinematics import Kinematics

STATUS_TXT = {"good": "Strength", "watch": "Watch", "work": "Work on", "info": "Baseline", "na": "n/a"}

CSS = """
.ksd { --ink: var(--body-text-color, #1d1d1f); --muted: var(--body-text-color-subdued, #6b7280);
       --card: var(--block-background-fill, #ffffff); --line: var(--border-color-primary, #e5e7eb);
       color: var(--ink); font-size: 15px; line-height: 1.5; }
.ksd h3 { margin: 18px 0 8px; font-size: 17px; }
.ksd .hero { display:flex; gap:18px; align-items:center; flex-wrap:wrap; border:1px solid var(--line);
       border-radius:14px; padding:16px 18px; background:var(--card); }
.ksd .score { font-size:42px; font-weight:800; line-height:1; }
.ksd .score small { font-size:15px; font-weight:600; color:var(--muted); }
.ksd .verdict { font-size:16px; font-weight:600; }
.ksd .sub { color:var(--muted); font-size:13px; }
.ksd .chain { display:flex; align-items:center; flex-wrap:wrap; gap:6px; margin-top:12px; }
.ksd .chip { border-radius:10px; padding:7px 11px; color:#fff; font-weight:700; font-size:14px; }
.ksd .chip span { display:block; font-weight:500; font-size:11.5px; opacity:.92; }
.ksd .link { font-size:12px; text-align:center; color:var(--muted); min-width:64px; }
.ksd .link b { display:block; font-size:16px; }
.ksd .ok { color:#1a7f37; } .ksd .bad { color:#cf222e; } .ksd .meh { color:#b26b00; }
.ksd .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:12px; }
.ksd .tile { border:1px solid var(--line); border-radius:12px; padding:10px 12px; background:var(--card); }
.ksd .tile .v { font-size:20px; font-weight:750; }
.ksd .tile .k { font-size:12px; color:var(--muted); }
.ksd .card { border:1px solid var(--line); border-left:5px solid #888; border-radius:12px;
       padding:12px 14px; margin:10px 0; background:var(--card); }
.ksd .card.good { border-left-color:#1a7f37; } .ksd .card.watch { border-left-color:#d4a72c; }
.ksd .card.work { border-left-color:#cf222e; } .ksd .card.info { border-left-color:#6e7781; }
.ksd .card .t { font-weight:700; font-size:15.5px; }
.ksd .card dl { display:grid; grid-template-columns:110px 1fr; gap:4px 10px; margin:8px 0 0; }
.ksd .card dt { color:var(--muted); font-size:13px; } .ksd .card dd { margin:0; }
.ksd .pill { display:inline-block; border-radius:999px; padding:1px 9px; font-size:12px; font-weight:700; color:#fff; }
.ksd .pill.good{background:#1a7f37} .ksd .pill.watch{background:#b88a00} .ksd .pill.work{background:#cf222e}
.ksd .pill.info,.ksd .pill.na{background:#6e7781}
.ksd .conf { font-size:12px; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:0 8px; }
.ksd table { width:100%; border-collapse:collapse; font-size:13.5px; }
.ksd th, .ksd td { text-align:left; padding:7px 6px; border-bottom:1px solid var(--line); vertical-align:top; }
.ksd th { color:var(--muted); font-weight:600; }
.ksd .honest { border:1px dashed var(--line); border-radius:12px; padding:12px 16px; }
.ksd .warn { background:rgba(212,167,44,.12); border-radius:10px; padding:8px 12px; font-size:13.5px; margin-top:10px; }
.ksd ul { margin:6px 0 0 18px; padding:0; }
"""

CANNOT_MEASURE = [
    ("Depth (towards / away from the camera)",
     "One camera sees across and up/down; depth comes from a learned estimate, so rotations of a side-on body are approximations."),
    ("Bat and ball", "The bat isn't tracked: the hands are used as a proxy, so true bat speed and bat-face angle are unknown. Impact is estimated from peak hand speed."),
    ("Forces and weight transfer", "Without force plates we can't see how hard each foot pushes or how weight shifts."),
    ("Eyes", "Where the eyes look (and when they track the ball) needs eye tracking, not a body camera."),
    ("Fine timing", "At 30 fps each frame is 33 ms - about the same as the gaps between segment peaks. Slow-motion helps a lot."),
]
FULL_SYSTEM_ADDS = [
    ("Real 3D rotation", "multi-angle high-speed cameras fitted to the player's calibrated digital twin"),
    ("Force and weight transfer", "force plates in the lab session, learned into match-day estimates"),
    ("Eye tracking", "where and when the batter watches the ball"),
    ("Exact bat speed and path", "bat sensors in the lab and a bat-keypoint model on match video"),
    ("Player-specific baselines", "technique health compared with the player's own best, across matches, matchups and workload"),
]


def _e(x) -> str:
    return html.escape(str(x))


# ------------------------------------------------------------------ insights
def sequence_insight(K: Kinematics) -> dict:
    seq = K.seq
    L = C.SHORT_LABELS
    issues, tips = [], []
    for l in seq["links"]:
        a, b = L[l["from"]], L[l["to"]]
        if l["status"] == "reversed":
            issues.append(f"your {b.lower()} peaked {abs(l['dt_ms']):.0f} ms before your {a.lower()} (out of order)")
        elif l["status"] == "simultaneous":
            issues.append(f"{a.lower()} and {b.lower()} peaked together (within one frame)")
        if l["gain"] < 1.0:
            issues.append(f"speed dropped from {a.lower()} to {b.lower()} (×{l['gain']:.2f}) - a power leak at that link")
        if not l["braked"] and l["status"] == "in order":
            issues.append(f"your {a.lower()} was still at {l['brake_ratio'] * 100:.0f}% of its peak when the "
                          f"{b.lower()} peaked - it didn't slow down to pass its speed on")
    gaps = [l["dt_ms"] for l in seq["links"]]
    if seq["ideal_order"] and not issues:
        what = (f"Your segments fired in the textbook order - pelvis, torso, lead arm, then hands - "
                f"with gaps of {', '.join(f'{g:.0f}' for g in gaps)} ms, and each one sped up the next.")
        status = "good"
        tips.append("This is the pattern to protect. Re-test it when tired, or under pressure, to see if it holds.")
    else:
        what = ("Firing order: " + " → ".join(L[s] for s in seq["order"]) + ". " +
                " ".join(i[0].upper() + i[1:] + "." for i in issues[:3]))
        status = "work" if any(l["status"] == "reversed" for l in seq["links"]) else "watch"
        if any(l["status"] == "reversed" and l["from"] == "pelvis" for l in seq["links"]):
            tips.append("Hips first: step-and-drive drill or rotational medicine-ball throws where the hips start before the chest.")
        if any(l["gain"] < 1 for l in seq["links"]):
            tips.append("Find the leaking link above and slow the drill down until that segment clearly speeds up the next one.")
        if any(not l["braked"] for l in seq["links"]):
            tips.append("Firm the front leg so the hips can stop and 'whip' the energy up the chain.")
        if not tips:
            tips.append("Rhythm drill: 'hips - chest - arms - bat' in slow motion, then build speed while keeping the order.")
    return {"status": status, "what": what,
            "why": "In an efficient swing each body segment reaches its peak speed just after the one below it, "
                   "from the inside out, so speed is amplified link by link (golf measures this with K-Vest). "
                   "Out-of-order peaks or speed drops are where power leaks.",
            "try": " ".join(tips)}


def headline(K: Kinematics, factors: list[dict], seqi: dict) -> str:
    good = [f["name"].lower() for f in factors if f["status"] == "good" and f["confidence"] != "low"][:2]
    work = [f["name"].lower() for f in sorted(factors, key=lambda f: {"work": 0, "watch": 1}.get(f["status"], 9))
            if f["status"] in ("work", "watch")][:2]
    parts = [("Your kinematic sequence looks efficient. " if seqi["status"] == "good" else
              "Your kinematic sequence shows where power leaks. ")]
    if good:
        parts.append(f"Strengths to protect: {' and '.join(good)}. ")
    if work:
        parts.append(f"Biggest opportunities: {' and '.join(work)}. ")
    parts.append("All values are estimates from one 2D video.")
    return "".join(parts)


def maybe_polish(text: str, facts: dict) -> str:
    """Optionally re-phrase the headline with Gemini (free tier). Never adds facts."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return text
    try:
        from google import genai  # optional dependency
        client = genai.Client(api_key=key)
        prompt = ("Rewrite this cricket batting coach summary in warm, plain English, 2-3 sentences. "
                  "Use ONLY the facts given; do not add numbers, causes or advice that are not present. "
                  "Keep the note that values are estimates from one 2D video.\n\n"
                  f"SUMMARY: {text}\nFACTS (JSON): {json.dumps(facts)[:4000]}")
        out = client.models.generate_content(model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                                             contents=prompt).text
        return out.strip() if out and len(out) < 900 else text
    except Exception:
        return text


# ------------------------------------------------------------------ HTML
def summary_html(K: Kinematics, factors: list[dict]) -> str:
    seq = K.seq
    score = seq["score"]
    cls = "ok" if score >= 75 else "meh" if score >= 50 else "bad"
    verdict = ("Textbook inside-out sequence" if seq["ideal_order"] else
               "Out-of-order sequence - power leak" if any(l["status"] == "reversed" for l in seq["links"]) else
               "Sequence partly separated")
    chips = []
    for i, seg in enumerate(seq["order"]):
        pk = seq["peaks"][seg]
        chips.append(f'<div class="chip" style="background:{C.SEGMENT_HEX[seg]}">{i + 1}. {_e(C.SEGMENT_LABELS[seg])}'
                     f'<span>{pk["deg_s"]:.0f}°/s · {pk["t_ms"]:+.0f} ms</span></div>')
        if i < len(seq["order"]) - 1:
            nxt = seq["order"][i + 1]
            dt = seq["peaks"][nxt]["t_ms"] - pk["t_ms"]
            gain = seq["peaks"][nxt]["deg_s"] / max(pk["deg_s"], 1e-6)
            ok = C.SEGMENTS.index(nxt) == C.SEGMENTS.index(seg) + 1 and dt >= 0.5 * seq["frame_ms"]
            chips.append(f'<div class="link"><b class="{"ok" if ok else "bad"}">→</b>+{dt:.0f} ms<br>×{gain:.2f}</div>')
    ph = K.phases
    xf = max((f["metrics"].get("x_factor_max_deg", 0) for f in factors if f["key"] == "x_factor"), default=0)
    tiles = [
        (f"{(ph['impact'] - ph['top']) / K.fs * 1000:.0f} ms", "Downswing (top → impact)"),
        (f"{ph['peak_hand_speed']:.1f} m/s", "Peak hand speed (rel. to pelvis)"),
        (f"{xf:.0f}°", "Max hip–shoulder separation"),
        (f"{1000 / K.fs:.1f} ms", f"Timing resolution ({K.fs:.0f} fps)"),
        (K.handed.split("-")[0], f"Batter ({K.hand_source})"),
        (f"±{K.consistency.get('overall_cv_pct', float('nan')):.0f}%", "Bone-length wobble (tracking quality)"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="v">{_e(v)}</div><div class="k">{_e(k)}</div></div>'
                         for v, k in tiles)
    warn = ""
    if K.warnings:
        warn = '<div class="warn"><b>Notes on this clip</b><ul>' + "".join(f"<li>{_e(w)}</li>" for w in K.warnings) + "</ul></div>"
    return f"""<div class="ksd">
<div class="hero">
  <div><div class="score {cls}">{score}<small> /100</small></div><div class="sub">sequence score (heuristic)</div></div>
  <div style="flex:1;min-width:240px"><div class="verdict">{_e(verdict)}</div>
  <div class="sub">Peak rotation speed of each segment, in the order they fired (times relative to estimated impact).</div>
  <div class="chain">{''.join(chips)}</div></div>
</div>
<div class="tiles">{tiles_html}</div>{warn}
</div>"""


def _card(name, status, conf, what, why, evidence, try_, value=""):
    ev = (f'{_e(evidence["label"])} · {evidence["t_ms"]:+.0f} ms from impact - see the evidence frames and replay'
          if evidence else "")
    val = f' <span class="sub">({_e(value)})</span>' if value else ""
    conf_html = f' <span class="conf">{_e(conf)} confidence</span>' if conf else ""
    return f"""<div class="card {status}"><div class="t">{_e(name)} <span class="pill {status}">{STATUS_TXT[status]}</span>{conf_html}{val}</div>
<dl><dt>What happened</dt><dd>{_e(what)}</dd><dt>Why it matters</dt><dd>{_e(why)}</dd>
{f'<dt>Evidence</dt><dd>{ev}</dd>' if ev else ''}<dt>What to try</dt><dd>{_e(try_)}</dd></dl></div>"""


def report_html(K: Kinematics, factors: list[dict], seqi: dict, head: str) -> str:
    def ev(f):
        e = dict(f["evidence"])
        e["t_ms"] = (_frame_of(K, e["phase"]) - K.phases["impact"]) / K.fs * 1000
        return e

    rank = {"work": 0, "watch": 1}
    workons = sorted([f for f in factors if f["status"] in rank],
                     key=lambda f: (rank[f["status"]], {"high": 0, "medium": 1, "low": 2}[f["confidence"]]))[:3]
    strengths = [f for f in factors if f["status"] == "good" and f["confidence"] in ("high", "medium")][:3]
    seq_card = _card("Kinematic sequence", seqi["status"], "", seqi["what"], seqi["why"],
                     {"label": "Downswing", "t_ms": (K.phases["top"] - K.phases["impact"]) / K.fs * 1000},
                     seqi["try"])
    parts = [f'<div class="ksd"><h3>Coach summary</h3><p>{_e(head)}</p>', seq_card]
    if workons:
        parts.append("<h3>Work-ons (biggest first)</h3>")
        parts += [_card(f["name"], f["status"], f["confidence"], f["what"], f["why"], ev(f), f["try"], f["value"]) for f in workons]
    if strengths:
        parts.append("<h3>Strengths to protect</h3>")
        parts += [_card(f["name"], f["status"], f["confidence"], f["what"], f["why"], ev(f), f["try"], f["value"]) for f in strengths]
    rows = "".join(
        f'<tr><td><b>{_e(f["name"])}</b></td><td>{_e(f["value"])}</td>'
        f'<td><span class="pill {f["status"]}">{STATUS_TXT[f["status"]]}</span></td>'
        f'<td>{_e(f["confidence"])}</td><td class="sub">{_e(f["camera_note"])}</td></tr>' for f in factors)
    parts.append(f"""<h3>All batting factors</h3>
<table><tr><th>Factor</th><th>Estimate</th><th>Verdict</th><th>Confidence</th><th>What one camera can / can't see</th></tr>{rows}</table>""")
    tw = K.twin
    twin_txt = (f"Skeleton calibrated to {_e(tw['source'])} (scale ×{tw['scale']:.2f}, height {tw['height_m'] * 100:.0f} cm)."
                if tw["used"] else
                f"No body measurements given: MediaPipe's default body scale was used (estimated height ≈ {tw['height_m'] * 100:.0f} cm). "
                "Add your measurements under 'Digital twin lite' for calibrated distances and speeds.")
    parts.append(f"""<h3>Honesty box</h3><div class="honest">
<p><b>These are estimates from a single 2D video.</b> Angles and speeds come from Google MediaPipe pose landmarks,
filtered with a zero-lag low-pass filter; depth is refined using constant bone lengths. Verdict thresholds are heuristics,
not elite norms. {twin_txt}</p>
<p><b>What one camera can't measure well</b></p><ul>{''.join(f'<li><b>{_e(a)}:</b> {_e(b)}</li>' for a, b in CANNOT_MEASURE)}</ul>
<p style="margin-top:10px"><b>What the full system adds</b></p><ul>{''.join(f'<li><b>{_e(a)}:</b> {_e(b)}</li>' for a, b in FULL_SYSTEM_ADDS)}</ul>
</div></div>""")
    return "\n".join(parts)


def _frame_of(K, phase):
    ph = K.phases
    return {"setup": ph["setup"][1], "top": ph["top"],
            "plant": ph["plant"] if ph["plant"] is not None else ph["top"],
            "impact": ph["impact"], "follow": ph["follow_end"]}[phase]


# ------------------------------------------------------------------ exports
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 3)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def metrics_json(K: Kinematics, factors: list[dict], seqi: dict, out_dir: str) -> str:
    data = {
        "disclaimer": "Estimates from a single 2D video. Thresholds are heuristic.",
        "batter": {"handedness": K.handed, "source": K.hand_source},
        "capture": {"effective_fps": K.fs, "frame_ms": 1000 / K.fs},
        "phases_ms_from_impact": {k: ((v - K.phases["impact"]) / K.fs * 1000 if isinstance(v, (int, np.integer)) else
                                      [(x - K.phases["impact"]) / K.fs * 1000 for x in v] if isinstance(v, tuple) else v)
                                  for k, v in K.phases.items() if k != "peak_hand_speed"},
        "peak_hand_speed_m_s": K.phases["peak_hand_speed"],
        "kinematic_sequence": {"order": K.seq["order"], "score": K.seq["score"], "ideal_order": K.seq["ideal_order"],
                               "peaks": K.seq["peaks"], "links": K.seq["links"], "insight": seqi},
        "factors": [{k: f[k] for k in ("key", "name", "value", "status", "confidence", "metrics", "what", "try")}
                    for f in factors],
        "digital_twin": K.twin, "tracking_consistency": K.consistency, "notes": K.warnings,
    }
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(data), f, indent=2)
    return path


def standalone_html(summary: str, report: str, chart_png: str, evidence_png: str, out_dir: str) -> str:
    def b64(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Batting Kinematic Sequence Report</title><style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
max-width:1100px;margin:24px auto;padding:0 16px;background:#fafafa}} img{{max-width:100%;border-radius:10px}}{CSS}</style></head>
<body><h2 style="margin-bottom:4px">Cricket batting · kinematic sequence report</h2>
<p style="color:#666;margin-top:0">Single-camera proof of concept · all values are estimates from one 2D video</p>
{summary}<h3 style="font-family:inherit">Kinematic sequence</h3><img src="data:image/png;base64,{b64(chart_png)}">
<h3>Evidence frames</h3><img src="data:image/png;base64,{b64(evidence_png)}">{report}</body></html>"""
    path = os.path.join(out_dir, "batting_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
