# Method — how the estimates are made

Written for engineers reviewing the demo. Every constant mentioned lives in `ksd/config.py`.

## 1. Pose
- **Google MediaPipe Pose Landmarker** (Tasks API, `full` model, VIDEO mode, temporal tracking).
- Up to 3 people per frame; the batter is chosen on first sighting (largest, most central) and then tracked by
  nearest hip centre, so umpires, keepers and non-strikers are ignored.
- Frames resized to ≤960 px; clips capped at 15 s / 900 frames.
- Landmarks with visibility < 0.35 are treated as missing and linearly interpolated.

## 2. Filtering
4th-order **zero-lag Butterworth** (filtfilt) on all coordinates, cutoff `min(12 Hz, 0.33·fs)`. The angular-speed
curves get a second, segment-specific low-pass (pelvis 7 Hz, torso 8, arm 10, hands 14) — heavy proximal
segments move at lower frequencies. Time uses the **true capture rate** (`fps × slow-motion factor`).

## 3. Hybrid 2.5D coordinates
- **x, y:** MediaPipe's 2D image landmarks (precise), converted to metres with a per-clip pixels-per-metre factor
  (90th percentile of image-length / 3D-length over torso and leg bones — the least foreshortened moments).
- **z (depth):** MediaPipe's learned 3D world landmarks.
- Everything is expressed relative to the hip centre, so camera pans don't move the numbers.

## 4. Rigid-segment depth refinement (digital twin in miniature)
Front-on, a side-on batter's hip and shoulder lines point almost straight at the camera, so their rotation is
mostly *depth* — the weakest signal from one camera. Bones keep a constant length L, so

  |z| = √(L² − x² − y²)

where x, y come from the precise 2D image. The learned depth supplies only the sign (towards/away), and the
magnitude only near square-on where the constraint is ill-conditioned (weight `(xy/L)⁴`). Applied to the pelvis
line, shoulder line and lead upper arm.

## 5. Segment speeds
| Segment | Vector | Speed |
|---|---|---|
| Pelvis | trail hip → lead hip | yaw rate about the vertical axis |
| Torso | trail shoulder → lead shoulder | yaw rate about the vertical axis |
| Lead arm | lead shoulder → lead elbow | 3D angular speed `|u × du/dt|` |
| Hands (bat proxy) | wrists → knuckles (both hands) | 3D angular speed — the hands are rigidly gripped to the handle |

## 6. Swing phases (automatic)
From hand speed relative to the pelvis: **impact (est.)** = peak hand speed; **top of backlift** = the slow
point just before the downswing (walk back below 25 % of peak, then to the local minimum); **setup** = quiet
period before the backlift; **follow-through** = hands fall below 20 % of peak, +0.25 s; **front-foot plant** =
lead foot moves then settles (< 30 % of its peak speed for 2 frames) before impact. No clear stride → treated as a
back-foot shot.

## 7. Kinematic sequence
- Peak of each segment in a window from just before the top of backlift to shortly after impact, located with
  **sub-frame parabolic interpolation**.
- For each link (pelvis→torso→arm→hands): timing gap and status (*in order*, *simultaneous* = within half a
  frame, *reversed*), **speed gain** (next peak / this peak; < 1 = power leak) and **braking** (this segment's
  speed when the next one peaks; > 85 % of its own peak = it didn't slow down to pass energy on).
- **Sequence score (heuristic):** 40 order + 30 gain + 30 braking.

## 8. Batting factors
Stance & balance, trigger, stride, front-leg bracing, head stability, hip rotation, hip–shoulder separation
(X-factor), hip–shoulder alignment, driving from the shoulder vs the wrist, elbow extension, wrist cock, swing
timing. Each has a status (strength / watch / work on / baseline), a **confidence** that depends on how well one
front-on camera sees it (and drops when tracking quality is poor), and a note on what the camera can't see.
Thresholds are heuristic and editable in `config.py` — calibrating them on elite players is the job of the full system.

## 9. Digital twin lite
Height (and optionally limb lengths) → scale factor from MediaPipe's body to the player's real body (Winter
segment ratios when only height is given). Scales distances and hand speed. Independently, the app reports
**bone-length wobble** — the coefficient of variation of each bone's length over the swing. Real bones don't
stretch, so this is a direct, honest measure of tracking quality.

## 10. Validation so far (synthetic)
A scripted 3D batter with known segment peak times is projected through a front-on pinhole camera, with
noise added (`tests/test_synthetic.py`, `ksd/synthetic.py`). 10 runs per cell:

| Noise (depth / pixels) | 30 fps | 60 fps | 120 fps | 240 fps |
|---|---|---|---|---|
| Moderate (1.5 cm / 1.2 px): correct order | 9/10 | 10/10 | 10/10 | 10/10 |
| Heavy (3 cm / 2 px): correct order | 4/10 | 10/10 | 7/10 | 9/10 |
| Reversed pelvis–torso swing never reported as ideal | 10/10 | 10/10 | 10/10 | 10/10 |

Read honestly: the maths recovers the sequence when tracking is decent, and 30 fps with poor tracking is not
enough — which is exactly why the full system uses high-speed, multi-angle capture. **Next validation step:**
compare against a marker-based or IMU capture of the same swings.

## 11. Known limitations
- Single 2D camera: depth is learned, not measured; rotation numbers are estimates.
- The bat is not tracked; hands are the proxy. Impact is estimated from peak hand speed, not ball contact.
- Hands gripping a bat are often occluded → wrist metrics are low confidence.
- 30 fps = 33 ms per frame, comparable to the gaps between peaks.
- Thresholds are heuristics, not elite norms. One shot, not a player's history.
- Pose models are trained mostly on everyday poses; extreme batting positions and pads/helmets can confuse them.

## 12. Roadmap to the full system
Bat keypoint model → multi-camera triangulation → per-player digital twin from a lab session (body scan,
markers, force plates, eye tracking, bat sensors) → player-specific baselines, matchups, pressure patterns and workload flags.
