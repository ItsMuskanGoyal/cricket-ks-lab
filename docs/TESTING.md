# Testing plan

## 1. Self-tests (no video, run any time)
```
python tests/test_synthetic.py      # 5 checks: order recovery, faulty sequence flagged, handedness, knee collapse, full pipeline
python analyze_cli.py --synthetic   # writes a full set of outputs for a scripted batter into ./results
```

## 2. Real clips — use footage you own
For a pitch to Google, record your own clips (or teammates' with permission) rather than downloading broadcast
footage, which is usually copyrighted.

**How to film**
- Phone on a tripod (or held very still) **behind the bowler's arm**, about 10–15 m away, roughly chest height.
- Whole batter in frame, head to feet, with some space above the bat.
- Use **slow-motion** (120 or 240 fps) if your phone has it. Good light helps the pose model.
- Trim each clip to one shot: 1 s before the backlift → 1 s after the follow-through.

**The five-clip test set**
| # | Clip | What it checks |
|---|---|---|
| 1 | Your best slow-motion straight/cover drive | Hero case: clean sequence, front-foot plant found |
| 2 | Same shot at normal 30 fps | App warns about timing resolution; order may blur |
| 3 | A left-handed batter | Auto-detect picks *Left-handed* |
| 4 | A back-foot pull or cut | "No clear stride" handled gracefully |
| 5 | A clip with other people in frame (keeper, umpire, net bowler) | Skeleton stays on the batter |

**For each clip, check and note in a table**
1. Skeleton sits on the batter in every frame of the replay (no jumping to another person).
2. Phase labels look right: scrub the replay — *DOWNSWING* starts when the bat starts coming down; *IMPACT (est.)* is within a couple of frames of bat-ball contact.
3. The firing order is plausible for what you see; on a deliberately "arms-only" swing it should get worse.
4. Each report sentence matches the video (e.g. if it says the head moved 10 cm sideways, can you see it?).
5. Bone-length wobble below ~12 % (otherwise confidence is lowered automatically).
6. Re-run the same clip: identical results (the pipeline is deterministic).

**Consistency check (strong evidence for engineers):** record 5 repetitions of the same drive. Peak times and
the order should be similar across repetitions. Then do 5 deliberately "arms-first" swings: the score should drop.
