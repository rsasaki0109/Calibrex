# KITTI camera-LiDAR v3: cross-start agreement rule (development protocol)

**Status: frozen before any replay of this rule on real inputs; screened once,
[not carried forward](kitti_camera_lidar_v3_start_agreement_results.md).** This page
fixes one opt-in rule, how it is screened on the 55 spent KITTI raw drives and
what the result can and cannot decide. It is development, not a pre-registration
or an audit. No evaluation data is read.

## Why this rule

On all 55 spent drives, the [current rule](kitti_camera_lidar_v3_development.md#all-55-spent-drives-captured-and-replayed-2026-10-31)
judges ten turned-run axes more than 0.5 deg off. The 2011_09_28_0002
[start trace](kitti_camera_lidar_v3_development.md#why-a-scalar-adoption-margin-is-insufficient-on-these-probes)
shows the mechanism behind the largest errors. A single-axis fit from one
profile start and a multi-axis fit from another are both evidenced. They agree
on pitch (−2.85 and −3.05 deg) but differ on roll by 1.55 deg, and the adopted
multi-axis fit reports that roll. No scalar adoption margin separates the two
cases. The 3.5 margin also moves an error to 0036.

## The rule

`EdgeAlignmentOptions(start_agreement_deg=0.5)`. Every search start whose
optimum observes at least one rotation DoF yields an **evidenced** optimum:
the candidate start and each distinct profile start. The adoption rule and the
final fit are unchanged. A rotation axis the final fit would report is
**withheld**, recorded as unobservable with a reason, when any other evidenced
optimum differs from the final fit on that axis by more than 0.5 deg. All
optima are corrections relative to the same candidate. An optimum that observes
no rotation DoF carries no evidence and is not compared.

0.5 deg is the check's existing observable rotation std bound
(`observable_rotation_std_deg`). It is not chosen from any result: two
supported optima farther apart than the std bound mean the axis is not
determined to that bound. The option defaults to `None`, which gives
bit-identical v2 behavior and is omitted from recorded options, so existing
captures and caches stay valid. A unit test reproduces the 0002 pitch +3
trace's expected effect. That traced effect motivated this rule, so 0002 is
not independent evidence for it.

## Screen

1. **Cached inputs, 23 drives.** Replay the current rule and the new rule on
   the 23 drives whose captured NPZ inputs remain. The current-rule replay must
   match the captured evidence (`1e-9`).
2. **Re-capture, 32 drives.** Run `tools/kitti_camera_lidar_v3_batch.py` with
   `--proposed-options '{"start_agreement_deg": 0.5}'` on the other 32 drives,
   using the existing pins and retry rules.
3. **Score all 55.** `tools/kitti_camera_lidar_v3_rule_screen.py` takes every
   spent drive exactly once and applies the criteria below.

The rule is **carried forward** to drafting a v3 pre-registration only if
every check holds on all 55 drives. The values are v2's gate values, fixed here
before the replay:

| Check | Requirement for the proposed rule |
| --- | --- |
| Baseline not flagged | no scored baseline is warn or fail |
| Baseline judged accuracy | every judged baseline axis within 0.75 deg |
| Turned judged accuracy | every judged turned-run axis within 1.5 deg |
| 3 deg turns flagged | at most 10 % of counted 3 deg trials missed |
| 1 deg turns flagged | at most 25 % of counted 1 deg trials missed |
| Judged-axis coverage | at least half of the scored baseline axes judged |
| Scored-drive coverage | at least two thirds of the 55 drives scored |
| Fewer large errors | fewer turned-run judged axes beyond 0.5 deg than the current rule (10) |
| No new large errors | none beyond 0.5 deg that the current rule did not have |
| No failed drives | no capture, parity or scoring failure |

Otherwise it is **not carried forward**. No tolerance, criterion or rule
detail changes after the screen. A different rule needs a new protocol.
Failures and every per-drive result are published either way, as before.

## Limits

All 55 drives are spent: they were used to motivate this rule and they include
v2's evaluation drives. A pass is development evidence only. It neither
supports a SOTA claim nor replaces a pre-registered test on unread recordings
(KITTI-360 after its exposure audit). Translation is still not judged. Running
the rule natively through `calibrex check` would need its CLI and schema wiring
and a real-driver confirmation, both deferred until the rule is carried
forward.

```bash
PYTHONPATH=src:. .venv/bin/python -c "from pathlib import Path; \
from tools import kitti_camera_lidar_v3_replay as r; \
r.compare_drives(Path('CACHE'), Path('OUT/cached'), drives=[...], \
proposed={'start_agreement_deg': 0.5})"
PYTHONPATH=src:. .venv/bin/python tools/kitti_camera_lidar_v3_batch.py run \
  docs/assets/kitti_camera_lidar_v3/spent_batch_20261031/pins.json WORK \
  --proposed-options '{"start_agreement_deg": 0.5}' --drive ...
PYTHONPATH=src:. .venv/bin/python tools/kitti_camera_lidar_v3_rule_screen.py \
  SCREEN.json OUT/cached/comparison.json LATEST_ATTEMPT_COMPARISONS...
```
