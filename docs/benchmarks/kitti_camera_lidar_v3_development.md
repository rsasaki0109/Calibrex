# KITTI camera-LiDAR v3 development

**Status: development incomplete.** This page records a post hoc development
proposal after [v2 was refuted, 8 of 9 gates](kitti_camera_lidar_v2_check.md).
It is not a pre-registration or a new audit. The v2 verdict and thresholds stay
as published. No new evaluation sensor data was read for this work.

## Proposed far-start adoption rule

`EdgeAlignmentOptions(start_scan_min_control_t=3.5)` adds a control margin to
the profile-start adoption rule. A far optimum must still observe more rotation
axes than the current fit, at least two by default, with no rotation at the search
bound. In addition, **every rotation axis it would report** must have a finite
held-out control `t >= 3.5`. This includes axes already observed at the current
fit. An axis that fails the ordinary observability rule does not veto adoption.

Rejection retains the current transform, jackknife and controls. It does not
remove axes from the rejected transform and report that transform. Fits started
at the candidate keep the ordinary `t >= 2.5` rule. The extra margin is recorded
in the artifact's `options` and thus participates in cache keys.

The new option defaults to `None`, which preserves v2 behavior. **3.5 is a
proposal to compare on spent data, not a validated default.** It is currently
available through the native Python estimator API; `calibrex check` continues
to construct the default options. For example, a development replay can pass
`options=EdgeAlignmentOptions(start_scan_min_control_t=3.5)` to
`estimate_from_frames` or `run_ros2_camera_lidar_edge`. The v2 frozen driver,
scorer and pre-registration are unchanged.

Synthetic regressions cover the original count and bound gates, all reported
axes including an already observed weak axis, the inclusive 3.5 boundary,
unobserved axes, missing/nonfinite controls, invalid margins and retention of
the current transform and evidence. These establish the selection behavior;
they do not measure real-data accuracy or coverage.

## Current rule on all 55 spent KITTI raw drives (post hoc)

The committed v2 development scores contain 37 drives, and its evaluation scores
contain 18. All 55 are now spent development data. The following is a descriptive
aggregation of those existing results, not a new evaluation or a change to v2's
verdict. Each drive uses its own day's published calibration as the reference,
not ground truth.

| Measure | Current v2 rule | Proposed 3.5 margin |
| --- | --- | --- |
| Scored drives | 52/55; 3 inconclusive, 0 failed | not replayed |
| KITTI baselines flagged | 0/52 scored drives | unknown |
| Maximum judged baseline axis error | 0.45000000000000573 deg | unknown |
| Maximum judged turned-run axis error | 1.502977473509024 deg | unknown |
| Baseline judged axes, scored drives (v2 denominator) | 125/156 = 0.8012820512820513 | unknown |
| Baseline judged axes, all attempted drives | 125/165 = 0.7575757575757576 | unknown |
| 3 deg turns flagged on baseline-judged axes | 122/125 = 0.976 | unknown |
| 1 deg turns flagged on baseline-judged axes | 125/125 = 1.0 | unknown |
| 3 deg turns judged right | 0/125 | unknown |
| Change in judged-axis rate, percentage points | reference | unknown |

The three inconclusive drives are `2011_09_28_0016`, `2011_09_28_0021` and
`2011_09_29_0026`. The scored-drive denominator excludes their axes, as v2 did;
the all-drive rate above shows their cost as well.

The reproducible inventory is
[`development_inventory.json`](../assets/kitti_camera_lidar_v3/development_inventory.json).
It records input SHA-256 digests, generator/estimator hashes and the Git commit.
Regenerate it from this checkout with:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python tools/kitti_camera_lidar_v3_development.py \
  --output docs/assets/kitti_camera_lidar_v3/development_inventory.json
```

The tool reads only the committed v2 scores and their corresponding saved final
evidence. It neither downloads recordings nor runs the estimator. Proposed-rule
metrics and coverage change are explicitly `null`.

## Saved far fits (post hoc diagnostic only)

Of 385 old runs (55 drives times seven), the repository contains final estimator
evidence for **133**: all 126 v2 evaluation runs and seven development runs of
`2011_09_26_0005`. Evidence for the other **252** development runs is absent.
Among the available evidence, 11 selected final fits came from profile starts;
six fail the proposed margin:

| Drive | Runs failing the 3.5 margin | Weak reported axes |
| --- | --- | --- |
| 2011_09_28_0002 | roll +3, pitch +3, yaw +3, pitch +1 | roll and pitch |
| 2011_09_30_0034 | roll +3, yaw +3 | roll |

On the v2 gate-failing yaw +3 run of `2011_09_28_0002`, roll's control is
`2.6493418318574165` and pitch's is `3.252264718453982`. On `2011_09_30_0034`,
roll's control is `2.8402831777931024` for roll +3 and `2.7521510201550727` for
yaw +3. Thus the margin would also reject saved fits beyond the known near miss.
Its effect on detection and coverage must be measured rather than assumed.

This screen says only whether those saved final fits clear the extra margin.
It does not contain the candidate-start fit, rejected alternatives, images,
clouds or estimator input caches. Rejecting a final fit cannot reveal which fit
a new run would select. Even a saved fit that clears the margin need not be the
new final fit because earlier adoption decisions can change. Consequently this
screen cannot supply new-rule flags, errors, coverage, detection rates or audit
thresholds.

## Work still required before pre-registration

The official S3 host is still blocked by this environment (HTTP 403 on a
metadata-only HEAD for a spent KITTI raw archive on 2026-10-15). No local replay
inputs were found. Access findings and the KITTI-360 exposure correction are in
[PR #235](https://github.com/rsasaki0109/Calibrex/pull/235); the synthetic-tested
KITTI-360 converter is in
[draft PR #236](https://github.com/rsasaki0109/Calibrex/pull/236).

Before step 3 can finish, regenerate estimator inputs from the 55 spent raw
drives, compare both rules on the same inputs, then confirm via the real
driver. Record every measure in the table above, including coverage loss and
missed turns. Confirm the KITTI-360 converter on spent 0000/0002 data.

Only then derive the v3 gates from the new development results using the
[handoff rules](../sota_handoff.md#how-an-audit-is-run). Evaluation archive pins,
disjoint 60 s interval selection and the preregistration remain pending metadata
access and an exposure audit. KITTI-360 0003 was already read by earlier research
and must be excluded, as documented in PR #235. No v3 thresholds are frozen here,
and no evaluation is authorized by this development page.
