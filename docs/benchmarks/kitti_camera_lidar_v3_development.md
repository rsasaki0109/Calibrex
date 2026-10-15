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
proposal to compare on spent data, not a validated default.** The native Python
API accepts `options=EdgeAlignmentOptions(start_scan_min_control_t=3.5)` in
`estimate_from_frames` or `run_ros2_camera_lidar_edge`. The same opt-in is
available in `calibrex check`, `estimate` and `drift`:

```bash
calibrex check BAG --pairs camera-lidar --camera-lidar-far-control-t 3.5 \
  --no-cache --fail-on never --output confirm/check.json
```

All three commands record `camera_lidar_start_scan_min_control_t` in their
result options and `start_scan_min_control_t` in estimator evidence. Cache keys
distinguish the new rule from the default. The CLI and result models reject
nonfinite values or values below the ordinary control threshold of 2.5. The
committed check, estimate and drift schemas include the optional field; older
artifacts remain valid without it. The v2 frozen driver, scorer and
pre-registration are unchanged.

Synthetic regressions cover the original count and bound gates, all reported
axes including an already observed weak axis, the inclusive 3.5 boundary,
unobserved axes, missing/nonfinite controls, invalid margins and retention of
the current transform and evidence. CLI regressions also cover all three result
schemas, early rejection of invalid values and cache separation, including
switching back to the default rule. These establish the selection behavior;
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

## Capture and paired replay on spent data

`tools/kitti_camera_lidar_v3_replay.py` rebuilds the development workflow without
reading any fresh evaluation recording. `capture` accepts only the fixed 55 spent
raw drive IDs from the committed v2 scores. Before opening sensor streams it
requires a KITTI raw conversion sidecar for that same single drive, `image_02`
and azimuth point times. Each actual candidate must match the sidecar's published
reference transform, or that reference with the declared 1 or 3 deg turn.

Capture runs the real `calibrex check` seven times with the current defaults and
cache disabled. It saves a **separate input for every run**, after synchronization
and motion correction but before edge preparation. A rotated candidate can change
visible-point timing and therefore pairing and corrected points; the baseline's
frames are not reused for its six injections. Images, points and per-point times
are stored losslessly in NPZ with a versioned JSON manifest, full payload SHA-256,
all option values, camera model, motion summary and original provenance. Loading
uses `allow_pickle=False`, verifies array types/shapes and refuses implicit option
schema migrations.

For one already converted spent bag:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python tools/kitti_camera_lidar_v3_replay.py capture BAG \
  --drive 2011_09_26_0005 --output /tmp/kitti-v3-inputs/2011_09_26_0005
```

The capture directory retains check outputs, evidence, logs, commands, elapsed
times, conversion provenance and a digest manifest. A solver exception is a
capture failure even when `--fail-on never` returns 0: the estimator must also
produce its evidence. Failed inputs/attempts are retained, and existing output
directories are never overwritten. A failed or interrupted capture stays
incomplete. These are development reruns; this tool is not the once-only v3
evaluation driver.

After captures exist for all 55 spent drives:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python tools/kitti_camera_lidar_v3_replay.py compare \
  /tmp/kitti-v3-inputs --output /tmp/kitti-v3-comparison-3p5 --min-far-control-t 3.5
```

Replay does not open the bag or use the network. Both rules receive identical
read-only input arrays and explicit captured settings; only the far-start margin
differs. It requires the default check recipe, checks every manifest digest and
calibration/injection identity, and verifies that the current-rule replay matches
the captured numerical evidence (absolute tolerance `1e-9`) and the real check's
axis verdicts. A parity failure, missing capture or corrupt input remains a failed
drive in **both** rule summaries. It is never silently dropped.

`comparison.json` reports both rules' flags, maximum errors, judged-axis rates,
1/3 deg detection rates and coverage change in percentage points. It reuses v2's
per-drive metric definitions and the native check's verdict rule, without invoking
v2's audit gates. `audit_thresholds` stays `null`. A `--drive DATE_ID` subset is
allowed for development but is always incomplete; exit 0 requires all 55 drives
with no failed captures, parity checks or drive scores. It never means
`supported`. A completed development comparison still needs real-driver
confirmation before v3 thresholds can be selected.

Tests use synthetic data only. They exercise real seven-check capture on a tiny
converted raw fixture, replay after deleting its bag, a converged synthetic scene
with all six actual injections, exact image/point/time round trips, corruption,
recipe migration, parity mismatches and preservation of solver/missing-input
failures. No synthetic metrics are used as the 55-drive development measurements.

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
