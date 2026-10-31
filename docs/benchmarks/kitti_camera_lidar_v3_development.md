# KITTI camera-LiDAR v3 development

**Status: development incomplete.** This page records a post hoc development
proposal after [v2 was refuted, 8 of 9 gates](kitti_camera_lidar_v2_check.md).
It is not a pre-registration or a new audit. The v2 verdict and thresholds stay
as published. No new evaluation sensor data was read for this work.

Real capture/replay progress is **55/55 spent drives** after the
[2026-10-31 batch](#all-55-spent-drives-captured-and-replayed-2026-10-31). The
current-rule replay reproduces v2's committed aggregate exactly. The proposed 3.5
margin loses eight 3 deg detections and still judges one turned axis 1.35 deg
off. It is **not adopted as a default**: it stays an opt-in candidate, with no new
claim or default change.

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

## All 55 spent drives captured and replayed, 2026-10-31

`tools/kitti_camera_lidar_v3_batch.py` ran the existing capture and single-drive
comparison on every spent drive in one cloud session (Linux, Python 3.12.3, four
CPUs). Archive size, ETag, frame counts and each day's calibration SHA-256 were
pinned in [`pins.json`](../assets/kitti_camera_lidar_v3/spent_batch_20261031/pins.json)
before any payload was read. They match the 0005 and 0002 pilot pins. Drives
with known problems ran first, so that their inputs would stay cached. This is
development on spent data, not an audit.

The [summary](../assets/kitti_camera_lidar_v3/spent_batch_20261031/summary.json)
is `complete`, with no failed captures, parity checks or drive scores.
The current-rule replay equals the committed v2 aggregate on every
measure in the table above.

| Measure, 55 spent drives | Current rule | Proposed 3.5 margin |
| --- | ---: | ---: |
| Scored drives | 52 (3 inconclusive) | 52 (same 3) |
| KITTI baselines flagged | 0 | 0 |
| Maximum judged baseline axis error | 0.450 deg | 0.450 deg |
| Maximum judged turned-run axis error | 1.503 deg (0002) | **1.350 deg (0036 yaw +3)** |
| Baseline judged axes, scored drives | 125/156 = 80.1 % | 125/156 = 80.1 % |
| 3 deg turns flagged on baseline-judged axes | 122/125 | **114/125** |
| 1 deg turns flagged on baseline-judged axes | 125/125 | 125/125 |
| Turned-run judged axes more than 0.5 deg off | 10 in 6 drives | 5 in 5 drives |

The margin removes the four 0002 roll errors (up to 1.503 deg) and the two
0057 pitch +3 errors. It also creates a new 1.350 deg yaw error on 0036 yaw +3,
where it rejects a far optimum and keeps a wrong current fit. Errors on 0023
(0.80 deg), 0043 (0.55 deg), 0093 (0.50 deg) and 2011_09_30_0016 (0.53 deg)
remain under both rules. It loses eight 3 deg detections: 0014 pitch and yaw,
0029 pitch, 0035 roll, 0101 yaw, 0002 pitch, and 0034 roll and yaw. Coverage is
unchanged.

Measured against v2's gate values, the margin would pass on these spent drives:
3 deg misses are 8.8 %, within 10 %, and the turned-run error is 1.350 deg,
within 1.5 deg. The current rule misses that turned-run error gate by 0.003 deg.
This is **not evidence for adoption**. The value 3.5 was proposed after reading
0002, which is v2 evaluation data, and these 55 drives include all of v2's
evaluation drives. The margin also moves a large error to 0036 rather than
removing it. The margin therefore stays an opt-in candidate. Any v3 rule still
needs development here and a pre-registered test on unread data.

Four large zip64 drives (2011_09_30_0018 and 0028, 2011_10_03_0027 and 0034)
first failed while fetching, through truncated transfers or connection resets
on some of their roughly 1000 ranges. All eight failed attempts are retained.
curl's `--retry` skips these errors. The first fix, `--retry-all-errors`
(`d94d7540`), let a retry append to the partial body. The range length check
rejected the result, so no bad data entered. The fix at `a7a83db0` sends a
fresh request per attempt and requires the exact length. Every range is still
checked against its ETag, length and CRC. The latest attempt of each drive
passes; capture and scoring code did not change between attempts. Each report
records its commit and frozen module hashes.

Captured NPZ inputs (0.21–0.74 GB per kept drive) stay on the batch machine only
while disk allowed. They are kept for 23 drives, including 0002, 0014, 0023,
0029, 0034, 0035, 0036 and the three inconclusive drives. The other 32 were
deleted after their comparison, with their digests recorded in each report;
those drives need re-capture before a new rule can be replayed. The
[drive receipts](../assets/kitti_camera_lidar_v3/spent_batch_20261031/drives/manifest.json)
and the [0005 smoke](../assets/kitti_camera_lidar_v3/spent_batch_20261031/smoke/manifest.json)
publish reports, check outputs, logs, evidence and comparisons. They exclude
images, clouds, bags, archive payloads and NPZ arrays. The smoke matched the
2026-10-22 pilot's input members digest and its scores within about 1e-14.

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

## Native spent-data pilot, 2026-10-23

The first real capture/replay pilot completed on **one of 55** spent drives,
`2011_09_26_0005`. Its 154 frames were fetched using the original archive's
size/ETag and the published calibration ZIP hash. The selected streams required
432,575,704 bytes, below the 512 MiB acquisition cap. No new evaluation data
was read. The pilot's source is `c43cc7df`; it is development, not an audit.

The first attempt at `3c3e1234` stopped before estimation in all seven checks:
the capture guard compared the rounded calibration matrix directly with its
quaternion-serialized TF representation. On the real baseline their maximum
matrix-element difference is `2.280962807521547e-8`, above the guard's `1e-8`.
The [original failure](../assets/kitti_camera_lidar_v3/native_pilot_0005/attempt_1/report.json)
and all seven failed-check receipts remain published.

Conversion now also records `camera.t_imu_velo`. The replay tool reconstructs
both TF edges using the converter's quaternion serialization before composing
the reference. The original `camera.t_cam_velo`, candidate guard tolerance,
estimator, scoring rules and audit gates are unchanged. A rounded calibration
without the extra provenance requires bag regeneration; legacy exact-rigid
snapshots remain readable. Regressions check decimal-rounded calibration,
all seven candidate/injection identities and refusal of a wrong candidate.

The retry verified all **468** retained members against their size/CRC and used
zero additional download bytes. It generated a new bag and separate output
directory, preserving the failed attempt. Seven real checks and fourteen
re-estimations completed. Every current-rule replay matched the captured
numerical evidence within `1e-9` and the native verdicts. The
[paired report](../assets/kitti_camera_lidar_v3/native_pilot_0005/attempt_2/comparison/comparison.json)
deliberately says `incomplete`, with no audit thresholds.

| Measure on 0005 only | Current rule | Proposed 3.5 margin |
| --- | ---: | ---: |
| Baseline judged axes | 3/3 | 3/3 |
| Baseline flagged | no | no |
| Maximum judged baseline error | 0.1000 deg | 0.1000 deg |
| Maximum judged turned-run error | 0.1009 deg | 0.1009 deg |
| 1 deg turns flagged | 3/3 | 3/3 |
| 3 deg turns flagged | 3/3 | 3/3 |
| Turned axes incorrectly judged right | 0/6 | 0/6 |

Coverage change is zero percentage points. There is **no measured improvement
on this drive**; it validates the real comparison pipeline. It cannot resolve
the v2 failures on other recordings or replace the all-55 denominator above.

Separate [failure](../assets/kitti_camera_lidar_v3/native_pilot_0005/attempt_1/manifest.json)
and [retry](../assets/kitti_camera_lidar_v3/native_pilot_0005/attempt_2/manifest.json)
manifests pin all public receipts and exact producer/source capsules by SHA-256.
Snapshot JSONs pin each private NPZ input; images, clouds, bags and NPZ files
are not republished. The retry's roughly 0.7 GB of exact input arrays stays
local for subsequent comparisons. Report schemas and committed integrity tests
check the failure/retry chain, native/replay parity and source bindings.

## Known-failure case, 2026-10-23

The second native comparison used spent `2011_09_28_0002`, the recording behind
v2's gate-failing far-start result. Source `46f4b53a` retained the same 600-frame
recipe, frozen proposed margin `3.5` and original archive/calibration pins.
This drive has 376 frames, all acquired for each required stream. Exact ranges
fetched 995,875,492 bytes below the 2 GiB cap, and every member passed CRC.
There were seven actual checks and fourteen paired re-estimations, without
capture or replay failures. The current-rule evidence and native verdicts
matched all seven captures. The
[comparison](../assets/kitti_camera_lidar_v3/native_development_20110928_0002/attempt_1/comparison/comparison.json)
is explicitly `incomplete`, with no audit thresholds.

| Measure on 0002 only | Current rule | Proposed 3.5 margin |
| --- | ---: | ---: |
| Baseline judged axes | 1/3 | 1/3 |
| Baseline flagged | no | no |
| Maximum judged baseline error | 0.0500 deg | 0.0500 deg |
| Maximum judged turned-run error | 1.5030 deg | 0.1000 deg |
| 3 deg controls flagged, on baseline-judged axes | 1/1 | **0/1** |
| 1 deg controls flagged, on baseline-judged axes | 1/1 | 1/1 |
| Counted turned axes incorrectly judged right | 0/2 | 0/2 |

This is a **tradeoff, not a demonstrated overall improvement**. The maximum
error is over axes still judged; its decrease does not establish accurate
calibration of unreported rotations. Baseline coverage changes by zero
percentage points, but the proposed pitch +3 deg result becomes inconclusive
with no judged rotation axes, losing the only counted 3 deg detection. The
denominator is one because baseline 0002 judges pitch only, not all three axes.

Two further runs through `calibrex check` from the same bag confirm the proposed
rule, with cache disabled and `--camera-lidar-far-control-t 3.5`:

- [yaw +3 deg](../assets/kitti_camera_lidar_v3/native_development_20110928_0002/confirmation_yaw3/report.json)
  was selected before inspecting the new replay results because it caused the
  published v2 error-gate failure;
- [pitch +3 deg](../assets/kitti_camera_lidar_v3/native_development_20110928_0002/confirmation_pitch3/report.json)
  is explicitly a **post hoc** selection after replay revealed the detection
  regression.

Both native runs match the replay's numerical evidence within `1e-9` and its
axis/overall verdicts. Both are inconclusive with no judged rotation axes.
Yaw is not a counted turn in the detection row because baseline yaw was not
judged. The pitch regression therefore also exists through the real driver;
it is not a replay-only discrepancy. No threshold was changed after observing
these results, and the existing default stays unchanged.

The public [attempt manifest](../assets/kitti_camera_lidar_v3/native_development_20110928_0002/attempt_1/manifest.json)
and two confirmation manifests bind reports, schemas, logs and exact source
capsules. Publication excludes images, clouds, bags and the private NPZ arrays.
The full 55-drive development comparison still has **53 pending recordings**;
these two deliberately selected probes cannot supply its aggregate detection,
coverage or new audit gates. Inspect the detection/coverage cost on further
spent recordings, including 0034, before considering adoption.

## Why a scalar adoption margin is insufficient on these probes

A [post hoc start trace](../assets/kitti_camera_lidar_v3/native_development_20110928_0002/start_trace/report.json)
at source `49449c4d` re-estimated the saved pitch +3 and yaw +3 inputs under both
existing rules, without downloading data or changing the estimator. Serial
instrumentation recorded each distinct fit and adoption decision. All four
outputs matched the earlier full numerical evidence, excluding provenance,
within `1e-9`. The [publication manifest](../assets/kitti_camera_lidar_v3/native_development_20110928_0002/start_trace/manifest.json)
binds the trace, artifacts and producer; estimator/replay source hashes resolve
to the already published `attempt_1` capsules. NPZ inputs remain private.

| Injected turn | Candidate-start correction, roll/pitch/yaw (deg) | Single-axis profile fit | Later multi-axis fit |
| --- | --- | --- | --- |
| Pitch +3 | (-0.90, +0.05, -0.10), no observed axes | (0, -2.85, 0), pitch observed, t=3.3816 | (-1.55, -3.05, -2.25), roll/pitch observed, minimum t=2.6211 |
| Yaw +3 | (-1.95, +1.00, +0.05), no observed axes | (0, +0.05, -2.95), pitch observed, t=2.9194 | (-1.50, -0.15, -5.30), roll/pitch observed, minimum t=2.6493 |

These are left-correction vectors relative to each injected candidate, not
absolute calibration angles. For both inputs the existing minimum of two
observed axes rejects the single-axis profile fit. The default then accepts
the multi-axis fit; margin 3.5 rejects it and retains the unobservable
candidate-start fit. The pitch-only profile fit is near the inverse injected
turn, but the yaw profile fit still does **not** observe yaw: correcting an
angle in a fit does not establish that the angle is measurable.

For these two multi-axis fits, any scalar margin low enough to retain the
pitch detection (`t <= 2.6211`) also admits the yaw run's erroneous roll fit
(`t = 2.6493`). Raising only this threshold cannot retain that pitch detection
while excluding that yaw-run fit. This statement concerns the traced candidate
sets, not every possible estimator. The pitch multi-axis fit also reports an
erroneous roll correction; retaining its detection alone would not fix accuracy.

Simply allowing one observed axis is not justified either: earlier spent 0056
showed a single-axis far optimum judging roll about 6.6 deg off, motivating the
existing two-axis requirement. The next development task is to evaluate a
different start-selection or axis-evidence rule on cached spent inputs, with
coverage and injected-turn detection measured together. These traces establish
a failure mechanism, not a new calibration method, new threshold or SOTA claim.

## Conditional-axis development probe

`tools/kitti_camera_lidar_axis_probe.py` tests a different search model: fit
one rotation coordinate while fixing the other five coordinates at the supplied
candidate. It profiles the existing range, refines the maximum, refits only
that coordinate in each leave-one-block-out fold, and records its held-out
control. It also repeats the full profile/refinement after fixing each other
rotation coordinate at plus/minus the existing 2 deg control amount. No new
confidence or sensitivity threshold is selected. This is a research diagnostic,
not an estimator/check backend; separate coordinate fits are not composed into
a calibration transform or native verdict.

```bash
python -m tools.kitti_camera_lidar_axis_probe SAVED_SNAPSHOT --output NEW_DIRECTORY
```

The [fixed batch](../assets/kitti_camera_lidar_v3/conditional_axis_probe/report.json)
ran on all seven cached inputs from each of 0005 and 0002: **14 inputs, 42
conditional-axis fits**, with no acquisition or estimation failures. The
parent commit is `ca55d701`; the then-uncommitted tool is separately pinned by
its exact source SHA-256 and capsule. Reference errors are computed after
fitting, undoing the capture's recorded injection; no reference enters search
or controls. The [manifest](../assets/kitti_camera_lidar_v3/conditional_axis_probe/manifest.json)
binds reports, schemas and sources, about 140 KB excluding the manifest.
Sensor payloads remain local. The batch producer asserts unchanged Git HEAD
and source bytes through completion.

| Conditional fit | Correction | Active-axis reference error | Conditional std | Control t | Nuisance correction span |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0002, pitch +3, fit pitch | -2.85 deg | +0.15 deg | 0.0417 deg | 3.3036 | 7.80 deg |
| 0002, yaw +3, fit yaw | -2.95 deg | +0.05 deg | 0.0417 deg | 0.8925 | 9.05 deg |
| 0005, baseline, fit pitch | -0.10 deg | -0.10 deg | 0.0250 deg | see receipt | 5.65 deg |

Using only the existing ordinary evidence conditions (detected control,
conditional std at most 0.5 deg, no search bound), 15 of 21 conditional fits
on 0005 and 3 of 21 on 0002 qualify. All 18 have active-axis reference errors
at most 0.15 deg on these selected cached inputs. The near-inverse pitch
correction is promising, but this is **not a measured gain in joint calibration
accuracy or a new SOTA result**. A correct yaw correction with a failed control
still supplies no evidence to report yaw. There is no native detection rate
from these separate fits, and no new drive has been captured.

The large nuisance spans show why conditional precision cannot establish
joint observability: pitch +3 on 0002 has a conditional std of just 0.0417 deg
yet its optimum moves across a 7.8 deg range when other axes are perturbed.
Even accurate baseline pitch on 0005 spans 5.65 deg. These spans include full
searches that may reach a bound; they are sensitivity probes, not uncertainty
intervals. A coupled synthetic ridge likewise passes conditional controls and
has 0.025 deg conditional std while the nuisance optimum spans 4 deg; the
uncoupled synthetic case spans zero. Tests retain both cases to prevent
mistaking conditional precision for joint identifiability.

The next algorithmic question is how to preserve useful axis evidence while
accounting for uncertain nuisance coordinates, with explicit coverage and
injected-turn detection costs. Neither the single-axis fit nor a threshold
on its nuisance span is promoted from these two recordings. The earlier 0056
false optimum and the remaining 53-drive comparison still need to be covered
before any adoption decision.

The follow-up [nuisance-profiled control experiment](kitti_camera_lidar_profile_controls.md)
refits other rotation axes on training blocks before scoring fixed held-out
poses. All 14 inputs completed, but controls alone still pass a wrong roll
basin. Neither diagnostic is promoted as a calibration method.

## Work still required before pre-registration

The [2026-10-23 local recheck](kitti360_camera_lidar_v3_access.md) verified
official KITTI-360 access and exact metadata ranges for 21 archives.
Published-pose and opt-in raw-OXTS conversion each passed on three frames from
spent 0000 and 0002. The earlier cloud HTTP 403 is historical evidence.
All 55 spent drives now have real captures and paired comparisons (above).
KITTI-360 conversion checks are not calibration measurements.

The scalar 3.5 margin is not adopted as a default (above). A frozen
[cross-start agreement rule](kitti_camera_lidar_v3_start_agreement_results.md)
was then screened on all 55 drives. It cut turned-run judged errors beyond
0.5 deg from 10 to 2 (maximum 0.532 deg), but it flagged only 66 % of 3 deg turns,
so it was not carried forward. A post hoc count suggests flagging cross-start
disagreement instead of only withholding axes; that needs its own frozen
protocol, with the 0029 baseline cost stated. Confirm any selected rule via the
real driver, and validate any additional converter behavior on spent data.

Only then derive the v3 gates from the new development results using the
[handoff rules](../sota_handoff.md#how-an-audit-is-run). Evaluation archive pins,
disjoint 60 s interval selection and the preregistration remain pending a full
exposure audit and development results. KITTI-360 0003 was already read by earlier research
and must be excluded, as documented in PR #235. No v3 thresholds are frozen here,
and no evaluation is authorized by this development page.
