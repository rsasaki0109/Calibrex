# KITTI-360 camera-LiDAR v3: exposure audit and segment rule

**Status, 2026-11-01: development only. There is no pre-registration and no
evaluation. This work has read no candidate sensor payload, but the owner cannot
rule out earlier reads outside git (see the exposure audit), so the candidates are
not usable as unread data.** This page
fixes two inputs of a future v3 check pre-registration: which KITTI-360
sequences are unread, and which 60 s segments of them would be evaluated.
Gates, the run recipe and verdict rules come later, from development runs on
spent sequences only. They will be frozen in a separate pre-registration PR
before any candidate image, point cloud, OXTS record or timestamp is read.

## Planned claim and its scope

The planned claim is that `calibrex check` judges the KITTI-360 vendor
camera-LiDAR calibration and flags deliberate 1° and 3° rotation errors on
unread KITTI-360 segments. It would use the default rule, with gates derived
before the run as in [v2](kitti_camera_lidar_v2_preregistration.yaml). A
secondary claim would measure the opt-in
[cross-start disagreement warning](camera_lidar_check.md#experimental-cross-start-disagreement-warning-2026-10-31-opt-in)
on the same runs. That warning's development numbers come from the data that
motivated it.

KITTI-360 is **one dataset family recorded on one platform**. Separate
sequences are not independent rigs. A v3 result would therefore be scoped to
the KITTI-360 rig. It cannot meet the final gate of the
[research plan](../research/camera_lidar_sota_plan.md), which needs two
dataset families and an independent rig. It cannot establish SOTA on its own.

## Exposure audit

The audit searched every commit on every ref of the repository (1,427 commits
on 2026-10-31). It looked for `2013_05_28_drive_NNNN` and for KITTI-360 near
each candidate ID (`git log --all -G`).

| Commit | What it does with candidates `0004`–`0007`, `0009`, `0010` |
| --- | --- |
| `3c3e1234` (2026-10-23) | Reads ZIP end records and central directories only. Sensor payload bytes: **0** ([access check](kitti360_camera_lidar_v3_access.md)). |
| `62f60251` (2026-10-28) | Reserves them in text as candidates only. |
| `5838af10` (2026-10-15) | Uses the name `2013_05_28_0004` in a unit test whose input path does not exist. |

Other hits for `drive_0004`, `drive_0005` and `drive_0009` are KITTI raw drives
(`2011_09_26_drive_NNNN`). They are unrelated recordings. No committed
result, asset, log or capture reads a candidate's sensor contents. `0003`
stays spent: the research plan records a development capture of it, and
there are no frame counts for it.

The audit has one limit. Work that never reached git cannot be searched, for
example a local workspace or another clone. Before the freeze, the repository
owner must confirm that no such work opened candidate sensor data. The freeze
PR records that confirmation.

**Owner's answer, 2026-11-01: opened, or cannot rule it out.** Asked whether work
outside git had opened or processed images, point clouds, OXTS records or
timestamps of 0004–0007, 0009 or 0010, the owner answered that it had, or that
they do not know. The candidates therefore **cannot be pre-registered as unread**.
The evaluation mode's guard stays locked, because it requires a recorded
confirmation. Any KITTI-360 v3 evaluation needs a narrower, confirmed exposure
statement first, or other unread recordings.

## Segment rule

The rule uses archive member counts only, from the committed
[frame summary](../assets/kitti360_sota_readiness/paired-frame-counts.json)
(sensor payload bytes 0):

- Start at the first frame ID common to the rectified `image_00` and Velodyne
  archives. Both ID ranges must be gap-free by count.
- Take 600 consecutive IDs (60 s at 10 Hz). `calibrex check` reads at most 48
  scans 1 s apart from a segment.
- Start the next segment 1200 IDs later, which leaves a 60 s gap. Keep every
  segment that fits inside the common range.
- Cut spent sequences by the same rule into development segments.

| Sequence | Role | Segments |
| --- | --- | ---: |
| `0004` | evaluation | 10 |
| `0005` | evaluation | 6 |
| `0006` | evaluation | 8 |
| `0007` | evaluation | 3 |
| `0009` | evaluation | 12 |
| `0010` | evaluation | 3 |
| `0000` | development (spent) | 10 |
| `0002` | development (spent, starts at ID 4391) | 12 |

That is 42 evaluation and 22 development segments. The
[segment list](../assets/kitti_camera_lidar_v3/kitti360_prep/segments.json)
records the rule, the generator's and the counts' SHA-256, and
`sensor_payload_bytes_read: 0`. The tool refuses to overwrite a written list.
A unit test checks that the committed rule reproduces it.

```bash
python tools/kitti360_camera_lidar_v3_segments.py \
  docs/assets/kitti360_sota_readiness/paired-frame-counts.json OUT.json
```

Segments are fixed by IDs, not by content. A stationary or featureless segment
stays in the list. Whether the check can judge it is part of the coverage
measure.

## Gate rule, fixed before the development results

This rule was committed while the first development segment was still being
fetched, before any KITTI-360 check result existed. The gates themselves are
not set yet. They follow from the rule once the development batch has run.

Development values come from the 22 spent segments, scored by the frozen v2
scorer under the default rule. KITTI raw's 55 spent drives are reported beside
them but set no gate, because the evaluation rig is KITTI-360. The rules are
v2's, unchanged:

- **Judged-axis error, baseline and turned runs.** 1.5 times the worst
  development value, rounded up to the next 0.25°.
- **Judged-axis coverage.** The development rate divided by 1.5, rounded down
  to the next 0.25.
- **Scored segments.** The development rate divided by 1.5, rounded down to a
  whole number of the 42 segments.
- **3° turns missed.** The one-sided 95 % exact binomial upper bound of the
  development miss rate (the rule of three when none is missed), rounded up to
  the next 0.05.
- **1° turns missed.** 0.25, v1's allowance.
- **Vendor calibration flagged, 3° turns judged right, failed segments.** None
  allowed.

The secondary claim measures the opt-in warning
(`--camera-lidar-start-agreement-deg 0.5`) on the same runs. Its baseline
warning rate, and its run-level 3° and 1° miss rates, get the binomial
allowance above. Its judged-axis errors get the 1.5× rule. The warning's
development numbers on KITTI raw come from the data that motivated it, so
only KITTI-360 development values set its gates too.

If the development batch already fails a gate that the rule derives from the
batch itself, for example a flagged vendor baseline, the pre-registration
records that. It is not tuned away.

## Development results, 2026-11-01

The [batch](../assets/kitti_camera_lidar_v3/kitti360_dev_20261031/summary.json) ran all
22 spent segments once from commit `57d3796a`, with no failed attempt. Each segment
read about 1.36 GB of exact member ranges. All eight spent archives' central
directories matched their committed SHA-256 bindings. Applying the rule above gives
these [gates](../assets/kitti_camera_lidar_v3/kitti360_dev_20261031/gates.json):

| Requirement | Development (default rule) | Gate |
| --- | --- | --- |
| Vendor calibration flagged | **1 of 22 (0002-s04)** | none allowed |
| Worst judged baseline axis | 0.60° | ≤ 1.0° |
| Worst judged turned-run axis | 0.65° | ≤ 1.0° |
| Judged baseline axes | 59 of 66 (0.89) | ≥ 0.5 |
| Scored segments | 22 of 22 | ≥ 28 of 42 |
| 3° turns missed | 4 of 59 | ≤ 0.15 |
| 3° turns judged right | 0 | none allowed |
| 1° turns missed | 0 of 59 | ≤ 0.25 |

| Requirement | Development (warning, 0.5°) | Gate |
| --- | --- | --- |
| Baselines warned | 1 of 22 | ≤ 0.2 |
| Worst judged baseline / turned axis | 0.60° / 0.65° | ≤ 1.0° / ≤ 1.0° |
| 3° turns missed by the run verdict | 2 of 59 | ≤ 0.15 |
| 1° turns missed by the run verdict | 0 of 59 | ≤ 0.1 |

**The default rule already fails its own vendor gate on development data.** On
0002-s04 the check judges yaw −0.60° from the KITTI-360 calibration, with a
reported std of 0.042°, so the 0.5° floor sets the tolerance and the axis warns.
Neighbouring segments read yaw −0.50° (0002-s03, judged pass) and +0.65°
(0002-s05, not judged). Every other judged yaw lies within ±0.25°. Roll reads
+0.25° to +0.40° wherever it is judged. A yaw that swings by more than a degree
between adjacent minutes, with a std of hundredths of a degree, points to an
overconfident estimate on those segments rather than a calibration offset. This
is a reading of development data. As stated above, the gate is not changed for it.

## Not yet decided

The following are decided only after development runs on the 22 spent
segments:

- timestamp-boundary checks and the failure policy for a segment;
- the gate values, by the rule above;
- the per-sequence reporting.

Spent segments may also be used to develop the batch tool. The pre-registration
must name the frozen code commit. It must also pin the archives (already pinned
by size and ETag in the
[range bindings](../assets/kitti360_sota_readiness/metadata-range-bindings.json))
and the reference calibration.
