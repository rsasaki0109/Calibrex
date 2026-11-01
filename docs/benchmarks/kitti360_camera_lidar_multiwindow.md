# KITTI-360 camera-LiDAR: multi-window check (development protocol)

**Status, 2026-11-01: run once; verdict `not carried forward`. All six criteria
held, but one of the 42 segments could not run (an archive data gap), and the
protocol required all 42. This is development, not a claim.** KITTI-360 holds no unread camera-LiDAR data here (the
[exposure audit](kitti360_camera_lidar_v3_prep.md#exposure-audit)), so nothing
on this page is a SOTA or held-out result. The 42 segments of 0004–0007, 0009
and 0010 have not been read by this development work. They serve as its
held-out check, run once under the rule below. The rule and the criteria are
committed before that run, and neither changes after it.

## Why

On spent segment 0002-s04 the check flags KITTI-360's calibration. It reads yaw
−0.60°, with a reported std of 0.042°. Both halves of that minute read the same
bias, so a block jackknife within one window cannot see it
([cause](kitti360_camera_lidar_v3_prep.md#cause-of-the-0002-s04-flag-post-hoc-spent-data-only)).
Disjoint windows of one recording see different scenes. Their spread measures a
bias a single window shares across its blocks, and their median is robust to one
biased window. Raising the 0.5° floor would hide the flag too, but it is a
threshold change and is not considered.

## Rule

`tools/kitti360_camera_lidar_multiwindow.py` combines the per-window results of
the unchanged default check. A recording of `K` disjoint 60 s windows gets one
verdict per rotation axis, over the windows whose check judges that axis:

- **Too few windows.** Fewer than `max(2, ceil(K / 2))` judging windows leaves the
  axis unchecked.
- **Center.** `m` is the median of the windows' candidate errors (candidate minus
  estimate).
- **Spread.** `s = 1.4826 · MAD` of those errors, `w` is the median reported std,
  and `σ = max(s, w)`. The median's std is `1.2533 · σ / sqrt(n)`.
- **Tolerance.** `max(3 · that std, 0.5°)`, the check's `sigma_k` and rotation
  floor.
- **Status.** Pass when `|m|` is within the tolerance, fail beyond twice it,
  otherwise warn.

The recording's verdict is its worst axis, or inconclusive when no axis is
judged. Turned runs use the same injection in every window. A trial counts when
the multi-window baseline judges the turned axis. It is flagged when the turned
run's multi-window status is warn or fail, as in v2. Accuracy uses the median of
the judging windows' estimate errors against the KITTI-360 calibration.

## Recordings and criteria

The exposed segments are combined two ways:

- **Sequence level.** Each sequence's segments, in order: 6 recordings, with
  `K` from 3 to 12.
- **Triple level.** Disjoint consecutive triples within a sequence: 13
  recordings (0004: 3, 0005: 2, 0006: 2, 0007: 1, 0009: 4, 0010: 1).

The unchanged single-window check is reported on the same 42 segments for
comparison. The rule is **carried forward** only if every one of the 42
segments runs and all six criteria hold:

| Criterion | Requirement |
| --- | --- |
| Vendor calibration | no sequence- or triple-level recording flags it |
| Baseline accuracy | every judged triple-level baseline axis within 1.0° |
| Turned-run accuracy | every judged triple-level turned-run axis within 1.0° |
| 3° turns | at most 0.15 of counted triple-level trials missed, none judged right |
| 1° turns | at most 0.25 of counted triple-level trials missed |
| Coverage | at least 0.5 of triple-level baseline axes judged |

The accuracy and detection bounds are the KITTI-360 development gates
([gates](../assets/kitti_camera_lidar_v3/kitti360_dev_20261031/gates.json)). The
vendor requirement is v2's. If carried forward, the rule becomes an opt-in
`calibrex check` option for long recordings. Otherwise it is recorded as not
carried forward.

## Design data

The rule was written while looking at the 22 spent segments only. On them, the
single-window check flags the vendor calibration once (0002-s04), as the v2
scorer does. A unit test pins that parity. The 7 spent triples and 2 spent
sequences flag it never. Their worst judged errors are 0.55° (baseline) and
0.60° (turned). They miss 2 of 19 counted 3° turns and 0 of 19 1° turns, and
judge 19 of 21 baseline axes. These are design data, not evidence.

## Result (run once, 2026-11-01)

The batch ran all 42 segments from the protocol commit `41ee8249`. The rule was
applied once to the
[published receipts](../assets/kitti_camera_lidar_v3/kitti360_exposed_20261101/multiwindow.json).
**41 of 42 ran.** 0009-s07 failed at fetch: the official OXTS archive has no
packet for frame 8849. The batch requires every member, so the attempt was not
retried, since a retry would fail the same way. The gap is visible in archive
metadata. The segment rule checked only image and scan continuity, which is a
lesson for any future segment rule. Without s07, 0009's third triple is s06, s08
and s09, giving 12 triples instead of 13.

| 41 scored segments | Single window (shipped check) | Triples | Sequences |
| --- | ---: | ---: | ---: |
| Recordings | 41 | 12 | 6 |
| Vendor calibration flagged | 0 | **0** | **0** |
| Worst judged baseline axis | 0.40° | 0.40° | 0.40° |
| Worst judged turned-run axis | **4.75°** | 0.65° | 0.65° |
| Judged baseline axes | 104 of 123 | 33 of 36 | 15 of 18 |
| 3° turns missed | 5 of 104 | 1 of 33 | 1 of 15 |
| 1° turns missed (judged right) | 4 of 104 (2) | 0 of 33 (0) | 0 of 15 (0) |

**Frozen verdict: `not carried forward`.** Every criterion holds at the triple and
sequence levels, but the protocol required all 42 segments to run. It stays
unchanged.

### After the verdict (post hoc, changes nothing)

- **No vendor flag to remove.** The single-window check also flags none of these
  41 segments. So the run does not show the rule removing a vendor flag. That
  benefit was seen only on the design data (0002-s04).
- **Large errors removed.** The rule removes the single-window check's largest
  errors. In 0007-s02 under a 3° roll turn, a single window judges roll 4.75° from
  the reference with a reading of the wrong sign (−1.75°). The trial is not
  counted, because the baseline does not judge roll there. Two 1° turns are
  judged right by a single window (0006-s05 roll, 0009-s01 yaw). No triple or
  sequence does either.
- **A constant offset from the vendor calibration.** Every triple and sequence
  that judges roll puts the estimate +0.35° to +0.40° from the vendor
  calibration in roll, and pitch 0° to −0.20°. That matches 0000 and 0002.
  A difference this constant across the rig is a property of the rig or of the
  method, not of a scene. Which one this data cannot tell.

All KITTI-360 sequences are now spent. Shipping the rule as an opt-in check
option would be a product decision on development numbers, as with the
[disagreement warning](camera_lidar_check.md#experimental-cross-start-disagreement-warning-2026-10-31-opt-in).
It would not be a validated rule.
