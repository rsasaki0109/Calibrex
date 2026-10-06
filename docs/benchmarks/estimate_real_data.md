# `calibrex estimate` on real data

`calibrex estimate <bag> --output DIR` (`slac.bag_estimate/v0.1`) estimates a
calibration for a bag that has none and exports only what the data observed.
This page collects every real-data run of it: what was estimated, how it
compares with the reference, and whether the exported `frames.yaml` still
checks out on a **different** recording with `calibrex check --tf`.

These are tool-validation runs, not SOTA claims. No audit is attached, and a
`pass` here covers only the axes that were judged. Each result states its data
split. Every dataset used was already on disk; nothing was downloaded.

| Pair | Data | Estimate vs reference | Round trip on another recording |
| --- | --- | --- | --- |
| `camera-imu` | Hilti 2022 exp21 | rotation 0.15-0.33 deg, time offset -0.08 / -0.21 ms from Kalibr | exp07: pass |
| `imu-lidar` | Koide indoor_easy_01 | rotation 0.08 / 0.66 / 0.07 deg from `/tf_static` | indoor_easy_02: pass |
| `lidar-vehicle`, `ins-lidar` | KITTI 0005 + 0014 + 0022 | pitch and yaw within 0.1 deg of the 5-drive result; `ins-lidar` roll and pitch within 0.08 deg of `calib_imu_to_velo` | 0009 and 0015: pass on the one axis the short drive constrains, the rest inconclusive |
| `imu-vehicle` | KITTI 0005 + 0014 + 0022 | **failed**: no axis constrained (roll 0.57, pitch 0.16, yaw 0.16 deg std against a 0.1 deg bound); a data limit, see below | not exported |
| `lidar-wheel_odometry` | KITTI 0005 + 0014 + 0022 | pitch and yaw as `lidar-vehicle` (same solver) | pass on yaw (0.003 and 0.095 deg) |
| `gnss-lidar`, `gnss-imu` | RTK-SLAM stadtgarten_seq2 | y within 0.3 and 0.7 cm of CAD; x and z unobservable | stadtgarten_seq1: `imu-lidar` pass (rotation only), `gnss-lidar` **warn** on a prior-filled x, `gnss-imu` inconclusive |

## Camera-IMU: Hilti 2022 (from PR #112)

Hilti exp21 against Kalibr. The Kalibr camchain is passed only as the prior,
for the intrinsics and the lever arm. The rotation and the time offset come from
the bag; the lever arm is marked `NOT MEASURED`.

| Camera | roll / pitch / yaw delta to Kalibr | Time offset (Kalibr 1.90 ms) |
| --- | --- | --- |
| cam1 | 0.148 / 0.332 / 0.185 deg | 1.827 ± 0.135 ms (-0.075 ms) |
| cam0 | 0.192 / 0.331 deg; yaw `control_not_detected`, taken from the prior and marked | 1.701 ± 0.133 ms (-0.206 ms) |

Round trip: the exported camchain on exp07 gets **pass** (cam1 is off by 0.015 /
0.283 / 0.137 deg against a tolerance of about 0.5 deg; cam0's yaw is judged
against the prior, not an estimate). On this data the Hesai `imu-lidar` z axis
came out `control_not_detected`, so the PandarXT-32 frame was **omitted** when
no prior was given. Hilti exp21 and exp07 are the development recordings; exp01-04 were
not used.

## IMU-LiDAR: Koide indoor_easy_01 (from PR #112)

Rigid scans, the bag's `/tf_static` as the prior. The rotation is observed with
std 0.25-0.29 deg; the lever arm is unobservable, so it comes from the prior and
is marked. Against `/tf_static` the estimate differs by 0.080 / 0.658 / 0.065 deg.
The round trip on **indoor_easy_02 passes** (0.239 / 0.948 / 0.464 deg against
the 1.5 deg rigid floor; x, y, z unchecked).

## Vehicle pairs: KITTI raw (development drives)

**Data.** 2011_09_26 drives 0005, 0009, 0014, 0015 and 0022 only; drives 0027 to
0059 (held out) were not touched. Drives are converted with `calibrex convert
kitti-raw` (see [the check tutorial](../tutorials/calibrex_check.md#converting-a-kitti-raw-drive));
`base_link` is the OXTS frame and `/oxts/twist` is declared as the wheel proxy.

| Role | Drives |
| --- | --- |
| Estimate (one pooled bag, 1268 sweeps) | 0005 + 0014 + 0022 |
| Round trip (never used for the estimate) | 0009 (44 s) and 0015 (30 s), each its own bag |

```bash
calibrex convert kitti-raw 2011_09_26_drive_0005_sync 2011_09_26_drive_0014_sync \
  2011_09_26_drive_0022_sync --calib-dir 2011_09_26 --output kitti_est_bag
calibrex estimate kitti_est_bag --vehicle-frame base_link --topic-kind /oxts/twist=wheel \
  --output est_kitti
calibrex check kitti_0009_bag --tf est_kitti/frames.yaml --vehicle-frame base_link \
  --topic-kind /oxts/twist=wheel --pairs lidar-vehicle,imu-vehicle,ins-lidar,lidar-wheel_odometry
```

The bag carries the vendor calibration as `/tf_static`, so it is also the rough
prior (`base_link` to `imu_link` identity, `imu_link` to `velo_link` =
`calib_imu_to_velo`). The estimate took 2 min 43 s.

**Estimate against the references** (degrees; the reference for `lidar-vehicle`
is the [5-drive result](kitti_lidar_vehicle.md), the reference for `ins-lidar`
is `calib_imu_to_velo`):

| Pair, axis | Estimate (3 drives) | Reported std | Reference | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `lidar-vehicle` roll | - | 0.51 | 0.22 ± 0.42 (unobservable) | - | unobservable; prior roll kept |
| `lidar-vehicle` pitch | +0.525 | 0.052 | +0.62 | -0.10 | observed |
| `lidar-vehicle` yaw | -0.313 | 0.094 | -0.26 | -0.05 | observed |
| `lidar-wheel_odometry` pitch / yaw | +0.525 / -0.304 | 0.052 / 0.091 | as `lidar-vehicle` | 0.00 / +0.01 | observed |
| `ins-lidar` roll (`T_imu_velo`) | -0.923 | 0.063 | -0.849 | -0.074 | observed |
| `ins-lidar` pitch | +0.118 | 0.039 | +0.116 | +0.002 | observed |
| `ins-lidar` yaw, x, y, z | - | 0.21 deg, 0.12 / 0.18 / 0.95 m | -0.044 deg, 0.81 / -0.31 / 0.80 m | - | unobservable; prior kept |
| `ins-lidar` time offset | +5.07 ± 2.81 ms | | | | estimated |
| `imu-vehicle` | - | roll 0.57, pitch 0.16, yaw 0.16 deg | - | - | **failed**: no axis constrained (all three `unobservable`, values withheld) |

The three drives are a subset of the five behind the reference, so the two are
not independent; the table says only that the same solver on fewer turns lands
near the earlier value (pitch two reported std away). The 0.5 deg pitch offset
of the OXTS frame from the motion-defined vehicle frame, reported on the
[KITTI LiDAR-Vehicle page](kitti_lidar_vehicle.md), shows up again (+0.525).
`imu-vehicle` failing on this pooled bag matches the five-drive result, where
every axis std is over its bound. It was investigated (next section): a data
limit, not a bug.

**What was exported.** Roll was never observed and the lever arms are not
estimated by the vehicle pairs, so both frames are written only because the
bag's `/tf_static` prior supplies those axes, and the YAML marks them:

```text
base_link -> velo_link  [lidar-vehicle]  NOT MEASURED roll, x, y, z (from --tf prior)
velo_link -> imu_link   [ins-lidar]      NOT MEASURED yaw, x, y, z (from --tf prior)
```

The exported `velo_link` rotation is (-0.849, +0.525, -0.313) deg: the roll is
the vendor's, not a measurement. Without the prior (the 0009 bag with its
`/tf_static` removed) the result is `lidar-vehicle` yaw -0.213 ± 0.041 deg only,
and **no frame is exported**; the omission reason lists the unobserved axes.

Composing the two exported edges gives `base_link` to `imu_link` as (+0.07,
+0.40, -0.36) deg. Its pitch and yaw come from the data; its roll is an artifact
of the two prior roll values cancelling (the five-drive `imu-vehicle` roll is
1.04 ± 0.46 deg, unobservable on these drives). Do not read that composed roll
as a measurement.

**Round trip** (`calibrex check <drive> --tf est_kitti/frames.yaml`, defaults,
tolerance 0.5 deg). The exported file overrides the bag's `/tf_static` with the
estimated tree; `base_link` stays the root.

| Drive | `lidar-vehicle` | `imu-vehicle` | `ins-lidar` | `lidar-wheel_odometry` |
| --- | --- | --- | --- | --- |
| 0009 (44 s) | pass, partial: yaw 0.092 deg (0.18 of tolerance); roll, pitch unchecked | inconclusive (no axis constrained) | inconclusive | pass, partial: yaw 0.095 deg |
| 0015 (30 s) | inconclusive | pass, partial: pitch 0.186 deg (0.37 of tolerance) | inconclusive | pass, partial: yaw -0.003 deg |

The overall verdict is `inconclusive` on both: every pass covers one axis
because short drives leave the others unconstrained, as in the
[Phase C1 table](../tutorials/calibrex_check.md#phase-c1-results-on-kitti-raw-development-drives-only).
No axis the data did not observe was reported as checked.

## GNSS pairs: RTK-SLAM stadtgarten

**Data.** RTK-SLAM (a hand-held Livox MID360 with an RTK receiver) sequences
were already spent for claims, so this is tool validation. The estimate runs on
`stadtgarten_seq2` and the round trip on `stadtgarten_seq1`. The spans are those
of the [`check` GNSS runs](../tutorials/calibrex_check.md#phase-c2-results-on-rtk-slam):
`--max-duration-s 120` for `imu-lidar`, and `--gnss-max-duration-s` 900 s on seq2
and 600 s on seq1. The reference is the CAD antenna offset in `calib.yaml` (IMU
origin to antenna phase centre (0.023, -0.023, 0.090) m) and the MID360 manual's
IMU position (`T_lidar_imu` translation (-0.011, -0.023, 0.044) m, identity
rotation); neither is metrology.

```bash
calibrex estimate stadtgarten_seq2 --max-duration-s 120 --gnss-max-duration-s 900 \
  --tf rtk_slam_eval/calib/calib.yaml --output est_sg2
calibrex check stadtgarten_seq1 --tf est_sg2/frames.yaml --max-duration-s 120 \
  --gnss-max-duration-s 600 --pairs imu-lidar,gnss-lidar,gnss-imu
```

Without `--tf`, the data observe only `y` of the antenna offset (and the IMU-LiDAR
rotation); `x`, `z` and the antenna orientation are not observed, so **no frame is
exported** and `next steps` asks for a prior. With `calib.yaml` as the prior,
`x`, `z` and the orientation come from it. The estimates on seq2 (about 15 min
for `imu-lidar` and 11 min for the GNSS windows; the estimator cache was not hit):

| Pair, axis | Estimate | Reported std | CAD / manual reference | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `imu-lidar` roll / pitch / yaw | -0.101 / +0.097 / -0.149 deg | 0.060 / 0.057 / 0.076 deg | 0 (identity) | 0.10 / 0.10 / 0.15 deg | observed |
| `imu-lidar` y (`T_lidar_imu`) | +0.0128 m | 0.0080 m | +0.023 m | -1.0 cm | observed |
| `imu-lidar` x, z | - | z 1.45 cm | +0.011 m, -0.044 m | - | x `control_not_detected`, z unobservable |
| `gnss-lidar` y | -0.0028 m | 0.0055 m | 0.000 m | -0.28 cm | observed |
| `gnss-lidar` x, z | - | 1.04 cm, 4.08 cm | 3.4 cm, 4.6 cm | - | unobservable (x std just over the 1 cm bound) |
| `gnss-imu` y (antenna in IMU) | -0.0157 m | 0.0097 m | -0.023 m | +0.73 cm | observed |
| `gnss-imu` x, z | - | 1.21 cm, 4.34 cm | 2.3 cm, 9.0 cm | - | unobservable |
| `gnss-imu` roll, pitch, yaw | - | - | - | - | unobservable (a point offset has no orientation) |
| `gnss-lidar` time offset | -23.4 ± 12.1 ms | | | | unobservable |

(The `gnss-lidar` CAD values combine the manual's IMU position with the antenna
offset, as on the [GNSS-LiDAR page](rtk_slam_gnss_lidar.md).) The estimates match
the earlier `check` numbers for this sequence (`gnss-lidar` y 0.28 cm, `gnss-imu`
y 0.73 cm from CAD). The known 1.7-2.2 cm offset of x from CAD is **not**
reproduced as an estimate: x is not constrained at this span, so it is left to
the prior and marked.

**Exported** (`frames.yaml`, root `imu`):

```text
imu -> lidar0         [imu-lidar]  NOT MEASURED x, z (from --tf prior)
imu -> gnss_antenna   [gnss-imu]   NOT MEASURED roll, pitch, yaw, x, z (from --tf prior)
```

The antenna frame is the prior's x, z and identity orientation with the
estimated y (-0.0157 m); the lidar frame has the estimated rotation and y.

**Round trip on stadtgarten_seq1** (`calibrex check`, the exported `frames.yaml` as
the only calibration; the overall verdict is `warn`). The `imu-lidar` and `gnss-imu`
pairs were first skipped as `frame_not_in_tree` until bug 1 below was fixed; the
numbers here are with the fix (run with the fixed `frames.yaml`, which differs from the
first export only by the added `/livox/imu: imu` mapping).

| Pair | Verdict | Judged axes (delta against the tolerance) | Not judged |
| --- | --- | --- | --- |
| `imu-lidar` | pass, partial | roll 0.117, pitch 0.018, yaw 0.080 deg (0.5 deg tolerance); the held-out controls detect 0.52-0.62 deg | x, y, z (std 1.2 / 1.8 / 1.7 cm over the bound) |
| `gnss-lidar` | **warn**, partial | x 2.14 cm against the 2.0 cm tolerance | y, z (std 1.4 / 1.3 cm); the antenna orientation |
| `gnss-imu` | inconclusive (`no_judgeable_axes`) | none: composed stds 1.4 / 2.3 / 2.1 cm | all |

The `warn` is the **known baseline** for this sequence, not a defect of the
estimate: x was not measured on seq2, so the exported value is the CAD prior's, and
seq1's own x estimate sits 2.1 cm from it (the same 2.15 cm `warn` that `check` gives
seq1 against `calib.yaml`). The exported file therefore carries exactly the
information of its prior for x and z, plus the IMU-LiDAR rotation and a y that
seq1 is not precise enough to confirm or contradict. The round trip confirms the
rotation; it does not confirm the lever arm.

## Two unexplained KITTI failures, explained

Both were seen on the pooled bag and left uninvestigated in the first version of
this page. Bags used: the pooled 0005 + 0014 + 0022 bag and the single drives
0009 and 0015 (development drives only).

**`imu-lidar` failed with "too few LiDAR rate intervals with gyro coverage".**
Root cause: the OXTS IMU runs at **10 Hz** (median interval 100 ms, max 110 ms on
every drive), and the estimator only uses a LiDAR interval when the gyro has no
hole longer than 50 ms (`GyroSeries.covers`, `max_gap_s=0.05`), i.e. it needs
a gyro of at least 20 Hz. On KITTI no interval ever qualifies, so the solver gets
zero intervals. It is not the pooling: the single drives 0015 and 0009 fail
identically (every one of their IMU gaps, 296 and 446, exceeds 50 ms), the pooled
bag has no reversed or repeated timestamps (the only large gap is 387 s at a
drive boundary, which the 10 Hz gaps already dwarf), and the topic mapping was
right (`/oxts/imu`, `imu_link`). This is a data limitation, so the pair is now
**skipped up front** with the reason `unsupported_sensor`: "/oxts/imu runs at
about 10.0 Hz (median interval 100 ms); imu-lidar ... needs gyro samples at least
every 50 ms (>= 20 Hz)". `calibrex check` on KITTI gives the same skip (the
[Phase C1 table](../tutorials/calibrex_check.md#phase-c1-results-on-kitti-raw-development-drives-only)
predates rigid-scan support, which made the pair run and fail instead). Regression
tests: `test_imu_lidar_skips_an_imu_too_slow_for_the_gyro_coverage_rule` and
`test_the_rate_gate_matches_the_solvers_coverage_rule`.

**`imu-vehicle` "failed, no axis constrained" on the pooled bag.** Root cause: a
genuine data limit, plus a reporting defect. The estimator solved on all 1268
motions, but every axis std is over the 0.1 deg observability bound:

| Bag | roll (deg) | pitch (deg) | yaw (deg) |
| --- | --- | --- | --- |
| 0015 (`estimate`) | unobservable, std 54.85 | **+0.409 +- 0.075, observed** | unobservable, std 0.199 |
| 0009 | +1.81, std 0.62 | +0.233, std 0.14 | -0.258, std 0.38 |
| pooled 0005 + 0014 + 0022 | +0.98, std 0.57 | +0.437, std 0.16 | -0.323, std 0.16 |

The pooled std is the jackknife over blocks: pitch differs between drives (the
single-drive pitch of the [`check` table](../tutorials/calibrex_check.md#phase-c1-results-on-kitti-raw-development-drives-only)
is 0.18 on 0005, 0.47 on 0014 and 0.41 on 0015), so pooling does not shrink it
below the bound. The `check` pass on 0015 and the `estimate` failure on the pool
are consistent: the options are the same, and `calibrex estimate` on 0015 alone
also gives pitch +0.409 +- 0.075 (observed). Pooled drives are not mishandled:
the motions are split at gaps (`split_at_gaps`) and every drive gets its own
blocks. The defect: `estimate` replaced every axis of such a failed pair with
`no_estimate` and the bare reason "no axis was constrained", dropping each axis's
status and std. It now keeps `unobservable` with the reported std per axis
(values stay withheld) and lists them in the reason: "no axis was constrained by
the data: roll unobservable (reported std 0.574 deg); pitch unobservable
(reported std 0.16 deg); yaw unobservable (reported std 0.164 deg)". Regression
test: `test_pair_that_solved_but_constrained_nothing_keeps_its_axis_stds`.

The rest of the pooled estimate is unchanged by both fixes (same
`lidar-vehicle`, `ins-lidar` and `lidar-wheel_odometry` numbers as the table
above). The same run also shows `gnss-lidar` skipped on `/oxts/fix` (no RTK-grade
fixes) and `gnss-imu` skipped as `degenerate_frames`.

## Bugs this validation found

1. **The root frame's topics were not exported** (`calibrex estimate`, fixed
   here). With the IMU as the export root (the `calib.yaml` prior), the IMU
   topic's mapping `/livox/imu -> imu` was left out of `topic_frames`, because
   only exported entries were considered and the root is not an entry. Checking
   another bag with `--tf frames.yaml` then skipped `imu-lidar` and `gnss-imu`
   as `frame_not_in_tree`. Regression test:
   `test_topic_of_the_root_frame_is_mapped_in_the_export`.
2. **The progress line of `estimate` started at the machine's uptime**
   (`[44h00m elapsed]` on the first line): the timeline's pair clock was only set
   by the first pair, and `estimate` prints before any pair. It now starts at
   construction. Regression test:
   `test_stage_before_the_first_pair_measures_from_construction`.

No wrong sign, wrong axis status or wrongly rooted export was found in the
vehicle or GNSS pairs: the exported rotations follow the `check` conventions
(KITTI `T_imu_velo` roll -0.923 against `calib_imu_to_velo` -0.849), and an axis
the data did not observe was never written as measured. Additional unit tests pin
the KITTI and RTK-SLAM shapes: a vehicle-root export keeps the prior roll and
marks it, without a prior it is omitted, and a one-axis GNSS lever arm keeps the
prior for the rest.

## Limitations

- Every vehicle and GNSS export here needed a prior for most axes. A bag without
  a vendor `/tf_static` yields **no** vehicle or GNSS frame on these short or
  hand-held recordings.
- Only the development drives of KITTI and the already spent RTK-SLAM sequences
  were used, with one estimate and one or two round-trip bags each. There is no
  repeat over splits and no uncertainty on the deltas beyond the reported std.
- The composed `base_link` to `imu_link` of the two vehicle edges inherits both
  prior roll values, which cancel; it is not a measured roll.
- `lidar-wheel_odometry` reports its time offset on a 10 Hz lattice (66-70 ms,
  std 0.00 or 5.6 ms, status `estimated` or `unobservable`). That is not
  meaningful for KITTI's OXTS proxy, and it is not exported.
- `imu-lidar` cannot run on KITTI: the 10 Hz OXTS gyro is below the 20 Hz the
  estimator's coverage rule needs, so it is skipped with that reason (see above).
  The pooled `imu-vehicle` stays `failed`: its axes are not constrained at this
  span.
