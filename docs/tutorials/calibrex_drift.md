# Detect calibration drift (`calibrex drift`)

`calibrex check` judges a bag against a calibration you hand it. `calibrex drift`
answers a different question when you have several recordings of the **same
rig** (different days, after a service, after a bump): did the extrinsic
calibration change between them? No reference calibration is needed.

```bash
calibrex drift day1/ day2/ day3/ --output drift/                        # every pair with an estimator
calibrex drift day1/ day2/ --pairs imu-lidar --max-duration-s 120 --output drift/ --html drift.html
calibrex drift a/ b/ c/ --vehicle-frame base_link --topic-kind /oxts/twist=wheel --output drift/
```

Give the bags in recording order. The flags of `calibrex estimate` apply
(`--tf` prior, `--frame-map`, `--vehicle-frame`, `--topic-kind`, `--pairs`,
`--max-duration-s`, `--cache-dir`, `--no-cache`, ...).

## What it does

1. Every bag is estimated with the machinery of [`calibrex estimate`](calibrex_check.md)
   (the per-bag `slac.bag_estimate/v0.1` artifact is written to
   `drift/<bag-name>/bag_estimate.json`). The
   [estimator cache](calibrex_check.md#estimator-cache) is shared with `check` and `estimate`,
   so re-running `drift` on bags it has seen costs seconds.
2. For each pair and each axis that the data **observed** in at least two bags
   (an axis that was `unobservable`, `control_not_detected` or not estimated
   never enters a test), with estimates `x_i` and standard deviations `s_i`:
    * pairwise difference `d_ij = x_i - x_j` against the tolerance
      `max(3 * sqrt(s_i^2 + s_j^2), floor)`. The floor is the minimum detectable
      change of the axis type, the same policy as `check`: 0.5 deg rotation,
      1.5 deg rotation for `imu-lidar` estimated on rigid scans (no deskew), 2 cm
      translation (`--rotation-floor-deg`, `--rigid-scan-rotation-floor-deg`,
      `--translation-floor-m`, `--sigma-k`);
    * a chi-square homogeneity test of the inverse-variance weighted mean
      (`--chi2-alpha`, default 0.01);
    * the axis is `drift` when a pair exceeds its tolerance **and** the
      chi-square test rejects homogeneity, `stable` when not, and
      `inconclusive` when fewer than two bags observed it.
3. **Clock offset.** The per-pair time offset of each bag's estimate (`time_offset` of the
   `slac.bag_estimate`) is compared with the same two tests, as the row `dt` in ms. Only a
   bag whose offset the estimator marks `estimated` (with a finite std) enters; an
   `unobservable` offset is skipped like an unobserved axis. The floor is per pair type:
   **1 ms for `camera-imu`, 2 ms for every other pair** (`--time-offset-floor-ms` sets one
   floor for all). `lidar-wheel_odometry` is never compared: its offset sits on the odometry
   sample lattice (KITTI's OXTS proxy gives 66 to 70 ms) and is not a clock-offset
   measurement; the output says so.
4. Pair verdict: `drift` if any axis **or the clock offset** drifts, else `stable` if any
   was tested, else `inconclusive`. Overall verdict: the worst pair
   (`stable < inconclusive < drift`). Exit status 1 on `drift` by default;
   `--fail-on inconclusive|never` changes it.
5. **Which bag moved.** With three or more bags the deviating bags are those
   whose removal restores consistency (fewer than half the bags may be
   removed; if the bags split instead, nobody is blamed). With two bags a
   difference cannot be attributed, and the output says so. For each deviating
   bag the artifact gives its offset from the consensus per axis and, when all
   three rotation axes were observed, the angle of the rotation between the two
   estimated rotations, and its clock-offset change. The attribution uses the axes and
   the clock offset of the pair together. When only the clock offset differs the next steps
   say so (check the time synchronisation of that recording), because re-calibrating the
   extrinsic would not help.

## Reading the output

```text
imu-lidar: drift  T_livox_lidar_livox_imu
  axis   stadtgarten_seq1  ...  construction_seq2           sg1_z1p0   max|diff|  min detectable  status
  roll    -0.218 +- 0.058  ...    -0.151 +- 0.053    -0.219 +- 0.058      0.176           0.500  stable
  pitch   +0.114 +- 0.058  ...    +0.081 +- 0.050    +0.112 +- 0.058      0.123           0.500  stable
  yaw     -0.229 +- 0.069  ...    -0.106 +- 0.073    -1.229 +- 0.069      1.123           0.500  drift
  sg1_z1p0 vs the others: roll -0.038 deg, pitch +0.047 deg, yaw -1.047 deg; rotation 1.05 deg
  deviating bag(s): sg1_z1p0
```
(Five recordings of one MID360 rig; the last is a copy with its IMU rotated by 1 deg about z.
The `...` columns are elided here.)

`min detectable` is the smallest pairwise tolerance among the bags that
observed the axis: **a change smaller than that cannot be flagged**, so a
`stable` verdict means "no change larger than this was found", not "nothing
changed". The numbers are the estimators' own standard deviations (analytic or
block jackknife), which are not guaranteed to be calibrated; the floor is what
protects against over-trusting a small std. Rotation axes are compared as the
estimators report them (rotation-vector components in degrees about the parent
frame's axes), which is exact enough for the sub-degree to few-degree changes
this is meant for.

`drift/calibration_drift.json` is `slac.calibration_drift/v0.1`
([schema](../reference/schemas.md)): the bag references (path, digest, the
per-bag estimate's path and sha256, cache hits and runtime), per pair and axis
every bag's observation and the tests, the thresholds, the next steps and the
provenance. `--html FILE` writes a self-contained report with the per-axis
estimates and the minimum detectable change drawn as a band.

## Known-bad control: a shifted IMU clock

`tools/shift_stamps_in_bag.py` copies a bag and adds a known offset to the header stamp of
every message of the chosen topics, which is what a changed clock offset (driver update, PTP
or hardware-timestamping change) does to a sensor's stamps:

```bash
python tools/shift_stamps_in_bag.py my_bag/ my_bag_dt/ --topic /alphasense/imu --shift-ms 2 \
  --max-duration-s 130
calibrex drift my_bag/ other_bag/ third_bag/ my_bag_dt/ --pairs camera-imu --output drift/
```

The estimators read the **header stamp** of Imu, Image and PointCloud2 messages (the bag's
log time only when the header stamp is zero), so that is what is shifted; the log (receive)
time is left alone unless `--shift-log-time` is given, and per-point time fields of a cloud
are not touched. Shifting the IMU by `+d` moves the estimated `camera-imu` offset by `+d`.

## Known-bad control: a remounted IMU

To check that the detector sees a real change, `tools/rotate_imu_in_bag.py`
copies a bag and rotates every IMU `angular_velocity` and `linear_acceleration`
vector by a known rotation, which is exactly what a physically remounted IMU
measures:

```bash
python tools/rotate_imu_in_bag.py my_bag/ my_bag_remounted/ --axis 0 0 1 --angle-deg 1.0 \
  --max-duration-s 130        # optional: keep only the first 130 s (a small copy)
calibrex drift my_bag/ other_bag/ my_bag_remounted/ --pairs imu-lidar --output drift/
```

Results on real data, including the smallest rotation that was detected and
the false-alarm behaviour on unmodified recordings, are on
[Drift on real data](../benchmarks/drift_real_data.md).

## Limits

* The bags must be of the same rig: frames and sensors are matched by pair and
  frame names, and a pair whose frames differ between bags is not compared.
* Only axes the estimators **observe** are compared. Short or low-excitation
  recordings leave most axes unobservable (a KITTI drive constrains one or two
  of the vehicle pair's axes), and the pair is then `inconclusive` rather than
  `stable`.
* Intrinsics are not compared. Time offsets are compared only where the estimator reports
  one as `estimated`; the floors (1 ms `camera-imu`, 2 ms otherwise) were set from the
  recording-to-recording scatter of the two datasets on the
  [evaluation page](../benchmarks/drift_real_data.md#clock-offsets), not from a larger study.
* A real change smaller than the minimum detectable change is not detected, and
  an underestimated std on a dataset the estimator handles poorly can produce a
  false alarm; the floors exist for the second case, and the evaluation page
  reports what was seen.
