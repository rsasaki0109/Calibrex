# `calibrex drift` on real data

[`calibrex drift`](../tutorials/calibrex_drift.md) (`slac.calibration_drift/v0.1`) tests whether
the extrinsic calibration of a rig changed between recordings of it. This page
collects the real-data runs behind it: the false-alarm behaviour on unmodified
recordings of one rig, and the detection of a known injected change.

These are tool-validation runs, not SOTA claims. No audit is attached, and the
RTK-SLAM sequences were already spent for earlier claims. Every dataset was
already on disk; nothing was downloaded, and no held-out KITTI drive
(0027 to 0059) was touched. Estimates are the `calibrex estimate` machinery;
the per-bag artifacts are written next to the drift artifact.

## Same-rig recordings, nothing modified (expected: `stable`)

| Set | Pair, axes compared | Bags | Result | Largest pairwise \|z\| | Largest difference | Floor (min detectable) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| RTK-SLAM MID360, `stadtgarten_seq1`, `stadtgarten_seq2`, `construction_seq1`, `construction_seq2`, first 120 s | `imu-lidar` roll / pitch / yaw | 4 | **stable** (chi-square p 0.21 / 0.34 / 0.56) | 1.9 / 1.6 / 1.2 | 0.18 / 0.12 / 0.12 deg | 0.5 deg |
| the same | `imu-lidar` y | 2 | stable | 1.2 | 1.4 cm | 3.7 cm (sigma-limited) |
| the same | `imu-lidar` x, z | 1 each | inconclusive (observed in one bag) | - | - | - |
| Koide `indoor_easy_01`, `indoor_easy_02` (rigid depth-camera scans) | `imu-lidar` roll / pitch / yaw | 2 | **stable** (p 0.97 / 0.065 / **0.003**) | 0.04 / 1.8 / **2.95** | 0.01 / 0.68 / 0.91 deg | 1.5 deg (rigid-scan floor) |
| the same | `imu-lidar` x, y, z | 0 | inconclusive (unobservable on rigid scans) | - | - | - |
| KITTI dev drives `0009`, `0015`, pool `0005+0014+0022` (converted bags) | `lidar-wheel_odometry` yaw | 3 | **stable** (p 0.60) | 0.9 | 0.09 deg | 0.5 deg |
| the same | `lidar-vehicle` yaw | 2 | **stable** (p 0.36) | 0.9 | 0.09 deg | 0.5 deg |
| the same | `lidar-vehicle` / `lidar-wheel_odometry` pitch, `ins-lidar` roll, pitch, `imu-vehicle` pitch | 1 each | inconclusive | - | - | - |
| the same | `imu-vehicle` (roll, yaw), `ins-lidar` yaw and x, y, z | 0 | inconclusive (unobservable) | - | - | - |

False-alarm behaviour, stated plainly:

* **No false drift in any of the three sets** at the default thresholds. The overall verdict is
  `stable` for RTK-SLAM and Koide and `inconclusive` for KITTI (the worst pair, because
  one-drive and pooled KITTI estimates leave most axes of `imu-vehicle` and `ins-lidar`
  unobserved in two bags; the pairs that could be compared are `stable`).
* **The floor is doing real work on Koide.** The yaw of the two Koide recordings differs by
  0.91 deg with a combined std of 0.31 deg (z = 2.95, chi-square p = 0.003): without the
  1.5 deg rigid-scan floor this would be flagged. The reported std of the depth-camera
  `imu-lidar` estimate is clearly smaller than the recording-to-recording scatter, so the
  default for rigid scans is conservative on purpose; two recordings are not enough to
  say more.
* On RTK-SLAM the four recordings agree to 0.18 deg, within the reported stds (std 0.05 to 0.08
  deg), so there the 0.5 deg floor is a policy choice, not a rescue.

## Known-bad control: a remounted IMU

`tools/rotate_imu_in_bag.py` copies a bag and rotates every IMU `angular_velocity` and
`linear_acceleration` vector by a known rotation `R` (exactly what an IMU remounted by `R`
measures); all other messages are byte-identical. The copies are written outside the repository
(for RTK-SLAM only the first 130 s of `stadtgarten_seq1`, `--max-duration-s 130`). `calibrex drift`
was then run on the four unmodified RTK-SLAM bags **plus** one modified copy (five bags), first 120 s,
`imu-lidar` only. The reference is the original bag's estimate, so the *twin* column is the exact
effect of the injection on the estimate (the same data, only the IMU rotated).

| Injected rotation | Twin: modified minus original estimate | Verdict | Deviating bag | Change reported (vs consensus of the others) | Largest difference / floor |
| --- | --- | --- | --- | --- | --- |
| 0.25 deg about z | yaw -0.250 deg (angle 0.250) | **stable** (not flagged) | - | - | 0.37 / 0.5 deg |
| 0.5 deg about z | yaw -0.500 deg | **drift** | the modified bag | 0.55 deg (yaw -0.547) | 0.62 / 0.5 deg |
| 1.0 deg about z | yaw -1.000 deg | **drift** | the modified bag | 1.05 deg (yaw -1.047) | 1.12 / 0.5 deg |
| 2.0 deg about z | yaw -2.000 deg | **drift** | the modified bag | 2.05 deg (yaw -2.047) | 2.12 / 0.5 deg |
| 1.0 deg about x | roll -1.000 deg | **drift** | the modified bag | 1.04 deg (roll -1.037) | 1.12 / 0.5 deg |

Findings:

* **The injected rotation is recovered to within its noise.** The estimate moves by exactly the
  injected angle (0.250, 0.500, 1.000, 2.000 deg; to three decimals), on the axis that was
  rotated (the IMU frame of the MID360 is aligned with the LiDAR frame, so a rotation about the
  IMU z shows up as yaw). The other two rotation axes move by 0.00 to 0.004 deg. Against the
  *consensus of the other recordings*, the change reads 0.04 to 0.05 deg larger than injected,
  which is the offset (about 0.05 deg) of the original bag from the others.
* **Smallest rotation detected with the default thresholds: 0.5 deg** (the rotation floor). The
  0.25 deg change *is* measured (z = 3.7 against the others, chi-square p = 0.002) and is **not
  flagged**, because it is below the 0.5 deg minimum detectable change by construction. This is a
  policy, not an estimator limit: with `--rotation-floor-deg 0.2` the four unmodified bags stay
  `stable` (tolerance 0.27 deg, sigma-limited) and the 0.25 deg bag is flagged `drift`
  (0.30 deg vs the others). The 0.5 deg detection is marginal in the sense that its difference
  (0.62 deg) is only 0.12 deg above the floor.
* **The deviating bag is named in all four detected cases** (five bags, so attribution is possible).
* With only two bags (the original and a modified copy) the verdict is `drift` and the output says the
  moved recording cannot be told; it asks for a third recording.

### A rigid-scan rig: Koide `indoor_easy_01` with its IMU remounted

`indoor_easy_01` (full 139 s) copied with the IMU rotated about z, compared with the two
unmodified bags (the original and `indoor_easy_02`):

| Injected rotation | Twin: modified minus original estimate (roll / pitch / yaw) | Verdict | Deviating bag | Largest difference / floor |
| --- | --- | --- | --- | --- |
| 2 deg about z | -0.87 / -1.39 / -1.54 deg (angle 2.000) | **drift** (yaw only, barely: 1.54 vs 1.5 deg) | the modified bag, after the pair-level attribution (below) | 1.54 / 1.5 deg |
| 3 deg about z | -1.30 / -2.09 / -2.31 deg (angle 3.000) | **drift** (pitch and yaw) | the modified bag | 2.31 / 1.5 deg |

The depth camera is not aligned with the IMU, so the rotation spreads over all three axes of
the parent frame. Smallest rotation detected on this rig with the default (rigid-scan) floor:
**2 deg, marginally** (1 deg is below the 1.5 deg floor by design; `check` has the same
resolution on this data). The change reported against the consensus of the other two bags is 1.4
and 2.4 deg for the 2 and 3 deg injections, less than the injected angle because the second
unmodified bag lies 1.1 deg from the original along the same direction: with two reference
recordings that scatter by 1 deg, only a change of a few degrees is unambiguous.

**A flaw found and fixed by this control.** The first version blamed the deviating bag per axis. On
the 2 deg case yaw alone puts the middle bag (`indoor_easy_02`) closer to the modified bag than to
the original, so the original `indoor_easy_01` was named. Attribution is now done over all
axes of the pair together (the bag whose removal leaves the smallest total chi-square; fewer than
half the bags may be removed), which names the modified bag; the regression test
`test_attribution_uses_all_axes_of_the_pair_together` uses these numbers.

## Runtimes

Estimator runs dominate and are cached (key: bag digest, estimator options, version and a content
hash of the estimation source), so a re-run costs seconds. Heavily loaded shared machine; first
runs are one process at a time.

| Run | First run | Cached re-run |
| --- | ---: | ---: |
| RTK-SLAM, 4 bags, `imu-lidar`, 120 s each | 5329 s (2066 + 1351 + 954 + 953 s per bag) | 21 s (43 s while another job ran) |
| + one modified RTK-SLAM copy (5 bags) | 1050 to 1150 s (the new bag; the four others 3 to 25 s) | 21 to 47 s |
| Koide, 2 bags, `imu-lidar` (full 139 s) | 378 s (178 + 198) | 8 to 14 s |
| + one modified Koide copy (3 bags) | 177 to 184 s (the new bag) | 13 s |
| KITTI, 3 bags, four vehicle pairs | 254 s (48 + 42 + 161) | 3 to 4 s |

## Reproducing

```bash
R=/path/to/rtk_slam/ros2
calibrex drift $R/stadtgarten_seq1 $R/stadtgarten_seq2 $R/construction_seq1 $R/construction_seq2 \
  --pairs imu-lidar --max-duration-s 120 --output drift_rtk --html drift_rtk.html
python tools/rotate_imu_in_bag.py $R/stadtgarten_seq1 sg1_z1p0 --axis 0 0 1 --angle-deg 1.0 \
  --max-duration-s 130
calibrex drift $R/stadtgarten_seq1 $R/stadtgarten_seq2 $R/construction_seq1 $R/construction_seq2 \
  sg1_z1p0 --pairs imu-lidar --max-duration-s 120 --output drift_rtk_z1p0
calibrex drift kitti_0009_bag kitti_0015_bag kitti_pool3_bag --vehicle-frame base_link \
  --topic-kind /oxts/twist=wheel --output drift_kitti
```

(KITTI bags are converted as in [the estimate page](estimate_real_data.md#vehicle-pairs-kitti-raw-development-drives).)

## Limitations

* Only the IMU extrinsic was perturbed. A remounted *LiDAR* or camera is a different physical
  change with the same effect on a relative extrinsic, but it was not injected here, and
  translation drift was not injected at all (the lever arm is observed only by a few pairs and
  bags: RTK-SLAM `imu-lidar` y in two bags, std 0.8 to 0.9 cm).
* The injection is a *synthetic* remount of one real recording, so the modified bag shares its
  data with the original. It shows that the test sees a known change and attributes it, not that
  real-world drift behaves the same.
* A `stable` verdict means no change above the minimum detectable change was found on the axes
  that were observed in two bags. KITTI pairs are mostly `inconclusive` for lack of observed axes.
* The estimators' stds are not guaranteed calibrated (see the Koide yaw above), so the floors, not
  the stds, set what is flagged on rigid-scan data. Rotation axes are differences of rotation-vector
  components, a small-angle approximation (the angle in the table is computed from the rotations).
* Three or more bags are needed to say which one moved, and the recording-to-recording
  scatter limits attribution when only two reference bags exist.
