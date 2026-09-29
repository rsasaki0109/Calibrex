# RTK-SLAM GNSS-LiDAR Lever Arm

This page reports Calibrex's native GNSS antenna lever-arm and clock-offset
calibration against LiDAR odometry. The data are the
[RTK-SLAM dataset](https://rtk-slam-dataset.github.io/) (University of
Stuttgart), which pairs a hand-held Livox MID360 with an RTK GNSS receiver.
The evidence artifacts are
[`rtk_slam_gnss_lidar_stadtgarten_seq1.yaml`](../assets/rtk_slam_gnss_lidar_stadtgarten_seq1.yaml)
and
[`rtk_slam_gnss_lidar_stadtgarten_seq2.yaml`](../assets/rtk_slam_gnss_lidar_stadtgarten_seq2.yaml)
(`slac.gnss_lidar_lever_arm/v0.1`).

## Results

The lever arm is the antenna phase centre in the LiDAR frame. The reference
combines the IMU position from the MID360 manual with the CAD antenna offset
from `calib.yaml`. It is used only for this comparison, and it is not
metrology. The clock offset follows `t_gnss = t_lidar + dt`.

Both sequences are `inconclusive`: the horizontal lever arm is calibrated,
z is not, and neither determines the clock offset. Sweeps are
motion-compensated (see below).

| Quantity | seq1 (89 windows) | seq2 (38 windows) | CAD reference |
| --- | --- | --- | ---: |
| x | **5.1 ± 0.7 cm**, estimated | **5.2 ± 1.0 cm**, estimated | 3.4 cm |
| y | -1.3 ± 0.9 cm, estimated | -0.2 ± 0.5 cm, estimated | 0.0 cm |
| z | 7.6 ± 1.3 cm, unobservable | 4.2 ± 4.0 cm, unobservable | 4.6 cm |
| clock offset | -6.6 ± 9.2 ms, unobservable | -23.8 ± 12 ms, unobservable | — |

Every 5 cm lever-arm control is detected on held-out windows in both
sequences (Δχ² 33-278). The held-out median displacement residual
(2.7-2.8 cm) matches the training residual, which is about the RTK noise
level.

### Reading the results

- **The two sequences agree on x (5.1 and 5.2 cm).** Both sit about 1.7 cm
  from the CAD value (1.8-2.4 σ), in the same direction. Antenna phase
  centres are poorly defined by CAD, so a consistent offset of this size is
  plausible. Until an independent measurement settles it, treat the CAD
  difference as a tension to resolve, not as an error in either value.
- **y agrees with the CAD reference** in both sequences (within 1.4 σ).
- **z needs more rotation about horizontal axes** than either walk provides.
- **Motion compensation mattered.** Without it (the first version of this
  page, PR #69), seq1 gave x = 7.2 ± 0.6 cm, 6.6 σ from CAD, and both
  sequences reported a ~42 ms clock offset. That offset was an artefact:
  undeskewed sweeps are stamped at the start of the sweep, but their geometry
  sits mid-sweep, about 50 ms later. With deskewing the offset is
  indistinguishable from zero.

## Method

1. **Streaming segmentation.** The rosbag2 database, about 30 GB per sequence,
   is read once. LiDAR odometry runs only while RTK-fixed epochs cover the
   scans, restarts after every gap, and is cut at unreliable registrations
   and into 10-second windows.
2. **Motion-compensated scan-to-local-map odometry.** Each Livox sweep is
   deskewed from per-point `offset_time` under a constant-velocity model and
   registered to the union of the last five scans. Against RTK, the
   one-second distance error fell from 7.3 cm (scan to scan) to 2.2 cm with
   the local map, and then to 1.5 cm with deskewing.
3. **Variable projection.** For a trial lever arm and clock offset, each
   window's ENU alignment is the closed-form weighted Procrustes rotation.
   The robust outer fit therefore has four unknowns, and its covariance
   already marginalizes the alignments.
4. **Held-out evidence.**
   - Every third window is held out.
   - An 8-group window jackknife sets the reported std when it is larger
     than the analytic std.
   - Each estimated quantity is shifted by 5 cm or 20 ms, and held-out
     windows must detect the shift.

Reproduce with:

```bash
calibrex gnss-lidar rtk-slam \
  --bag ros2/stadtgarten_seq1 --rtk rtk_slam_eval/data/stadtgarten_seq1/rtk.txt \
  --calib rtk_slam_eval/calib/calib.yaml --output stadtgarten_seq1.yaml
```

## Findings along the way

- **Interpolating noisy GNSS biases the clock offset.** A linear blend of two
  noisy epochs has less variance than either epoch. An estimator that
  interpolates the GNSS track therefore prefers offsets that land between
  epochs: a synthetic 30 ms offset was estimated at 47 ms. Calibrex
  interpolates the smooth odometry to the observed GNSS epochs instead. It
  also scales the pose-noise variance by the blend fraction, so the fit no
  longer favours offsets that average two noisy poses.
- **Relative displacements see the clock offset only through velocity
  changes.** Under constant velocity, shifting time leaves every displacement
  unchanged. Hand-held walking provides the accelerations that make the
  offset observable; steady vehicle cruising would not.
- **The `T_lidar_imu` convention in `calib.yaml` had to be pinned down.**
  Its translation is the negative of the IMU position given in the MID360
  manual, so it maps LiDAR-frame points into the IMU frame. The opposite
  reading moves the reference by up to 9 cm, and it lies farther from the
  estimate in every axis.

## Limitations

- Deskewing assumes constant velocity within a sweep.
- Only RTK-fixed epochs are used: 54 % of seq1 and 40 % of seq2.
- LiDAR odometry has no gravity or heading. The per-window ENU alignment is
  estimated, which removes any information such an alignment would carry.
- One rig in one dataset family is not enough for a SOTA claim, so the
  [SOTA leaderboard](sota_leaderboard.md) is unchanged.
