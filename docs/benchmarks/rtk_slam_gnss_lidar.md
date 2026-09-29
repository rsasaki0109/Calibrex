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

**`stadtgarten_seq1`: `inconclusive`** (89 windows from 8518 RTK-fixed scans)

| Quantity | Status | Estimate | Reported std | CAD reference | Difference | Held-out control |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| x | estimated | 7.2 cm | 0.6 cm | 3.4 cm | +3.8 cm | 5 cm detected (Δχ² 528) |
| y | estimated | 0.4 cm | 0.45 cm | 0.0 cm | +0.4 cm | 5 cm detected (Δχ² 435) |
| z | unobservable | 6.1 cm | 1.3 cm | 4.6 cm | +1.5 cm | 5 cm detected (Δχ² 57) |
| clock offset | estimated | 43.7 ms | 1.2 ms | — | — | 20 ms detected (Δχ² 51) |

**`stadtgarten_seq2`: `warn`** (37 windows from 3462 RTK-fixed scans)

- The lever arm stays unobservable (reported std 1.6-1.9 cm).
- The clock offset is 40.3 ± 4.2 ms, but held-out windows do not detect a
  20 ms shift. The artifact therefore warns rather than claims it.

In both sequences the held-out median displacement residual (2.2-2.9 cm)
matches the training residual. That is about the RTK noise level.

### Reading the results

- **The clock offset agrees across sequences** (43.7 and 40.3 ms).
- **y agrees with the CAD reference to 0.4 cm**, within its uncertainty.
- **x disagrees with the CAD reference by 3.8 cm (6.6 σ).** The data cannot
  tell whether the reference or the estimate is off. Antenna phase centres
  are poorly defined by CAD, and the MID360 sweeps are registered without
  motion compensation, which could act like a lever arm under fast rotation.
  Until an independent measurement settles it, treat x as a tension rather
  than as a calibrated value.
- **z needs more rotation about horizontal axes.** Only seq1 provides enough
  to bring z near the 1 cm observability threshold.

## Method

1. **Streaming segmentation.** The rosbag2 database, about 30 GB per sequence,
   is read once. LiDAR odometry runs only while RTK-fixed epochs cover the
   scans, restarts after every gap, and is cut at unreliable registrations
   and into 10-second windows.
2. **Scan-to-local-map odometry.** Each Livox scan is registered to the union
   of the last five scans. Against RTK, this cut the one-second distance error
   from 7.3 cm (scan to scan) to 2.2 cm.
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

- MID360 sweeps are not motion-compensated.
- Only RTK-fixed epochs are used: 54 % of seq1 and 40 % of seq2.
- LiDAR odometry has no gravity or heading. The per-window ENU alignment is
  estimated, which removes any information such an alignment would carry.
- One rig in one dataset family is not enough for a SOTA claim, so the
  [SOTA leaderboard](sota_leaderboard.md) is unchanged.
