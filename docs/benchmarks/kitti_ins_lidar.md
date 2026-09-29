# KITTI INS-LiDAR Hand-Eye

This page reports Calibrex's native INS-LiDAR trajectory hand-eye calibration
on KITTI raw. The evidence artifact is
[`kitti_ins_lidar_2011_09_26.yaml`](../assets/kitti_ins_lidar_2011_09_26.yaml)
(`slac.ins_lidar_hand_eye/v0.1`). The run pools drives `0005` and `0009` of
2011-09-26, which share one calibration (597 Velodyne sweeps, 165 motions).

**Verdict: `inconclusive`, a partial calibration.** Roll, pitch, and the
clock offset are constrained by the data, and held-out blocks detect a
known-bad shift of each. Yaw and the three translations are not reliably
determined by these two drives, and the artifact says so instead of reporting
numbers that look precise.

## Results

The reference is KITTI's `calib_imu_to_velo.txt`, inverted to `T_imu_velo`.
It is used only for this comparison, never in the fit, and it is itself an
estimate rather than independent metrology.

| DoF | Status | Estimate | Reported std | KITTI reference | Difference | Held-out control |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| roll | estimated | -0.974 deg | 0.062 deg | -0.849 deg | -0.124 deg | 1 deg detected (Δχ² 17.5) |
| pitch | estimated | 0.056 deg | 0.072 deg | 0.117 deg | -0.060 deg | 1 deg detected (Δχ² 148.9) |
| yaw | unobservable | -0.211 deg | 0.514 deg | 0.043 deg | -0.254 deg | 1 deg detected (Δχ² 93.8) |
| x | unobservable | 0.871 m | 0.637 m | 0.811 m | 0.060 m | 0.1 m not detected |
| y | unobservable | -0.243 m | 0.408 m | -0.307 m | 0.064 m | 0.1 m not detected |
| z | unobservable | 0.576 m | 3.600 m | 0.803 m | -0.227 m | 0.1 m not detected |
| clock offset | estimated | +4.0 ms | 1.9 ms | — | — | 20 ms detected (Δχ² 100.9) |
| reference scale | estimated | 1.0001 | 0.0042 | — | — | — |

The reported std is the larger of the analytic std and the block-jackknife
spread. For yaw the data do respond to a 1 deg shift, but leaving out a single
5-second block moves the estimate by about 0.5 deg. The estimate depends on
one or two turns, so yaw stays unobservable.

The held-out median motion residual is 0.089 deg in rotation and 0.079 m in
translation, which matches the training fit (0.090 deg, 0.107 m).

## Method

1. **LiDAR odometry from LiDAR alone.** A vectorized point-to-plane
   scan-to-scan registration starts from a constant-velocity guess, scaled
   across dropped frames. It never uses the INS as a prior, so the LiDAR
   motions cannot inherit the extrinsic under test.
2. **Joint solve.** `A(t + dt) X = X B` is solved for `X = T_ins_lidar`, the
   clock offset `dt`, and a reference trajectory scale, with Huber IRLS.
   Translation noise grows with motion length.
3. **Observability.** Each DoF is `estimated`, `prior`, or `unobservable`
   according to its reported std. A declared prior (for example
   `--prior z=0.80:0.02` from a CAD model) can constrain a DoF, and the
   artifact labels it as prior rather than data.
4. **Held-out evidence.**
   - Every third 5-second block is held out.
   - The block jackknife runs over the training blocks.
   - Each DoF is moved by a known-bad amount (1 deg, 0.1 m, or 20 ms).
   - A `pass` is refused while any extrinsic DoF is unobservable.

Reproduce with:

```bash
calibrex ins-lidar kitti \
  2011_09_26/2011_09_26_drive_0005_sync 2011_09_26/2011_09_26_drive_0009_sync \
  --output kitti_ins_lidar_2011_09_26.yaml
```

## Findings along the way

- **Positional frame pairing is wrong after dropped frames.** Drive `0009`
  lacks Velodyne files 177-180. Its timestamp file keeps a blank line for
  each missing file, so pairing sweeps with OXTS packets by directory position
  shifts every later frame. The median Velodyne-OXTS time difference becomes
  394 ms, and single-step motion residuals triple. Calibrex pairs frames by
  file index and interpolates the INS at LiDAR timestamps.
- **Rotation-first hand-eye fails silently on vehicles.** Near-planar driving
  aligns all rotation axes with the vertical. A Park-Martin solve on these
  drives reported "converged, rank 3" with a yaw error of up to 13 deg, and its
  holdout closure did not flag it. Solving rotation and translation jointly
  recovers yaw from the translation equations. Only translation along the
  vertical axis stays unobservable, which synthetic tests confirm.
- **The OXTS track and the LiDAR disagree on distance.** On drive `0005` the
  Mercator-projected OXTS steps are about 3 % shorter than the LiDAR-measured
  ones and 2.4 % shorter than OXTS's own velocity integral. On `0009` the
  mismatch is below 0.4 %. Without the scale nuisance this mismatch is
  absorbed by the lever arm; on `0005` it shifted the lateral translation by
  about 0.5 m.
- **Trajectory errors are correlated in time.** Lateral and vertical residuals
  grow roughly linearly with motion length. The analytic covariance is
  therefore overconfident: on drive `0009` alone it gave yaw ±0.06-0.08 deg
  while landing 0.38-0.42 deg from the reference. The block jackknife is what keeps the
  reported uncertainty honest.
- **Absolute timestamps froze the clock offset.** At UNIX times near
  1.3e9 s, float64 resolves only about 0.2 µs, which is finer than the
  optimizer's finite-difference step. The solver now works in epoch-relative
  time; a regression test covers it.

## Limitations

- Velodyne sweeps are registered as rigid snapshots, without motion
  compensation within a sweep.
- Drive `0001` has no OXTS timestamps. It is excluded rather than paired by
  position.
- Two drives from one day are one rig and one dataset family. A SOTA claim
  needs at least two dataset families, so the
  [SOTA leaderboard](sota_leaderboard.md) keeps `ins-lidar` at `no_claim`.
- For weakly constrained DoFs, the analytic std varies by up to about 20 %
  between numerically equivalent runs, because Huber weights can switch for
  residuals near the threshold. The jackknife and the larger-of rule absorb
  this variation.
