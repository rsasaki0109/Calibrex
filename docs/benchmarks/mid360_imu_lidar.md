# MID360 IMU-LiDAR Rotation

This page reports Calibrex's native calibration of the Livox MID360 built-in
IMU against LiDAR odometry. The run estimates the rotation of `T_lidar_imu`,
the clock offset (`t_imu = t_lidar + dt`), and the gyro bias. The reference is
the MID360 design value: IMU axes aligned with the LiDAR. It is used only for
this comparison, and it is not a measurement.

Two recordings from two dataset families are used:

| Recording | Motion | Artifact |
| --- | --- | --- |
| [RTK-SLAM](https://rtk-slam-dataset.github.io/) `stadtgarten_seq2` | hand-held, 15 min | [`mid360_imu_lidar_rotation_rtk_slam_seq2.yaml`](../assets/mid360_imu_lidar_rotation_rtk_slam_seq2.yaml) |
| [Zenodo 14841855](https://zenodo.org/records/14841855) "Driving SLAM Test with Livox MID360" (K. Koide, CC BY 4.0) | vehicle, 277 s | [`mid360_imu_lidar_rotation_driving_slam.yaml`](../assets/mid360_imu_lidar_rotation_driving_slam.yaml) |

## Results

**RTK-SLAM seq2 (hand-held): `pass`.** Every rotation axis is estimated, and
held-out windows detect every known-bad shift.

| Quantity | Estimate (relative to design) | Reported std | Held-out control |
| --- | ---: | ---: | --- |
| roll | -0.080 deg | 0.025 deg | 1 deg detected (Δχ² 870) |
| pitch | +0.139 deg | 0.024 deg | 1 deg detected (Δχ² 1024) |
| yaw | -0.155 deg | 0.031 deg | 1 deg detected (Δχ² 643) |
| clock offset | +9.82 ms | 0.03 ms | 10 ms detected |
| gyro bias | (0.0156, -0.0025, 0.0017) rad/s | ≤ 0.0002 rad/s | — |

- The held-out median rate residual is 0.012 rad/s, the same as the training
  residual.
- The gyro-deskew iteration converged: in the last pass, every axis moved by
  at most 0.006 deg.

**Driving (vehicle): `inconclusive`.**

- Roll is estimated at -0.31 ± 0.09 deg.
- Pitch (± 0.31 deg) and yaw are not.
- A vehicle rotates almost only about the vertical axis, so a 1 deg yaw shift
  is invisible to held-out windows (Δχ² 3.0). The yaw value wanders between
  deskew passes, and the artifact reports it as unobservable.
- The clock offset is +2.3 ± 0.2 ms.

The two recordings come from different MID360 units, so their rotations need
not agree.

## What made the hand-held result possible

The first version of this page (PR #70) reported the hand-held recording as
inconclusive: yaw sat 1.4 deg from the design value, and the held-out residual
was 2.5 times the training residual. It attributed this to coning error in a
mean-rate approximation. **That diagnosis was wrong.** Replacing mean rates
with on-manifold gyro preintegration fixes coning in synthetic data, but it
left the real-data result unchanged (yaw -1.39 deg).

The residual grew with the rotation rate: 0.007 rad/s below 0.3 rad/s,
0.18 rad/s above. The cause was **deskewing**. Sweeps were motion-compensated
with the LiDAR's own previous motion at constant velocity, which cannot
follow fast hand-held changes of rotation. Deskewing with gyro rotations
instead cut the residual during rotation by about ten times, and the rotation
moved to within 0.16 deg of the design value.

Deskewing with the gyro feeds the extrinsic under test into the LiDAR
odometry, so the evaluation guards against the estimate merely confirming its
own deskew model:

1. **Iteration.** Pass 0 deskews with LiDAR-only motion. Each later pass
   deskews with the previous estimate. Passes continue until every axis moves
   less than 0.25 of its reported std, up to five passes.
2. **Convergence gate.** A `pass` is refused, and downgraded to `warn`, while
   the last pass still moves any axis by more than its reported std. Before
   this gate existed, an unconverged second pass (yaw still moving 0.22 deg)
   had been labelled `pass`.
3. **Feedback check.** A final pass deskews with the estimate rotated by
   2 deg on purpose. The artifact records the fraction of that error that
   survives in the result (`deskew_feedback_ratio`): 0.20 for the hand-held
   recording and 0.04 for the vehicle. Because the fraction is well below 1,
   the fixed point is not an artefact of its starting value; the iteration
   converges geometrically at that rate.

## Comparison with LI-Init

The first audited SOTA claim of Calibrex compares this calibration with
[LI-Init](https://github.com/hku-mars/LiDAR_IMU_Init) (commit `66b157a`,
GPL-2.0). LI-Init runs only in a container
(`tools/external/li_init/`), on the same recordings converted to ROS 1 bags,
and its result is imported as a `slac.external_calibration_run/v0.1`
artifact ([seq2](../assets/li_init/rtk_slam_seq2_external_run.yaml),
[driving](../assets/li_init/driving_external_run.yaml)).

| Recording | LI-Init result |
| --- | --- |
| RTK-SLAM seq2 | refined rotation (1.25, -4.03, -0.45) deg from design, clock offset -55.8 ms |
| Driving | no result: its excitation meter stopped at 81/73/72 % |

**Pre-registration.** The claim, metric, requirements, and thresholds were
committed in
[`mid360_imu_lidar_preregistration.yaml`](mid360_imu_lidar_preregistration.yaml)
before any held-out score of LI-Init was computed. The audit protocol pins
that file's SHA-256, and no threshold has changed since.

**Scoring.** Each method's rotation and clock offset are held fixed, the
LiDAR odometry is recomputed with sweeps deskewed by that method's own
extrinsic, and only the gyro bias is refit on the train windows. Every method
is scored on the same held-out time spans: every third 10-second window of the
design-reference odometry. Each span is one paired benchmark split.

| Method (RTK-SLAM seq2, 29 held-out spans) | Mean of span medians | Median | p95 |
| --- | ---: | ---: | ---: |
| Calibrex | 0.0123 rad/s | 0.0117 rad/s | 0.018 rad/s |
| MID360 design value (sanity row) | 0.0299 rad/s | 0.0285 rad/s | 0.052 rad/s |
| LI-Init | 0.148 rad/s | 0.106 rad/s | 0.347 rad/s |

The paired improvement of Calibrex over LI-Init is 0.136 rad/s (95 %
bootstrap CI 0.090-0.185). On the driving recording, Calibrex's held-out span
median is 0.0052 rad/s over 9 spans.

**Audit: `supported`** (4/4 gates,
[protocol](../assets/mid360_imu_lidar_sota_protocol.yaml),
[result](../assets/mid360_imu_lidar_sota_audit.yaml)).

- Calibrex returns an estimate on every held-out span of both recordings.
- Its paired improvement over LI-Init has a 95 % CI lower bound ≥ 0.
- Its mean held-out residual on RTK-SLAM is ≤ 0.02 rad/s.

**The first build of this audit was `refuted`, by a pairing bug.**

- Each method's own deskewing changes its odometry segmentation. LI-Init's
  run had 89 windows instead of 88, so its every-third-window held-out choice
  landed on different stretches of the recording, about 11 s away.
- The builder took the union of all held-out windows and counted a window
  missing from Calibrex's list as a Calibrex failure, which gave a failure
  rate of 0.5.
- The fix scores every method on the same time spans (`holdout_spans_s`), and
  the builder now refuses scores taken on different spans.
- The thresholds and requirements were not touched. On its own segmentation,
  LI-Init's held-out median is 0.089 rad/s; that is a different set of spans,
  and it is also far above Calibrex's.

What this claim does **not** say:

- It is a single LI-Init run with its default MID360 parameters, on one
  hand-held recording where it returned a result.
- Held-out fit to gyro-deskewed LiDAR odometry is the metric, not agreement
  with a surveyed ground truth. No ground truth exists for these units.
- LI-Init also estimates translation; this claim covers only rotation and
  clock offset.

## Method

1. **Odometry.** Scan-to-local-map point-to-plane LiDAR odometry runs on
   deskewed sweeps: first with LiDAR-only constant velocity, then with gyro
   rotations mapped by the current extrinsic estimate.
2. **Rotation increments.** For consecutive scans, the LiDAR rotation
   increment is compared with the preintegrated gyro increment
   `R dR_imu(b, dt) R^T`. The raw gyro is integrated once; interval
   increments and first-order bias Jacobians come from cumulative sums, so
   every evaluation is vectorized.
3. **Fit.** A Procrustes initialization handles any mounting. It is followed
   by Huber IRLS over the rotation, gyro bias, and clock offset.
4. **Held-out evidence.**
   - Every third 10-second window is held out.
   - An 8-group window jackknife sets the reported std when it is larger
     than the analytic std.
   - Each rotation axis is shifted by 1 deg and the clock offset by 10 ms,
     and held-out windows must detect each shift (Δχ² ≥ 9).

Recording conventions are explicit per stream profile:

| Profile | Point topic | Point time | Acceleration |
| --- | --- | --- | --- |
| `rtk-slam` | `/livox/points` | `offset_time`, seconds after the header | m/s² |
| `livox-ros-driver2` | `/livox/lidar` | `timestamp`, absolute nanoseconds | g |

```bash
calibrex imu-lidar livox BAG_DIR --profile rtk-slam \
  --dataset-family rtk_slam --dataset-license "see the RTK-SLAM dataset page" \
  --output seq2.yaml
```

## Limitations

- The design reference is not a measurement.
- The translation of `T_lidar_imu` is not estimated. It needs the
  accelerometer and a full LiDAR-inertial model.
- Gyro deskewing makes the LiDAR odometry depend on the IMU. The feedback
  ratio quantifies this dependence but does not remove it.
- The external baseline is a single tool (LI-Init). The
  [SOTA leaderboard](sota_leaderboard.md) records the claim with the scope
  above.
