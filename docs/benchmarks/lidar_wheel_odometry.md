# LiDAR-Wheel Odometry

`calibrex lidar-wheel` calibrates a LiDAR against wheel odometry. Wheel
odometry means the vehicle's forward speed and yaw rate
(`calibrex.solvers.lidar_wheel_solver`). The command estimates:

- the rotation of `T_vehicle_lidar`;
- the wheel-speed scale `s` (`v_true = s v_wheel`, a tyre-radius error);
- the clock offset (`t_wheel = t_lidar + dt`);
- the LiDAR's lever ahead of the turning axle.

It builds on the non-holonomic constraints of [lidar-vehicle](kitti_lidar_vehicle.md),
adding two matches in the vehicle frame: the forward speed and the yaw rate.

```bash
# Any LiDAR trajectory (TUM) and wheel CSV (t, speed, yaw_rate):
calibrex lidar-wheel trajectory --trajectory lidar.tum --wheel wheel.csv --output lidar_wheel.yaml
```

The evidence is the same as for lidar-vehicle: 10-second held-out blocks and
an 8-group block jackknife. Known-bad controls must raise the held-out
chi-square by at least 9: a 1 deg rotation, a 20 ms clock offset, and a 1 %
speed scale. The clock offset is first scanned from the yaw rate alone,
because wheel data interpolated between samples leaves the cost with kinks;
a gradient start at zero stopped 20 ms short in the tests.

## KITTI, with a stand-in for wheels

KITTI raw has no wheel odometry. `calibrex lidar-wheel kitti` substitutes the
OXTS forward speed `vf` and yaw rate `wu`. The speed scale and the clock
offset therefore describe the OXTS navigation solution, not wheels. The run
used the development drives 0005, 0009, 0014, 0015, and 0022, and 1998
motions.

Verdict: `warn`.

| Parameter | Estimate | Reported std | Held-out control |
| --- | ---: | ---: | --- |
| roll | 0.22 deg | 0.42 deg (unobservable) | Δχ² 47 |
| pitch | 0.61 deg | 0.04 deg | Δχ² 993 |
| yaw | -0.26 deg | 0.05 deg | Δχ² 815 |
| lever | 0.65 m | 0.03 m | — |
| speed scale | 1.0034 | 0.0022 | **1 % not detected (Δχ² -45)** |
| clock offset | 73.9 ms | 4.1 ms | 20 ms detected (Δχ² 58) |

- **The rotation matches** the motion-only [lidar-vehicle](kitti_lidar_vehicle.md)
  estimate within 0.004 deg.
- **The speed scale is flagged.** The held-out blocks fit better with the
  scale 1 % higher, so train and held-out disagree on it (odometry scale
  varies between drives), and the policy warns instead of passing.
- **The clock offset does not describe the timestamps.** 74 ms is far from
  the +4 ms that the [INS-LiDAR hand-eye](kitti_ins_lidar.md) finds between
  OXTS poses and the LiDAR. It is consistent with the OXTS velocity output
  being filtered and lagging its pose.

[Artifact](../assets/kitti_lidar_vehicle/lidar_wheel_proxy_dev_2011_09_26.yaml).

No SOTA claim is made for lidar-wheel_odometry. Its first audit, on NCLT, is refuted
(below).

## A wheeled robot: NCLT (development)

NCLT (the University of Michigan North Campus Long-Term dataset; Carlevaris-Bianco, Ushani
and Eustice, IJRR 2016; Open Database License) records a Segway that drives at 1–1.5 m/s.
It carries a Velodyne HDL-32E mounted upside down, wheel encoders, a FOG and an IMU.
`tools/nclt_to_rosbag2.py` converts a window of a session to a rosbag2:

- the raw Velodyne hits in the sensor frame, one scan per turn of the head;
- NCLT's 100 Hz odometry (wheels, FOG heading, IMU tilt) as `nav_msgs/Odometry`;
- NCLT's own LiDAR calibration (Table 4) as `/tf_static`.

The reference is not independent. NCLT estimated its calibration in a SLAM problem with
odometry factors and LiDAR scan matching.

Development on session 2012-01-08 (spent) found three faults, all fixed:

- **Upside-down LiDAR.**
  - The solver took "up" to point along the sensor's z, so the estimate came out turned
    by 180 deg.
  - The wheel yaw rate now decides which way is up. The sensor's z is used only when the
    wheels never turn.
- **Speed gate.**
  - The 2 m/s gate kept almost nothing of a robot at 1–1.5 m/s.
  - `calibrex check` and `estimate` keep motions from 0.5 m/s, ten times the velocity
    noise.
  - The solver's default, and the frozen KITTI benchmarks above, keep 2 m/s.
- **Scan period.**
  - The HDL-32E turns at about 7.4 Hz. Fixed 100 ms scans held three quarters of a turn,
    so scan matching saw each direction one turn (135 ms) apart and the LiDAR odometry
    came out a third too fast (speed scale 1.30).
  - With one scan per turn, the LiDAR's displacement over 50 scans (7 s) is 0.999 of the
    ground truth's (median).

`calibrex check` on 2012-01-08 from 60 s, against NCLT's calibration:

| Window | Motions | Rotation error | Roll / pitch / yaw error (std) | Lever | Speed scale | check |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 10 min | 4388 | 1.38 deg | 0.78 (0.42) / 1.13 (0.77) / 0.10 (0.19) deg | -0.012 m | 0.972 | inconclusive |
| 20 min | 8731 | 0.76 deg | -0.15 (0.55) / 0.74 (0.27) / 0.05 (0.14) deg | 0.001 m | 0.976 | inconclusive |
| 30 min | 13048 | 0.60 deg | -0.07 (0.31) / 0.60 (0.19) / 0.04 (0.12) deg | 0.015 m | 0.976 | inconclusive |

What the development runs show:

- Yaw and the lever agree with NCLT's calibration. NCLT's lever is 0.002 m.
- Pitch carries the error. At 30 minutes it is 0.60 deg, 3.2 times its reported std,
  but within check's fail bound of 1.11 deg.
- `check` is inconclusive at every window length: no axis reaches the 0.1 deg std that
  the vehicle pairs require before they judge.
- The clock offset is 0.10 s. It is reported, not judged. The scans are stamped at the
  start of their turn.

[Development records](../assets/nclt_lidar_wheel/development/summary.json).

The audit on five unseen sessions, one hour each, was
[pre-registered](nclt_lidar_wheel_preregistration.yaml). It tested two claims:

- rotation within 0.5 deg and lever within 0.20 m on every session;
- `check` never fails NCLT's calibration, and no axis is beyond check's fail bound.

At 30 minutes the development session missed the 0.5 deg gate on its pitch.

## Result: accuracy refuted (0/5 sessions), check honesty supported

The [pre-registered](nclt_lidar_wheel_preregistration.yaml) audit ran once, on 2026-11-03.
It ran from a worktree at the pre-registration commit (a2e297e8), whose `src/` and
`tools/` equal `frozen_code_commit` (640a3389). Each session's window was one hour,
starting 60 s after its first Velodyne packet.

| Session | Turns/s | Rotation error | Roll / pitch / yaw error (std) | Lever error | Speed scale | Clock offset | check |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 2012-02-02 | 7.4 | 1.20 deg | 0.80 (0.30) / 0.90 (0.45) / 0.05 (0.13) deg | 0.11 m | 0.979 | 76 ms | inconclusive |
| 2012-05-11 | 8.2 | 2.79 deg | -1.37 (0.28) / 2.43 (0.83) / 0.15 (0.16) deg | 0.13 m | 0.988 | 84 ms | inconclusive |
| 2012-08-04 | 8.2 | 1.09 deg | 0.95 (0.30) / 0.52 (0.81) / 0.06 (0.16) deg | 0.19 m | 0.966 | 116 ms | inconclusive |
| 2012-11-04 | 7.4 | 0.87 deg | -0.11 (0.36) / -0.86 (1.00) / -0.08 (0.11) deg | 0.01 m | 0.943 | 136 ms | inconclusive |
| 2013-02-23 | 10.7 | 0.74 deg | -0.13 (0.21) / 0.58 (0.87) / 0.43 (0.08) deg | 0.21 m | 0.941 | 58 ms | pass (yaw only) |

Every session converged.

- **Accuracy: refuted, 0 of 5 sessions.**
  - The rotation errors were 0.74–2.79 deg, nearly all in roll and pitch.
  - Yaw was within 0.15 deg on four sessions and 0.43 deg on the fifth.
  - The lever was within 0.20 m on four sessions and 0.21 m off on 2013-02-23.
- **Check honesty: supported.**
  - `check` never failed NCLT's calibration.
  - No axis was beyond check's fail bound 2 x max(3 std, 0.5 deg).
  - On four sessions no axis reached the 0.1 deg std the vehicle pairs need before
    they judge, so `check` said inconclusive.
  - On 2013-02-23 it judged yaw and passed it: 0.43 deg against the 0.5 deg floor.
- **The reported stds are too small on this data.**
  - The judged yaw was 5.6 of its stds off.
  - Roll was 4.9 and 3.2 of its stds off on 2012-05-11 and 2012-08-04.
  - The 0.5 deg floor, not the stds, kept these within check's tolerance.
- **The head's spin rate differs between sessions** (7.4–10.7 turns per second).
  Scans of one turn each followed it; fixed-length scans would not have.
- **The reference has its limits.**
  - NCLT gives one calibration for 13 months of sessions, with no stated uncertainty.
  - A remount would show here as an error.
  - The consistent yaw suggests no large one; roll and pitch are what this method and
    the reference disagree on.

For a wheeled robot like this one, then:

- `lidar-wheel_odometry` measures the LiDAR's heading (yaw) to about half a degree.
- It does not measure roll and pitch to better than 1–2.5 deg in an hour of driving.
- `check` mostly stays inconclusive rather than giving a wrong verdict.

[Run records](../assets/nclt_lidar_wheel/evaluation/summary.json): per session the check
artifact, the evidence artifact, the score and the log. The five sessions are spent.

## Development after the audit: the stds hold within a recording, the vehicle frame moves between recordings

Why did errors reach 5.6 reported stds? The six spent sessions (2012-01-08 and the five
evaluation sessions) served as development data. Each session's hour, and each of its two
halves, was fitted separately (`calibrex.evaluation.lidar_wheel` with check's options; the
full-hour fits reproduce the audit's numbers).

- **Within a recording, the stds are honest.** The two halves of a session differ by
  their stds' expected amount:

  | Axis | Half difference / its expected std (six sessions) | RMS |
  | --- | --- | ---: |
  | roll | 1.77, 0.28, 0.05, -0.57, -0.71, 0.40 | 0.84 |
  | pitch | 1.27, 0.36, 0.62, 2.03, 0.13, 0.39 | 1.03 |
  | yaw | -0.37, 0.12, 0.48, 0.95, -0.28, -0.84 | 0.59 |

- **Between recordings, the estimate moves by more than the stds.** This is measured
  without the reference: it is the spread of the six full-hour estimates around their
  mean.

  | Axis | Spread (SD) | RMS reported std | Excess | Mean error vs NCLT |
  | --- | ---: | ---: | ---: | ---: |
  | roll | 0.85 deg | 0.31 deg | 0.80 deg | -0.05 deg |
  | pitch | 1.06 deg | 0.74 deg | 0.76 deg | +0.64 deg |
  | yaw | 0.17 deg | 0.12 deg | 0.12 deg | +0.11 deg |

What this means:

- Each session holds its own roll and pitch through the hour: 2012-05-11's halves agree
  at -1.28 and -1.31 deg of roll; 2012-08-04's at +0.82 and +1.05 deg.
- What changes between sessions is the vehicle frame the method measures against. That
  frame is the ground and the motion, not the chassis. A Segway's chassis tilt against
  the ground can change with tyre pressure, load and balancing, and the LiDAR mount can
  move over 13 months.
- On average, NCLT's single calibration agrees on roll (mean -0.05 deg).

So the stds are not wrong, and inflating them would not be honest. They describe how well
one recording fixes its own vehicle frame, and they do that correctly. A chassis
calibration checked against one recording of a robot like this one can differ by about
0.8 deg in roll and pitch for reasons that are not the LiDAR mount.

[Records](../assets/nclt_lidar_wheel/development_halves/halves.json) (per-session capture
and fit JSONs; the scripts as text).

The converter also resumes a cut-off transfer of the 15–19 GB Velodyne archives, with a
Range request pinned by If-Match; a proxy tunnel dropped one after 4 GB.
