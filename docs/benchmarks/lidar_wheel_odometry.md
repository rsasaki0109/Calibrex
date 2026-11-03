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

No SOTA claim is made for lidar-wheel_odometry.

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

[Development records](../assets/nclt_lidar_wheel/development/).

The audit on five unseen sessions, one hour each, is
[pre-registered](nclt_lidar_wheel_preregistration.yaml). It tests two claims:

- rotation within 0.5 deg and lever within 0.20 m on every session;
- `check` never fails NCLT's calibration, and no axis is beyond check's fail bound.

At 30 minutes the development session missed the 0.5 deg gate on its pitch.
