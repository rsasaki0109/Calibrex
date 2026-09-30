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
