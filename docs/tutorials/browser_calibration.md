# Browser Calibration

**[Open the browser calibration page](../app/index.html)**

The page calibrates an IMU against a trajectory of another sensor. It runs
entirely in your browser: Calibrex, NumPy, and SciPy run under
[Pyodide](https://pyodide.org), and no data leaves the page. It estimates:

- the rotation of `T_lidar_imu`;
- the clock offset (`t_imu = t_sensor + dt`);
- the gyro bias;
- the lever arm, from the accelerometer.

The computation, held-out evidence, and artifacts are the same as for the
[MID360 IMU-LiDAR benchmark](../benchmarks/mid360_imu_lidar.md). The only
difference is that the sensor motion comes from a trajectory you already
have, instead of Calibrex's own LiDAR odometry.

## Inputs

| Input | Format |
| --- | --- |
| IMU CSV | `t, gx, gy, gz, ax, ay, az` per line: seconds, rad/s, and m/s² or g. Commas or spaces; a header line is allowed. |
| Sensor trajectory | TUM: `t x y z qx qy qz qw` per line, poses `T_world_sensor`, `#` for comments. For example, a LiDAR odometry or SLAM output. |

Both inputs must share one clock, up to the constant offset that is
calibrated. The trajectory is cut into 10-second windows, and a gap longer
than 0.5 s starts a new block. At least three windows are needed, and a few
minutes of motion that rotates about every axis gives the best results.

Press **Use a synthetic example** to see a hand-held recording with a known
truth being calibrated. It takes about 10 s to load Python and about 10 s
more to calibrate.

## Outputs

- **Verdicts.** The rotation and the lever arm each get a pass, warn, fail,
  or inconclusive verdict, with the reasons.
- **Per-quantity table.** Each quantity has its reported std, its status
  (estimated or unobservable), and whether held-out windows detected the
  known-bad control.
- **Extrinsic.** The 4×4 `T_lidar_imu` and the clock offset.
- **Artifacts.** Downloadable, schema-valid `slac.imu_lidar_rotation/v0.1`
  and `slac.imu_lidar_translation/v0.1` artifacts with provenance, including
  the SHA-256 of your inputs.

## Checking the page

`tools/check_browser_page.mjs` runs the exact Python snippets embedded in the
page under Pyodide in Node. See the script header for how to run it.

## The same thing on the command line

```bash
calibrex imu-lidar trajectory --imu imu.csv --trajectory odometry.tum \
  --output-rotation rotation.yaml --output-translation translation.yaml
calibrex validate translation.yaml
```

Use `--acceleration-unit g` for Livox drivers, and `--mid360-reference` to
compare the result with the Livox MID360 design extrinsic.

## Limitations

- **The trajectory is not checked.** If it comes from a LiDAR-inertial
  odometry, it already depends on the IMU and on the extrinsic that odometry
  assumed.
- **Motion must excite every axis.** Planar or yaw-only motion leaves axes
  unobservable, and the verdict says so.
- **Accelerometer scale and axis misalignment are not estimated.**
