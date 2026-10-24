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

Looking for `calibrex check` or `calibrex estimate` instead? The [bag check page](../app/check.html)
plans, checks and estimates on your own rosbag2 in the browser; see
[the workflow](workflow.md#in-the-browser) and
[Plan and run a bag in the browser](calibrex_check.md#plan-and-run-a-bag-in-the-browser).

The bag check page includes a live dashboard with a separate card for each CPU
worker: current sensor pair, stage, recent processed-item counts and stage history.
Counts restart on a new stage or pass and are activity indicators, not residuals
or accuracy measurements. IMU rotation solves also emit a separate live curve of
the noise-normalized training residual RMS, starting at initialization and then
at each robust IRLS round. Each independent solve (including uncertainty refits)
starts a new curve; curves are not joined across fits or workers. These residuals
can increase during robust reweighting and are not a held-out accuracy verdict.
Other solvers show processing activity without inventing a residual curve.
After a run, the axis explorer shows candidate errors
against tolerances in Check mode, or measured values with 1σ uncertainty in
Estimate mode. Unchecked and unobserved axes retain their reasons; prior values
are not shown as measurements. The original JSON and HTML reports remain available.

### Explore the real-data sample

**Explore sample** opens the NTU VIRAL tnp_01 point-cloud comparison immediately,
without waiting for Python. Drag to orbit, compare Before/After, move the reveal
slider, or play a reveal. Cyan is the reference LiDAR map; orange is the second
LiDAR. These are two saved transforms: an intentionally perturbed start and the
Calibrex estimate from the full development recording. The reveal moves a divider;
it does not show a live solve or optimizer iterations. The 19.1 → 4.4 cm residuals
come from the full saved subset in the linked provenance manifest; the browser
uses a reduced, ground-filtered display subset. NTU VIRAL data is CC BY-NC-SA 4.0.

**Plan the tiny bag** is separate: its 2 s synthetic bag tests topic discovery and
planning, and does not provide enough motion for a calibration. **Check your bag**
opens the input flow for an actual browser run on your own recording.

Rebuild the digest-bound display asset with
`python tools/build_alignment_demo.py` (requires the development image dependencies).

### Inspect the rig in 3D

The bag check page draws the candidate frame tree in an interactive 3D view.
Drag or use arrow keys to rotate, use +/− to zoom, select XY/XZ views, or choose a
frame to inspect its position in the root frame and its local quaternion.
The grid is a reference XY plane, not reconstructed point-cloud geometry.
Positions share a metric scale; axis glyph lengths are illustrative.
Disconnected trees have separate selectors: their relative placement is unknown.

After Estimate, the viewer draws only frames in the exported tree. Amber dashed
edges indicate axes filled from a prior, with **NOT MEASURED** shown in the
selected frame's details. Descendant poses inherit that indication. Omitted
frames and unexported pair placeholders are not drawn. A Check view remains a
view of the deployed candidate, not a new estimate or an accuracy verdict.
The viewer requires no WebGL, external 3D library or additional network request.

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
