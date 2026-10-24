# Explore the real-data demos

Open the [browser gallery](../app/check.html) and select a method tab. Eleven
panels now cover the original five sensor-pair demos and six added explorations.

| Panel | What to try | Data and interpretation |
|---|---|---|
| LiDAR–LiDAR | Orbit the point clouds and slide the comparison | Saved NTU VIRAL alignment |
| IMU–LiDAR | Play or stop the deskew recording | Real MID360 sweep |
| Camera–IMU | Compare measured motion windows, explore rotation × time and inspect residuals | Measured Hilti camera/gyro series; saved baseline, fixed gyro bias |
| Camera–LiDAR | Change six offsets and compare the projections | Real KITTI returns and photographs; manual experiment |
| Fisheye stereo | Keep the photographed pair while switching models; orbit 3D and inspect matches | Real images; supplied KB4 and newly fitted MEI/DS; no depth truth |
| Method comparison | Switch four native board methods | Shared ACFR fit/holdout split; separate targetless A2D2 evidence |
| Robot hand–eye | Switch eight methods and inspect motion closure | Measured ETHZ poses; closure is not absolute accuracy |
| GNSS tracks | Zoom receiver tracks and change a candidate lever arm | Measured comma2k19 fixes; fused path is not truth |
| Wheels / INS | Select each wheel or the body-axis view | Measured wheel speeds and fused camera FRD velocity |
| Radar tracks | Select a snapshot and change candidate yaw | Real processed CAN targets; reported relative speed, not raw Doppler |
| Live calibration | Run, step, stop or reset a six-parameter fit | Actual browser updates on measured ACFR centres; holdout excluded from fitting |

In **Live calibration**, switch **Observation condition** between full coverage,
two measured captures, and four explicitly injected fit outliers. Compare
**Least squares** with **Huber**, then run or single-step the fit. The rank warning
blocks underconstrained geometry; the unchanged holdout curve shows how outlier
bias differs from numerical convergence. These are controlled changes to a real
board dataset, not simulated sensor recordings or a literal vehicle-motion test.

In **Camera–IMU**, the time slider changes which IMU samples are averaged
over the fixed camera intervals, while the rotation slider injects one
camera-frame rotation around the saved estimate. The displayed offset convention
stays consistent across the range. Choose all 59 observations or either of two
non-overlapping 20-observation windows selected by camera-rate RMS. Their
side-by-side heatmaps share a colour scale; switching windows preserves the
injected parameters. Higher RMS does not guarantee calibration accuracy. The 3D residual heatmap,
time profile, selected-axis curves and numerical interval inspection update
together. The near-minimum contour is not a statistical confidence region, and
the grid minimum is not asserted to be calibration truth. See the
[clock experiment guide](../assets/camera_imu_lock/timing.md) for integration,
sign convention, source records and interpretation.

The [source guide](../assets/evidence-demos/README.md) describes coordinate
conventions, licenses, reproducible producers and the limits of each view.
Each tab clearly distinguishes saved results, manual geometry and actual live
optimization. The [backlog](../browser_demo_backlog.md) records the scope delivered
for the seven requested additions.

In **Fisheye stereo**, click a sparse depth dot or use the match slider and
previous/next buttons. Both rectified display crops update together at a responsive
magnification, with pixel coordinates, an 11×11 neighbourhood outline and the
saved correspondence at the cross. The 41×41 context helps inspect repeated
patterns. Saved epipolar guides remain visible; this does not recompute ZNCC
or certify the match. Missing border context is padded rather than copied from
the other camera. Estimated depth has no independent ground-truth validation.

The stereo **3D** view back-projects each saved match into the rectified left
camera using `X = (u − (width−1)/2) Z / f`, `Y = (v − (height−1)/2) Z / f` and
the existing `Z = fB/d`. Coordinates are metres, X right, Y down, Z forward;
right-image vertical mismatch remains a separate diagnostic. The metric grid
lies on the camera Y=0 plane; it is a coordinate guide, not an inferred floor. Drag or use orbit
sliders to rotate the orthographic view. Click a point to select its image
match; the match slider and previous/next buttons provide keyboard access.

The default display includes Z ≤ 1 m. Choose Z ≤ 3 m or **All estimated depths**
to include farther estimates; the count always reports points outside the
display range. This is a display filter, not an outlier rejection or quality
claim. **Focus selected match** reveals a selected point outside the range and
centres the view on it. **Reset 3D view** fits the current depth range. Colours
clip at 0.3–3 m, while coordinates keep their full values. The view shows sparse
estimated matches, not a dense reconstruction or independently validated depth.

All KB4 / MEI / Double Sphere models now support pairs `000`, `014` and `027`.
Changing a model keeps the photographed pair selected; the saved rectification,
match set, 3D view and close-ups update together. Raw photographs stay identical
for that pair. Match counts and model-specific match sets do not establish a
model accuracy ranking; there is no independent depth truth. The fit-level
excluded-capture reprojection diagnostic is shared across that model’s pairs.
