# Explore the real-data demos

Open the [browser gallery](../app/check.html) and select a method tab. Eleven
panels now cover the original five sensor-pair demos and six added explorations.

| Panel | What to try | Data and interpretation |
|---|---|---|
| LiDAR–LiDAR | Orbit the point clouds and slide the comparison | Saved NTU VIRAL alignment |
| IMU–LiDAR | Play or stop the deskew recording | Real MID360 sweep |
| Camera–IMU | Explore rotation × time, click heatmap cells and inspect residuals | Measured Hilti camera/gyro series; saved baseline, fixed gyro bias |
| Camera–LiDAR | Change six offsets and compare the projections | Real KITTI returns and photographs; manual experiment |
| Fisheye stereo | Switch KB4 / MEI / Double Sphere and inspect a match | Real images; supplied KB4 and newly fitted MEI/DS; no depth truth |
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
and all 59 intervals stay consistent across the range. The 3D residual heatmap,
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
