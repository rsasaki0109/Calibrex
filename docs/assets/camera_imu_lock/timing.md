# Camera–IMU rotation and clock-offset experiment

The browser's **Camera–IMU** tab uses the committed real Hilti 2022 `exp21`
camera-rate and gyro series. The original [provenance](series.provenance.json)
records extraction, topics, source hashes and Hilti SLAM Challenge 2022 terms.
No rendered angular rates or new recordings are substituted.

Select a plot axis, inject −50 to +50 ms relative to the saved time estimate,
and vary −5 to +5 degrees around a selected camera-frame rotation axis. The rate
plot compares camera-derived rates
with gyro means over the **same camera intervals**; the next plot shows their
selected-axis difference. The lower plot shows the three-axis residual RMSE
over the selected 59- or 20-observation window at the selected rotation. Its vertical scale and
those of the two other plots stay fixed while either injection changes. Click
the sensitivity curve to change time only; **Zero rotation** changes rotation
only, and **Reset both** returns both injections to zero.

## Measured motion windows

Compare the full 59 exported observations with two non-overlapping, contiguous
windows of 20 observations each. The producer chooses the pair with the largest
difference in camera-derived three-axis angular-rate RMS, breaking ties by the
first ascending pair of start indices. Gyro residuals and calibration estimates
do not enter this selection. The higher-rate window is exported observations
40–59 (34.15 deg/s RMS); the lower-rate window is observations 15–34
(14.41 deg/s RMS). These identifiers refer to the exported subset.

The two small heatmaps show both windows side by side. Selecting a window keeps
the injected time and rotation unchanged and updates the main heatmap, curves,
observation count, timestamp range and numerical interval inspection. Colour
and curve scales remain shared across windows. Each window uses its fixed
intervals throughout the ±50 ms range, without extrapolation.

These are post-hoc development windows. Higher camera-rate RMS measures motion
amplitude, potentially including tracking noise; it does not establish richer
excitation, observability or accuracy. Equal counts do not make their motions,
durations or noise identical. Comparing their residual minima or contour areas
is not an independent accuracy ranking or uncertainty estimate.

## Rotation × time heatmap

The heatmap evaluates 41 rotation angles (0.25° spacing) and 101 clock shifts
(1 ms spacing). Every cell uses the same measured camera intervals within the selected window.
The colour scale is shared by all measured windows and X/Y/Z rotation slices. Clicking a cell updates both
controls and the interval curves; equivalent keyboard access is available
through the two sliders. The plot axis and injected rotation axis are independent.

The white cross marks the selected cell; the cyan square marks the saved
estimate. The cyan contour encloses cells within **+0.5 deg/s** of that slice's
minimum. This is a displayed residual contour, **not a confidence region,
covariance estimate, success gate or proof of observability**. Its width can
reflect noise and model errors as well as similar parameter combinations.
**Inspect slice minimum** selects the lowest displayed grid value; it does not
run a full calibration optimizer or assert the true clock/extrinsic solution.

Only one rotation axis and time vary in each slice. The other two rotation
components and IMU-frame gyro bias are not refitted. A narrow region in this
two-parameter view does not establish that the complete calibration is uniquely
constrained. Other parameter directions, unseen motions and independent data
remain untested.

The clock convention is:

`t_IMU = t_camera + saved_offset + injected_shift`.

Positive injection samples later IMU timestamps for the same camera interval.
The saved offset is approximately `+1.701 ms`. Zero time injection restores that
estimate; it does not assert clock truth. Rotation is applied as
`R_candidate = R_camera_axis(injected_angle) × R_saved` using the right-hand
rule. The saved rotation supplies the baseline; the saved **IMU-frame gyro bias**
remains fixed. Rotating the already bias-corrected camera-frame mean is
equivalent to applying this candidate rotation to the original IMU mean.

Gyro means use the difference of sampled trapezoidal cumulative integrals,
linearly interpolated at the two shifted endpoints and divided by interval
duration. This matches the existing recorded-comparison metric. Both camera
and gyro curves show **interval means**, not instantaneous measurements. No
extrapolation or changing interval subset is permitted over the slider range.

This is a development-window sensitivity experiment, not a new optimization,
independent accuracy benchmark or calibration verdict. Camera tracking noise,
rotation error, bias and timing can all contribute to residuals. Even the
displayed-window minimum would not establish the true recording-wide clock
offset. The earlier GIF remains linked as a separate saved rotation/time/bias
comparison; it is not presented as actual optimization progress.

Rebuild with `PYTHONPATH=src:tools python tools/build_camera_imu_timing_demo.py`.
The [generated asset](../../app/samples/camera_imu_timing.json) binds original
series, producer and existing metric source to SHA-256. Reference values are
computed with the existing Python interval metric at −50, −10, 0, +10 and +50 ms,
and checked against the browser implementation. Additional rotation/time
references use the original Python metric with the left-multiplied
candidate rotation `R_camera_axis × R_saved`. The fast surface calculation uses
rotation-invariant vector energies and cross-moments; tests compare it with
direct vector rotation and residual evaluation. All updates run locally after
the measured series loads; no animation starts automatically.

Window references also compare the browser against original IMU cumulative
integrals and the saved bias/rotation, independently of camera-frame moments.
