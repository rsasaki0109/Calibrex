# Camera–IMU clock-offset experiment

The browser's **Camera–IMU** tab uses the committed real Hilti 2022 `exp21`
camera-rate and gyro series. The original [provenance](series.provenance.json)
records extraction, topics, source hashes and Hilti SLAM Challenge 2022 terms.
No rendered angular rates or new recordings are substituted.

Select camera X/Y/Z, inject −50 to +50 ms relative to the saved time estimate,
or click the sensitivity curve. The upper plot compares camera-derived rates
with gyro means over the **same camera intervals**; the next plot shows their
selected-axis difference. The lower plot shows the three-axis residual RMSE
over all 59 displayed intervals. Its vertical scale and those of the two other
plots stay fixed while the time slider moves.

The clock convention is:

`t_IMU = t_camera + saved_offset + injected_shift`.

Positive injection samples later IMU timestamps for the same camera interval.
The saved offset is approximately `+1.701 ms`. Zero injection restores that
estimate; it does not assert clock truth. Saved camera-from-IMU rotation and
IMU-frame gyro bias remain fixed throughout the experiment.

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
and checked against the browser implementation. All updates run locally after
the measured series loads; no animation starts automatically.
