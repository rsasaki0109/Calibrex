# Stereo round-trip and disparity-band diagnostic

## Frozen diagnostic protocol

Use all 12 distinct spent frames from the re-rectification comparison, with its
saved rotations, projection matrices, image remaps and SGBM settings/version.
Recompute left disparity and require exact equality of the restored depth with
its hashed saved array. Compute an independent reverse map by matching horizontally
flipped right/left images with the same positive disparity range, then flip back.
This is an additional image-based diagnostic, not a replacement for SGBM's existing
internal `disp12MaxDiff` check.

For each left match, sample right disparity at rounded `u_left - disparity`.
Absolute forward-minus-reverse disparity <= 1 pixel is consistent; > 1 is
inconsistent; absent/out-of-domain reverse matches are separately missing.
Verify remap support at both ends. No threshold fitting is performed.

Project LiDAR using the dataset reference into the newly rectified left camera,
z-buffer the nearest point per pixel and retain all finite stereo depths in the
existing 0.1–80 m interval. **Do not apply a reference-depth agreement gate or a
2,000-point cap.** Visibility from the right camera is not established by this
left-image z-buffer. Reference uncertainty, timing and occlusions remain factors.

Report all/consistent/inconsistent/reverse-missing populations in fixed disparity
bands [0,8), [8,16), [16,32), [32,64), [64,128) pixels. Depth difference is LiDAR z
minus stereo z; disparity difference is measured minus reference-predicted.
Report counts, medians and absolute-error 90th percentiles, using null for empty
sets. No optimizer, calibration verdict or production frontend is changed.

The diagnostic is committed before examining results. This is post-hoc development
on spent data; even reverse-consistent matches are not independent ground truth.
Low disparity amplifies pixel errors into metre errors, and different populations
must not be called a paired accuracy improvement.
