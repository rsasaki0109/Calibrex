# Metric stereo motion validation and36-pair pilot

Pre-registered2026-10-30. Camera-only; no LiDAR arrays, target extrinsics or OXTS
motion used. First validate metric PnP on18 known-motion correspondence controls:
xyz rotation ±1deg and same-axis translation ±.1m, clean and disparity-noise seeds
730/731 (fB390, sigma.5px,600points z5–30m, .5px target noise and10% random pixel
outliers). Seed construction in frozen driver. Require ALL18 rotation<.2deg,
translation<.05m. Controls validate geometry/scale/robust PnP, not image matching
or full stereo inference on rendered data. Failure blocks real execution with
control outputs retained; no gate relaxation or resampling after outcome.

Run all36 previously frozen short pairs, including eight earlier support failures.
Recompute grayscale stereo with unchanged re-rectification/SGBM/restoration and
camera projection matrices; cache all54 unique frame depths. For18 centers require
exact depth equality to archived SGBM. Match equalized left images with ORB2000,
Hamming cross-check. Sample original-camera stereo depth at nearest keypoint pixel,
finite(.1,80]m; backproject at subpixel keypoint ray. No new quality mask.

Estimate target-from-source using OpenCV EPNP RANSAC1000 trials,2px,.999,seed0,
then inlier-only LM. Estimate both directions independently using each source's
stereo depth; record complete inlier identities and estimated4x4 transforms.
Internal support gate: >=30inliers, >=50% valid-pair inlier ratio and >=6 image
cells in both images for BOTH directions; reverse*forward cycle<=.5deg/.1m.
Keep every pair and explicit failure, independent of previous essential screen.
Both directions share images/matches/stereo error; cycle agreement is not accuracy
or ground truth. Metric scale derives from camera baseline only, but is not
independently validated on real motion. No claim of a trusted temporal selector.

Report controls, all36 real outcomes if controls pass, depth hashes, exact center
parity, motion magnitudes, internal support and failures. Pin source/image/calibration
hashes before fitting. No fresh data, extrinsic fit, ICP, model threshold retuning,
production adoption or SOTA claim. Controlled gate and real support are distinct.
