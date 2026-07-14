# Levinson--Thrun online camera--LiDAR calibration research note

## Scope and primary source

Calibrex implements Jesse Levinson and Sebastian Thrun, *Automatic Online
Calibration of Cameras and Lasers*, RSS 2013, DOI
[`10.15607/RSS.2013.IX.029`](https://doi.org/10.15607/RSS.2013.IX.029). The
[RSS proceedings paper](https://roboticsproceedings.org/rss09/p29.pdf) is the
primary implementation source. No upstream implementation code is copied into
`src/calibrex`.

The typed, ROS-independent core is
`src/calibrex/solvers/levinson_thrun_online_solver.py`. It monitors or tracks
`T_camera_lidar`, with
`p_camera = R_camera_lidar p_lidar + t_camera_lidar`, over a rolling frame
window. Dataset decoding and the A2D2 specialization remain in the native
adapter.

## Paper contract and declared specializations

The implementation follows the paper's equations and search procedure:

1. Equation 1 forms a grayscale camera-edge image from the maximum absolute
   difference to the eight neighboring pixels, then spreads the response with
   `alpha = 1/3` and `gamma = 0.98`.
2. Equation 2 retains the nearer return at an organized LiDAR beam
   discontinuity, rejects jumps below 0.3 m, and weights retained points by
   the square root of the positive range jump.
3. Equation 3 sums the projected camera-edge response multiplied by those
   discontinuity weights over the most recent `w` frames. This raw sum is the
   search objective. A support-normalized companion is reported only for
   holdout comparison and diagnostics.
4. The center is compared with all `3^6 - 1 = 728` non-center perturbations in
   the radius-one six-DoF grid. `F_C` is the fraction with a strictly lower
   objective than the center.
5. Equation 4 converts `F_C` to a calibrated probability using the paper's
   reported correct distribution `(mu=0.997, sigma=0.014)` and incorrect
   distribution `(mu=0.505, sigma=0.14)`.
6. Tracking selects the best of all 729 candidates only when it improves the
   current center; otherwise it keeps the current transform.

The paper writes the spilloff distance with a max-coordinate expression. The
finite implementation therefore uses an eight-neighbor Chebyshev spread. It
also bounds propagation by a declared `max_radius`; this is a numerical work
budget, not an alteration of the local edge or decay equations. Pinhole and
OpenCV fisheye projection are implemented without a ROS or OpenCV runtime
dependency.

Frames are assigned by a seeded train/shadow-holdout split. Only train frames
enter the rolling monitor and update selection. Holdout frames score the final
transform, the twelve signed controls, and numerical curvature, and can never
choose an update.

## Falsification, drift, and observability

- Twelve signed controls apply the configured +/- translation along x/y/z and
  +/- rotation about roll/pitch/yaw. Each records the untouched holdout
  objective drop and detectability.
- The full central-difference Hessian of the **negative** support-normalized
  holdout objective reports rank, spectrum, negative directions, and positive
  condition number. It is numerical objective curvature, explicitly not
  covariance.
- Insufficient frames or projected points remain insufficient evidence.
  Rank, probability, and controls are separate diagnostics and cannot promote
  a limited public run to an accuracy claim.

Synthetic tests cover the correct local optimum, a local full-six-DoF offset,
a sudden 10 cm/0.25 degree miscalibration, and gradual three-axis rotational
drift of 0.02 degree per axis per frame. The local tracker recovers translation
within 11 mm and rotation within 0.11 degree. On gradual drift it applies at
least three updates, reduces mean rotation error below 60% of the fixed
identity baseline, and keeps the final combined rotation error below 0.25
degree. These are deterministic regression contracts, not a reproduction of
the paper's complete driving experiment.

## Public A2D2 execution: intentionally inconclusive

The public example reuses two authenticated camera/LiDAR pairs from A2D2
sequence `20180810_150607` and the official `cams_lidars.json`:

```bash
python3 tools/download_public_dataset.py a2d2_pandey_mutual_information
calibrex calibrate examples/public_datasets/a2d2_pandey_mutual_information/levinson_online_config.yaml
calibrex validate outputs/a2d2_levinson_thrun_online/result.yaml --json
calibrex verify outputs/a2d2_levinson_thrun_online/bundle.json
```

The A2D2 view-filtered NPZ files do not preserve organized beam neighborhoods,
so the adapter uses the dataset's `pcloud_attr.boundary` selection with unit
weights instead of claiming to recompute Equation 2. The distributed points
are also already registered into the corresponding camera view. Finally, only
two frames are available, so the example uses `w=1`, not the paper's robust
`w=9` window. All three limitations are recorded in metrics, warnings, and
provenance.

The authenticated run projects 1,827 train and 1,842 holdout boundary points.
Its center worsening fraction is `0.52885`, Equation 4 probability is about
`1.55e-243`, six of twelve signed controls are detected, and numerical
curvature is rank 6 with positive condition number `58.53`. Thus the monitor
rejects a calibrated interpretation under this specialization. Calibrex
reports WARN/INCONCLUSIVE: full curvature rank cannot override the incomplete
window, substituted depth-edge extraction, or non-independent extrinsics. The
gate and paper distributions are not weakened to force a PASS.

## Provenance contract

Every result records the DOI and primary PDF, transform/camera-axis
conventions, rolling-window and grid parameters, seeded train/holdout frame
IDs, complete monitoring timeline, all controls, curvature payload, extraction
specializations, input paths/sizes/SHA-256 digests, pinned-digest verification,
and the Apache-2.0/CC BY-ND license boundary.
