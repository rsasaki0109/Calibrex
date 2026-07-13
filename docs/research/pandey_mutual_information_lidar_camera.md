# Pandey targetless camera--LiDAR mutual-information research note

## Scope and primary source

Calibrex implements Gaurav Pandey, James McBride, Silvio Savarese, and Ryan
Eustice, *Automatic Targetless Extrinsic Calibration of a 3D Lidar and Camera
by Maximizing Mutual Information*, AAAI 2012, DOI
[`10.1609/aaai.v26i1.8379`](https://doi.org/10.1609/aaai.v26i1.8379). The
[author-hosted primary paper](https://robots.engin.umich.edu/publications/gpandey-2012a.pdf)
is the implementation source. No upstream implementation code is copied into
`src/calibrex`.

The typed, ROS-independent core is
`src/calibrex/solvers/pandey_mutual_information_solver.py`. It estimates
`T_camera_lidar`, with
`p_camera = R_camera_lidar p_lidar + t_camera_lidar`, from synchronized
luminance images and calibrated LiDAR reflectivity. The A2D2 filesystem and
pipeline boundary is isolated in
`native_pandey_mutual_information_solver.py`.

## Mathematical and numerical contract

For every candidate six-vector `[x, y, z, roll, pitch, yaw]`, transformed
LiDAR points are projected through the declared calibrated camera model and
the image is sampled bilinearly. The aggregate reflectivity/luminance samples
define

```text
I(X;Y) = H(X) + H(Y) - H(X,Y).
```

This follows equations 1--8. The paper's continuous Gaussian Parzen estimate
is specialized to a finite soft 2D histogram convolved with a Gaussian whose
bandwidth is derived from the sample covariance. Four-bin deposition and
bilinear image sampling keep the finite objective responsive to sub-pixel
motion. Raw MI in nats is optimized; normalized MI is separately reported for
cross-run interpretation.

Equations 9--11 are implemented as central numerical gradients and normalized
Barzilai--Borwein ascent. Step clipping and monotonic backtracking are explicit
Calibrex numerical safeguards, not claims about the paper. Optimization uses
only seeded train frames. Unchanged holdout frames are never used for stopping.

The implementation supports pinhole and OpenCV fisheye projection without an
OpenCV runtime dependency. It also declares whether input coordinates use the
standard optical-z convention or A2D2's forward-x/left-y/up-z camera view.

## Falsification and observability

- Twelve signed known-bad candidates apply ±5 cm x/y/z and ±5°
  roll/pitch/yaw to the fitted six-vector.
- Each control records held-out normalized MI, score drop, and detectability.
- The full central-difference Hessian of **negative** held-out MI reports rank,
  spectrum, negative directions, and positive condition number.
- That Hessian is an objective-curvature diagnostic. It is explicitly not a
  covariance estimate and does not claim the paper's Fisher-information or
  Cramér--Rao bound result.
- Constant reflectivity, insufficient projected support, and too few train
  frames remain explicit weak or insufficient evidence; they are not promoted
  to successful calibration.

The synthetic evaluation uses five independently textured images, varied
depth from 3--14 m, 900 reflective points per frame, a full six-DoF truth, a
deterministic 80/20 frame split, and noisy reflectivity. From a local
1 cm/0.3° perturbation, the native solver recovers translation within 11 mm
and rotation within 0.1°, improves train MI, scores the untouched holdout,
produces all 12 controls, and materializes the six-dimensional curvature.

## Public A2D2 evidence and limitation

The reproducible public example range-fetches frames 60 and 61 from A2D2
sequence `20180810_150607`, plus the official `cams_lidars.json`. Every member
has a pinned SHA-256 digest and stays below 3.1 MB, so the 48--54 GB tar objects
do not need to be downloaded in full:

```bash
python3 tools/download_public_dataset.py a2d2_pandey_mutual_information
calibrex calibrate examples/public_datasets/a2d2_pandey_mutual_information/config.yaml
calibrex validate outputs/a2d2_pandey_mutual_information/result.yaml --json
calibrex verify outputs/a2d2_pandey_mutual_information/bundle.json
```

The [official A2D2 tutorial](https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/tutorial.ipynb)
states that these LiDAR points are already mapped into the corresponding
camera view and supplies their undistorted pixel rows and columns. Therefore
this run is useful real-image/real-reflectivity execution, holdout, known-bad,
and curvature evidence, but it is **not independent extrinsic accuracy
evidence**. Calibrex records that limitation in the manifest, metrics,
provenance, warning list, and public verdict.

With the unchanged generic MI gate, the two-frame run reports normalized
holdout MI `0.03686`, below the existing `0.1` WARN threshold. Ten of twelve
signed controls lower MI, and the objective Hessian is rank 6, but the public
verdict remains FAIL. Full numerical rank and probe response are not allowed to
override weak modality dependence or the pre-registration limitation. The
estimate is 0.0270 m and 0.743° from A2D2's distributed identity registration;
those are non-independent diagnostics. A future PASS requires a public raw,
unregistered, multi-view sequence with independently measured extrinsics; the
gate is not weakened for this dataset.

## Provenance contract

Every result records the DOI and primary PDF, transform and camera-axis
conventions, KDE specialization, optimizer safeguards, train/holdout frame
IDs, iteration trace, all probes, full curvature payload, input paths/sizes/
SHA-256 digests, A2D2 tutorial URL, pre-registration limitation, and the
Apache-2.0/CC BY-ND license boundary.
