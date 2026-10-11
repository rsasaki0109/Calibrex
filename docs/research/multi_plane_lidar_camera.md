# Multi-plane LiDAR-camera calibration research note

## Scope and implementation decision

Calibrex treats multi-plane calibration as a correspondence solver, not as a
checkerboard detector. The ROS-independent native core consumes oriented plane
pairs and estimates `T_camera_lidar`, with
`p_camera = R_camera_lidar p_lidar + t_camera_lidar`. Image, point-cloud, and
target feature extraction remain adapter responsibilities.

The plane-only native implementation is
`src/calibrex/solvers/planar_board_lidar_camera_solver.py`. Calibrex also runs
an independent centre+normal baseline in
`src/calibrex/solvers/point_plane_lidar_camera_solver.py`. This is not a
duplicate: board centres constrain tangential displacement that plane equations
alone do not measure. The two transforms are retained separately and compared;
the independent baseline does not silently replace the configured output.
The same capture centres also feed a normal-free Horn absolute-orientation
baseline in `src/calibrex/solvers/horn_point_lidar_camera_solver.py`. This third
estimate isolates what the 3D centre correspondences constrain without using
any plane-normal information.

## Primary literature

- Zhang and Pless, *Extrinsic calibration of a camera and laser range finder*,
  IROS 2004, DOI `10.1109/IROS.2004.1389752`. This establishes the planar-target
  correspondence constraint and motivates the native method provenance.
- Fu et al., *LiDAR-Camera Calibration Under Arbitrary Configurations:
  Observability and Methods*, IEEE TIM 2020, DOI
  `10.1109/TIM.2019.2931526`. This motivates explicit minimal-observability and
  scattered-target diagnostics rather than accepting observation count alone.
- Verma et al., *Automatic extrinsic calibration between a camera and a 3D
  Lidar using 3D point and plane correspondences*, ITSC 2019, DOI
  `10.1109/ITSC.2019.8917108`. This is an independent multi-pose plane/centre
  comparison baseline.
- [Horn, *Closed-form solution of absolute orientation using unit
  quaternions*](https://doi.org/10.1364/JOSAA.4.000629), JOSA A 4(4), 1987.
  The [author-hosted paper](https://people.csail.mit.edu/bkph/papers/Absolute_Orientation_Scanned.pdf)
  derives a symmetric closed-form least-squares solution for corresponding 3D
  points using the maximum eigenvector of a symmetric quaternion objective.
- Zhou, Li, and Kaess, *Automatic Extrinsic Calibration of a Camera and a 3D
  LiDAR using Line and Plane Correspondences*, IROS 2018. It demonstrates why
  plane-only calibration needs diverse poses and why boundary lines can restore
  constraints in a one-pose workflow. Calibrex implements that distinct method
  in `planar_board_line_plane_solver.py`.

No paper implementation is copied into `src/calibrex`; the solver is an
independent NumPy implementation under the repository's Apache-2.0 license.

## Independent Verma centre+normal baseline

For capture `i`, the extracted features satisfy

```text
c_camera_i = R_camera_lidar c_lidar_i + t_camera_lidar
n_camera_i = R_camera_lidar n_lidar_i
```

Verma et al. optimize these features with a genetic algorithm. Calibrex does
not copy that optimizer: it implements a deterministic weighted Procrustes fit
with capture-level Huber IRLS. The centred 3D point covariance and scaled
normal covariance share one rotation solve; translation follows from the
weighted centre means. The method provenance explicitly calls this an
independent baseline rather than claiming source equivalence.

The six-column local Jacobian stacks centre residuals
`[-R[c_lidar]_x, I]` and scaled normal residuals
`[-s R[n_lidar]_x, 0]`. Its singular spectrum, rank, condition number, and
weakest direction are serialized. Fitting uses only seeded train captures.
Unchanged holdout captures report centre, normal, and induced plane-offset
RMSE. Twelve known-bad candidates apply both signs of ±5 cm x/y/z and ±5°
roll/pitch/yaw in the camera frame.

ACFR's 19-row capture contract supplies camera centre/normal at rows 0/1 and
LiDAR centre/normal at rows 6/7. These are oriented correspondences; their sign
is preserved and recorded rather than inferred from an estimate. The corner
rows feed only the line+plane row below, through an adapter that declares and
records how their identities are derived; the source row order is never
trusted as a correspondence.

## Independent Horn point-only baseline

For corresponding board centres, Calibrex minimizes

```text
sum_i w_i ||c_camera_i - (R_camera_lidar c_lidar_i + t_camera_lidar)||^2.
```

Weighted centroids remove translation. With centred LiDAR-to-camera
cross-covariance `S`, Horn's symmetric `4 x 4` matrix `N(S)` turns the rotation
problem into `max_(||q||=1) q^T N q`; the unit eigenvector associated with the
largest eigenvalue gives `R`, and
`t = mean(c_camera) - R mean(c_lidar)`. Calibrex uses the rigid specialization:
extrinsic scale is fixed to exactly one. Horn's centred RMS scale ratio is
serialized only as a unit-consistency diagnostic and is never applied to the
transform.

The native implementation is an independent NumPy derivation with
capture-level Huber IRLS. It consumes only rows 0 and 6 of each ACFR capture;
normals and corners are not passed to the solver. Fit and evaluation use the
same deterministic capture split as the plane-only and centre+normal methods.
The reported diagnostics include centred-point singular values and rank,
quaternion maximum eigengap and normalized eigengap, and the singular spectrum,
rank, condition number, and weakest direction of the local six-column point
Jacobian `[-R[c_lidar]_x, I]`. Collinear centres are rejected. Non-collinear
planar centres remain valid because three corresponding points can constrain a
rigid 3D transform even when their centred covariance has rank two.

Unchanged holdout centres report metric RMSE. Twelve falsification controls
apply both signs of ±5 cm x/y/z and ±5° roll/pitch/yaw in the camera frame; a
control is detected only when held-out RMSE rises by more than 5 mm. The solver
also has an explicit forward/reverse correspondence symmetry test, as required
by Horn's coordinate-frame symmetry argument.

## Line+plane row and its adapter-side edge extractor

Zhou, Li and Kaess's line+plane method needs corresponding board boundary
lines. ACFR's rows 2-5 (camera) and 8-11 (LiDAR) hold the four board corners in
millimetres; rows 12-18 are not consumed (they appear to hold corner angles,
LiDAR edge lengths, the range and the capture index). The corners carry no identity, so
`calibrex.solvers.acfr_board_edges` derives one in three recorded steps:

1. Each corner loop is ordered counter-clockwise about its own oriented board
   normal. In ACFR's file the camera loop is clockwise and the LiDAR loop
   counter-clockwise, so their row orders run in opposite directions.
2. A rectangular board is symmetric under a 180° turn about its normal, and no
   row says which way round the LiDAR saw it. Among the cyclic shifts of the
   LiDAR loop, the adapter picks the one whose oriented edge directions agree
   best with the camera's under a prior rotation. The backend passes the
   plane-only rotation fitted on the train captures, so no line information
   enters the prior. The alternatives are 90° or 180° away.
3. A paired edge whose camera and LiDAR lengths differ by more than 15 % rejects
   the capture. That is half the 28 % short/long difference of the 610 x 850 mm
   board, so a 90° mislabeling cannot pass. Rejected captures are listed.

Each boundary line runs through the midpoint of its two corners along the loop
direction. The native solver fits on the train captures of the same seeded
split as the other rows (the backend checks that the frame IDs match). It
reports held-out boundary-distance and boundary-direction RMSE, plane closure
as a diagnostic, and the rotation and translation spectra. Twelve controls
apply ±5 cm x/y/z and ±5° roll/pitch/yaw in the camera frame. A control counts
as detected when held-out boundary distance rises by more than 5 mm or
direction by more than 0.1°.

On the 40 ACFR captures every corner pair matches; 6 captures take the
symmetric LiDAR labeling (largest paired edge-length mismatch 5.4 %, smallest
direction-agreement margin 3.9 of a possible 8). With the default seed-0 split:

| Evidence | Train | Holdout | Gate | Verdict |
| --- | ---: | ---: | ---: | --- |
| Boundary-distance RMSE | 0.0126 m | 0.0139 m | ≤ 0.03 m | PASS |
| Boundary-direction RMSE | 1.413° | 1.509° | ≤ 3.0° | PASS |
| Plane-normal RMSE | 1.272° | 1.712° | diagnostic | recorded |
| Plane-offset RMSE | 0.0150 m | 0.0288 m | diagnostic | recorded |
| Known-bad detectable fraction | — | 12/12 (1.0) | ≥ 0.8 | PASS |
| Rotation / translation rank, condition | — | 3 / 3, 7.448 | rank 3 | PASS |

Its translation is approximately `[0.0234, -0.1860, -0.0898] m`, 2.07 cm and
0.26° from the plane-only output; like the other inter-method deltas, this
stays an ungated WARN diagnostic. The symmetry step is load-bearing: keeping
the source row order for the 6 symmetric captures raises the held-out
boundary distance to 0.37 m, far over the gate. The opt-in test
`tests/integration/test_acfr_line_plane.py` runs both cases on the downloaded
file.

## Mathematical contract

For corresponding unit-normal planes represented by `n^T p + d = 0`,

```text
n_camera = R_camera_lidar n_lidar
n_camera^T t_camera_lidar = d_lidar - d_camera
```

Rotation is a weighted orthogonal-Procrustes fit. Translation is weighted
least squares over the camera normals. Capture-level Huber IRLS prevents a
single bad plane capture from dominating. For unoriented extractor output,
normal signs are aligned using the declared initial transform and every flip is
recorded. Extractors that publish consistently oriented pairs can instead use
`normal_sign_policy: preserve`, avoiding an external calibration estimate just
to choose signs.

Three observations are not sufficient evidence by themselves. The stacked
camera normals must have rank three and an acceptable condition number. Nearly
parallel targets must therefore produce `degenerate_normals`, not a nominal
6-DoF estimate.

## Falsification and evaluation contract

- Split whole captures deterministically into train and holdout sets.
- Fit only on train captures.
- Report held-out angular normal closure and metric plane-offset closure.
- Apply ±5 degree roll/pitch/yaw and ±5 cm x/y/z known-bad transforms.
- A probe is detectable only when held-out normal RMSE rises by more than
  0.1 degree or offset RMSE rises by more than 5 mm.
- Record method ID, DOI, transform convention, normal-sign policy, split IDs,
  singular spectrum, condition number, and probe outcomes.

## Public real-data acceptance

The reproducible public example uses 40 extracted real checkerboard poses from
the Apache-2.0 ACFR VLP-16 quick-start dataset, pinned to upstream commit
`ecda574ed902913fbc2ae50f92ad3356f318431c`. The downloader verifies
`poses.csv` against SHA-256
`024bc6ed9009652761e9c0df49b106d325a10c88d41a80e9b36ea55fd567e110`.
The upstream ROS/PCL extractor remains an external adapter boundary; no
upstream implementation is copied or executed by the native solver.

```bash
python3 tools/download_public_dataset.py acfr_vlp_plane_poses --output-dir data/public
calibrex calibrate examples/public_datasets/acfr_vlp_plane_poses/config.yaml
calibrex validate outputs/acfr_vlp_plane_poses/result.yaml --json
calibrex verify outputs/acfr_vlp_plane_poses/bundle.json
```

The default protocol (seed 0, 20% capture holdout) produced:

| Evidence | Train | Holdout | Gate | Verdict |
| --- | ---: | ---: | ---: | --- |
| Plane-normal RMSE | 1.262° | 1.755° | ≤ 3.0° | PASS |
| Plane-offset RMSE | 0.0140 m | 0.0274 m | ≤ 0.05 m | PASS |
| Known-bad detectable fraction | — | 10/12 (0.833) | ≥ 0.8 | PASS |
| Normal rank / condition number | — | 3 / 3.758 | rank 3 | PASS |

The independent point+plane baseline on the exact same split produced:

| Evidence | Train | Holdout | Gate | Verdict |
| --- | ---: | ---: | ---: | --- |
| Board-centre RMSE | 0.0119 m | 0.0117 m | ≤ 0.03 m | PASS |
| Board-normal RMSE | 1.281° | 1.681° | ≤ 3.0° | PASS |
| Plane-offset RMSE | 0.0141 m | 0.0276 m | diagnostic | recorded |
| Known-bad detectable fraction | — | 12/12 (1.0) | ≥ 0.8 | PASS |
| Joint rank / condition number | — | 6 / 7.447 | rank 6 | PASS |

Its output translation is approximately `[0.0181, -0.1825, -0.0895] m`.
Relative to the plane-only output, the independent result differs by 0.0176 m
and 0.219°. Those deltas remain WARN diagnostics because no independent
metrology threshold was predeclared. Consequently the method-specific gates
PASS while the combined run is honestly WARN rather than promoting agreement
to ground truth.

The normal-free Horn point-only baseline, again on the exact same capture
split, produced:

| Evidence | Train | Holdout | Gate | Verdict |
| --- | ---: | ---: | ---: | --- |
| Board-centre RMSE | 0.00877 m | 0.00867 m | ≤ 0.03 m | PASS |
| Known-bad detectable fraction | — | 12/12 (1.0) | ≥ 0.8 | PASS |
| Centred geometry rank / condition | — | 3 / 9.368 | rank ≥ 2 | PASS |
| Joint rank / condition | — | 6 / 15.696 | rank 6 | PASS |
| Quaternion normalized eigengap | — | 0.6801 | ≥ 1e-4 | PASS |
| RMS camera/LiDAR scale ratio | — | 1.0029 | diagnostic | recorded |

Its rigid translation is approximately `[0.0100, -0.1412, -0.0848] m`.
Relative to the configured plane-only output it differs by 0.0277 m and
1.280°; relative to the centre+normal baseline it differs by 0.0424 m and
1.071°. These are retained as WARN comparison diagnostics because the dataset
does not provide independent metrology and no acceptance threshold was
predeclared. The Horn method-specific gates nevertheless PASS without weakening
the configured defaults.

The configured plane-only output `T_camera0_lidar0` translation is approximately
`[0.0220, -0.1661, -0.0842] m`. Its method-specific evidence gates PASS. The
result is schema-valid and its evidence bundle verifies the downloaded input
digest; the combined quality remains WARN solely because the inter-method
deltas intentionally have no acceptance threshold.
This evaluates upstream extracted observations, not independent raw-image and
raw-point-cloud feature extraction; that distinction is recorded in result
provenance.

A result lacking normal diversity remains INCONCLUSIVE; gates must not be
weakened to force a PASS.
