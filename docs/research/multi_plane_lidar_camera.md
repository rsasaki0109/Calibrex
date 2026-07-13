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
is preserved and recorded rather than inferred from an estimate. The remaining
corner rows are not treated as line correspondences: their discrete ordering is
not declared by the adapter, and a naive ordering produced a clear real-data
failure. Calibrex records the rows it consumes and does not invent corner
identities.

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
