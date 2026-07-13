# Calibration Method Portfolio

Calibrex supports multiple calibration methods behind one result and evidence
contract. A method is not considered complete merely because it returns a
matrix: it also needs typed inputs, provenance, observability diagnostics,
holdout evaluation, and known-bad controls.

## Implemented and In Progress

| Method | Role | Status | Independent evaluation |
| --- | --- | --- | --- |
| Native fixed-trajectory LiDAR point-to-plane | LiDAR extrinsic refinement | Implemented | voxel-plane holdout and 6-DoF controls |
| Online motion-compensated LiDAR point-to-plane | Streaming LiDAR extrinsic refinement | Implemented | rolling holdout and adoption gates |
| Open3D RGB-D SLAC | External RGB-D trajectory/calibration adapter | Adapter boundary | Calibrex result/report metrics |
| Koide-style direct LiDAR-camera | External targetless baseline | Adapter boundary | projection, edge, depth-edge, and controls |
| Radar Doppler candidate consistency | Radar yaw evidence | Implemented | frame holdout, four yaw controls, sensor policy |
| Robust Radar ego velocity | Scan-wise motion estimation from Doppler | Native solver | synthetic truth, outliers, LOS rank diagnostics |
| Radar-to-trajectory yaw | Radar extrinsic rotation from paired velocities | Native solver | deterministic holdout, direction diversity, four yaw controls |
| Planar-board LiDAR-camera plane alignment | LiDAR-to-camera 6-DoF extrinsic | Native solver | capture holdout, normal/offset closure, normal-span diagnostics |
| Planar-board LiDAR-camera line+plane | One-pose-capable LiDAR-to-camera 6-DoF extrinsic | Native solver | plane/edge closure, separate rotation/translation spectra, edge-angle gate |
| Robust point-to-point ICP | Generic local 3D registration | Native solver | spatial-block holdout, mutual/trimmed matches, 3D-spread and frozen-pair diagnostics |
| Open3D Generalized ICP | Optional probabilistic registration baseline | MIT adapter | train-only source fit, common Calibrex spatial holdout/rematching metrics |
| PCL/Autoware NDT | External registration baseline | Subprocess/precomputed adapter | declared train isolation, common Calibrex spatial holdout/rematching metrics |
| Park-Martin motion hand-eye | Trajectory-derived `AX = XB` extrinsic | Native solver | motion holdout, axis spectrum, translation-system spectrum and closure |
| Tsai-Lenz motion hand-eye | Independent separable `AX = XB` baseline | Native solver | shared motion holdout, two system spectra, 12 known-bad controls |
| Daniilidis dual-quaternion hand-eye | Simultaneous `AX = XB` rotation/translation baseline | Native solver | shared holdout, 8D nullspace/Study diagnostics, 12 known-bad controls |
| LiDAR-IMU rotation consistency | Supplied rotation evaluation | Implemented | held-out angular-rate and gravity evidence |
| Anchored temporal offset | One-dimensional time calibration evidence | Implemented | injected offsets and adapted/anchored comparison |

The native Radar ego-velocity solver uses Huber IRLS on
`d_i = -u_i^T v`. It reports rank and condition diagnostics from the LOS normal
matrix and refuses a 3D result when the scan geometry does not span three
velocity directions. Automotive scans may therefore be valid only for a
future planar or yaw-specific stage; Calibrex must not silently promote them to
full 3D support.

## Next Native Methods

1. Multi-plane LiDAR calibration using plane normal and offset constraints.

## Optional Adapter Methods

- Open3D Generalized ICP
- PCL or Autoware NDT through a subprocess boundary
- Kalibr spatiotemporal camera-IMU calibration
- AprilTag or ChArUco observation extraction
- external visual and LiDAR odometry producers for hand-eye calibration

External implementations remain optional adapters. Their version, command,
license boundary, input digests, and output role must be recorded. GPL or
license-uncertain code must not be copied into `src/calibrex`.

## Research Sequence

The intended sequence is compositional:

```text
Radar Doppler returns
  → robust scan velocity
  → Radar-to-trajectory rotation
  → angular-rate-excited lever arm
  → joint spatial/temporal evidence
```

Likewise, target-based LiDAR-camera calibration supplies an independently
checkable reference for later edge and mutual-information targetless methods.
Every new optimizer must use a train split and be judged on an unchanged
holdout split with explicit perturbation controls.

## Planar-board LiDAR-camera research basis

The native plane-only solver follows the geometric contract introduced by
[Zhang and Pless (IROS 2004)](https://doi.org/10.1109/IROS.2004.1389752): an
oriented camera plane corresponds to an oriented LiDAR plane. For
`p_camera = R p_lidar + t`, the constraints are
`n_camera = R n_lidar` and
`n_camera^T t = d_lidar - d_camera`. Rotation is estimated by weighted
orthogonal Procrustes and translation by weighted least squares. Robust IRLS
weights whole captures, so a dense plane cannot silently dominate merely by
containing more points.

Plane count alone is not an observability test. The camera-plane normal matrix
must have rank three for full translation, and near-parallel normals remain
ill-conditioned. This agrees with the explicit observability analysis of
[Fu et al.](https://doi.org/10.1109/TIM.2019.2931526) and the minimum
three-pose plane/centre method evaluated by
[Verma et al.](https://doi.org/10.1109/ITSC.2019.8917108). Calibrex therefore
reports the singular spectrum, rank, condition number, capture split, and every
normal sign selected using the supplied initial transform.

Plane-only calibration does not claim one-shot observability. A finite board's
boundary lines supply the missing in-plane constraints. The implemented
line-plus-plane method follows
[Zhou, Li, and Kaess](https://doi.org/10.1109/IROS.2018.8593660) and can be
observable from one pose when two non-parallel board edges are reliably
measured. Plane plus one edge, or two parallel edges, remains rank five because
translation along the shared edge direction is unobservable. Calibrex reports
separate rotation and translation singular spectra and rejects weak edge-angle
geometry. LCE-Calib likewise combines
point-to-plane and point-to-line refinement
([Jiao et al.](https://arxiv.org/abs/2303.09825)).

The closed-form initializer assumes the extractor supplies stable edge IDs and
consistent oriented directions. An unlabeled rectangle has a 180-degree
permutation ambiguity, and a square can additionally admit 90-degree
permutations. These discrete hypotheses are not silently resolved; the policy
is recorded in provenance and such inputs require a labeled checkerboard,
fiducial, or an adapter that enumerates and evaluates permutations.

Feature extraction stays outside the native core. GPL implementations such as
FAST-Calib, `plycal`, and several ROS checkerboard packages may only be invoked
through optional adapters; their code is not copied into `src/calibrex`.

## Native robust point-to-point ICP

The native ICP follows the refreshed nearest-neighbour loop of
[Besl and McKay](https://doi.org/10.1109/34.121791) and the fixed-overlap
trimming principle of
[Chetverikov et al.](https://doi.org/10.1109/ICPR.2002.1047997). Each iteration
transforms the source, recomputes nearest neighbours, optionally retains only
mutual pairs, applies a distance gate and stable distance-ranked trim, then
solves the retained rigid alignment by an SO(3)-corrected SVD. This is a local
registration method; the result does not claim global convergence.

Point holdout is spatial rather than an independent random-point split. Source
points are grouped into deterministic voxels before train/holdout selection so
neighbouring samples of the same surface do not trivially leak across the
evaluation boundary. Held-out source points are evaluated with fresh target
nearest neighbours.

The solver reports the fixed-pair six-dimensional information spectrum, but it
does not interpret its inverse as calibration covariance. As shown by
[Bonnabel, Barczyk, and Goulette](https://doi.org/10.1109/ACC.2016.7526532),
ordinary point-to-point fixed-correspondence Hessians can be misleading when
ICP rematches points. The native solver therefore conservatively rejects
collinear and planar retained geometry instead of reporting full confidence
from a formally rank-six frozen-pair Jacobian.

Effective diagnostics now perturb all six transform directions and rerun the
complete nearest/mutual/gate/trim correspondence policy. They report rematched
black-box objective curvature and correspondence-pair Jaccard stability, not a
covariance matrix. Six independent initialization probes additionally flag a
distinct transform with equivalent spatial holdout error as a symmetry
ambiguity. Local curvature and global multi-start falsification remain separate
diagnostics because either can pass while the other fails.

Open3D GICP and PCL/Autoware NDT are optional adapters with distinct method
IDs. Their library-specific fitness values are retained as raw provenance, not
treated as directly comparable metrics; Calibrex recomputes common spatial
holdout and rematching metrics for cross-backend comparisons. An external NDT
result that does not declare train/holdout isolation is explicitly warned.

## Motion hand-eye calibration

The native hand-eye solver implements the separable Lie-algebra construction
of [Park and Martin](https://doi.org/10.1109/70.326576). Inputs are typed
relative-motion pairs that explicitly satisfy `A X = X B`. With
`alpha = Log(R_A)` and `beta = Log(R_B)`, rotation is the weighted orthogonal
Procrustes map `alpha = R_X beta`. Translation is then solved from
`(R_A - I)t_X = R_X t_B - t_A`.

Motion count is not used as a substitute for observability. One rotation-axis
family leaves the hand-eye twist about that axis ambiguous, while the stacked
translation system also leaves translation along the shared axis unobservable.
The solver requires at least two independent rotation-axis directions and rank
three in `stack(R_A - I)`. It rejects identity and sub-threshold rotations and
reports separate rotation-axis and translation singular spectra, rank, and
translation condition number.

Evaluation compares `A X` and `X B` without refitting and reports rotation
geodesic and translation closure on train and holdout motions. If relative
motions were derived from absolute trajectories, the upstream builder must
split disjoint absolute pose or temporal blocks before constructing pairs;
randomly splitting a dense all-pairs motion set would leak the same poses into
both sets. That builder remains separate from the core solver so eye-in-hand,
eye-to-hand, and robot-world/hand-eye conventions cannot be mixed implicitly.

[Tsai and Lenz](https://doi.org/10.1109/70.34770) is implemented as an
independent modified-Rodrigues/linear-translation baseline with the same typed
motion split and closure evaluation as Park-Martin. The simultaneous
dual-quaternion method of
[Daniilidis](https://doi.org/10.1177/02783649922066213) is also implemented as
an independent 8D nullspace solver. It enforces unit and Study constraints,
reports the full singular spectrum and nullspace separation, and selects real
constrained candidates by train closure. OpenCV `calibrateHandEye` is suitable
for an optional adapter; Kalibr is a distinct camera-IMU spatiotemporal
workflow and is not labeled as this native `AX = XB` method.
