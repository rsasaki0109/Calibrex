# Hand-eye calibration baselines research note

## Shared contract

All native hand-eye baselines consume typed relative-motion pairs satisfying
`A X = X B`. `X` maps the `B` motion frame into the `A` motion frame. Absolute
trajectory builders must split disjoint temporal blocks before constructing
pairs; randomly splitting a dense all-pairs set would leak poses across train
and holdout.

Every baseline uses the same deterministic pair-level split and is evaluated by
unfitted `A X` versus `X B` rotation and translation closure. This common
evaluation is deliberately separate from each method's estimator.

## Tsai and Lenz

Primary reference: R. Y. Tsai and R. K. Lenz, *A New Technique for Fully
Autonomous and Efficient 3D Robotics Hand/Eye Calibration*, IEEE Transactions
on Robotics and Automation 5(3), 1989, DOI `10.1109/70.34770`.

For unit quaternions of corresponding rotations, with equal scalar components,
the vector equation is

```text
[q_A.vector + q_B.vector]_x p_X = q_B.vector - q_A.vector
p_X = q_X.vector / q_X.scalar
```

The stacked system solves the modified Rodrigues parameter `p_X`; normalization
recovers the unit quaternion. Translation then follows from

```text
(R_A - I) t_X = R_X t_B - t_A.
```

The implementation reports the singular spectra, ranks, and condition numbers
of both systems. A single rotation-axis family is rejected rather than emitted
as a nominal 6-DoF calibration.

Tsai-Lenz is a separable estimator: rotation error propagates into translation.
It remains valuable as an algorithmically independent comparison with the
Park-Martin Lie-algebra rotation solver.

## Daniilidis dual quaternion

Primary reference: K. Daniilidis, *Hand-Eye Calibration Using Dual
Quaternions*, International Journal of Robotics Research 18(3), 1999, DOI
`10.1177/02783649922066213`.

The method represents rotation and translation as a unit dual quaternion. For
`Q = q + epsilon q'`, every motion contributes

```text
[ L(a) - R(b)          0       ] [q ] = 0
[ L(a') - R(b')  L(a) - R(b)   ] [q']
```

to an 8D homogeneous system. The native solver takes its two-dimensional SVD
nullspace and solves the quadratic Study constraint `q dot q' = 0`; unit
normalization enforces `q dot q = 1`. It reports all eight singular values,
observable rank and condition number, sixth/seventh singular-direction gap,
candidate count, and final constraint errors.

This is not described as the Tsai-Lenz or Park-Martin separable solution. The
baselines share only typed inputs, deterministic split, closure evaluation,
and falsification protocol.

## Horaud and Dornaika closed form

Primary reference: R. Horaud and F. Dornaika, *Hand-Eye Calibration*,
International Journal of Robotics Research 14(3), 1995, DOI
`10.1177/027836499501400301`, equations (16), (18), (27), and (29). The authors
also provide the article text as arXiv `2311.12655`.

For the classical formulation, corresponding nonzero motion rotations provide
unit axes satisfying

```text
n_A = R_X n_B.
```

Writing `R_X n_B = q_X * n_B * conjugate(q_X)` turns every axis pair into a
positive semidefinite four-dimensional quadratic form. The unit quaternion
`q_X` is the eigenvector of their weighted sum associated with its smallest
eigenvalue. Translation is then solved from the same `AX = XB` linear equation
used by the other separable baselines.

Calibrex records the axis singular spectrum and rank, all four quaternion
objective eigenvalues, the smallest-eigenvalue gap, its normalization by the
largest eigenvalue, and the translation spectrum. The eigengap represents the
width of the closed-form minimum rather than claiming covariance. A single
rotation-axis family produces a repeated minimum and is rejected. This is an
independent NumPy implementation; no source implementation from the paper or
another package is copied.

## Andreff, Horaud, and Espiau incremental Kronecker method

Primary reference: N. Andreff, R. Horaud, and B. Espiau, *On-line Hand-Eye
Calibration*, Second International Conference on 3-D Digital Imaging and
Modeling, 1999, DOI `10.1109/IM.1999.805374`, equations (10), (13), (14), and
(15). The [primary PDF is hosted by
INRIA](https://perception.inrialpes.fr/Publications/1999/AHE99/3dim99.pdf).

Using row-major vectorization, the paper rewrites the rotational part of each
motion as the homogeneous system

```text
(I_9 - R_A tensor R_B) vec(R_X) = 0.
```

Two nonparallel rotation axes give eight observable directions and a
one-dimensional kernel. Calibrex incrementally accumulates the 9x9 normal
matrix, extracts the kernel by symmetric eigendecomposition, applies the
paper's determinant normalization, and records the correction required by a
nearest-SO(3) projection. Translation is then solved conditionally from

```text
(I_3 - R_A) t_X = t_A - R_X t_B.
```

The online state also accumulates a 3x3 translation information matrix, its
camera-motion right-hand side, and a 3x9 rotation/translation coupling. It can
therefore update without retaining raw motions. Every accepted train motion
records the evolving axis rank, rotation-kernel rank and width, translation
rank, and the rank with which translations alone constrain `vec(R_X)`.

The paper explicitly warns that solving all twelve rotation-matrix and
translation coefficients at once is physical-unit dependent and does not
guarantee an orthogonal rotation. Calibrex records
`full_12_variable_solution_executed: false` and follows the paper's two-stage
solution. Synthetic controls recover exact truth with rotations as small as
0.05 degrees. Three independent pure translations correctly report rank 9 for
rotation-from-translation information but no observable hand-eye translation,
so they cannot be emitted as a full calibration.

## Falsification protocol

- Whole relative-motion pairs are split into train and holdout.
- The estimator fits train motions only.
- Holdout reports rotation and translation closure without refitting.
- ±5 degree roll/pitch/yaw and ±5 cm x/y/z transform controls are mandatory.
- A control is detectable when rotation closure rises by more than 0.1 degree
  or translation closure rises by more than 5 mm.
- Pure translation, sub-threshold rotation, a single rotation-axis family, and
  rank-deficient translation systems must not return `converged`.

All implementations are independent NumPy code. No paper or third-party source
implementation is copied into `src/calibrex`.

## Public real-data evaluation

The public comparison uses ETHZ ASL's real robot-arm pose streams from the
hand-eye benchmark, DOI `10.3929/ethz-c-000788527`. The small upstream archive
is pinned to BSD-3-Clause repository commit
`966cd92518f24aa7dfdacc8ba9c5fa4a441270cd` and SHA-256
`2454578f731e656a940ddf51017b56e8d535c58d16a50629b85326005dedd6c3`.
The dataset record permits non-commercial use. Calibrex parses the two pose
CSVs directly and does not copy or execute the upstream ROS implementation.

```bash
python3 tools/download_public_dataset.py ethz_hand_eye_robot_arm_real \
  --output-dir data/public
calibrex calibrate \
  examples/public_datasets/ethz_hand_eye_robot_arm_real/config.yaml
calibrex verify outputs/ethz_hand_eye_robot_arm_real/bundle.json
```

Timestamp alignment retained 1,688 pairs within 10 ms. Fourteen relative
motions were constructed from disjoint absolute-pose blocks; thirteen pass the
shared one-degree comparison filter and produce a common 10-train/3-holdout
split. The recorded absolute-pose reuse count is zero.

| Method | Holdout rotation | Holdout translation | Known-bad detection |
| --- | ---: | ---: | ---: |
| Park-Martin | 0.938° | 0.0172 m | 6/12 |
| Tsai-Lenz | 0.945° | 0.0174 m | 6/12 |
| Daniilidis | 0.939° | 0.0174 m | 6/12 |
| Horaud-Dornaika | 0.985° | 0.0190 m | 6/12 |
| Andreff-Horaud-Espiau | 0.939° | 0.0172 m | 6/12 |

All five methods pass the declared closure gates of 2° and 3 cm. Horaud-
Dornaika has axis rank 3/3 and normalized quaternion eigengap 0.7758, passing
the predeclared 0.001 minimum-width gate. Andreff has rotation rank 8/8,
translation rank 3/3, normalized rotation width 0.4836, and SO(3) projection
correction 0.0003515. Its kernel-to-eighth singular-value ratio is 0.05005.
These pass the predeclared 0.0001 minimum-width, 0.25 maximum-nullspace-ratio,
and 0.05 maximum-projection gates. The five methods share exactly the same
train/holdout IDs. They do not
reach the unchanged 0.75 known-bad detectable-fraction gate, so the public run
is honestly **INCONCLUSIVE**, not PASS. This indicates limited falsification
power in the selected motion blocks despite low closure residuals. The
schema-valid result and evidence bundle record the negative finding, input
digest, split IDs, spectra, and solver provenance.
