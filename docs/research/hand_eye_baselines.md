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
three implementations share only typed inputs, deterministic split, closure
evaluation, and falsification protocol.

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
motions were constructed from disjoint absolute-pose blocks; the recorded
absolute-pose reuse count is zero.

| Method | Holdout rotation | Holdout translation | Known-bad detection |
| --- | ---: | ---: | ---: |
| Park-Martin | 0.938° | 0.0172 m | 6/12 |
| Tsai-Lenz | 0.945° | 0.0174 m | 6/12 |
| Daniilidis | 0.939° | 0.0174 m | 6/12 |

All three methods pass the declared closure gates of 2° and 3 cm. They do not
reach the unchanged 0.75 known-bad detectable-fraction gate, so the public run
is honestly **INCONCLUSIVE**, not PASS. This indicates limited falsification
power in the selected motion blocks despite low closure residuals. The
schema-valid result and evidence bundle record the negative finding, input
digest, split IDs, spectra, and solver provenance.
