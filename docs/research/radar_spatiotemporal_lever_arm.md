# Radar lever-arm and temporal calibration research note

## Scope and primary references

Calibrex estimates a Radar translation (lever arm) and constant Radar clock
offset from timestamped scan-wise Radar ego velocities and a reference-body
kinematic trajectory. Rotation is deliberately supplied by an independent
rotation calibration stage; the solver does not pretend that all six spatial
degrees of freedom are jointly recovered.

The measurement contract follows the velocity formulations used by:

- Wise et al., *A Continuous-Time Approach for 3D Radar-to-Camera Extrinsic
  Calibration*, IEEE ICRA 2021, arXiv `2103.07505`. The work uses Radar
  velocity measurements, a continuous-time platform trajectory, and derives
  observability properties for targetless extrinsic calibration.
- Wise, Cheng, and Kelly, *Spatiotemporal Calibration of 3D
  Millimetre-Wavelength Radar-Camera Pairs*, IEEE T-RO 2023, arXiv
  `2211.01871`. It analyzes joint spatial/temporal identifiability and the
  motions needed for infrastructure-free calibration.
- Chen et al., *RIs-Calib: An Open-Source Spatiotemporal Calibrator for
  Multiple 3D Radars and IMUs Based on Continuous-Time Estimation*, arXiv
  `2408.02444`. It motivates joint spatial/temporal Radar-inertial calibration
  and continuous-time evaluation at asynchronous measurement times.

No source code from either project is copied. This module is an Apache-2.0
native implementation of the declared rigid-body velocity model.

## Measurement and time conventions

For known `R_body_radar`, the timestamped constraint is

```text
R_body_radar v_radar(t_r) =
  v_body(t_r + dt_radar) + omega_body(t_r + dt_radar) x t_body_radar
```

The time convention matches `calibrex.core.time`:
`radar_time + dt_radar = reference_time`. Reference linear and angular
velocities are linearly interpolated. Measurements without interpolation
support across the entire declared offset search range are excluded before the
train/holdout split, so candidate offsets cannot win merely by dropping hard
samples.

At each clock-offset grid point, translation is a three-column weighted linear
least-squares problem. The clock offset is selected by profiling this linear
solution over a bounded, explicitly resolved grid. A boundary optimum is
reported as `offset_at_boundary`, not as ordinary convergence.

## Observability and falsification

The singular spectrum, rank, and condition number of the stacked
`[omega]_x` matrix diagnose lever-arm observability. Rotation about only one
fixed axis leaves translation along that axis unobservable and is rejected.
The profiled clock objective reports a central finite-difference curvature;
zero curvature is evidence that the trajectory lacks temporal excitation, not
a covariance estimate.

Lever-arm rank alone is insufficient: a time perturbation can be reproduced by
a linear combination of translation perturbations. Calibrex therefore also
forms a four-column local Jacobian ordered as `(tx, ty, tz, dt)`. Translation
columns are scaled by the declared 0.10 m known-bad probe and the time column
by the declared 20 ms probe, so its spectrum and condition number are
dimensionless with an explicit scale convention. The reported
`time_translation_subspace_coupling` is the fraction of the scaled time column
explained by the translation-column subspace. A value of one and joint rank
three means complete local compensation; the numerical candidate is retained
for diagnosis but is not applied as a calibration result.

Train and unchanged holdout scan IDs are serialized. Known-bad probes apply
both signs of 0.10 m perturbations independently to x/y/z and both signs of a
20 ms clock perturbation. Each probe records the holdout RMSE increase and a
declared detection margin. Synthetic tests prove exact truth recovery,
train/holdout isolation, all eight perturbations, single-axis degeneracy, and
clock-search boundary reporting. A separate adversarial trajectory has
full-rank lever-arm excitation but an exactly confounded time column; it proves
that the joint-rank gate rejects a near-zero-residual false convergence.

## nuScenes public-data adapter

The `radar_spatiotemporal_velocity` factor connects the native contract to an
official, locally downloaded nuScenes mini tree without the nuScenes SDK. It
re-estimates planar Radar ego velocity from static raw `vx`/`vy` returns rather
than treating the dataset's ego-motion-compensated `vx_comp`/`vy_comp` as an
independent measurement. Consecutive `ego_pose` positions and orientations
provide body-frame linear and angular velocity. Every consumed PCD and metadata
table receives a SHA-256 provenance record.

nuScenes automotive Radar has essentially planar LOS support, while ordinary
road motion is dominated by yaw. Consequently, full three-axis lever-arm rank
is not assumed. The adapter publishes an honest `INCONCLUSIVE` result when the
motion has rank two, retaining counts, spectrum, rejected scans, and raw input
digests. Even with rank-three angular motion, the scaled joint lever-arm/time
Jacobian must have rank four and remain below the configured condition limit.
The fixed Radar rotation and every solver threshold are serialized with the
input digests. A local dataset absence is likewise `unavailable`; the project
does not redistribute nuScenes data or bypass its terms-of-use flow.
