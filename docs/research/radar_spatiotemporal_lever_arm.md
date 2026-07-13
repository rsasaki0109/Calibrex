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

Train and unchanged holdout scan IDs are serialized. Known-bad probes apply
both signs of 0.10 m perturbations independently to x/y/z and both signs of a
20 ms clock perturbation. Each probe records the holdout RMSE increase and a
declared detection margin. Synthetic tests prove exact truth recovery,
train/holdout isolation, all eight perturbations, single-axis degeneracy, and
clock-search boundary reporting.

## Current limitation

The native solver consumes already-estimated scan-wise Radar ego velocity and
a reference kinematic trajectory. Connecting this contract to public nuScenes
raw Radar/ego-pose tables, with raw-file digests and dataset-specific timing
limitations, is the next adapter layer. Until that evaluation exists, this is
synthetic algorithm evidence rather than a public-real-data calibration claim.
