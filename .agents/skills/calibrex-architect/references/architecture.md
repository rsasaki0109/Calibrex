# Architecture Reference

Calibrex core is a calibration problem compiler:

- Dataset adapters normalize logs.
- Sensor models define typed streams and parameters.
- Frontends create candidate measurements and correspondences.
- Problem builders compile variables, factors, priors, and gauges.
- Backends solve through optional implementations.
- Evaluation, comparison, and visualization consume stable result, comparison,
  and report sidecar schemas.

The core must not depend on ROS, Autoware, Kalibr, Open3D, GTSAM, or Ceres.
Those integrations belong in adapters or optional extras.

All generated calibration evidence should preserve provenance:

- producer, execution mode, role, and evidence level
- dataset source, sequence, and cached/recomputed status
- metric and protocol versions
- train/holdout split or limitation notes
- external tool command, version, license, and adapter version when available
