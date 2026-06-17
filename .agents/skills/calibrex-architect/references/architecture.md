# Architecture Reference

Calibrex core is a calibration problem compiler:

- Dataset adapters normalize logs.
- Sensor models define typed streams and parameters.
- Frontends create candidate measurements and correspondences.
- Problem builders compile variables, factors, priors, and gauges.
- Backends solve through optional implementations.
- Evaluation and visualization consume a stable result schema.

The core must not depend on ROS, Autoware, Kalibr, Open3D, GTSAM, or Ceres.
Those integrations belong in adapters or optional extras.
