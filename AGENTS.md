# Calibrex Agent Notes

Calibrex is a long-term robotics calibration framework, not a collection of
one-off calibration scripts.

Non-negotiable project rules:

- Keep the core ROS-independent.
- Keep GPL code out of `src/calibrex`.
- Public APIs require type hints.
- Configs and results must stay schema-validatable.
- Calibration changes must include evaluation or tests.
- Every generated result must include provenance.
- Prefer adapters for Kalibr, Open3D, Autoware, and other ecosystem tools.

When in doubt, preserve schema stability and evaluation quality over short-term
algorithmic convenience.
