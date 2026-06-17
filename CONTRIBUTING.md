# Contributing

Calibrex is schema-first and evaluation-first.

Every PR that changes calibration behavior must include:

- tests
- docs or examples when user-facing behavior changes
- schema updates when config or result shape changes
- evaluation impact when a solver, factor, frontend, or metric changes

Core rules:

- Do not add ROS message types to `src/calibrex/core`.
- Do not copy GPL code into the core package.
- Keep external tools behind adapters or subprocess boundaries.
- Preserve `T_parent_child`, meters, nanoseconds, and quaternion `xyzw`.
