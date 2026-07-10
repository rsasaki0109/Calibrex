# Problem Builder

Calibrex compiles configs into a backend-neutral graph problem before any
solver-specific adapter runs.

The compiled problem contains:

- variables: trajectory, extrinsics, intrinsics, time offsets, IMU bias, maps,
  control grids, and velocity blocks
- factors: backend-neutral residual descriptors
- priors: transform, timing, and root pose priors
- gauges: explicit constraints that remove free global modes
- observability: static pre-solver diagnostics

Inspect a compiled problem:

```bash
Calibrex compile examples/configs/minimal.yaml --json
```

Solver adapters translate this representation into backend-native structures
for scipy, GTSAM, Ceres, Open3D, or future backends.
