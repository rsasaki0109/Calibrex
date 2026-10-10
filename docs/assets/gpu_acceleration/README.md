# CUDA backend evidence

The `*.benchmark.json` files are schema-valid timing reports, with raw paired
timings and input/configuration/code hashes. Local interpreter/dataset paths
and host names are normalized for publication. Each report retains the original
local report SHA-256 in `provenance.source_artifacts.local_report_before_publication`.
See [the tutorial](../../tutorials/gpu_acceleration.md) for workloads, measurement
commands and limitations. Raw KITTI sensor data, model checkpoints and depth maps
are not included.

## Converged frozen-protocol trial

The two `kitti_raw_converged_*.trace.json` files preserve the complete native
candidate-trace schema for trial `se3-0.5deg-0.5m-000` from the existing
KITTI raw 0018 six-DoF protocol. Both backends converged after
325 evaluations. Initial/output transforms, all candidate
parameters/objectives/MI records/acceptance decisions, status, stopping reason
and outcome are exactly equal. Solver identity, measured runtime and provenance
timestamps/commands differ as expected for separate executions.

The frozen protocol permits 800 evaluations and uses 200 perturbations; only
trial 000 was rerun here, with no protocol changes. This does not establish a
200-trial recovery rate. Calibration error is measured against the
dataset-provided reference, rather than independent metrology.

- Problem SHA-256: `a3ceb77714ba08aa83f3a16d09c9841542569a988437fb7277e663fc62f1a111`
- Original frozen protocol SHA-256: `659f6758e33d6b7ae7c5665e96c3114e4ab5141bbfe6ad05d780a2118465238b`
- Original NumPy trace SHA-256 before path normalization: `13c84a50cf47a671a3f0288c076521db54049222f2829f33fa809fc597c83567`
- Original CUDA trace SHA-256 before path normalization: `e592d900c8ee84935a9d3bdaa709dc92dc435eb19b2963e76a50e0b0eab4810f`

The command's local interpreter/dataset paths are normalized in the public trace
copies; all numerical values and input/protocol digests remain unchanged.

With the existing problem and original frozen protocol, reproduce this trial using:

```bash
python tools/run_borer_six_dof_chunk.py problem.yaml six-dof-protocol.yaml \
  --trace-dir outputs/parity-numpy --start 0 --stop 1 --projection-backend numpy
python tools/run_borer_six_dof_chunk.py problem.yaml six-dof-protocol.yaml \
  --trace-dir outputs/parity-cuda --start 0 --stop 1 --projection-backend cuda
```
