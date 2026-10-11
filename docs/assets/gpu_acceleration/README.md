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

## Single-estimation CLI with holdout

The two `kitti_raw_estimate_*.result.json` files are standard
`slac.result/v0.1` results from `camera-lidar estimate-six-dof`, using the same
KITTI raw 0018 problem digest as above. The last five of its 25 frame IDs were
reserved before execution for holdout; histogram ranges and optimization use
only the other 20 frames. These files retain all 157 evaluations, separate
initial/output/reference transforms, train/holdout objectives and full backend
identity. Both backends converged, with exact equality of all numerical solver
records, output transforms and initial/final holdout evaluations.

| Single invocation | Input verification/loading | Backend preparation | Solve | Holdout evaluation |
| --- | ---: | ---: | ---: | ---: |
| NumPy | 2.352 s | <0.001 s | 29.591 s | 0.105 s |
| CUDA | 2.350 s | 0.618 s | 1.847 s | 0.006 s |

This is an end-to-end smoke check on the same NVIDIA CUDA GPU, not a repeated timing
benchmark. The declared problem prior is the dataset reference; this run does
not demonstrate recovery from an independently measured prior or establish
calibration accuracy. Quality remains `warn`, with no estimated covariance or
observability rank. No acceptance threshold was selected from holdout data.
The output also passed standard result validation, HTML rendering and ROS TF
export.

Paths are normalized; numerical values and original input/configuration digests
are unchanged. `run.provenance.source_artifacts.local_result_before_publication`
records each original YAML result digest:

- NumPy: `abc103cca3618d48bff3ac144ebf772a9e22fcfe3f7d114b85305ceedfb02dd4`
- CUDA: `2ab5b71d38c192807e1be41bbc9a7dce02f12e65f3e43ef0d061430ba4912f0d`

Reproduce each invocation with the existing problem, changing `BACKEND` to
`numpy` or `cuda`:

```bash
calibrex camera-lidar estimate-six-dof problem.yaml \
  --projection-backend BACKEND \
  --holdout-frame-id 0000002301 --holdout-frame-id 0000002416 \
  --holdout-frame-id 0000002531 --holdout-frame-id 0000002646 \
  --holdout-frame-id 0000002761 --output outputs/estimate-BACKEND.yaml --json
```

## Large integrated inputs and cache comparison

`kitti_raw_cache_projection.benchmark.json` and
`kitti_raw_cache_six_dof.benchmark.json` compare NumPy, the default 256 MiB CUDA
input cache and a 2048 MiB CUDA input cache in one process with rotating timing
order. The existing KITTI raw 0018 radius-5 integrated problem contains
33,173,268 points across 25 frames and uses the existing frozen FeatDepth
provider. No depth maps or sensor inputs were regenerated.

All measured feature arrays, MI records and complete solver results/traces
match NumPy exactly. Each search is capped at 13 evaluations, with three
timing repetitions per implementation. This measures latency and numerical
equivalence; it does not establish convergence or calibration accuracy.

The default cache retains six frames (268,045,632 bytes) and uploads
884,222,944 input bytes per full candidate evaluation. The larger cache retains
all 25 frames (1,152,268,576 bytes), with zero large input uploads during
evaluation. The repeated-search medians are 2.384 s and 0.404 s respectively.
Upload counters exclude small pose/bin-edge uploads and returned histograms.
Both CUDA projectors remain resident together: allocation-pool/device readings
are aggregate snapshots, while cached-input bytes and upload counters are
per projector. Loading/verification and cache preparation are outside the
reported search timing. NumPy search times vary substantially in this run;
individual timings remain in the reports.

- Integrated problem SHA-256: `869e64208ef5f6b6387f3f052ded88a9d80ef27995bcadd534bf259063fca5c0`

Each report records the original local report digest before path/host
normalization in `provenance.source_artifacts.local_report_before_publication`.
Numerical records and original input/configuration/code digests are unchanged.
Measurements preceded the feature commit: `git_commit` identifies the base
checkout, while `source_artifacts` hashes identify the modified measurement code.
Raw sensor data, model checkpoints and depth maps remain external. Reproduce
with the same verified integrated problem:

```bash
python tools/benchmark_d2d_projection.py \
  --problem problem-integrated-r5.yaml --backends numpy cuda \
  --cuda-cache-mib 256 --compare-cuda-cache-mib 2048 \
  --repeats 3 --solve-evaluations 13 --solver-repeats 3 \
  --output-dir outputs/d2d-cache-comparison
```
