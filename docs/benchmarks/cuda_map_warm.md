# Complete warm CUDA map comparisons

`tools.benchmark_cuda_map_warm` measures complete calibration and evidence APIs
after each worker's own prime. It uses the frozen source trees and raw-input
bindings from `tools.benchmark_cuda_map_pipeline`, alternates variant order
across three independent process pairs, and gives every worker an initially
empty, separate CuPy cache. Each worker then makes three complete warm calls.

All calls retain map preparation, fits, covariance, holdout, signed controls and
jackknife. The worker verifies its actual imports against the frozen manifest.
The parent requires exact complete evaluation equality for every prime/warm
pair. A warm call compiling another program fails the run. BLAS, OMP and MKL
thread counts remain one. Core result/config schemas and runtime APIs are
unchanged.

NVRTC counts and durations are recorded separately for each call. The reported
ratio is the median of each worker's three warm API times, then the median
across independent workers. Imports and file saves are outside the API scope.
Saved phase summaries, protocol and final summary have a dedicated JSON schema;
complete calibration artifacts retain their core schema and provenance.

The parent checks GPU compute processes before each worker and every two seconds
while it runs. A foreign or unknown process stops the owned worker and retains
its attempt, observations and log. Initial interference prevents a worker from
starting. These observations do not guarantee continuous GPU exclusivity,
detect all graphics activity or reserve CPU resources. Hardware models and
process IDs are omitted from saved observations.

Install the development dependencies for process-tree tracking. Windows virtual
environment launchers can spawn the interpreter that owns the CUDA context;
both the launcher and its descendants belong to the worker. The runner tracks
their process identities, ignores exited processes in driver snapshots, waits
for owned descendants to finish, and terminates only that captured worker tree
when interrupted. It retains ownership across workers so teardown is not
mistaken for unrelated work.

First freeze and measure the source pair with the public KITTI copied-sensor
execution control, as described in the [map fusion benchmark](cuda_map_fusion.md).
Then use its local output directory:

```bash
python -m tools.benchmark_cuda_map_warm \
  --comparison-dir outputs/my-map-source-comparison \
  --output outputs/my-map-warm-comparison \
  --repeats 3 --warm-calls 3
```

The referenced raw files must still match the cold comparison's input hashes.
The public evidence archive contains results and frozen source bundles; it does
not redistribute those scans. This workload measures copied-sensor execution
and evidence, not independent two-sensor accuracy or reference odometry.

## Shared spatial-module experiment

A candidate compiled construction, generic traversal and a register-based
11-neighbor traversal together (`682eed0`), compared with the retained fusion
runtime (`0d66a49`). Three fresh-process pairs kept complete evaluations exactly
equal and retained 9,147,116 query rows per worker, including 762,118 rows through
the candidate's default-normal traversal. Observed NVRTC counts stayed at 17
per worker in both variants. The observations did not show a substantial
complete-pipeline benefit.

A check after the cohort found a foreign GPU compute process. The cohort was
not observed from its start, so those timings are unqualified exploratory data
and establish no accepted acceleration. The candidate is reverted; the
[retained map fusion results](cuda_map_fusion.md) remain the supported evidence.
The local frozen cohort keeps both source trees and full outputs for future
qualification with the guarded warm runner.

The corrected observer completed two independent warm process pairs before a
later process observation stopped the cohort. Their eight complete evaluation
pairs match exactly, with zero new compilations in every warm call. The
descriptive ratio is approximately 1.08x. This incomplete cohort supplies no
accepted acceleration and the candidate remains reverted. CPU pilots also
found only about 1.3x from four-thread spline evaluation, while streaming and
adjoint plane derivatives did not improve complete fixed-input fit time. Those
pilots leave the core runtime unchanged.
