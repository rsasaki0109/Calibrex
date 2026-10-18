# TUM diagnostic execution: rejected acceleration candidates

The complete, uninstrumented public TUM three-window workload takes **31.577
seconds before and 31.259 seconds with the candidate (1.010x)**, using medians
from three fresh workers per variant. This does not establish a useful whole
pipeline acceleration. **The candidate runtime changes are not adopted.**

An earlier process-parallel probe candidate gives **27.042 to 26.429 seconds
(1.023x)** in a separate four-worker screen. Its extra process lifecycle and
transport complexity is also rejected. These comparisons use different screens;
their absolute times and ratios must not be combined.

This is execution research on the already spent public TUM `freiburg1_xyz`
development sequence. It makes no accuracy, convergence or SOTA claim. The host
was unreserved, and individual worker times vary substantially.

## Complete production comparison

Both variants run the original public configuration with three native window
workers and one BLAS/OpenMP thread. The 35 actual input files are hashed before
execution and again afterward against the existing public inventory. All
primary and replication windows, ablations, optimization, rematching, holdout,
curvature, multistart, probes and report saving remain enabled.

| Fresh worker | Runtime | Whole worker seconds | Calibration API seconds |
| --- | --- | ---: | ---: |
| 01-current | candidate | 32.032 | 27.265 |
| 02-prior | preceding | 36.064 | 30.346 |
| 03-prior | preceding | 31.577 | 26.736 |
| 04-current | candidate | 31.259 | 26.611 |
| 05-current | candidate | 28.881 | 23.943 |
| 06-prior | preceding | 28.720 | 24.201 |

Whole worker time includes fresh process startup, imports, public-input
verification, child window startup, transport, calibration, reports and evidence
saving. The API timer covers the calibration call, including its window
processes and report saving. Numerical tracing hooks are absent from these six
workers. The default NumPy path leaves the optional CUDA adapter unloaded.

The preceding runtime is `dddf26a422ee320736d7cd8e1380733064095c8a`. The rejected
candidate is `ae943a932e220813650fc5174e3c60a5d5b5f526`. Their 392- and
394-module source capsules, imported-source identities, original records and
all attempts are preserved in the [archive](../assets/tum_diagnostic_execution/index.json).
Execution policy and every numerical result field agree exactly; only run IDs,
timestamps and Git identities are excluded from the independent comparison.

## What was tried

The first candidate lowered the packed-query threshold for captured native
plane geometry from 64 to eight points, batched native diagnostic residuals,
enabled native residual-dimension metadata on NumPy, and reused bounded factory
signature caches during reassociation. An initial **traced** six-worker screen
gave 45.900 to 40.335 seconds (1.138x). That measurement includes trace overhead
and is not evidence for the production speedup.

A second candidate added spawned CPU workers for independent reassociation
probes on top of the first candidate. It preserved probe order, fixed populations
and complete results, with
a shared pool per window and bounded worker-local contexts. The separate
production screen's 1.023x result does not justify adopting this machinery.
The [source capsule](../assets/tum_diagnostic_execution/sources-probes.tar.gz)
preserves the experiment without adding this API to the maintained runtime.

The final candidate replaced intermediate mutable plane accumulation with
bounded array construction of immutable normal-free point-cloud snapshots.
Scalar addition order, the first twelve normal candidates, first-tie selection,
normal rejection and floating-point bytes were checked. Correctness alone did
not translate into a worthwhile complete-workload improvement.

## Profiling limits and next target

The baseline cProfile run contains profiling overhead. Its final comparison
initially failed because the comparison omitted two generated artifact
destination paths; a separate, preserved comparison verifies those paths before
normalizing them and confirms the complete result. The original failed
launcher and its logs are retained.

The first lightweight stage profile failed to flush its timing records on
process exit. The second profile saved records, but wrapping the map builder
changed its function identity and disabled the candidate's native map path.
Neither profile is used to support a speed claim. The third profile excludes
that hook and preserves the native path. All three attempts are retained in the
[development archive](../assets/tum_diagnostic_execution/development-screens.tar.gz).

The valid stage profile shows a useful structural regression: the candidate
builds 184–220 RGB-D evaluation plans per window, versus 20–22 previously.
Repeated reassociation and diagnostic scoring construct new plans around each
perturbed state, offsetting the savings from array evaluation. The next design
should share validated geometry across a batch of probe states and avoid
rebuilding their residual plans, while retaining the complete fixed population,
native callbacks, tie order and exact evaluation history. Merely increasing the
process count is not the next implementation step.

## Correctness and offline audit

A separate supplementary trace compares two complete runs. Each retains **24
solves, 168 Jacobian evaluations, 1,624 residual evaluations, 150 frontend records,
1,498 diagnostic records and 492 fixed-population scoring events**. Complete
results and ordered numerical histories agree, including residual/Jacobian
array hashes and fixed-population score bytes. These traces establish work and
result parity; their timings are not the production latency comparison.

The [supplementary records](../assets/tum_diagnostic_execution/supplementary-traces.tar.gz)
and [machine-readable summary](../assets/tum_diagnostic_execution/summary.json)
can be checked without a GPU or downloading the dataset:

```powershell
python tools/audit_tum_diagnostic_execution.py docs/assets/tum_diagnostic_execution
```

The auditor checks every indexed byte, fixed source and input pins, imported
paths, all six worker records, actual medians, complete results, and the
supplementary work populations and histories. Generated records include
provenance and JSON Schemas. Development records include unsuccessful attempts
so that an exploratory timing cannot be mistaken for the accepted production
comparison. The maintained runtime retains the preceding implementation.

The rejected candidate's frozen capsule passes **658 Windows tests**, including
34 real-device cases, and **624 Linux/Python 3.10 tests**, with the optional CUDA
test module skipped. The nine archive and rehashed-tampering tests pass on each
platform. Strict typing passes for the candidate's 394 modules on both platform
targets. All 129 public schema exports and their ledger match exactly. The
[validation record](../assets/tum_diagnostic_execution/validation.json) binds these
checks to the rejected capsule and separately verifies that the restored
maintained runtime matches the preceding 392-module source capsule.
