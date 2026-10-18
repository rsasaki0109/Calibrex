# Reusing joint Radar evaluation work

The complete prepared-input seven-DoF Radar calibration workflow changes from
**9.230905 to 4.094905 s (2.254242x)**.
The solver API changes from **7.305833 to 2.174879 s
(3.359191x)**. All six complete numerical results, optimizer
histories, weighting contracts and holdout probes match exactly.

## Workload and scope

This repeats the [indexed joint Radar workload](joint_radar_reference_indexing.md)
with the byte-identical control: all 2,762 OXTS reference samples of KITTI raw
drive `2011_09_30_drive_0018_sync` and 2,758 generated Radar-frame velocities.
The control retains deterministic 0.0001 m/s noise and unequal weights.
**Radar velocities are generated execution controls.** This is performance
and parity evidence, not independent Radar accuracy, a LiDAR pipeline speed
claim or a SOTA result. No locked accuracy cohort is evaluated.

The solver still fits all seven variables from an identity transform and
zero offset, with the 80-iteration bound, bounded clock, initial-residual gate,
finite-difference Jacobians, Huber LM, grouped train/holdout split and complete
diagnostics. Every run converges after five iterations, retaining 2,099 eligible
measurements and the same 659 rejected IDs, and evaluates fourteen signed
holdout probes. Measurement budgets and stopping rules are unchanged.

## Change and work counts

Extrinsic finite differences repeatedly decode the same transform for every
measurement while leaving the reference clock unchanged. One solve now owns
a shared 32-entry transform decoder; each measurement owns a separate two-entry
reference-interpolation cache. Memory stays bounded as iteration count grows.
Caches expire with the solve and never share initial transforms, reference
trajectories or options across solves. Binary floating-point keys retain
signed zeros. Quaternion normalization moves into the shared transform state;
the same two Hamilton products rotate each measured velocity in the original
operation order. The public APIs and result/schema contracts are unchanged.

| Work per complete run | Prior | Reused |
| --- | ---: | ---: |
| Factor evaluations | 162,024 | 162,024 |
| Reference interpolations | 171,502 | 41,799 |
| SE(3) tangent decodes, including final diagnostics/probes | 162,037 | 91 |

The portable audit requires identical factor evaluation counts in all six
workers, identical work counts across repeats of each variant, and fewer
interpolations/decodes in the reused variant. The previous profiled and joint
indexing archives retain their original audit contracts and results.

## Complete fresh-process measurements

Three alternating pairs use frozen sources, actual Git identities, identical
dependencies/interpreter and one BLAS thread. The host is unreserved, and
variation is retained without discarding or relabeling any run as warmup.
Whole-process timing includes startup, input parsing/construction, gates,
the complete solve, work counters, diagnostics, serialization and saving.
Downloading, control generation, source freezing and audits are untimed.

| Pair (execution order) | Prior whole s | Reused whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (prior, reused) | 8.115166 | 3.845855 | 2.110x |
| 2 (reused, prior) | 9.230905 | 4.094905 | 2.254x |
| 3 (prior, reused) | 9.298377 | 4.508278 | 2.063x |

The **2.254242x** headline is the ratio of three-run
medians for this workload. Baseline: `d235e388dff403424447bca7548147b3e61ce06d`.
Measured runtime: `4a1fd5004256f4da0bbc4bc392250a72c2fbbec5`. Only the joint Radar
solver changes in the complete runtime source manifests.

## Validation and provenance

Every worker retains its complete output, accepted/rejected optimizer steps,
initial rejection IDs, train/holdout assignment, full rank/condition/family
diagnostics and probes. Numerical parity uses no tolerance or normalization;
source identity lives in a separate provenance wrapper. New regression tests
cover residual bytes across cache eviction, signed zero keys, interleaved clock
probes, separate measurement/initial-transform ownership, unsupported-clock
errors, and complete weighted solves from two initial transforms. The related
suite passes 112 cases. Fourteen new archive checks exercise complete parity,
counter consistency, unchanged evaluation budgets and tamper rejection.
Strict core typing, Ruff, strict documentation and all public schema exports
plus their ledger are verified separately.

The [complete archive](../assets/joint_radar_evaluation_reuse/README.md)
retains six workers, source capsules/manifests, input/download/generator
identities, the actual measured launcher, timings and validation receipts.
Derived data retains KITTI attribution and CC BY-NC-SA 3.0 separately from
Apache-2.0 source code.

## Reproduce

```bash
python -m tools.benchmark_radar_reference_indexing \
  --audit-only docs/assets/joint_radar_evaluation_reuse
```

Remeasure with the original control and generator:

```bash
python -m tools.benchmark_radar_reference_indexing --solver joint --joint-work-counters \
  --baseline-commit d235e388dff403424447bca7548147b3e61ce06d \
  --control docs/assets/joint_radar_evaluation_reuse/control.json \
  --control-generator docs/assets/joint_radar_evaluation_reuse/control-generator.py.txt \
  --output-dir outputs/joint-radar-reuse-new --repetitions 3
```
