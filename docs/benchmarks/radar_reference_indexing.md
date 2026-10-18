# Indexed reference timestamps for Radar clock profiling

The complete prepared-kinematics calibration workflow changes from
**90.695003 to 11.618504 seconds (7.806083x)** on the public-data-derived
execution control below. The solver API changes from **88.714816 to
6.589154 seconds (13.463765x)**. This is CPU acceleration of the profiled
lever-arm/clock solver, enabled automatically through its existing API.

## Workload and scope

Public KITTI raw drive `2011_09_30_drive_0018_sync` supplies all 2,762 OXTS
samples. A deterministic control generates 2,758 Radar-frame velocities,
one per interior timestamp with full clock/probe support. It declares a
17-degree yaw, lever arm `[0.42, -0.18, 0.27]` m, clock offset 0.023 s,
deterministic 0.0001 m/s sinusoidal noise, and unequal measurement weights.
The zero-based OXTS fields are `vf,vl,vu [8:11]` and `wf,wl,wu [20:23]`;
original timestamp differences are computed in integer nanoseconds before
conversion to seconds. All generation code and raw file identities are retained.

**These are generated Radar velocities, not independently measured Radar.**
The workload establishes performance and numerical parity. It does not
establish Radar accuracy, uncertainty coverage, LiDAR odometry speed, or a
SOTA claim. No locked accuracy cohort is opened or updated.

The unchanged default search evaluates all 201 offsets over +/-0.1 s at
0.001 s steps. It retains the same sorted IDs, train/holdout split, weights,
weighted linear least squares, best-candidate selection, rank/condition
checks, time curvature, coupled observability and eight signed holdout probes.
No measurements, search candidates or diagnostics are reduced.

## Change

Previously each profile candidate scanned reference timestamps from the first
sample for every measurement. With K clock candidates, M measurements and N
reference samples, this lookup cost grows as O(K M N). The solver now builds
one finite timestamp tuple and selects the same two bracketing samples with
`bisect_left`: O(N + K M log N) lookup work and O(N) additional memory. Sorting
and the least-squares work are unchanged.

The original interpolation function evaluates that pair with its original
arithmetic. Exact-knot queries retain the interval on the left, including
duplicate-clock behavior; they do not bypass interpolation by returning a
stored knot. Non-finite reference clocks retain the original scan path.
The public API, options, result format and standalone RMSE helper are unchanged.

## Complete fresh-process measurements

Three alternating pairs use the same interpreter and installed NumPy/SciPy,
one BLAS thread, and frozen source trees with real Git identities. The host
was unreserved and variation is material; all six timings are retained.
Whole-process time includes imports, prepared-control JSON loading, object
construction, calibration, diagnostics, capture of every profile candidate,
serialization and saving. Downloading, control generation, source verification
and external auditing are outside both timed workflows.

| Pair (execution order) | Prior whole s | Indexed whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (prior, current) | 104.309565 | 11.618504 | 8.978x |
| 2 (current, prior) | 66.392337 | 15.813232 | 4.199x |
| 3 (prior, current) | 90.695003 | 8.016925 | 11.313x |

The reported **7.806083x** is the ratio of the two three-run medians. It is
specific to this reference length and workload; the complete timings expose
the host variation. No run is discarded or relabeled as a warmup.

The baseline source is `2f4c4fdc80fc7469c2d03e9f2c759da66c7cbd2e`; the measured source
is `559df53946079719ac2052559138dabcedabbeb9`. Across the complete runtime manifests,
only `solvers/radar_spatiotemporal_lever_arm_solver.py` changes.

## Parity and validation

Every run retains the full output and all 201 candidate translations, MSEs
and singular spectra. All six outputs and all 1,206 candidate records match
exactly, including the final holdout/probe values, coupled diagnostics,
options and measurement IDs. Numerical values receive no tolerance or
normalization. Commit identity is stored separately in the provenance wrapper.

Regression tests check floating-point bytes at exact knots, nextafter
neighbors, duplicate timestamps, endpoint/outside queries and epoch-sized
times; complete weighted solver parity; two-sample interpolation support;
and non-finite-clock fallback. Related calibration tests and archive tests
also exercise the Radar service, joint solver and tamper rejection. Strict
core typing and schema export parity are checked separately. The final
targeted suite passes 77 tests; strict typing checks all 378 core source
files, Ruff and the strict documentation build pass, and the 129 exported
schemas plus registry ledger (130 files) match the committed contents.

The [complete archive](../assets/radar_reference_indexing/README.md) retains
the inputs, original download receipts, source capsules/manifests, actual
launcher and helpers, all six workers, numerical traces, audit and validation
receipts. The indexed derived numerical data carries KITTI attribution and
CC BY-NC-SA 3.0 terms separately from Apache-2.0 source code.

## Reproduce

Audit the committed results without downloading data or executing calibration:

```bash
python -m tools.benchmark_radar_reference_indexing \
  --audit-only docs/assets/radar_reference_indexing
```

To remeasure, materialize the OXTS directory from the original public
`2011_09_30_drive_0018_sync` archive (URL and SHA-256 identities are in
`input-download.json`). The retained range-fetch generator selects only
OXTS members, avoiding a whole-drive download. Regenerate a control with the
current launcher, then run both source versions in fresh workers:

On a fresh checkout, the first two commands fetch the OXTS files. With that
directory already present, start at `--prepare-oxts`.

```bash
python -c "from pathlib import Path; Path('.tmp-calibrex-fetch-oxts-reference-v1.py').write_bytes(Path('docs/assets/radar_reference_indexing/input-fetcher.py.txt').read_bytes())"
python .tmp-calibrex-fetch-oxts-reference-v1.py
python -m tools.benchmark_radar_reference_indexing \
  --prepare-oxts data/public/kitti_oxts_reference_v1 \
  --output-dir outputs/radar-control-new
python -m tools.benchmark_radar_reference_indexing \
  --baseline-commit 2f4c4fdc80fc7469c2d03e9f2c759da66c7cbd2e \
  --control outputs/radar-control-new/control.json \
  --output-dir outputs/radar-reference-new --repetitions 3
```
