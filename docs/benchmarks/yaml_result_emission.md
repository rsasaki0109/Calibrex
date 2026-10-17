# Safe YAML result emission

Writing the complete public TUM result takes **3.763 → 0.819 seconds** through
`write_mapping`, a **4.60x** improvement. Atomic writing takes
**3.902 → 0.834 seconds**, a **4.68x** improvement. All twelve exported files have
identical bytes across both implementations.

These measurements export an already computed result. The separate full
calibration comparison gives **46.270 → 38.724 seconds (1.195x)** on two
alternating pairs. That whole-pipeline effect is modest; the export ratios apply
to result writing.

## Implementation and compatibility

The existing PyYAML dependency supplies optional `CSafeDumper` support. Plain
data uses that safe emitter when its representer and resolver tables match the
Python safe dumper. Python emission remains available for missing LibYAML,
custom representers or alias policy, unsupported objects and lone surrogates.
JSON writing follows the same code path as before.

Native emission constructs the complete YAML string before writing to the
stream. A native emitter error can therefore fall back before any partial prefix
is written. This temporarily retains the YAML string in memory; the measured
file is 3,984,581 bytes. The atomic writer retains its temporary file, flush,
fsync, replacement and failure-cleanup behavior.

Tests check parsed values, float bits, ordering, aliases and cycles, Unicode,
custom hooks, fallback, LF newlines, unchanged JSON bytes and atomic failures.
YAML formatting can differ for other inputs; consumers should bind checksums to
the bytes actually saved. All 129 public schema exports and their ledger remain
unchanged.

## Complete export workload

Each fresh worker loads the unchanged 5,339,931-byte JSON result derived from
the public TUM fr1/xyz development sequence and saves its complete contents once
through each public writer. Three alternating pairs produce six workers and
twelve exports. Source capsules contain the exact Git bytes; only
`calibrex/core/io.py` changes.

| Scope | Prior seconds | Current seconds | Prior/current |
| --- | ---: | ---: | ---: |
| Fresh export worker: imports, input loading, both writes and worker evidence | 8.444 | 2.450 | 3.446x |
| One complete `write_mapping` call | 3.763 | 0.819 | 4.597x |
| One complete `write_mapping_atomic` call, including fsync | 3.902 | 0.834 | 4.677x |

| Pair | Current worker seconds | Prior worker seconds | Execution order |
| --- | ---: | ---: | --- |
| 1 | 2.450 | 8.475 | current, prior |
| 2 | 2.448 | 8.384 | prior, current |
| 3 | 2.646 | 8.444 | current, prior |

Every file is reloaded and compared against the complete input, preserving
mapping/list order, JSON numeric types, signed zero and all calibration fields.
The [byte-parity record](../assets/yaml_result_emission/export-byte-parity.json)
also records the shared SHA-256 of all twelve outputs. The source calibration's
provenance remains in those files; each worker separately identifies the writer
runtime that produced the copy.

## Full calibration check

The unchanged configuration retains three windows, all full-resolution input
pixels, sampling and iteration limits, ablations, holdouts, known-bad controls,
reassociation, curvature and multistart. Both variants use the existing three
window processes with one BLAS/OpenMP thread each.

| Fresh calibration worker | Whole seconds |
| --- | ---: |
| 01-current | 40.203 |
| 02-prior | 47.214 |
| 03-prior | 45.327 |
| 04-current | 37.244 |

The medians give a 16.3% reduction in this exploratory full-pipeline comparison.
The host was unreserved; these small samples provide no confidence interval or
hardware-independent guarantee. Whole-worker timings include startup, child
processes, factor transport, traces, result writing and aggregation.

The supplementary audit rebuilds aggregate traces from all raw parent and child
records. Every run matches the original published history: 75 complete decoded
images, 75 ordered point samples, 24 optimizer results, 168 Jacobians, 1,624
residual arrays and 1,498 diagnostics. All 492 fixed-population probe evaluations
also agree. Complete schema-valid results retain grades, failed controls, ranks,
warnings and correspondence lineage.

Only independently verified Git identities, generated run IDs/dates and verified
output/config paths and hashes are normalized. Inputs, calibration options and
the execution policy are checked before comparison.

## Evidence and reproduction

- Prior runtime: `2ab25954f4af1c9b8229d332487d9910f793564a`.
- Measured writer runtime: `6f22f69f67bcafbe2d7f1e698f34f1c509aef833`.
- [Export protocol](../assets/yaml_result_emission/protocol.json),
  [all export attempts](../assets/yaml_result_emission/attempts.json),
  [export summary](../assets/yaml_result_emission/summary.json).
- [Complete calibration checks](../assets/yaml_result_emission/calibration-validation.json),
  [all calibration attempts](../assets/yaml_result_emission/calibration/attempts.json).
- [Validation record](../assets/yaml_result_emission/validation.json) and
  [complete archive inventory](../assets/yaml_result_emission/index.json).

```powershell
python tools/benchmark_yaml_emission.py --audit-only docs/assets/yaml_result_emission
python tools/audit_tum_yaml_emission.py docs/assets/yaml_result_emission
python tools/benchmark_yaml_emission.py --baseline-commit 2ab25954f4af1c9b8229d332487d9910f793564a --output outputs/new-yaml-emission-comparison
```

The input result is already used development evidence. This is a runtime change
with no accuracy or SOTA claim.
