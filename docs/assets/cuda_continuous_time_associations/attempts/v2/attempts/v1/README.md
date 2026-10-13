# CUDA continuous-time LiDAR associations

Capture the native uniform voxel neighborhoods and adaptive spatial topology once.
Both maps query the same coordinate upload on an owned CUDA stream. Uniform neighborhood
order, nearest normal rejection and inclusive gates retain native FP64 arithmetic.
Adaptive ties, tiny distances and gate boundaries use the existing CPU repair.
Diagnostics stream XYZ without constructing every pose and observation.

## Complete-process comparisons against the preceding GPU version

Each workload has five fresh paired subprocesses in alternating order. Both methods use
CUDA finite differences; the baseline retains CPU associations. Initially empty independent
Numba/CuPy caches, initial compilation, every slow run and every outlier are included.
Wall includes replay, all clock candidates/refinements, transfers, diagnostics, hashes and
saved artifacts. Parent audits are outside the timer.

| Public workload | Train / holdout targets | Preceding GPU median | Candidate median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|---:|
| 120 frames, up to 8,000 points/frame | 768,000 / 192,000 | 121.270540 s | 87.924570 s | 1.379x | 1.379x |
| 120 frames, up to 400 points/frame | 38,400 / 9,600 | 16.867248 s | 18.493694 s | 0.912x | 1.107x |
| Default sampling caps, CPU dispatch | 6,000 / 3,200 | 6.477332 s | 6.284855 s | 1.031x | 0.998x |

Small budgets add no GPU work. Kernel/phase timings do not establish the same
complete-process ratio. These descriptive measurements do not establish a portable speedup.

| Workload | API medians | Solver medians | All-thread CPU medians |
|---|---:|---:|---:|
| 120 frames, up to 8,000 points/frame | 118.220670 -> 85.177026 s (1.388x) | 81.714399 -> 51.801816 s (1.577x) | 174.875000 -> 113.093750 s (1.546x) |
| 120 frames, up to 400 points/frame | 13.032843 -> 13.626851 s (0.956x) | 9.036709 -> 8.942251 s (1.011x) | 20.250000 -> 18.625000 s (1.087x) |
| Default sampling caps, CPU dispatch | 3.747747 -> 3.562632 s (1.052x) | 2.890792 -> 2.716652 s (1.064x) | 7.750000 -> 7.421875 s (1.044x) |

### 120 frames, up to 8,000 points/frame

Status `max_iterations`, rank 6; retained train/holdout correspondences 582,119/67,602. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 132.344839 s | 89.377082 s | 1.481x |
| 2 | candidate -> baseline | 120.203982 s | 79.703879 s | 1.508x |
| 3 | baseline -> candidate | 125.633335 s | 103.031409 s | 1.219x |
| 4 | candidate -> baseline | 98.483385 s | 78.604546 s | 1.253x |
| 5 | baseline -> candidate | 121.270540 s | 87.924570 s | 1.379x |

[dense definition](dense.definition.json), [aggregate](dense.benchmark.json), [complete measurement](dense.sparse.repeat-01.measurement.json), [saved result](dense.sparse.repeat-01.result.yaml).

### 120 frames, up to 400 points/frame

Status `max_iterations`, rank 6; retained train/holdout correspondences 29,181/3,451. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 22.807361 s | 18.493694 s | 1.233x |
| 2 | candidate -> baseline | 23.182989 s | 31.423578 s | 0.738x |
| 3 | baseline -> candidate | 16.867248 s | 15.230354 s | 1.107x |
| 4 | candidate -> baseline | 13.752384 s | 11.595538 s | 1.186x |
| 5 | baseline -> candidate | 14.636972 s | 21.429495 s | 0.683x |

[full definition](full.definition.json), [aggregate](full.benchmark.json), [complete measurement](full.sparse.repeat-01.measurement.json), [saved result](full.sparse.repeat-01.result.yaml).

### Default sampling caps, CPU dispatch

Status `converged`, rank 6; retained train/holdout correspondences 4,773/2,526. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 17.651374 s | 13.999704 s | 1.261x |
| 2 | candidate -> baseline | 5.782925 s | 6.113695 s | 0.946x |
| 3 | baseline -> candidate | 6.534609 s | 6.800862 s | 0.961x |
| 4 | candidate -> baseline | 6.477332 s | 5.727973 s | 1.131x |
| 5 | baseline -> candidate | 6.275412 s | 6.284855 s | 0.998x |

[default definition](default.definition.json), [aggregate](default.benchmark.json), [complete measurement](default.sparse.repeat-01.measurement.json), [saved result](default.sparse.repeat-01.result.yaml).

## Separate preliminary phase diagnostics

On the dense population, numeric query call sums fall from 67.896332 to 4.035025 s (16.827x).
These are two instrumented runs. Calls overlap across concurrent candidates; nested timers
overlap too. They are not critical-path fractions or pooled with formal measurements.
The association-only profile precedes streamed diagnostics and binds its modified working
tree by exact source hashes. Historical diagnostic comparison additionally excludes only
the provenance git commit. Complete original results and both exact source snapshots remain.

[Prior profile](profiles/prior/profile.json), [association-only profile](profiles/association_only/profile.json).

## Actual GPU work in the final source

A separate standard-capture diagnostic records 52 GPU query batches, 1,248,000 uniform and 1,248,000 adaptive point queries. It uploads 105,635 bytes of captured maps and 59,904,000 bytes of query inputs. Native CPU repair handles 0 uncertain rows. No complete uniform/adaptive query batch falls back to CPU.

The same run records 129 GPU differences over 3,782,178 correspondence rows, 26 retained observation uploads (103,523,608 bytes), 207 CPU standalone residual calls and 129 ordered CPU normal calls.

Standalone residuals, trajectory interpolation, adaptive repairs and ordered reductions remain
on CPU. Small capture budgets remain CPU-only and do not import CuPy. Source/input/HEAD guards
pass, and the complete diagnostic result matches the final formal result.

[Diagnostic](diagnostic.json), [measurement](diagnostic.measurement.json), [launcher](diagnostic-launcher.py.txt).

## Reproduction and provenance

Measured candidate `4a7a5f327c9c6b41b978a633275ed84e5e4e5493`; preceding CUDA baseline `a1ff874ab0dbc39c2dacc9607f40841c610997d7`. Publication adds docs/evidence.

```bash
pip install -e ".[dev,cuda,numba]"
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope cuda \
  --baseline-commit a1ff874ab0dbc39c2dacc9607f40841c610997d7 \
  --baseline-query-backend cuda --candidate-query-backend cuda \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_dense_capture_config.yaml \
  --repetitions 5 \
  --output-dir outputs/public-glim-cuda-associations-dense-reproduction
```

Use the public-dataset tutorial to download and checksum-verify the GLIM bag and trajectory,
then run the declared restamping/copied-topic conversion. The same sensor is duplicated as an
execution control; this supplies no independent two-sensor extrinsic/clock accuracy evidence.
All capture, train/holdout, solver, profile and robust-loss settings remain fixed in each pair.

The benchmark archives twelve baseline modules, including the old packing and CUDA adapters.
Both execution overrides, actual module locations, source/input/config hashes, cache file
hashes and every complete saved/API value, status, trace and count are retained.
Comparison normalizes path separators and excludes only the exact bound execution note.
Portable measurements change only module location metadata; definitions also replace command
and source locations. Original hashes and changed fields are recorded in the manifest.

Validation: 455 relevant local tests, including actual CUDA association and optimizer checks;
strict typing for 355 modules; 126 schema files byte-identical to the preceding stage.

[Index](publication-index.json), [conversion lineage](conversion-provenance.json),
[benchmark launcher](benchmark-launcher.py.txt). All archived source bytes retain their exact
original line endings; committed-code comparisons normalize source line endings only.
