# GPU associations and compiled CPU replay for continuous-time LiDAR

Capture the native uniform voxel neighborhoods and adaptive spatial topology once.
Both maps query the same coordinate upload on an owned CUDA stream. Uniform neighborhood
order, nearest normal rejection and inclusive gates retain native FP64 arithmetic.
Adaptive ties, tiny distances and gate boundaries use the existing CPU repair.
Diagnostics stream XYZ without constructing every pose and observation.
Native per-point capture times, trajectory/base composition and source transforms also batch
on the optional compiled CPU adapter. Rounding, failure prefixes and logical counters match.

## Complete-process comparisons against the preceding GPU version

Each workload has five fresh paired subprocesses in alternating order. Both methods use
CUDA finite differences; the baseline retains CPU associations and scalar replay. Independent
Numba/CuPy caches, initial compilation, every slow run and every outlier are included.
Wall includes replay, all clock candidates/refinements, transfers, diagnostics, hashes and
saved artifacts. Parent audits are outside the timer.

| Public workload | Train / holdout targets | Preceding GPU median | Candidate median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|---:|
| 120 frames, up to 8,000 points/frame | 768,000 / 192,000 | 126.761159 s | 69.113193 s | 1.834x | 1.657x |
| 120 frames, up to 400 points/frame | 38,400 / 9,600 | 10.849183 s | 9.257591 s | 1.172x | 1.048x |
| Default sampling caps, CPU dispatch | 6,000 / 3,200 | 8.128536 s | 9.054054 s | 0.898x | 0.872x |

Small budgets add no GPU work. Kernel/phase timings do not establish the same
complete-process ratio. These descriptive measurements do not establish a portable speedup.

| Workload | API medians | Solver medians | All-thread CPU medians |
|---|---:|---:|---:|
| 120 frames, up to 8,000 points/frame | 122.457685 -> 65.568554 s (1.868x) | 89.488570 -> 52.911092 s (1.691x) | 182.484375 -> 99.687500 s (1.831x) |
| 120 frames, up to 400 points/frame | 8.416714 -> 6.652229 s (1.265x) | 5.579711 -> 4.097385 s (1.362x) | 15.156250 -> 11.875000 s (1.276x) |
| Default sampling caps, CPU dispatch | 4.781339 -> 4.739577 s (1.009x) | 3.491127 -> 2.467725 s (1.415x) | 9.421875 -> 10.156250 s (0.928x) |

### 120 frames, up to 8,000 points/frame

Status `max_iterations`, rank 6; retained train/holdout correspondences 582,119/67,602. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 103.215496 s | 64.228128 s | 1.607x |
| 2 | candidate -> baseline | 141.827398 s | 91.374901 s | 1.552x |
| 3 | baseline -> candidate | 131.563788 s | 79.409100 s | 1.657x |
| 4 | candidate -> baseline | 126.761159 s | 69.113193 s | 1.834x |
| 5 | baseline -> candidate | 124.050141 s | 56.566337 s | 2.193x |

[dense definition](dense.definition.json), [aggregate](dense.benchmark.json), [complete measurement](dense.sparse.repeat-01.measurement.json), [saved result](dense.sparse.repeat-01.result.yaml).

### 120 frames, up to 400 points/frame

Status `max_iterations`, rank 6; retained train/holdout correspondences 29,181/3,451. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 18.388750 s | 23.190575 s | 0.793x |
| 2 | candidate -> baseline | 9.704557 s | 9.257591 s | 1.048x |
| 3 | baseline -> candidate | 10.786194 s | 7.621274 s | 1.415x |
| 4 | candidate -> baseline | 10.849183 s | 8.170953 s | 1.328x |
| 5 | baseline -> candidate | 18.364678 s | 19.335320 s | 0.950x |

[full definition](full.definition.json), [aggregate](full.benchmark.json), [complete measurement](full.sparse.repeat-01.measurement.json), [saved result](full.sparse.repeat-01.result.yaml).

### Default sampling caps, CPU dispatch

Status `converged`, rank 6; retained train/holdout correspondences 4,773/2,526. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 20.835861 s | 34.053724 s | 0.612x |
| 2 | candidate -> baseline | 7.478140 s | 10.341684 s | 0.723x |
| 3 | baseline -> candidate | 10.192716 s | 8.959855 s | 1.138x |
| 4 | candidate -> baseline | 7.891478 s | 9.054054 s | 0.872x |
| 5 | baseline -> candidate | 8.128536 s | 8.041445 s | 1.011x |

[default definition](default.definition.json), [aggregate](default.benchmark.json), [complete measurement](default.sparse.repeat-01.measurement.json), [saved result](default.sparse.repeat-01.result.yaml).

## Separate preliminary phase diagnostics

On the dense population, numeric query call sums fall from 67.896332 to 4.035025 s (16.827x).
These are two instrumented runs. Calls overlap across concurrent candidates; nested timers
overlap too. They are not critical-path fractions or pooled with formal measurements.
The association-only profile precedes streamed diagnostics and binds its modified working
tree by exact source hashes. Historical diagnostic comparison additionally excludes only
the provenance git commit. Complete original results and both exact source snapshots remain.

[Prior profile](attempts/v1/profiles/prior/profile.json), [association-only profile](attempts/v1/profiles/association_only/profile.json).

## Actual GPU work in the final source

A separate standard-capture diagnostic records 52 GPU query batches, 1,248,000 uniform and 1,248,000 adaptive point queries. It uploads 105,635 bytes of captured maps and 59,904,000 bytes of query inputs. Native CPU repair handles 0 uncertain rows. No complete uniform/adaptive query batch falls back to CPU.

The same run records 129 GPU differences over 3,782,178 correspondence rows, 26 retained observation uploads (103,523,608 bytes), 207 CPU standalone residual calls and 129 ordered CPU normal calls.

Standalone residuals, trajectory interpolation, adaptive repairs and ordered reductions remain
on CPU. Small capture budgets remain CPU-only and do not import CuPy. Source/input/HEAD guards
pass, and the complete diagnostic result matches the final formal result.

[Diagnostic](diagnostic.json), [measurement](diagnostic.measurement.json), [launcher](diagnostic-launcher.py.txt).

Compiled CPU replay records 121 batches over 51,000 rows.
This CPU work is separate from GPU association/linearization accounting. Timestamp rounding,
native source world points, target poses/extrapolation values and logical counters match.

## Preserved preceding complete comparison

The association/streamed-diagnostic candidate alone measured only 1.379x dense whole-wall gain.
Its standard-capture ratio of medians was 0.912x, with slower candidate pairs retained.
All 30 complete trials, both preliminary profiles, exact sources and actual-device diagnostic
remain [in the preceding index](attempts/v1/publication-index.json), separate from final timings.
A [working-tree replay profile](profiles/replay/profile.json) also retains its exact sources
and complete result. It is instrumented and not pooled with formal comparisons.

## Reproduction and provenance

Measured candidate `12500f035fc059f30f40af61dd42f4ef26bdb03a`; preceding CUDA baseline `a1ff874ab0dbc39c2dacc9607f40841c610997d7`. Publication adds docs/evidence.

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
Comparison normalizes path separators and excludes only the exact bound solver and replay execution notes.
Portable measurements change only module location metadata; definitions also replace command
and source locations. Original hashes and changed fields are recorded in the manifest.

Validation: 546 relevant local tests, including actual CUDA association and optimizer checks;
strict typing for 356 modules; 126 schema files byte-identical to the preceding stage.

[Index](publication-index.json), [conversion lineage](conversion-provenance.json),
[benchmark launcher](benchmark-launcher.py.txt). All archived source bytes retain their exact
original line endings; committed-code comparisons normalize source line endings only.
