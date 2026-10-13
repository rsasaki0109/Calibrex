# GPU associations and compiled CPU replay for continuous-time LiDAR

Capture the native uniform voxel neighborhoods and adaptive spatial topology once.
Both maps query the same coordinate upload on an owned CUDA stream. Uniform neighborhood
order, nearest normal rejection and inclusive gates retain native FP64 arithmetic.
Adaptive ties, tiny distances and gate boundaries use the existing CPU repair.
Diagnostics stream XYZ without constructing every pose and observation.
Frames with at least 4,096 selected points batch capture-time trajectory/base calculations
on the optional compiled CPU adapter. Smaller frames retain scalar replay without extra JIT.

## Complete-process comparisons against the preceding GPU version

Each workload has five fresh paired subprocesses in alternating order. Dense/standard methods
use CUDA differences; default caps use CPU. The baseline retains CPU associations and scalar replay.
Independent
Numba/CuPy caches, initial compilation, every slow run and every outlier are included.
Wall includes replay, all clock candidates/refinements, transfers, diagnostics, hashes and
saved artifacts. Parent audits are outside the timer.

| Public workload | Train / holdout targets | Preceding GPU median | Candidate median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|---:|
| 120 frames, up to 8,000 points/frame | 768,000 / 192,000 | 113.100676 s | 61.548794 s | 1.838x | 1.727x |
| 120 frames, up to 400 points/frame | 38,400 / 9,600 | 18.195873 s | 14.291126 s | 1.273x | 1.107x |
| Default sampling caps, CPU dispatch | 6,000 / 3,200 | 5.671808 s | 5.456039 s | 1.040x | 1.054x |

Small budgets add no GPU work. Kernel/phase timings do not establish the same
complete-process ratio. These descriptive measurements do not establish a portable speedup.

| Workload | API medians | Solver medians | All-thread CPU medians |
|---|---:|---:|---:|
| 120 frames, up to 8,000 points/frame | 109.680328 -> 58.592788 s (1.872x) | 73.805746 -> 46.042372 s (1.603x) | 160.484375 -> 85.203125 s (1.884x) |
| 120 frames, up to 400 points/frame | 13.696047 -> 11.098158 s (1.234x) | 9.749505 -> 7.102957 s (1.373x) | 19.906250 -> 16.046875 s (1.241x) |
| Default sampling caps, CPU dispatch | 3.172026 -> 3.085743 s (1.028x) | 2.376252 -> 2.274642 s (1.045x) | 6.750000 -> 6.531250 s (1.033x) |

### 120 frames, up to 8,000 points/frame

Status `max_iterations`, rank 6; retained train/holdout correspondences 582,119/67,602. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 104.532851 s | 67.069371 s | 1.559x |
| 2 | candidate -> baseline | 147.618909 s | 51.782368 s | 2.851x |
| 3 | baseline -> candidate | 116.630315 s | 67.520951 s | 1.727x |
| 4 | candidate -> baseline | 113.100676 s | 61.548794 s | 1.838x |
| 5 | baseline -> candidate | 92.061607 s | 59.253833 s | 1.554x |

[dense definition](dense.definition.json), [aggregate](dense.benchmark.json), [complete measurement](dense.sparse.repeat-01.measurement.json), [saved result](dense.sparse.repeat-01.result.yaml).

### 120 frames, up to 400 points/frame

Status `max_iterations`, rank 6; retained train/holdout correspondences 29,181/3,451. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 28.353358 s | 72.676153 s | 0.390x |
| 2 | candidate -> baseline | 19.394306 s | 19.170858 s | 1.012x |
| 3 | baseline -> candidate | 13.736185 s | 12.404626 s | 1.107x |
| 4 | candidate -> baseline | 18.195873 s | 14.291126 s | 1.273x |
| 5 | baseline -> candidate | 13.118011 s | 10.623346 s | 1.235x |

[full definition](full.definition.json), [aggregate](full.benchmark.json), [complete measurement](full.sparse.repeat-01.measurement.json), [saved result](full.sparse.repeat-01.result.yaml).

### Default sampling caps, CPU dispatch

Status `converged`, rank 6; retained train/holdout correspondences 4,773/2,526. These statuses are retained.

| Pair | Order | Preceding GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 17.270732 s | 15.580718 s | 1.108x |
| 2 | candidate -> baseline | 5.671808 s | 5.382022 s | 1.054x |
| 3 | baseline -> candidate | 5.992307 s | 5.313003 s | 1.128x |
| 4 | candidate -> baseline | 5.480828 s | 5.456039 s | 1.005x |
| 5 | baseline -> candidate | 5.432669 s | 6.314291 s | 0.860x |

[default definition](default.definition.json), [aggregate](default.benchmark.json), [complete measurement](default.sparse.repeat-01.measurement.json), [saved result](default.sparse.repeat-01.result.yaml).

## Separate dense CPU/hybrid control

Three fresh paired subprocesses compare the preceding packed/parallel CPU implementation
with the final hybrid. This secondary cohort retains its cold/slower runs and is separate
from the five-pair GPU/GPU comparisons above.

Whole-process medians: 167.319674 -> 66.829418 s (2.504x); median paired ratio 2.816x.

| Pair | Order | CPU wall | Hybrid wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | CPU -> hybrid | 220.142010 s | 76.609320 s | 2.874x |
| 2 | hybrid -> CPU | 161.490250 s | 66.829418 s | 2.416x |
| 3 | CPU -> hybrid | 167.319674 s | 59.410116 s | 2.816x |

CPU source `eb50c99878706a45eb0e584f3573507313cdc172`; candidate source `b9eadf0f2a14c1c8993d85bcb2da29a73878893c`.
Compiled scope archives ten baseline modules. The captured packed builder is shared; its
new GPU callback is disabled for the CPU method. Its existing numeric selection stays native.
Source/module locations and complete output/split identities are verified in each process.

[CPU/hybrid definition](cpu_control/dense.definition.json), [aggregate](cpu_control/dense.benchmark.json).

## Separate preliminary phase diagnostics

On the dense population, numeric query call sums fall from 67.896332 to 4.035025 s (16.827x).
These are two instrumented runs. Calls overlap across concurrent candidates; nested timers
overlap too. They are not critical-path fractions or pooled with formal measurements.
The association-only profile precedes streamed diagnostics and binds its modified working
tree by exact source hashes. Historical diagnostic comparison additionally excludes only
the provenance git commit. Complete original results and both exact source snapshots remain.

[Prior profile](attempts/v2/attempts/v1/profiles/prior/profile.json), [association-only profile](attempts/v2/attempts/v1/profiles/association_only/profile.json).

## Actual GPU work in the final source

A separate dense-capture diagnostic records 52 GPU query batches, 24,960,000 uniform and 24,960,000 adaptive point queries. It uploads 105,635 bytes of captured maps and 1,198,080,000 bytes of query inputs. Native CPU repair handles 0 uncertain rows. No complete uniform/adaptive query batch falls back to CPU.

The same run records 129 GPU differences over 75,503,015 correspondence rows, 26 retained observation uploads (2,066,730,664 bytes), 207 CPU standalone residual calls and 129 ordered CPU normal calls.

Standalone residuals, trajectory interpolation, adaptive repairs and ordered reductions remain
on CPU. Small capture budgets remain CPU-only and do not import CuPy. Source/input/HEAD guards
pass, and the complete diagnostic result matches the final formal result.

[Diagnostic](diagnostic.json), [measurement](diagnostic.measurement.json), [launcher](diagnostic-launcher.py.txt).

Compiled CPU replay records 120 batches over 960,000 rows.
This CPU work is separate from GPU association/linearization accounting. Timestamp rounding,
native source world points, target poses/extrapolation values and logical counters match.

## Preserved preceding complete comparisons

The association/streamed-diagnostic candidate alone measured only 1.379x dense whole-wall gain.
Its standard-capture ratio of medians was 0.912x, with slower candidate pairs retained.
All 30 complete trials, both preliminary profiles, exact sources and actual-device diagnostic
remain [in the initial index](attempts/v2/attempts/v1/publication-index.json), separate from final timings.
The subsequent 64-row replay dispatch measured 1.834x dense whole-wall improvement, but
regressed default budgets: 8.128536 -> 9.054054 s (0.898x). Its first default candidate took
34.053724 s versus 20.835861 s for the baseline; all five pairs remain.
That cold/small-frame cost motivated the final 4,096-point crossover and small-budget
factory bypass. [All 30 preceding trials](attempts/v2/publication-index.json) remain separate.

A [working-tree replay profile](attempts/v2/profiles/replay/profile.json) also retains its exact sources
and complete result. It is instrumented and not pooled with formal comparisons.

## Reproduction and provenance

Measured candidate `b9eadf0f2a14c1c8993d85bcb2da29a73878893c`; preceding CUDA baseline `a1ff874ab0dbc39c2dacc9607f40841c610997d7`. Publication adds docs/evidence.

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

Validation: 546 relevant local regression tests for the combined implementation, followed by 96, including actual CUDA association and optimizer checks;
crossover/pipeline checks; strict typing for 356 modules; 126 schema files byte-identical to the preceding stage.

[Index](publication-index.json), [conversion lineage](conversion-provenance.json),
[benchmark launcher](benchmark-launcher.py.txt). All archived source bytes retain their exact
original line endings; committed-code comparisons normalize source line endings only.
