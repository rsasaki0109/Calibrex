# Native registration preparation and multi-start execution

Native registration now forms voxel centroids and ICP information matrices with
NumPy arrays, reuses each solve's training coordinates, and batches target-normal
PCA calculations. Larger point-to-point clouds also execute independent
multi-start diagnostics with bounded CPU threads. These changes extend the
previous exact spatial-neighbor acceleration and use existing dependencies.

## Preserved behavior

Voxel preprocessing still rejects nonfinite XYZ rows, sorts integer voxel keys
lexicographically, emits their centroids, and applies the same deterministic
uniform cap. `np.add.at` accumulates each coordinate in input order from zero;
the implementation does not replace sequential sums with pairwise reductions.
Only selected centroids become Python point objects. Coordinates whose voxel
keys cannot fit int64 use the original scalar path, preserving arbitrary
precision keys and exceptions on invalid division. Extra input fields remain
ignored. No voxel size or sampling cap changes as part of acceleration.

Point ICP fills the same 3N-by-6 Jacobian in the same row order, including signed
zeros, and retains its SVD, rank and condition checks. It materializes training
coordinates once per solve. It does not substitute the squared-condition-number
normal equations for the Jacobian SVD.

Plane ICP uses the same indexed neighborhoods, coordinate means, covariance
matrices and 3-by-3 eigensolves. Neighborhood temporaries are batched to at most
65,536 neighborhood entries, or one target row when its requested neighborhood
exceeds that limit. Normal validity thresholds and eigenvector signs retain
the existing operations. The complete neighbor-index output still scales with
target count times neighborhood count. No peak-memory measurement is claimed.

`RobustPointToPointIcpOptions.multi_start_workers=0` selects at most four threads
and at most the reported logical CPU count when the source has at least 1,500
points; smaller clouds use one worker. `1` selects sequential execution and a
positive explicit value is capped at the six trials. The native registration
comparison accepts the same option in `registration_backend_comparison`.
Each trial owns its arrays and indexes; nested multi-start is disabled. Results
retain x/y/z/roll/pitch/yaw order. The scoped executor joins running work before
returning or propagating an exception. Concurrent solves can consume more CPU
and temporary memory than sequential execution; callers already parallelizing
cloud pairs can select one worker.

All correspondence gates, mutual checks, trimming, optimization iterations,
spatial holdout, rematching diagnostics, six multi-start trials, normal diagnostics
and twelve point-to-plane pose probes remain enabled with the existing settings.
No GPU, new optional dependency or schema migration is required.

## Public workload and timing scope

The indexed reference is commit `9fe0d4f6ff866bed771b62ccbe7b7e57924b12bb`.
The measured implementation is `41c820f4f55e8ead78dd367c33c7194f42758ad9`.
`baseline.*.txt` contains its four exact reference Git blobs, including the
neighbor helper. Reference plane diagnostics are bound to the reference point
solver and neighbor implementation. `measured.*.txt` archives the seven measured
source files as canonical Git blobs; original working-file hashes are retained
separately.

Inputs are public KITTI raw drive `2011_09_30_drive_0018_sync`, using
`0000000000:0000000115` and `0000000115:0000000230` target/source scan pairs.
These pairs share one scan and are 115 frame indices apart. They are not
independent accuracy trials or an adjacent-motion or two-sensor calibration
benchmark. Identity initialization uses no reference pose; motion distortion
is not compensated. Raw scans remain external.

Every measured call freshly reads and SHA-256-verifies both full scans,
decodes XYZ, computes the existing 0.5 m voxel centroids with a fixed
500/1,500/3,000-point cap, and runs both complete native solvers with `as_dict`.
`wall_seconds` includes these stages; their nonoverlapping times sum to wall.
Prepared-point/output digesting and benchmark serialization are outside wall.
This is a raw-scan native registration workload, not the full `calibrex calibrate`
CLI, PCD parsing, dataset inspection, optional GICP/NDT or report-writing latency.

Settings retain a 1.5 m gate, mutual matching, 0.8 trimming, 0.2 spatial holdout
with 1 m voxels, seed 0, and a 20-iteration limit. The three methods are the
already indexed reference, vectorization with point multi-start forced sequential,
and vectorization with automatic multi-start workers. Plane solver options are
identical across methods; only point ICP's execution worker control differs.
The reference accepts the same common options object and does not use the added
worker field. Plane results retain that common-options metadata unchanged.

Two pairs, three caps, five repetitions and three methods produce 90 final
invocations. Method order rotates by case and repetition, with no separate
warmup. Process/library/OS caches persist, and other work shares the host.
Prepared float64 positions are bound bit-for-bit, including signed zeros;
stable IDs have separate digests. Every complete solver payload must match
the indexed reference and all repetitions for its case.

All 90 final complete numerical payloads match, and prepared position digests
match bit-for-bit. Five-repeat raw-scan workload medians improve
**1.95–3.17x** over the already indexed reference. This gain includes
fresh input read/verification and preprocessing, plus both complete solvers.
The earlier spatial-search speedups are not multiplied into these ratios.
No invocation or timing outlier is discarded.

| Target:source frame | Cap per cloud | Fit / holdout | Indexed reference median | Vectorized sequential median | Vectorized automatic median | Reference / automatic | Reference / sequential / automatic ranges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0:115 | 500 | 403 / 97 | 1.780698 | 0.751342 | 0.562525 | 3.17x | 1.530465–4.255877 / 0.528011–0.952019 / 0.467755–0.929834 |
| 0:115 | 1500 | 1197 / 303 | 2.799606 | 1.087240 | 1.013277 | 2.76x | 1.787245–4.390253 / 0.918843–1.701962 / 0.823972–1.841960 |
| 0:115 | 3000 | 2392 / 608 | 4.409817 | 2.405734 | 1.850283 | 2.38x | 3.393604–8.122340 / 2.146330–3.855594 / 1.569247–3.061013 |
| 115:230 | 500 | 392 / 108 | 2.312225 | 0.888662 | 0.860638 | 2.69x | 1.898826–3.187636 / 0.848013–0.947454 / 0.772792–1.041256 |
| 115:230 | 1500 | 1208 / 292 | 1.419733 | 0.789381 | 0.729356 | 1.95x | 1.386330–2.602388 / 0.756822–0.936914 / 0.661151–1.341446 |
| 115:230 | 3000 | 2414 / 586 | 2.368904 | 1.331356 | 1.045963 | 2.26x | 2.113699–2.387236 / 1.311410–1.356887 / 0.975013–1.167380 |

Seconds for each nonoverlapping stage are summarized below. Each cell lists
reference / vectorized sequential / vectorized automatic medians. Stage medians
need not sum to median total wall; each individual trial's stages do sum to its
wall. The reports retain every stage value rather than inferring stage costs
by subtracting unrelated invocations.

| Target:source frame | Cap per cloud | Read/verify/decode/voxel medians | Point ICP medians | Plane ICP medians |
| --- | ---: | --- | --- | --- |
| 0:115 | 500 | 1.324477 / 0.246337 / 0.195424 | 0.329410 / 0.291446 / 0.226995 | 0.166122 / 0.183759 / 0.134894 |
| 0:115 | 1500 | 0.945300 / 0.167805 / 0.189880 | 1.473458 / 0.724065 / 0.609480 | 0.380847 / 0.195371 / 0.246815 |
| 0:115 | 3000 | 0.996100 / 0.191078 / 0.180127 | 2.774763 / 1.774917 / 1.019180 | 0.638953 / 0.448986 / 0.515400 |
| 115:230 | 500 | 1.564998 / 0.254889 / 0.246393 | 0.526453 / 0.469211 / 0.428976 | 0.223027 / 0.149377 / 0.164111 |
| 115:230 | 1500 | 0.498475 / 0.125497 / 0.149423 | 0.714516 / 0.534723 / 0.435863 | 0.206899 / 0.137090 / 0.155037 |
| 115:230 | 3000 | 0.541396 / 0.153082 / 0.127895 | 1.379953 / 0.915988 / 0.639415 | 0.383173 / 0.261740 / 0.277027 |

At the 3,000-point cap, pair 0:115 changes from 4.409817 to 1.850283 s
(2.38x) and pair 115:230 from 2.368904 to 1.045963 s (2.26x). Automatic
multi-start adds 1.27–1.30x over vectorized sequential total wall at that cap.
At 1,500 points its additional total-wall gain is 1.07–1.08x. Both 500-point
methods execute sequentially; their timing difference reflects host variation,
not threading. Vectorization and preprocessing account for most of the gain.

Timing variation is substantial: the reference 0:115 3,000-point case spans
the range shown above, and first 500-point reference preparation costs
3.470830 s. These observations remain in the reports. No host settings or
unrelated processes were changed for the experiment.

Point ICP converges for both 500-point cases and reaches the existing
20-iteration limit for the four larger cases. Plane ICP reaches that limit in
all six cases. These statuses, full histories and all diagnostics are identical
across methods. Environment: Windows, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1,
12 reported logical CPUs; the automatic parallel cases select four workers.


Five repetitions characterize observed timing variation, not confidence or a
portable speed guarantee. Bootstrap fields are schema summaries only. Equality
preserves prior solver behavior; it does not establish correct extrinsics or
observability. A trial marked successful means execution and payload parity,
including a solver that reaches its iteration limit.

## Reproduction and retained evidence

```bash
python tools/benchmark_registration_preparation.py \
  /path/to/2011_09_30_drive_0018_sync \
  --output-dir outputs/public-kitti-registration-preparation-v1 \
  --pair 0000000000:0000000115 --pair 0000000115:0000000230 \
  --sizes 500 1500 3000 --repeats 5
```

The output directory must be new. The tool hashes the same bytes it decodes,
checks that measured HEAD/source bytes stayed fixed, archives pinned source,
retains all trials, and writes standard benchmark definition and aggregate
schemas. `output_sha256` binds canonical JSON containing prepared-cloud digests
and the full point/plane results, rather than a separately saved result file.

[registration.definition.json](registration.definition.json) and
[registration.benchmark.json](registration.benchmark.json) contain all final
invocations and their standard aggregate. The two `pilot-v*.registration.*`
report pairs retain twelve tool-pilot invocations. Complete payload strings
are stored once in definitions; aggregate trial notes refer to the corresponding
definition and retain the same output digest. All other fields match the standard
aggregator exactly, checked after restoring those notes from the definitions.
Pilot v1 preserves two
failures: the sequential control recorded `multi_start_workers=1` in plane
option metadata while the others recorded `0`. Every other field matches the
final case; the tool was corrected to pass identical plane options without
rewriting those failed payloads or statuses. Pilot v2 passes full equality and
its source-byte hashes match the later measured commit. Pilot medians are
separate from final results.

The twelve earlier ad-hoc exploratory invocations are preserved in
`exploratory-v*.records.txt` with full payloads, prepared points and provenance
wrappers, plus their archived launchers. They excluded fresh-read verification
and did not capture complete candidate source-byte hashes; their timings are
not attributed to the final commit. Their solver numerical fields and prepared
point bits match final cases. Exploratory v1 predates the worker option and
therefore omits that field in plane option metadata. No exploratory timing
enters the final medians. These text wrappers are not standard benchmark schemas.

`publication-launcher.py.txt` records normalization and validation. Local paths
become `${WORKSPACE_ROOT}` and `${DATASET_ROOT}`; all original numerical payloads,
input/config/output digests and report-byte digests remain retained. Published
archive hashes and canonical Git hashes are separate from original measured
working-file hashes. Six published JSON reports validate against their schemas;
aggregates are recomputed from their definitions. All 114 final, pilot and
exploratory payloads are audited, including the explicitly retained metadata
differences.

Tests cover input-order rounding, float64 position bits, stable IDs and caps,
empty/nonfinite/extra-field inputs, large integer keys and scalar exceptions,
batch boundaries and duplicate/planar normal eigensystems, exact Jacobian row
bits and SVD, complete multi-start payloads and direction order, invalid worker
counts, and joining active workers after a failure. Existing native ICP,
registration comparison and adapter tests also pass.
