# Exact spatial neighbor search for native ICP

Native robust point-to-point ICP and Chen–Medioni point-to-plane ICP now use
scoped spatial indexes instead of repeatedly constructing complete pairwise
distance matrices. Target neighborhoods, forward/reverse mutual matches,
spatial holdout, trimming, all optimization iterations, rematching diagnostics,
multi-start trials and known-bad-pose probes retain their existing definitions.
The optimization is automatic in both solver APIs and their
`native_registration_comparison` caller. SciPy is already a core dependency;
this change adds no dependency or GPU requirement.

## Numerical and memory contract

An invocation owns an immutable copy of each indexed reference cloud. The
fixed target index is reused during optimization and diagnostics; transformed
source indexes are rebuilt for mutual checks. Reverse matches need only be
queried for target points proposed by the forward search. No global cache or
retained external array is introduced. Nested multi-start solves own their
own indexes.

The index proposes neighbors using
[SciPy's `cKDTree.query`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html)
with `eps=0`. Candidate squared distances are recomputed using the original
float64 NumPy subtraction, square and three-coordinate sum. Ordering is by
that squared distance and then the original reference index, preserving
`argmin` and stable `argsort` tie behavior.

The search requests one additional neighbor. If its distance can meet the
selection boundary within a 64-machine-epsilon guard, a guarded
[radius query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html)
collects possible boundary ties for exact recomputation and ordering. The guard
also covers tiny squared distances. Radius candidates are handled one row at
a time. Duplicate-heavy geometry can still require exhaustive work, but it
does not retain a complete cloud-pair candidate matrix. Nonfinite coordinates
or magnitudes above 1e150 use exhaustive NumPy row batches, preserving native
NaN/overflow behavior. Each batch contains at most 65,536 point pairs, or one
reference-cloud row when the cloud itself exceeds that limit.

Ordinary working memory scales with clouds and requested neighbor count,
including a private cloud copy, the tree and candidate/distance arrays. The
default normal neighborhood count is 12. Requesting every target as a normal
neighbor can itself require quadratic output. No measured peak-RSS reduction
is claimed here. Normal covariance formation and eigensolves are unchanged.
No float32 conversion, approximate-neighbor setting or new sample reduction
is used by this optimization.

## Public workload and timing scope

The measured implementation is commit
`d2f49e4aa8cbeafb64bc92d6dca347b18323d1a2`; the reference is
`f959c684c3f4e4ec24a884cd4cea8545d44baa4a`. The two archived
`baseline.*.txt` files are its exact Git blobs. The old plane solver's common
candidate evaluation and source splitting are explicitly bound to the old
point solver, so its diagnostics also use the original exhaustive path.

Inputs are the public KITTI raw drive `2011_09_30_drive_0018_sync` Velodyne
files `0000000000.bin` and `0000000115.bin`. These sampled scans are 115 frame
indices apart. They are not an adjacent-motion accuracy benchmark or a
two-sensor ground-truth calibration. The initial transform is identity; no
reference extrinsic or motion enters optimization. Scan motion distortion is
not compensated.

The existing native adapter's 0.5 m voxel centroids and deterministic caps of
500, 1,500 and 3,000 points are prepared once and supplied identically to both
methods. Caps and preprocessing are part of the frozen benchmark input, not
changes made by the acceleration. Options retain mutual correspondences,
0.8 trimming, a 1.5 m distance gate, 0.2 spatial holdout with 1 m voxels,
seed 0, and a 20-iteration limit. Point ICP retains its six multi-start solves
and all rematching diagnostics; plane ICP retains normals, common evaluation
and twelve known-bad-pose probes. Complete statuses and diagnostic payloads
remain visible even when a solver reaches its iteration limit.

Each algorithm/cap runs each implementation three times, alternating order.
There is no separate warmup; process, library and OS caches persist. The
Windows host is shared with other work. `wall_seconds` includes the full
solver API call and `as_dict`: index/normal preparation, optimization, holdout
and every diagnostic. Input capture/downsampling and benchmark serialization
are excluded and the one-time preparation cost is recorded separately. These
are solver timings, not full `calibrex calibrate` CLI latency.

All 36 final complete numerical payloads match the pinned implementations
exactly, including poses, iteration histories, holdout results, normal
diagnostics, degeneracy, rematching, multi-start trials and pose probes.
Three-repeat solver medians improve **3.53–32.87x**.
Seconds and ranges retain all invocations; no outlier is discarded.

| Solver | Cap per cloud | Fit / holdout | Previous median | Indexed median | Ratio | Previous / indexed ranges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Point-to-point | 500 | 403 / 97 | 1.513741 | 0.428366 | 3.53x | 0.956516–1.918112 / 0.299062–0.492570 |
| Point-to-point | 1,500 | 1197 / 303 | 15.394562 | 0.765273 | 20.12x | 14.060144–18.238222 / 0.741916–1.653190 |
| Point-to-point | 3,000 | 2392 / 608 | 123.730614 | 4.057658 | 30.49x | 108.148124–163.601396 / 2.483133–4.649123 |
| Point-to-plane | 500 | 403 / 97 | 0.955779 | 0.223673 | 4.27x | 0.899353–0.958425 / 0.216413–0.284186 |
| Point-to-plane | 1,500 | 1197 / 303 | 9.482266 | 0.557271 | 17.02x | 8.230578–10.911877 / 0.499274–1.070583 |
| Point-to-plane | 3,000 | 2392 / 608 | 18.512043 | 0.563242 | 32.87x | 16.097518–21.133982 / 0.501249–0.633115 |

The 500-point point-to-point case converges; the other five algorithm/cap
cases reach the unchanged 20-iteration limit. Those statuses and every
diagnostic are identical between implementations. Timed calls reaching
the limit count as successful execution/parity trials, not successful
calibration recovery.

The point-to-point 3,000-point case changes from 123.731 s to 4.058 s;
point-to-plane changes from 18.512 s to 0.563 s. Host variation is
substantial: the former baseline spans 108.148–163.601 s and its indexed
calls span 2.483–4.649 s. This does not establish a portable speedup.

Input capture and preparation of all three cap pairs cost **1.894322 s**
once, outside solver-call timings. A single complete application invocation
must also account for its own input preparation, other backends and result
serialization. Environment: Windows, Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1.

The retained four-trial pilot measured point-to-point 0.677572 / 0.143118 s
and point-to-plane 0.446720 / 0.113647 s at the 500-point cap. All four full
payloads match their corresponding final-run case. They are not included
in final medians.

Three repetitions on one scan pair describe timing variation, not independent
accuracy trials or statistical confidence. Bootstrap fields are schema summaries
only. The same convergence or iteration-limit status and full diagnostic output
must match the pinned implementation. Equality preserves existing behavior;
it does not establish correct cross-sensor extrinsics or scene observability.

## Reproduction and evidence

```bash
python tools/benchmark_icp_neighbors.py \
  /path/to/2011_09_30_drive_0018_sync \
  --output-dir outputs/public-kitti-icp-neighbors-v1 \
  --pair 0000000000:0000000115 --sizes 500 1500 3000 --repeats 3
```

The output directory must be new, preserving earlier attempts. Raw scans stay
external; each byte buffer is hashed and decoded from the same read. Source
files, prepared point IDs/positions and frozen settings have separate digests.
The standard benchmark schemas contain every timed trial and complete solver
payload. `output_sha256` binds its canonical numerical JSON, not a separately
serialized result file.

[point.definition.json](point.definition.json) and
[plane.definition.json](plane.definition.json) retain all final trials. Their
`*.benchmark.json` files are recomputed by the standard aggregator. The
single-repetition 500-point pilot is retained in `pilot.*` definitions and
aggregates. It ran before committing the measured files; its source-byte
hashes match the final run. The pilot's indexed method is bound to their later
source commit, with the original checkout HEAD explicitly recorded. Its
timings are separate from final medians.

`publication-launcher.py.txt` records normalization and validation. Local path
prefixes are replaced by `${WORKSPACE_ROOT}` and `${DATASET_ROOT}`; full
numerical payloads and original input/config/output digests are unchanged.
Original raw-report digests, measured Windows worktree hashes, canonical Git
blob hashes and published source-archive hashes are kept separately.

Unit coverage includes duplicate and grid ties, near-boundary distances,
stable trimming and mutual gates, source-array ownership, empty queries,
nonfinite/underflow/overflow fallbacks, and complete solver payload equality
with exhaustive NumPy calculations and all diagnostics. Existing native
registration comparison and adapter tests also pass.
