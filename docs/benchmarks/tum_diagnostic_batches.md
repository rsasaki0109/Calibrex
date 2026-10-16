# Batched TUM calibration diagnostics

The complete native TUM RGB-D joint pipeline on the existing public fr1/xyz
development capture improves from **339.164822 to
186.922117 seconds (1.814x)**. The entire
`run_calibration` call improves from **336.058079 to
183.564714 seconds (1.831x)**. This comparison
starts from the already accelerated RGB-D graph in
[the previous change](rgbd_evaluation_reuse.md).

## What changes

After accelerating the optimizer, repeated correspondence diagnostics dominated
the remaining time. The adapter recalculated lattice weights and per-point
transforms, rebuilt target geometry during pose-only probes, and recreated
identical residual factors for each diagnostic state.

The native adapter now captures the fixed samples and lattice weights once per
window. It transforms ordered point arrays with the existing scalar-compatible
Hamilton-product kernel, uses the existing voxel query context for ordered
batch queries, and reuses immutable factors when their inputs are unchanged.
Ray-depth factors are keyed by measurement row and target voxel within one
captured map. Two-sided XYZ factors also include the selected raw target-point
index and the exact bytes of the world normal, which changes with target pose
or deformation. Changing poses, fields and correspondence support remain part
of every evaluation.

At most two corrected XYZ point-cloud/map states are retained for a window.
Their keys use offset bytes, including signed zero. Captures with more than
65,536 sampled points skip state retention. Each factor cache holds at most
the smaller of 65,536 entries and twice the query population, with a minimum
budget of 64 entries. These bounds limit retained entries; all requested
queries still run, and larger populations are not truncated. Captured geometry
is read-only, and all scratch belongs to one diagnostic window.

Lattice reductions retain Python's original floating-point `sum`, including
the supported Python versions' summation behavior. Depth addition order,
transform normalization, nearest-voxel tie order, raw-point lineage, residual
normalization, unmatched penalties and scalar objective accumulation are
preserved. Fixed/rematched curvature and all five initial-value searches run
the same candidates and stopping rules. Dependencies, ROS independence,
public signatures, and config/result schemas remain unchanged. These are CPU
measurements; this change introduces no GPU backend.

## Full public workload

The unchanged [public TUM configuration](https://github.com/rsasaki0109/Calibrex/blob/a2a13db31f6f21c2b90aaa28779f9c1047b6b7d6/examples/public_datasets/tum_rgbd_freiburg1_xyz/native_joint_slac_config.yaml)
uses three windows starting at 60, 180 and 300, each with three map frames and
eight query frames. The primary solve retains 2,466 data factors, eight pose
priors, 56 dimensions, nine LM iterations and the original holdout and probes.
All 24 solves, scalar/full-XYZ ablations, local-rotation updates, two-sided
pairs, cross-window transfers, reassociation, curvature, multistart and report
saving are included.

Every worker records 168 full Jacobians, 1,624 residual passes, 405 two-sided
reassociation states, 102 ray-depth reassociation states and 991 objective
evaluations. All four full calibration results, all raw numeric-array hashes,
24 optimizer-result hashes and 1,498 diagnostic events match exactly. A
diagnostic event records the input-state and correspondence-state digests, or
the exact hexadecimal objective value and its candidate digest. Only run IDs,
wall-clock dates and independently verified source identities are excluded
from complete-result comparison.

The 802-file input inventory pins all 798 depth PNGs and four text files.
Dataset pixels are not republished. This capture is development data. Existing
reference failures, warnings, rank limitations and SOTA verdicts remain as
recorded; execution speed does not establish an accuracy improvement.

## Timings and scope

Two alternating pairs use fresh processes, frozen source capsules with actual
Git identities, and one BLAS thread on an unreserved host. Whole-worker time
includes imports, source checks, the complete pipeline, trace capture,
serialization, compression and saving. API time includes the complete
`run_calibration` call and saved reports. Downloading, freezing, hashing the
input inventory, and external audits are outside the timed workers. All four
attempts are retained. Ratios divide the two-run medians and depend on host
load and capture geometry; previous speedups are not multiplied into them.
The 24 optimizer calls themselves measure 78.493785
seconds before and 95.962572 seconds afterward
(0.818x). Their implementation and numeric histories are
unchanged; this change accelerates the surrounding diagnostics. Host load was
not controlled, and these four runs do not establish a confidence interval.

| Pair (order) | Previous whole s | Batched whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (current, prior) | 286.530200 | 144.751217 | 1.979x |
| 2 (prior, current) | 391.799445 | 229.093017 | 1.710x |

| Complete diagnostic stage | Previous s | Batched s | Speedup |
| --- | ---: | ---: | ---: |
| Two-sided XYZ reassociation and signed probes | 80.552006 | 27.491603 | 2.930x |
| Ray-depth reassociation and signed probes | 85.763641 | 29.241404 | 2.933x |
| Holdout rematching, curvature and multistart | 74.871827 | 5.306373 | 14.110x |

Ray-depth factor construction drops from 257,458
to 23,769 calls; two-sided XYZ factory calls
drop from 165,805 to
69,951. The complete
correspondence and objective evaluation populations remain identical.

## Reproduce and verify

```console
python tools/download_public_dataset.py tum_rgbd_freiburg1_xyz
python tools/benchmark_tum_diagnostic_batches.py --baseline-commit a2a13db31f6f21c2b90aaa28779f9c1047b6b7d6 --output outputs/tum-diagnostic-replay --repetitions 2
python tools/benchmark_tum_diagnostic_batches.py --audit-only docs/assets/tum_diagnostic_batches
```

The offline auditor requires no dataset download. It verifies the fixed config,
802 input fingerprints, all three windows, source capsules and imported-module
hashes, complete results and raw histories, work counts, diagnostic timings,
factory counts, medians and the published byte inventory. Source manifests
contain 383 previous and 384 current Python modules. Only the native TUM
adapter and the new private `solvers/_tum_geometry.py` module change.

Regression tests compare against the original scalar algorithms, including
exact geometry bytes, correspondences, normals, field/pose changes, signed
zero, cache eviction, sparse-query fallback, curvature and every multistart
search. Validation details and full logs are bound in the archive.

- [Audited summary](../assets/tum_diagnostic_batches/summary.json)
- [Protocol and source identities](../assets/tum_diagnostic_batches/protocol.json)
- [Every attempt](../assets/tum_diagnostic_batches/attempts.json)
- [Complete byte inventory](../assets/tum_diagnostic_batches/index.json)
- [Validation and provenance](../assets/tum_diagnostic_batches/validation.json)
- [Public input fingerprints](../assets/tum_diagnostic_batches/inputs.json)
- [Schema byte parity](../assets/tum_diagnostic_batches/schema-parity.json)
