# CUDA mutual point-to-point ICP controls

The optional adapter replaces forward and reciprocal nearest-neighbor searches
with FP64 CUDA proposals over owned immutable spatial trees. It reuses the raw
source tree by moving queries into its frame. Original NumPy world-frame
distances and source-index tie ordering determine the result. Ambiguous rows
are repaired natively; small, non-finite or extreme-coordinate searches fall
back to the native evaluator. Pose updates, SVDs, reductions, spatial holdout
and all six multi-start diagnostics remain native. The two captured trees
are shared across starts; CPU execution imports no CuPy.

Primary observations use the final measured source `85a3f0d037dbbc29d3ecbd8a50ae721ada34f8bc`.
They use public KITTI Raw scan 0000000115 (127,158 points), sequence
2011_09_30_drive_0018_sync. Its raw SHA-256 is
`213948bd8c54c925aca7eb9f3b3c36e95e586b52ee8d9d64056c4fc4aeb12b2d`.
The target is a deterministically transformed copy with 2 mm Gaussian noise;
seed, transform, native options and exact inputs are bound in every record.
This establishes execution equivalence on public data, not accuracy between
independently recorded sensors. No personal rosbag is needed.

| Scope | Native median | CUDA median | Ratio of medians |
|---|---:|---:|---:|
| Whole worker | 46.777627 s | 24.407067 s | 1.917x |
| Complete solver | 43.813992 s | 21.320884 s | 2.055x |

| Pair | Order | Native whole wall | CUDA whole wall |
|---|---|---:|---:|
| 1 | native, cuda | 46.777627 s | 24.407067 s |
| 2 | cuda, native | 51.313975 s | 27.782913 s |
| 3 | native, cuda | 44.984903 s | 22.316691 s |

Whole worker includes interpreter startup, file loading, point preparation,
full solve, diagnostics, source/input guards, hashes and result saves. Solver
time includes optional CUDA import, initial upload, tree construction, cold
kernel compilation and all calibration diagnostics; it excludes input
preparation and result serialization. Every worker has its own initially
empty CUDA disk-cache directory. OS caches persist. All three alternating
pairs, cold trials and slower observations are included. No cohort ratios
are pooled or multiplied; these are local observations rather than portable
confidence estimates.

All final native calibration result bytes match between CPU and CUDA, including pose, complete
history, correspondence count, source split IDs, holdout, rematching curvature,
six multi-start trials and symmetry diagnostics. Both methods retain default
bounded automatic diagnostic parallelism. BLAS/OpenMP internal threads are
set to one in both fresh worker environments; every final record verifies
the loaded BLAS runtime thread count. This avoids nested oversubscription.

Two earlier runs without that thread budget ended in access violations: one
CUDA worker and one native CPU worker before any CUDA execution. Their logs
and measured sources are retained; no completed calibration model exists for
those failed workers. A successful unbounded GPU diagnostic and a sequential
comparison are also retained, separately from the primary result. The root
cause of the unbounded-thread failures remains unresolved. The sequential
comparison does not establish a gain against the existing parallel CPU path.

The bounded precursor and pre-guard comparisons are retained in full. The
final source adds a tested native fallback for extreme world coordinates;
the final primary cohort was rerun afterwards. The earlier raw-source search
probe remains explicitly kernel-only; its speed is not a solver speed claim.

The archive retains 22 complete worker results and 2 failed workers.
Identical complete JSON result bytes are stored once per digest, with an
explicit reference from every worker record. No fields are removed for parity.
Adapter provenance is retained separately in each CUDA measurement.
Source snapshots retain exact measured bytes and normalized Git source
identity. Only private paths in failed logs were replaced by portable markers;
original log digests are retained. Captured sources, measurements, logs and result bytes are digest-bound in
`publication-index.json`.

Validation: 198 distinct targeted tests across the final change and fallback
checks, mypy on 357 source modules, and 126 byte-identical exported schemas.
Real-device cases cover duplicates, near ties, wide coordinate scales, large
translations, immutable captures, small/empty fallback and complete native
calibration outcomes with sequential and parallel starts.

Reproduce after installing the optional CUDA extra and benchmark inspection helper:

```shell
python -m pip install -e ".[cuda]" threadpoolctl
python tools/benchmark_cuda_point_to_point_icp.py --input path/to/0000000115.bin --output outputs/icp-control --repeats 3
python tools/audit_cuda_point_icp_archive.py
```

The benchmark sets the worker BLAS thread environment before Python starts.
The adapter does not modify process-global BLAS settings. For direct use,
set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS before starting
Python when enabling parallel diagnostic starts.

The portable publisher splits its privacy-pattern literal without changing
behavior. Original and portable publisher digests are retained.
