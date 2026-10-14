# Exact GPU query candidate experiments

The public-data diagnostic captures the sixth registration of the initial
GLIM Livox odometry pass. It compares exact sorted insertion with max-heap
candidate management, and tree leaf sizes 16, 8 and 4. Runtime adapters keep
their existing implementation. This experiment supplies a repeatable way to
reject small or workload-specific gains before changing calibration behavior.

| Isolated query | Existing median | Candidate median | Ratio |
| --- | ---: | ---: | ---: |
| 17 neighbors, leaf size 16, sorted → heap | 15.471 ms | 9.673 ms | 1.599x |
| 11 neighbors, leaf size 16, sorted → heap | 7.652 ms | 5.574 ms | 1.373x |
| 2 neighbors, sorted, leaf size 16 → 8 | 1.253 ms | 0.806 ms | 1.555x |
| 2 neighbors, sorted, leaf size 16 → 4 | 1.253 ms | 0.785 ms | 1.596x |

All 18 candidate configurations return bit-identical distances and source-index
order. The target has 46,520 points; the two-neighbor query has 19,578 moving
points. The 11- and 17-neighbor queries use every target point. Heap comparison
orders candidates by distance and then original index, and sorts the retained
heap before returning neighbors. Equal and roundoff-close subtree boxes retain
the existing conservative pruning policy. FP64 arithmetic and disabled FMA are
preserved.

These are warmed GPU-event timings: five batches of ten launches for each
configuration, in the order stored in the measurement. Imports, CPU projection,
tree construction, capture, uploads, compilation, normal PCA, robust systems,
pose updates, fitting and evaluation are outside the timed scope. Different
leaf sizes change preparation work, which this experiment does not compare.
The fixed candidate order, single captured registration and unreserved host
load make these exploratory observations. **They do not establish a complete
calibration speedup.** The two-neighbor experiment also uses the standalone
query shader, not the fused odometry system shader.

The [source-bound archive](../assets/cuda_spatial_candidates/README.md) retains
every timing sample, both actual shaders, generator, schema, imported-source
manifest and exact source archive. Captured point arrays are kept in the local
output and their hash is recorded; public bag data is not redistributed.
The CPU-only audit checks all file hashes, imported bytes, baseline kernel
identity, the heap shader definition, complete candidate coverage and medians.
Actual-device tests also compare exhaustive neighbors for repeated points,
ties, changed coordinate scales and neighbor counts 1, 2, 11 and 17.

```bash
python -m tools.benchmark_cuda_spatial_candidates \
  --output outputs/cuda-spatial-candidates
python -m tools.audit_cuda_spatial_candidates docs/assets/cuda_spatial_candidates
```

Use a fresh output directory and the original public
`data/public/glim_versatile/livox.bag`. Capture uses six seconds, 0.03 m voxels,
three local-map scans, a 1 nm coordinate grid and returned-motion grids of
1 micrometre translation and 0.1 microradian rotation-vector components.
The capture deliberately stops before completing calibration; no calibration
verdict or independent extrinsic truth is produced. For complete comparisons,
use the [full calibration benchmark](cuda_normal_preparation.md).
