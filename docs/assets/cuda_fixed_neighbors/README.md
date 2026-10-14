# Rejected fixed-neighbor CUDA trial

The prototype is rejected. Three alternating fresh-process pairs with separate
initially empty CuPy caches produce whole medians of
20.702 s (previous) and
23.979 s (prototype),
0.863x. API medians are
15.661 s and
19.231 s,
0.814x. Active runtime changes are reverted.

The query compiler specializes the requested neighbor count and expands sorted
candidate updates into constant slots. Point membership, FP64 distance arithmetic,
distance/point-ID ordering, tree traversal and numerical repair policies are
preserved. All three complete API evaluations (transforms, covariance, signed
controls, holdout, jackknife and decisions) match exactly. Each worker retains
six samples, 762,118 map rows and 9,147,116 device query rows. No workload budget
is reduced. GPU parity tests for all 1–17 neighbor counts remain frozen here.

An earlier fixed-order shared-cache pilot appeared much faster: the previous
worker ran first with a cold cache and the prototype reused the generated
compilation cache. That pilot cannot establish acceleration. Warm query kernel
improvements likewise do not establish complete calibration speed. Cold/full
measurements take precedence. A separate compiler diagnostic measured only
about 0.9 s for the two specialized query modules; this does not explain all
whole-worker variation, and CPU resources were not reserved.

Public KITTI raw sequence `2011_09_30_drive_0018_sync`, scans 115, 230 and 345,
forms a copied-sensor identity execution control with fixed 0.01 m noise and
generated rigid poses. It does not establish independent two-sensor extrinsic
accuracy or include reference-odometry generation. Full worker time includes
imports, input reading, preparation, compilation, fits, holdout, signed controls,
jackknife, hashes, saved artifacts and frozen-import verification. Raw scans
are not redistributed. Workers use one BLAS/OMP/MKL thread each.

No foreign or unknown CUDA compute process was detected in 17
ten-second observations. Periodic sampling does not establish continuous GPU
exclusivity, detect all graphics activity or reserve CPU resources. Only the
owned cohort would be terminated on interference. Process IDs and names are
not published.

Baseline commit: `efd6ce8bf6a1e5b4331a61d9cb436fd3bd2d7e29`. Measured rejected source:
`1bb392b23d142982105eeda7241989e9491576c4`. The index binds both full source trees,
all six workers, launchers, schemas, device tests, observer and publisher.

```bash
python -m tools.audit_cuda_fixed_neighbors docs/assets/cuda_fixed_neighbors
```

Reproduction requires the measured commit above, the three raw inputs matching
the protocol hashes and the frozen launcher. The active runtime retains the
previous generic query implementation.

The bound [cold profile](cold-profile/summary.json) records 63 NVRTC compilations, 10.808 s within 14.517 s of profiled complete API execution. It uses the generic query override; actual checkout EOLs and all imported source hashes are retained. This is initialization evidence, not a new speed comparison.
