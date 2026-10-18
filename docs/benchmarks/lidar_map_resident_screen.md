# Resident odometry in LiDAR map calibration: rejected

Connecting the resident geometry pipeline to the reference odometry of
LiDAR–LiDAR map calibration did not improve complete execution time over the
existing CUDA backend. The candidate is **not adopted**. Production source
remains the runtime already covered by PR #270.

| Triad | CPU (s) | Existing CUDA (s) | Resident candidate (s) | CUDA / candidate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 47.083 | 18.070 | 18.761 | 0.963x |
| 2 | 47.667 | 16.747 | 18.836 | 0.889x |
| 3 | 54.222 | 16.308 | 17.457 | 0.934x |
| Median time | **47.667** | **16.747** | **18.761** | **0.893x** |

All three candidate workers were slower than their CUDA counterparts. The
ratio of median times means a **12.0% increase** in complete process time.
The candidate is 2.541x faster than CPU in this cohort, but the existing CUDA
implementation is faster still. That CPU ratio does not establish an
incremental benefit from the candidate and is not an adoption claim.

## Workload and scope

Three fresh cyclic triads replay six seconds of the spent public GLIM Livox
bag. The same point-cloud topic supplies reference and target scans. The
target receives fixed Gaussian measurement noise (0.01 m, seed 20261018),
avoiding a zero-residual copied-sensor fit. This is an execution control with
no independent dual-LiDAR accuracy evidence.

Reference odometry uses 0.03 m voxels, three local-map scans, a 1e-9 m geometry
grid, a 1e-6 m translation grid and a 1e-7 rad rotation grid in every arm.
The reference pass is performed once. Map fitting stays on CPU in all arms,
including the training fit, four block jackknife fits, full covariance,
held-out scoring and all six axis controls. Eleven target samples span four
training blocks and two held-out blocks; all 60 reference scans are retained.
Sampling and filtering follow the same declared options in every worker.

Whole-process time includes startup, imports, fresh compilation, both bag
streams, target-noise generation, reference odometry, map construction,
calibration/evidence, full saves, input/source verification and hashes of
prepared sample arrays. Every process has an independent initially empty
CuPy compiler cache and one BLAS thread. The host was unreserved. There is no
retained-cache, longer-recording or portable-device speed claim.

The predeclared speed gates require a median CPU/candidate ratio of at least
2 and a median CUDA/candidate ratio above 1. The second gate fails. No timing
worker is discarded, and no favorable replacement cohort is substituted.

## Quality and source retention

All 30 CPU/GPU full-fit comparisons retain the original map numerical
tolerances, including covariance: abs 1e-9 / rel 1e-6 generally and abs 1e-13 /
rel 1e-5 for covariance. Status, iteration and correspondence counts, policy,
degrees of freedom and control decisions match exactly. Existing/candidate
CUDA artifacts, all 15 paired full fits and every prepared pose/map/target
array digest match all float bits after only declared execution metadata and
run provenance are normalized. The copied-sensor policy status is `pass`;
that does not provide independent accuracy evidence for a two-LiDAR rig.

The candidate passed 100 relevant host/device tests, strict mypy for Linux
and Windows over 396 modules, and Ruff. Its two new test files exercise
option validation, CLI provenance and invalid-grid errors, synthetic room
calibration, complete sample/evidence parity, both deskew modes and stream
restoration on decoding failure.

The [bound archive](../assets/lidar_map_resident_screen/README.md) retains all
nine workers, every complete fit, the original launcher/protocol and the
candidate qualification. Its complete prior source is the already published
[source capsule](../assets/resident_scan_validation/current.sources.tar.gz);
a small candidate delta replaces exactly three modules. A complete manifest
checks all 396 reconstructed modules, preserving source identity while
avoiding duplicate unchanged source archives. The experimental source is
retained for inspection and is absent from the production changes.

The [CPU-only auditor](../assets/lidar_map_resident_screen/auditor.py.txt)
checks input/source/import/kernel/cache/thread bindings, full numerical and
evidence parity, sample digests and all original timings. It independently
recomputes the failed adoption gate. Tests reject altered payloads, a
rebound adoption claim, forged kernel identity and a rebound source delta.

```sh
python -m tools.audit_lidar_map_resident_screen \
  docs/assets/lidar_map_resident_screen --check
```
