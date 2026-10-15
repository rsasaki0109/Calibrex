# Deferred overlap of target preparation with CUDA source work

**Not adopted.** The two complete-worker pairs change direction and do not
establish a stable end-to-end speed gain. This CPU-thread prototype was reverted
to revision `88e4a3a1db0e67dc538e3faf73c9a12b570ab3de`. Candidate
`7fe26771f6fb2a8017be54b9053b112fcaf2efc0` and its 51 passing device/unit
tests remain archived for a controlled repeat.

The later [asynchronous normal preparation](cuda_normal_overlap.md) submits GPU
normal work before CPU source preprocessing without an additional CPU thread.
Its separate exact-output comparison supports a smaller whole-worker improvement.

Sequential CUDA odometry previously built its local map, captured the target
tree and uploaded the target before starting source deskew and downsampling.
Those stages use independent inputs. The prototype prepares one target in
a single worker while the caller performs its source work.

Its worker receives a snapshot of the recent-scan container and reference pose.
The caller waits for that target before the first registration iteration;
subsequent deskew rounds reuse the same target. No later scan is prefetched,
and at most one preparation is outstanding per odometry instance. Source
rotation callbacks continue to run on the caller's thread.

The target factory captures the caller's CUDA device and owns one private
stream. It synchronizes that stream before publishing prepared arrays to the
caller's registration stream, including on errors. Reusing the private stream
lets CuPy reuse its target buffers. A result snapshot shuts down the idle
executor; continuing the same odometry instance creates a fresh executor.
No process-global pool, dependency or public option is added. The native CPU
path creates no preparation executor and does not import CuPy.

Map/point selection, geometry, trees, normal repairs, GPU kernels, iteration
and stopping policies are unchanged. The optimization overlaps their existing
work; it does not reduce samples or validation. It can require more simultaneous
CPU working memory than executing both stages in sequence. The number of
active preparations is bounded, rather than process RSS.

## Complete public-data comparison

| Scope | Sequential preparation | Overlapped preparation | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 97.706 s | 86.766 s | 1.126x |
| Unchanged four CPU fits, including covariance | 16.384 s | 18.928 s | 0.866x |
| Unchanged CUDA normal preparation | 7.259 s | 5.895 s | 1.231x |

| Pair | Sequential complete worker | Overlapped complete worker | Observed ratio |
| --- | ---: | ---: | ---: |
| 1 (sequential first) | 72.916 s | 76.616 s | 0.952x |
| 2 (overlap first) | 122.497 s | 96.917 s | 1.264x |

The median ratio is descriptive. One pair is slower and the other is faster,
while unchanged CPU fit and GPU normal timings also vary substantially.
These observations do not establish a stable full-calibration improvement;
adoption is deferred pending a repeat under less host contention.

Two alternating fresh-process pairs replay the original public GLIM Livox
bag's first six seconds with four map rounds and one BLAS thread. Every worker
starts with a separate empty CuPy compilation cache. Complete-worker timing
includes startup/imports, decoding, all odometry passes, preparation and worker
shutdown, association, fits/covariance, noise updates, evidence, hashes, saves
and source verification. Background host load is unreserved. Unchanged fit and
normal timings are retained to expose variation in other stages.

Both variants retain 0.03 m odometry voxels, three local-map scans, an opt-in
1 nm coordinate grid and returned-motion grids of 1 micrometre translation and
0.1 microradian rotation-vector components. These grids remain disabled by
default, and native CPU odometry remains the default backend. CUDA odometry
uses one preparation worker plus its caller; native query-worker settings are
unchanged.

The [bound archive](../assets/overlapped_target_preparation/README.md) retains
all four complete results, every full fit, measurements/logs, both source
capsules/manifests and the frozen launcher/schema/publisher. Every physical and
evidence field, initial estimate, execution identity and GPU work counter must
match exactly after excluding only provenance timestamps, commands and source
revisions. Every fit record must match exactly except for wall time. The
CPU-only archive audit independently checks this requirement.

The clip supplies no independent extrinsic truth and its verdict remains
INCONCLUSIVE. The observed complete-worker ratio is specific to this workload;
it is not multiplied by results from other cohorts.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit 88e4a3a1db0e67dc538e3faf73c9a12b570ab3de \
  --require-exact-results --max-seconds 6 --max-rounds 4 --repeats 2 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-overlapped-target-preparation-pairs
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/overlapped_target_preparation
```

Use committed runtime sources and a new output directory. Keep HEAD, the
benchmark launcher and comparison schema fixed until all workers finish.
Run from the archived candidate revision above to reproduce this experiment;
the current runtime does not enable the candidate.

An existing local Windows cache-home test fails with both current and frozen
baseline sources because it expects changing `HOME` to redirect `Path.home()`.
Its source-bound baseline receipt, log and reproduction launcher are retained
in the archive. The assertion and production cache-path policy are unchanged.
