# Native TUM initial correspondences

Preparing the initial graph for **27,544 correspondences** on the public TUM
`freiburg1_xyz` sequence takes **18.419 → 6.614 seconds (2.785x)**. A fresh
worker including the complete XYZ optimizer takes **35.863 → 23.565 seconds
(1.522x)**. These are medians of three alternating pairs, comparing exact Git
source capsules after the [field-column change](rgbd_field_columns.md).

This change accelerates CPU preparation. It preserves all eleven decoded
images, every sampled point, map geometry, initial correspondence, factor,
optimizer evaluation and final result. Both variants retain the same
`max_iterations` outcome after 20 iterations; this is an execution comparison
on already spent development data, with no accuracy or SOTA claim.

## Implementation

The native initial matcher previously constructed the same camera pose for
each point and repeatedly calculated a voxel plane's normal during neighboring
queries. Native frame points now share one pose and use array transforms in
chunks of at most 4,096 points. The transform follows the original quaternion
arithmetic order, preserving float bytes rather than substituting a different
rotation formula.

A local query context captures each plane's centroid and normal once and uses
the existing bounded voxel matcher. World-frame queries retain the selected
geometry; two-sided pair queries also retain the voxel identity and exact
distance. Map construction, candidate order, gates, sample limits and factor
identities stay the same. No persistent cache, dependency or public configuration
field is added.

The fast path accepts ordinary native frames, transforms, points and planes.
Small point sets, custom methods, subclasses and unsupported values retain
scalar behavior. Tests cover signed zero, chunk boundaries, sparse maps,
custom callbacks and complete world-frame and pair graph parity.

## Dense graph and every attempt

The configuration selects three map frames and eight query frames starting at
paired index 180. Both variants use maximums of 50,000 map points, 20,000 query
points and 4,000 correspondences per query frame. The actual map contains
78,855 records and 645 voxels; the optimizer receives 27,544 data factors and
ten prior/regularization factors. All input bytes are checked against the
existing public-data inventory, with one BLAS/OpenMP thread.

| Fresh worker | Preparation seconds | Optimizer seconds | Whole worker seconds |
| --- | ---: | ---: | ---: |
| 01-current | 6.312 | 12.374 | 22.803 |
| 02-prior | 18.419 | 13.139 | 35.863 |
| 03-prior | 24.018 | 13.345 | 41.677 |
| 04-current | 7.391 | 14.692 | 27.173 |
| 05-current | 6.614 | 12.463 | 23.565 |
| 06-prior | 16.159 | 11.346 | 31.293 |

Preparation includes depth decoding, sampling, map and correspondence creation,
parameter and factor construction, and frontend trace hashing. Whole-worker
latency also includes startup, imports, input verification, the optimizer and
evidence saving. Optimizer medians are 13.139 → 12.463 seconds; that unchanged
stage is not the target of this change. The host was unreserved, so timing
variation does not support a hardware-independent guarantee.

Each worker retains 24 frontend events covering all pixel and sample bytes,
ordered map geometry, complete measurement records, factor metadata and blocks.
It also retains 21 complete residual/Jacobian evaluations, 81 residual
evaluations and the complete result, including all 60 known-bad probes.
These values and histories agree exactly across all six workers.

## Original multi-window suite

The original smaller example was also run in four fresh alternating workers,
using the same three window processes in both variants. Its median whole time
is **33.968 → 32.363 seconds (1.050x)**. The dense-graph ratios above must not be
applied to this smaller complete suite.

| Fresh worker | Complete suite seconds |
| --- | ---: |
| 01-current | 33.176 |
| 02-prior | 33.619 |
| 03-prior | 34.317 |
| 04-current | 31.550 |

Every run retains the same 24 solves, 168 Jacobian evaluations, 1,624 residual
evaluations and 492 fixed-population probes. The complete calibration results,
depth samples, two-sided pair solves, reassociation, holdouts, curvature and
multistart histories agree. The [full-suite record](../assets/tum_initial_correspondences/full-suite-screen.tar.gz)
includes parent and child records, source manifests, configuration, protocol,
launcher and trace worker. Its runtimes match the dense comparison's source
capsules. This supplementary experiment is an exploratory screen.

## Evidence and verification

- Prior runtime: `2ae6cb775e002cd21d2af6ccbef133600b27a0fd`.
- Measured runtime: `b654ee4da6de5c9222be8fcdb23c43eb20bb4f2d`.
- [All dense attempts](../assets/tum_initial_correspondences/attempts.json),
  [summary](../assets/tum_initial_correspondences/summary.json),
  [configuration](../assets/tum_initial_correspondences/config.yaml),
  [measurement provenance](../assets/tum_initial_correspondences/provenance.json),
  [full-suite summary](../assets/tum_initial_correspondences/suite-summary.json).
- [Complete file inventory](../assets/tum_initial_correspondences/index.json),
  [record schemas](../assets/tum_initial_correspondences/record-schemas.json),
  [validation](../assets/tum_initial_correspondences/validation.json).

```powershell
python tools/audit_tum_initial_correspondences.py docs/assets/tum_initial_correspondences
# From the repository root, with the already downloaded public TUM sequence:
Copy-Item docs/assets/tum_initial_correspondences/launcher.py.txt .tmp-calibrex-tum-initial-measurement-v1.py
python .tmp-calibrex-tum-initial-measurement-v1.py
```

The launcher creates a fresh `outputs/tum-initial-pairs-v1` directory and refuses
to reuse it. Keep existing measurements and select a new output path in the
copied launcher for another experiment. The saved launcher hash identifies the
exact measurement code. The independent audit checks both workloads, including
source and input fingerprints, complete populations, results and timing medians.
