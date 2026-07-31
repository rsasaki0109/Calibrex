# Native I2I performance and recovery plan

This note records the July 2026 performance change to the native Pandey-style
Camera-LiDAR mutual-information solver and the evidence required before making
an accuracy claim.

## Implemented change

The Gaussian smoothing of the finite 2D MI histogram previously iterated over
every output bin in Python. It now creates a NumPy sliding-window view and
contracts every window with the same kernel in one `einsum`. A scalar-reference
unit test requires agreement at `1e-13` relative and absolute tolerance.

On the development host, 100 repetitions on a `32 x 32` histogram measured:

| Implementation | Mean time per convolution |
| --- | ---: |
| Scalar Python windows | 10.71 ms |
| Vectorized NumPy contraction | 0.78 ms |

This local microbenchmark is approximately 13.8 times faster for the isolated
kernel. It is not a claim that complete calibration is 13.8 times faster;
projection, numerical gradients, probes, and curvature evaluation remain in
the end-to-end runtime.

## Coarse-to-fine initializer

The optional initializer performs deterministic signed coordinate searches at
fixed translation and rotation resolutions. Search and validation frames are
separate subsets of the solver's training partition. A proposed initialization
is accepted only when it does not reduce either aggregate training MI or the
internal validation MI. The local optimizer is also run from the original
initial estimate. The coarse-started endpoint replaces that baseline endpoint
only when aggregate and every per-frame training MI are non-decreasing. The
final benchmark holdout remains untouched.

The default solver behavior remains the versioned
`pandey_mutual_information_bb_ascent/v0.1` path. Coarse search must be enabled
explicitly and records every level, evaluation count, proposed transform,
validation score, and acceptance decision in provenance.

## KITTI recovery protocol

`calibrex kitti benchmark-i2i` consumes the digest-locked 20-frame KITTI raw
input artifact and compares:

- the scalar Gaussian-convolution reference with native I2I
  numerical-gradient/Barzilai-Borwein optimization;
- vectorized Gaussian convolution with deterministic, safety-gated
  coarse-to-fine initialization and the same local optimizer.

Both methods receive the same six `+0.10 m` or `+10 deg` single-axis
perturbations, seeded fit/holdout split, iteration budget, and point sample.
The schema-valid benchmark retains translation error, rotation error, held-out
normalized MI, recovery rate, failures, runtime, input digest, and producer
provenance.

## Official KITTI raw result

The 31 July 2026 run used the KITTI-hosted
`2011_09_26_drive_0005_sync.zip`, selected the preregistered frame indices
`0:154:8`, and locked 44 files under aggregate input SHA-256
`80987aa55ee30c53134607bd0d5562e002fa963fd6d2028e61f0cda47acae9db`.
The run used Python 3.12.3, NumPy 2.5.1, and an Intel Core i5-1145G7. Each
method used 2,000 deterministically sampled points per frame and 20 local
iterations.

| Metric | Scalar reference | Vectorized safe coarse | Change |
| --- | ---: | ---: | ---: |
| Mean runtime per trial | 4.4674 s | 2.5990 s | **1.719x faster** |
| Mean translation error | 0.0740715 m | 0.0740715 m | 0 |
| Mean rotation error | 5.1457413 deg | 5.1457413 deg | 0 |
| Mean holdout normalized MI | 0.0123131 | 0.0123131 | 0 |
| Recovery rate at 0.05 m / 1 deg | 0/6 | 0/6 | 0 |

All six paired trials produced identical transform-error, holdout-MI, and
recovery metrics. The optimized path therefore passes the preregistered
non-degradation and runtime gates. The result establishes a performance
improvement, not a solver-accuracy improvement: neither method recovered any
of the six large perturbations under the strict threshold. This poor capture
range is the reason coarse search remains opt-in and why a geometry-based D2D
objective is the next accuracy item.

The generated benchmark definition and aggregate are schema-valid and record
the input digest, complete executed Python-source digest (including
uncommitted changes), config/output hashes, execution host, Python/NumPy
versions, and coarse acceptance decisions.

## Research basis and boundaries

- Borer et al., [From Chaos to Calibration](https://arxiv.org/abs/2311.01905),
  motivates a future geometry-based D2D objective and initialization-recovery
  evaluation.
- Koide et al.,
  [General, Single-shot, Target-less, and Automatic LiDAR-Camera Extrinsic
  Calibration Toolbox](https://arxiv.org/abs/2302.05094), motivates
  multi-frame direct registration and an external MIT-licensed baseline.
- Liu et al.,
  [adaptive-voxelization implementation](https://github.com/hku-mars/mlcc),
  motivates correspondence acceleration. Its GPL-2.0 code must not enter
  `src/calibrex`; only an independent implementation or subprocess adapter is
  acceptable.

The next accuracy change should add a solver-neutral depth-map input contract
and D2D MI objective. Learned depth estimation remains an external adapter with
model/version/weight-digest/license provenance.
