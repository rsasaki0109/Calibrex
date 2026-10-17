# TUM factor and graph preparation

On the **27,544-correspondence public TUM XYZ graph**, reusing native frame
construction and validation reduces preparation from **5.667 to 3.991 seconds
(1.420x)**. Fresh worker latency improves from **14.529 to 12.208 seconds
(1.190x, 16.0% less elapsed time)** compared with the preceding CUDA runtime.

A direct control using the preceding runtime's NumPy backend takes **21.241
seconds**, giving **1.740x** whole-worker speedup for the current CUDA path.
This is a new comparison of complete workers; earlier benchmark ratios are
not multiplied. The optimizer alone is **10.851 to 4.622 seconds (2.348x)**
against that NumPy control, and **4.824 to 4.622 seconds (1.044x)** against the
preceding CUDA runtime.

All complete results, frontend records and numeric histories agree exactly.
The outcome remains 20 iterations with `max_iterations`. The input is the
already spent TUM `freiburg1_xyz` development sequence. This post-hoc execution
comparison makes no accuracy or convergence claim.

The **original smaller multi-window suite shows no speedup**: its medians are
**30.126 to 30.520 seconds (0.987x)**. Its complete results and evaluation
histories also agree. The dense-graph gains should not be applied to that suite.

## Changes

The native TUM builder shares each frame's immutable initial pose and sensor
identity across its measurements. It retains the original normalized quaternion
bits and creates every candidate and retained measurement in the original order.
Custom frame properties, constructors, identity methods, geometry functions and
non-native numeric inputs retain their existing construction path.

Factory transform signatures are reused within one synchronous builder, with at
most 128 entries. The cache compares immutable tuple identities, preserves float
bytes including signed zero, isolates nested builders and releases references
on return or error. Graph preparation separately validates shared initial frame
pairs once, with a fresh cache bounded by the existing 128-group limit. Replaced
leaves are validated again; non-native and out-of-range frames retain fallback.

The rigid lattice gauge batches its independent NumPy cross products, then adds
the vectors in the original left-to-right order. A replaced cross-product
function retains its original per-control calls. No correspondences, probes,
iterations or scalar residual dimensions are removed.

These changes accelerate CPU work around the existing optional CUDA backend.
NumPy remains the default. Enabling CUDA uses the
[existing configuration](cuda_rgbd_joint.md#enable-the-optional-backend), including
its 4,096-point initialization threshold and 64 MiB owned-data budget. The CUDA
kernel source and numerical policy remain the same. Public schemas are unchanged.

## Dense graph: every fresh worker

The prior runtime is `cbc69784089af8e29035ffe22ad7d119ad92a005`; the current
runtime is `b29d1e75e764af89dc7c9bb41b4919634257b7a6`. `prior` and `current`
request CUDA; `numpy` uses the prior runtime's default NumPy backend. The two
configurations differ only by the explicit CUDA execution option. Each variant
occurs three times, with the same mean position in the nine-worker sequence.

| Worker | Backend | Preparation seconds | Optimizer seconds | Whole worker seconds |
| --- | --- | ---: | ---: | ---: |
| 01-current | CUDA | 3.979 | 5.418 | 13.208 |
| 02-numpy | NumPy | 6.345 | 10.851 | 21.241 |
| 03-prior | CUDA | 5.667 | 4.824 | 14.529 |
| 04-numpy | NumPy | 5.606 | 12.579 | 22.053 |
| 05-prior | CUDA | 5.172 | 4.313 | 13.140 |
| 06-current | CUDA | 3.991 | 4.622 | 12.208 |
| 07-prior | CUDA | 6.462 | 5.390 | 15.999 |
| 08-current | CUDA | 4.336 | 3.905 | 11.885 |
| 09-numpy | NumPy | 6.608 | 9.429 | 19.762 |

All workers retain three map frames, eight query frames starting at paired index
180, 78,855 map records, 645 voxels, 27,544 data factors and ten priors or
regularizers. The maxima remain 50,000 map points, 20,000 query points and 4,000
correspondences per query frame. BLAS/OpenMP use one thread.

Each worker has a fresh CuPy compiler-cache path, and persistent CUDA driver
caching is disabled. The optimizer timer includes optional imports, GPU startup,
compilation, transfers and trace hashing. Whole-worker time also includes imports,
public-input verification, preparation and evidence saving. The host was
unreserved, so these medians are observations rather than hardware-independent
guarantees. The NumPy workers leave the CUDA adapter unloaded.

All 13 actual input files match the existing public inventory. Each worker saves
all 24 frontend events, covering every image, ordered sample, map geometry and
complete initial measurement and factor metadata. It also saves 21 complete
residual/Jacobian evaluations and 81 residual evaluations, including shapes,
state hashes, ordered factors and all array bytes' hashes. The complete payloads
match the previous published result, including all 60 known-bad probes.

Each CUDA worker retains 21 linearizations, 606 adapter evaluations and 1,090
kernel launches, with no budget fallback. Its owned-data bound is 10,364,160 bytes
and largest column chunk has 91,296 elements. Adapter evaluations include
prior-only parameter probes; the complete graph trace remains 21/81.

## Original smaller suite

Both variants here use default NumPy execution and three native window
processes. Every run retains 24 solves, 168 Jacobian evaluations, 1,624 residual
evaluations and 492 fixed-population probes, including pair matching,
reassociation, holdout, curvature and multistart histories.

| Worker | Runtime | Whole suite seconds |
| --- | --- | ---: |
| 01-current | current | 30.232 |
| 02-prior | prior | 30.829 |
| 03-prior | prior | 29.423 |
| 04-current | current | 30.807 |

The [complete screen](../assets/tum_factor_preparation/full-suite-screen.tar.gz)
and [summary](../assets/tum_factor_preparation/suite-summary.json) retain this
negative timing result alongside the exact result and history comparisons.

## Evidence

The [raw-file inventory](../assets/tum_factor_preparation/index.json) binds the
[configuration](../assets/tum_factor_preparation/config-cuda.yaml),
[all attempts](../assets/tum_factor_preparation/attempts.json),
[summary](../assets/tum_factor_preparation/summary.json), source capsules,
input pins, execution records, raw traces and validation records. The
[development archive](../assets/tum_factor_preparation/development-screens.tar.gz)
also retains both exploratory profiles and the intermediate six-worker screen:
its 1.158x preparation and 1.044x whole-worker result motivated further work.
Profiler timings are not latency evidence.

The [source-bound validation](../assets/tum_factor_preparation/validation.json)
records 570 passing Windows tests, including 34 real-device cases, and 536
passing Linux/Python 3.10 tests with the optional CUDA module skipped. Ten
archive and coherent-tampering checks pass on each platform. Strict typing
passes for all 392 source modules on Linux and Windows targets, and all 129
public schema exports plus their ledger match exactly.

```powershell
python tools/audit_tum_factor_preparation.py docs/assets/tum_factor_preparation
# From the measured current runtime with the public TUM sequence downloaded:
Copy-Item docs/assets/tum_factor_preparation/launcher.py.txt .tmp-calibrex-tum-factor-preparation-measurement-v2.py
python .tmp-calibrex-tum-factor-preparation-measurement-v2.py
```

The launcher refuses to reuse `outputs/tum-factor-preparation-pairs-v2`. Choose
a fresh output path when repeating it; the saved launcher hash identifies the
exact code used for each comparison.
