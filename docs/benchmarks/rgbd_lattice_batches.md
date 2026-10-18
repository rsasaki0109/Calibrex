# Ordered RGB-D lattice batches and owned graph plans

The complete public TUM fr1/xyz pipeline improves from **134.924629
to 98.237561 seconds (1.373x)**. All 24 optimizer
calls together improve from **67.429389 to
23.706579 seconds (2.844x)**. This starts
from the already accelerated [batched diagnostic implementation](tum_diagnostic_batches.md).

## Implementation

After batching the diagnostics, repeated lattice correction and graph preparation
accounted for most of the remaining optimizer time in a representative full-XYZ
solve. The RGB-D graph now captures ordered point, depth and lattice-weight arrays.
It batches independent products, then converts each product row to ordinary Python
floats and retains Python's original `sum`. This preserves lattice order, zero
terms and the summation behavior of each supported Python version. Ray scaling,
depth bias addition, both XYZ pair endpoints and the existing two-Hamilton-product
transform kernel retain their arithmetic order. Numerical central differences,
step sizes and all residual evaluations are unchanged.

The native TUM adapter marks its private factor populations as owned for a
synchronous solve, including the reassociation solver's inner updates when both
data and regularization populations are owned. The optimizer reuses train and
holdout preparation only for these populations. Ordinary caller sequences are
validated and prepared afresh. Unsupported callbacks discard preparation caches
before executing user code. Value types, dimensions, finite ranges and original
geometry methods are checked on reuse. Initial poses are copied without another
quaternion normalization; geometry arrays are read-only.

Each solve retains at most two RGB-D plans and 65,536 factors in total. Larger
populations are still evaluated and skip retention. Each group retains at most
two corrected point-array states, keyed by parameter tuple identity while holding
the tuples themselves. Signed-zero states cannot alias. Scope tokens release the
plans on return and error, and isolate nested solves.

Plan construction drops from **1,792 to 62** per
complete pipeline. Both versions still accept all 1,792 requested array
evaluations. No frames, points, iterations, probes or candidate starts are removed.
These are CPU measurements. Public APIs, dependencies and config/result schemas
are unchanged; the core remains independent of ROS.

## Complete public workload

The unchanged [TUM configuration](https://github.com/rsasaki0109/Calibrex/blob/a5af90c1a869b376ab5be4e6c765c2144af25cba/examples/public_datasets/tum_rgbd_freiburg1_xyz/native_joint_slac_config.yaml)
uses windows starting at 60, 180 and 300. Each contains three map frames and eight
query frames. The primary solve keeps 2,466 data factors, eight pose priors, 56
dimensions and nine LM iterations. All scalar/full-XYZ ablations, local-rotation
updates, two-sided pairs, transfers, reassociation, holdouts, curvature, multistart,
known-bad probes and saved reports run without option changes.

Every run contains 24 optimizer calls, 168 full Jacobian passes, 1,624 residual
passes, 405 two-sided reassociation states, 102 ray-depth states and 991 shared
objective candidates. All raw numeric-array byte hashes, factor IDs/order, input
states, 24 optimizer-result hashes, 1,498 diagnostic events and complete schema-valid
results match exactly across all four runs. Only run IDs, wall-clock dates and
independently verified source identities are excluded from result comparison.
Correspondence factory counts also remain identical.

The 802-file inventory pins all 798 public depth PNGs and four text files. Pixels
are downloaded separately. This is existing development data; reference failures,
rank limitations and known-bad verdicts remain recorded. This execution comparison
makes no accuracy or SOTA claim.

## Measurements

Fresh processes run in alternating order with frozen Git source capsules and one
BLAS thread on an unreserved host. Worker time includes imports, complete pipeline
execution, trace capture, result compression and saving. API time covers the full
`run_calibration` call and reports. Downloading, source freezing, input inventory
hashing and independent audits are outside the timed workers. Ratios divide two-run
medians; host load varies, and four runs do not establish a confidence interval.
Previous speedups are not multiplied into these values.

| Scope | Previous s | Current s | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 134.924629 | 98.237561 | 1.373x |
| Complete calibration API | 131.899256 | 95.132785 | 1.386x |
| All 24 optimizer calls | 67.429389 | 23.706579 | 2.844x |

| Pair (order) | Previous whole s | Current whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (current, prior) | 137.166122 | 107.902947 | 1.271x |
| 2 (prior, current) | 132.683137 | 88.572174 | 1.498x |

| Complete diagnostic stage | Previous s | Current s | Speedup |
| --- | ---: | ---: | ---: |
| Two-sided XYZ reassociation and probes | 19.590910 | 22.472452 | 0.872x |
| Ray-depth reassociation and probes | 22.491763 | 13.469494 | 1.670x |
| Holdout rematching, curvature and multistart | 4.959758 | 5.037118 | 0.985x |

Two-sided reassociation and holdout rematching are slower in this sample. All
diagnostic inputs and outputs remain exact, and the host was unreserved. This
comparison establishes the reported optimizer and whole-pipeline measurements;
it does not establish a speedup for every stage. Work outside the optimizer
limits the complete-pipeline gain.

## Reproduce and verify

```console
python tools/download_public_dataset.py tum_rgbd_freiburg1_xyz
python tools/benchmark_rgbd_lattice_batches.py --baseline-commit a5af90c1a869b376ab5be4e6c765c2144af25cba --output outputs/rgbd-lattice-replay --repetitions 2
python tools/benchmark_rgbd_lattice_batches.py --audit-only docs/assets/rgbd_lattice_batches
```

The offline auditor verifies the config, input fingerprints, actual Git identities,
384 source modules per capsule, imported-module hashes, all complete results and
numeric histories, work counts, preparation counts, factory counts, timings,
medians and every published byte. Only six private runtime implementation files
change. Regression tests compare complete residuals, Jacobians and LM results,
including all five RGB-D factor families, 1/8/64-control lattices, cancellation,
tiny values, signed zero, scalar callbacks, cache bounds, exceptions and nesting.
Full cross-platform logs, source inventories and schema parity are bound below.

- [Audited summary](../assets/rgbd_lattice_batches/summary.json)
- [Protocol and source identities](../assets/rgbd_lattice_batches/protocol.json)
- [All attempts](../assets/rgbd_lattice_batches/attempts.json)
- [Complete byte inventory](../assets/rgbd_lattice_batches/index.json)
- [Validation and provenance](../assets/rgbd_lattice_batches/validation.json)
- [Public input fingerprints](../assets/rgbd_lattice_batches/inputs.json)
- [Public schema byte parity](../assets/rgbd_lattice_batches/schema-parity.json)
