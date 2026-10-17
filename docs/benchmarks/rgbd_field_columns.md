# Ordered RGB-D field finite differences

The complete native XYZ optimizer on a **27,544-correspondence public TUM
graph** takes **48.545 → 12.731 seconds (3.813x)**. Its fresh worker, including
input preparation, takes **70.087 → 34.386 seconds (2.038x)**. These are medians
of three alternating pairs using exact Git source capsules.

This workload is one large XYZ field solve, with all 20 configured iterations,
observability, robust weighting, Schur solving and 60 known-bad probes. It is
not the complete multi-window, multi-ablation calibration suite. Both versions
retain the same `max_iterations` outcome; faster execution makes no convergence,
accuracy or SOTA claim.

## Implementation

Native RGB-D field columns evaluate the same positive and negative perturbations
in bounded point chunks. Pose transforms remain unchanged across these field
probes and are computed once per group. XYZ probes reuse unchanged axis sums,
using byte keys to distinguish signed zero.

Large batches retain the supported host's ordered float summation, including
CPython 3.12 compensation. Small batches and other Python versions use ordinary
Python `sum`. The arithmetic follows the
[versioned CPython implementation](https://github.com/python/cpython/blob/v3.12.10/Python/bltinmodule.c).
Pose columns keep their existing path. Unsupported or custom factors retain
the existing scalar fallback.

Scratch point chunks hold at most 16,384 probe points and 131,072 product
elements. The implementation writes into the existing Jacobian rather than
retaining a second full residual matrix for every probe. No new dependency or
public configuration field is introduced.

## Workload and every attempt

The already spent TUM `freiburg1_xyz` development sequence supplies three map
frames and eight query frames starting at paired index 180. Each worker decodes
all eleven complete PNGs. Relative to the example, only the graph density changes:
50,000 maximum map points per frame, 20,000 query points per frame and 4,000
maximum correspondences per frame. Both variants use the same configuration,
27,544 data factors, ten prior/regularization factors, iteration limits, gates
and one BLAS/OpenMP thread.

| Fresh worker | Complete optimizer seconds | Whole worker seconds |
| --- | ---: | ---: |
| 01-current | 10.807 | 27.357 |
| 02-prior | 42.896 | 59.190 |
| 03-prior | 53.264 | 71.128 |
| 04-current | 12.731 | 34.386 |
| 05-current | 12.979 | 35.981 |
| 06-prior | 48.545 | 70.087 |

Whole-worker time includes startup, imports, input hash verification, depth
decoding, sampling, voxel maps, initial matching, graph construction, the full
optimizer and evidence saving. Numeric trace hashing occurs inside the optimizer
timer in both variants. The host was unreserved; the observed run-to-run variation
does not support a hardware-independent guarantee.

All six complete results match, including every parameter, factor ID, control,
rank, diagnostic and failure status. Each worker also retains 21 complete
residual/Jacobian evaluations and 81 complete residual evaluations. Shapes,
input states, ordered factor identities and every array's float bytes agree.

An [earlier full-suite prototype](../assets/rgbd_field_columns/small-workload-prototype.tar.gz)
improved the small original multi-window workload only 1.050x (39.436 → 37.553
seconds). That exploratory precursor is retained with its four runs and exact
prototype source. The large-graph ratios above must not be applied to that suite.

## Evidence and verification

- Prior runtime: `00fe74e00a72d8f619592627268df9a2408a1496`.
- Measured runtime: `63b80c3109b7c714fd09129cc653d439e881c138`.
- [All attempts](../assets/rgbd_field_columns/attempts.json),
  [summary](../assets/rgbd_field_columns/summary.json),
  [configuration](../assets/rgbd_field_columns/config.yaml),
  [measurement provenance](../assets/rgbd_field_columns/provenance.json).
- [Complete file inventory](../assets/rgbd_field_columns/index.json),
  [record schemas](../assets/rgbd_field_columns/record-schemas.json),
  [validation](../assets/rgbd_field_columns/validation.json).

```powershell
python tools/audit_rgbd_field_columns.py docs/assets/rgbd_field_columns
# From the repository root, with the already downloaded public TUM sequence:
Copy-Item docs/assets/rgbd_field_columns/launcher.py.txt .tmp-calibrex-rgbd-field-columns-measurement-v1.py
python .tmp-calibrex-rgbd-field-columns-measurement-v1.py
```

The launcher creates a fresh `outputs/rgbd-field-columns-pairs-v1` directory and
refuses to reuse it. Keep any earlier directory and choose a new output path in
the copied launcher when repeating a local experiment. The saved generator hash
then identifies that exact launcher.
