# Selective package and CLI initialization

Three alternating fresh-process pairs per command retain the complete CLI:

| Command | Prior s | Selective s | Speedup |
| --- | ---: | ---: | ---: |
| Top-level help | 1.970183 | 0.547647 | 3.597543x |
| Artifact validation | 1.856774 | 1.187515 | 1.563580x |
| Complete probabilistic refinement | 2.486135 | 2.145332 | 1.158858x |

This improves startup, especially light commands. The full refinement gain is
modest; it is not presented as a large numerical or GPU acceleration.
All 18 complete outputs and six complete calibration artifacts/traces match.
All 164 parser routes, 1,257 recorded argument actions, help, defaults, choices
and all 129 public schemas plus their ledger remain identical.

The prepared execution control uses nine public KITTI scans with 1,024 points
per frame (9,216 rows). Image correspondences are generated projections,
without independent accuracy evidence. The compressed correspondence input
retains its exact uncompressed protocol hash. Original scan hashes and fixture
construction metadata are in `fixture-protocol.json` and the input provenance.
Raw scan registration and control generation are outside these timings.

This archive includes all prepared inputs, actual source identities, full source
capsules/manifests, every timing/stdout/stderr/import record and result, complete
parser snapshots, schema parity receipts, source/checker pins and validation
receipts. `index.json` binds every publication byte other than itself.
Earlier package-only and pre-guard cohorts remain identified, with all their
timings and hashes in `validation/cohort-selection.json`. Only the final v3
cohort supplies the speed claims above.

See [the benchmark](../../benchmarks/selective_package_initialization.md).
Audit offline from the repository root:

```bash
python -m tools.audit_lazy_runtime_benchmark \
  docs/assets/selective_package_initialization --runtime-profile selective-packages
```

KITTI attribution: Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel
Urtasun. Original data and generated derivatives retain CC BY-NC-SA 3.0:
[dataset](https://www.cvlibs.net/datasets/kitti/) and
[license](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Calibrex source snapshots and measurement code retain Apache-2.0.
