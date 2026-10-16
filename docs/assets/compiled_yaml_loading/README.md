# Complete compiled YAML loading comparison

All eighteen timed CLI workers are retained: three alternating pairs each for
YAML-input multiframe refinement, correspondence validation and help control.
Whole-process medians are 10.648664 to 4.369567 s (2.437006x) for refinement and
15.926314 to 3.862511 s (4.123306x) for validation. Help is 2.153393 to 2.416475 s
(0.891130x); it reads no artifact and supplies no acceleration claim. The host
was unreserved and variation is material. No warmup, slow attempt or failed
worker is omitted. All workers exited successfully.

The [benchmark report](../../benchmarks/compiled_yaml_loading.md) explains
automatic selection, compatibility, complete timings and reproduction.
All six complete refinement results and their 325 trace entries match exactly,
including acceptance, candidate/selected transforms, holdout and input hashes.
Only creation timestamps, source commit metadata and the declared output path
are normalized. Validation/help outputs also match completely.

`protocol.json` binds the actual launcher/freezer, both source manifests,
commands and YAML input hashes. `input-transcoding.json` binds those inputs to
the original JSON identities and records installed dependencies. The original
fixture protocol and byte-exact builder bind all nine public KITTI raw scans.
The 9,216 image correspondences are generated from public geometry. This is
execution and parity evidence, not independent calibration accuracy or
uncertainty coverage evidence. Coordinates and generated input files are not
redistributed. No locked accuracy cohort is opened by this comparison.

Each worker retains stdout/stderr and actual core-module import paths. Every
refinement retains its full result. The only runtime change in the complete
source capsules is core/io.py. Each extracted tree had a real Git identity
backed by the local object store, retaining normal provenance during timing.
The source trees' Git directories are not redistributed.

Fixture conversion, source checking, schema exports and output auditing are
untimed. Timed work includes startup, imports, parsing/validation, all solves,
diagnostics and saving. The same installed LibYAML, PyYAML and numerical
dependencies and one BLAS thread are used in both variants. The prior runtime
selects the Python parser; the candidate selects the compatible safe C parser.

The archive keeps both frozen source capsules/manifests, the launcher, shared
auditor, YAML entry point, original fixture builder, all eighteen attempts,
complete results, schema parity and the final regression receipt. The latter
includes all 1,889 committed YAML files plus numerical and import checks.
`index.json` binds every other file by SHA-256, including empty stderr files.

Audit on CPU without the data, a GPU, or LibYAML:

```bash
python -m tools.benchmark_yaml_loading --audit docs/assets/compiled_yaml_loading
```
