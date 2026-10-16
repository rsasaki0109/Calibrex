# Compiled safe YAML loading

Large calibration artifacts can spend more time parsing YAML than solving the
calibration. `read_mapping` uses PyYAML's `CSafeLoader` when LibYAML is available,
with the same safe constructor and resolver tables as the Python loader. The
existing PyYAML dependency supplies this optional compiled parser; no new
dependency or command option is required.

If LibYAML is absent, or an embedding application has registered different
Python `SafeLoader` constructors or resolvers, parsing uses the original
`yaml.safe_load`. When the C parser raises a YAML error, the file is rewound and
parsed with the Python loader, preserving its acceptance and diagnostics for
those edge cases. Both loaders retain safe construction rules; Python object
tags remain unsupported. PyYAML documents the available C bindings and parser
differences in its [official documentation](https://pyyaml.org/wiki/PyYAMLDocumentation).

The JSON path and both ordinary and atomic writers retain their original
implementations. There is no new output encoding, formatting, schema or
calibration algorithm. The core stays independent of ROS.

## Compatibility checks

The complete committed YAML corpus is compared against the Python safe loader:
mapping order, Python types, float64 bytes (including signed zero and NaN),
collection alias topology, dates, timestamps, binary values, Unicode, merge
keys and repeated keys are retained. Additional tests exercise custom
constructors, multi-constructors and both resolver tables, missing LibYAML,
invalid syntax, forbidden object tags, Python-only surrogate escapes and the
existing top-level mapping requirement. JSON is checked independently.

The final checks pass 157 tests: 129 compatibility, numerical and import checks,
plus nine new archive audits, four existing archive checks and fifteen schema
registry tests. All 130 schema/ledger exports are byte-identical; strict mypy
checks 378 runtime files, Ruff and the strict documentation build pass. The
final runtime matches the measured candidate after newline normalization.

## Complete CLI comparison

The workload reuses nine public KITTI raw scans and 9,216 generated image
correspondences from the existing mixed-camera execution control. Both source
variants read the same 2,899,500-byte YAML correspondence artifact and the same
YAML initialization problem, transcoded without changing their mappings. Input
hashes bind the original JSON artifacts, resulting YAML and original fixture
builder. Raw coordinates and generated input files remain external.

This is a CPU execution and numerical-parity comparison. Generated image
correspondences do not provide independent calibration accuracy or uncertainty
coverage evidence. The existing solver's train/holdout split, confidence gates,
acceptance, covariance and all evaluation budgets remain in effect.

Three alternating fresh-process pairs run each complete CLI command. Timings
include Python startup, imports, input parsing and validation, all solver work,
diagnostics and output saving. Source verification, fixture conversion, schema
exports and external result audits run outside the timing. No warmup, failed
worker or slow attempt is discarded. Both variants use one BLAS thread on an
unreserved host; help is retained as a control that does not parse the artifact.

The measured comparison is recorded in the
[complete indexed archive](../assets/compiled_yaml_loading/README.md).

| Complete CLI command | Python parser median | Compiled parser median | Ratio |
| --- | ---: | ---: | ---: |
| Probabilistic multiframe refinement and save | 10.648664 s | 4.369567 s | **2.437006x** |
| Validate the correspondence artifact | 15.926314 s | 3.862511 s | **4.123306x** |
| Help control | 2.153393 s | 2.416475 s | 0.891130x |

All six complete refinement results and their 325 trace entries match exactly,
including candidate and selected poses, acceptance and holdout. Validation and
help outputs also match. Only creation timestamps, source commit metadata and
the declared output path are normalized. Input hashes and every numerical field
remain part of the comparison.

The host variation is material: native validation times range from 10.207014 to
18.280436 s, and help also becomes slower. All attempts are retained. The ratios
above divide the two medians; they are not portable guarantees or claims about
help, JSON inputs or raw-sensor feature extraction. Separate one-pair exploratory
probes are not pooled with this final-runtime comparison.

The measured baseline is `c1a269773802b858083b18089b3cbb5789950c6e` and the
candidate is `ec12eefd380f495edd219da40c0f29582ed04795`. The shared launcher gives
both extracted trees real Git identities backed by the local object store;
normal provenance runs inside the measured processes. The dependency receipt
records Python 3.12.10, PyYAML 6.0.3, NumPy 2.3.4, SciPy 1.17.1 and Pydantic
2.12.5. Both variants have the same installed compiled parser; the previous
implementation continues to select the Python parser.

## Reproduce and audit

Obtain the nine original scans identified by the archived fixture protocol under
KITTI's upstream terms, then use the existing fixture builder:

```bash
python -m tools.benchmark_probabilistic_refinement \
  --build-kitti-fixture path/to/2011_09_30_drive_0018_sync \
  --points-per-frame 1024 --output-dir outputs/projected-control
python -m tools.benchmark_yaml_loading \
  --baseline-commit c1a26977 \
  --correspondence outputs/projected-control/correspondence.json \
  --problem outputs/projected-control/problem.json \
  --repetitions 3 --output-dir outputs/yaml-loading-comparison
python -m tools.benchmark_yaml_loading --audit docs/assets/compiled_yaml_loading
```

The builder uses the scans present in the input directory; use the nine archived
identities to match this workload. Rebuilt hashes can differ with output paths,
timestamps and source provenance. Use fresh output directories and the recorded
candidate runtime to reproduce the comparison. The shared frozen-source CLI
harness also retains its original JSON default. The YAML wrapper selects YAML
conversion and limits its audit to the single core IO source change.

Auditing the retained comparison needs ordinary CPU dependencies without the
raw scans, generated inputs or optional accelerator packages. The archive
retains complete outputs, import paths, original source capsules/manifests,
input/transcoding identities, the frozen launcher and correctness receipts.
