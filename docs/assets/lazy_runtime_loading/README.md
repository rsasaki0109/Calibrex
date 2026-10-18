# Complete CLI comparison archive

All 18 workers are retained: three alternating pairs for each of help,
validation and probabilistic refinement. The
[benchmark report](../../benchmarks/lazy_runtime_loading.md) defines the scope
and normalization. `protocol.json` binds the input hashes, commits, frozen source
manifests, launcher, Python and thread settings. `attempts.json` retains every
duration and exit code; `summary.json` reports their medians.

`sources-*.tar.gz` contain only Calibrex Python sources. The complete raw result,
stdout/stderr and actual import paths are kept for every worker. The frozen
imports identify the executed code; the result's existing Git field identifies
the enclosing checkout. The input geometry and generated correspondence files
are excluded. `fixture-protocol.json` retains their construction and source
identities. `schema-parity.json` binds 130 byte-identical schema/ledger exports
to both measured source manifests and the retained export checker.

`validation/` retains the 155-test correctness run, the executed test sources,
the documentation build log and a receipt confirming that all 377 runtime
sources still match the measured candidate after newline normalization.

`index.json` hashes every other file. Files use their original bytes, including
empty stderr files. Run the CPU-only audit from the repository:

```bash
python -m tools.audit_lazy_runtime_benchmark docs/assets/lazy_runtime_loading
```
