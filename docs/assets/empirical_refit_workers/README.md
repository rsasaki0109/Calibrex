# Deterministic empirical-refit comparison

Six complete CLI runs retain three alternating pairs, each with twenty seeded
temporal-block refits. The prior runtime uses one process; the current runtime
uses four. Whole-process medians are 29.935852 s and 12.643885 s (2.367615x).
Every run retains 6,224 solver trace entries. All 120 refits and six uncertainty
results match exactly after the declared execution metadata normalization.

The [benchmark report](../../benchmarks/empirical_refit_workers.md) describes
usage, numerical behavior and limits. `protocol.json` binds the complete source
capsules, all three input identities, launcher/freezer/schema and dependencies.
`fixture-builder.py.txt` is the exact original builder named by the fixture
protocol, with nine raw KITTI hashes and deterministic point selection.
Coordinates, generated correspondence/problem inputs and nested Git metadata
remain external. The control uses public KITTI geometry with generated image
correspondences. It establishes execution parity, not real-image accuracy or
uncertainty coverage. All uncertainty policies remain inconclusive.

Each run retains its twenty refit YAML files, final uncertainty artifact,
stdout/stderr and imported-source paths. The four process initialization
receipts in each current run also bind child imports to the source capsule.
The launcher only observes imports and invokes the regular CLI. Both source
trees had real Git identities backed by the local object store; normal Git
provenance was retained during timing. No numerical operation was patched.

All attempts are retained; no timing warmup is removed. Both variants use one
BLAS thread per process on an unreserved host. The comparison includes process
startup, input validation, twenty refits, diagnostics, YAML output and pool
shutdown. Separate untimed checks validate every output and child hash before
normalizing timestamps, source commits, declared output paths, the worker flag
and verified child-output hashes. No numerical field is normalized.

`initialization-v1.log` records a Git compatibility failure before any timed
worker. `measurement.log` records all six successful v2 workers and the later
summary failure on an omitted optional reference. `postprocessing.json` and
`postprocessing.log` bind the corrected summary to the unchanged measurements;
`launcher.py.txt` is the measured launcher, while `auditor.py.txt` handles that
omitted field and additional audit bindings. No timing was repeated or dropped.

`validation/` retains regression sources/logs, schema parity and a receipt
binding the final runtime to the measured candidate. `index.json` binds every
other file by SHA-256, preserving bytes across checkouts. Audit with ordinary
CPU dependencies, without raw KITTI data or a GPU:

```bash
python -m tools.benchmark_empirical_refits --audit docs/assets/empirical_refit_workers
```
