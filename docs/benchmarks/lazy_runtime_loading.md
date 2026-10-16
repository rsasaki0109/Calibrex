# Selective runtime loading

The CLI imports each command's dependencies when that command runs. Solver
package exports resolve to their original objects on first access. Artifact
validation loads the selected model and schema generator, while listing kinds
uses lightweight declarations. This removes unrelated initialization from
short jobs without changing their parsing, validation, optimization or output.

| Complete fresh CLI command | Previous median | Current median | Ratio |
| --- | ---: | ---: | ---: |
| Probabilistic Camera–LiDAR refinement and save | 7.758 s | 3.003 s | 2.584x |
| Validate a 9,216-correspondence artifact | 6.975 s | 1.821 s | 3.830x |
| Help | 6.460 s | 1.820 s | 3.549x |

Each case retains three alternating fresh-process pairs: prior/current,
current/prior, prior/current. All 18 attempts succeeded on an unreserved host
with one BLAS thread. No warmup or slower attempt is removed. Timing includes
interpreter startup, normal CLI parsing and execution, and a small import-path
receipt. Freezing and checking sources, external result validation and schema
export comparison occur outside the timed process. The previous runtime already
includes [batched pose evaluation](probabilistic_pose_batching.md); its different
worker timings must not be pooled or multiplied with this comparison.

The workload reuses that benchmark's nine public KITTI raw scans and 9,216
generated image correspondences, with mixed camera models and fixed noise.
These correspondences test execution and parity; they are not independent
camera measurements or an accuracy evaluation. Every refinement field and all
325 trace entries agree across all six runs, including acceptance and holdout.
Only creation time, source-commit provenance and the CLI's declared output path
are normalized. Validation and help outputs also agree completely.

All 130 schema and ledger exports are byte-identical between frozen runtimes.
Accepted versions still come from model `Literal` annotations: namespace hints
only prioritize candidate models, and a complete scan handles older or unrelated
namespaces. Aliases, migrations, retired-version errors, digest verification and
the public `SCHEMA_REGISTRY` tuple remain available. Concurrent first access
returns stable cached entries, and imports happen outside the registry cache lock.
Long-running jobs that already loaded these dependencies should expect less benefit.

The final runtime passes 155 targeted checks covering CLI dispatch, public
solver identities, concurrent schema loading, migrations, digest checks,
committed artifact readability and complete refinement parity. Four additional
archive checks pass after binding the final correctness receipt.

The [indexed archive](../assets/lazy_runtime_loading/README.md) contains every
attempt, full refinement output, CLI stdout/stderr, actual import paths, source
capsules and manifests, the launcher, input hashes and schema parity receipt.
Raw dataset coordinates, generated correspondence files and model weights are
not redistributed. The CLI's `git_commit` field records the enclosing checkout;
the separate frozen-source manifests and import receipts identify the runtime
actually executed for each variant.

## Reproduce

Use the fixture builder and nine scan identities documented in the
[pose batching benchmark](probabilistic_pose_batching.md). At measured candidate
`c0402f431902c18cb6f5c61b8aaa6d505ca2823a`, run:

```bash
python -m tools.benchmark_lazy_runtime_loading \
  --baseline-commit e7b89520f5391005c62016fd2549d39de0693c09 \
  --correspondence outputs/projected-control/correspondence.json \
  --problem outputs/projected-control/problem.json \
  --repetitions 3 --output-dir outputs/cli-comparison
python -m tools.audit_lazy_runtime_benchmark docs/assets/lazy_runtime_loading
```

Use a new output directory. The CPU-only audit verifies the inventory and source
capsules, limits runtime changes to the three loading files, checks every output
and input binding, and recalculates medians from all attempts.
