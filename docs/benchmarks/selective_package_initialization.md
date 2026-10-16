# Selective package and CLI initialization

Fresh CLI help falls from **1.970183 to 0.547647 seconds (3.597543x)**.
Validation falls from **1.856774 to 1.187515 seconds (1.563580x)**; complete
probabilistic refinement falls from **2.486135 to 2.145332 seconds (1.158858x)**.
These are ratios of three-run medians. The substantial benefit is in startup
and light commands; full calibration still needs its numerical dependencies.
This does not claim a large solver or GPU acceleration.

## Change and compatibility

The root, data and evaluation packages expose the same public objects through
on-demand imports. Type-checking imports, `__all__`, `dir()`, star imports,
class/function identity, pickling and concurrent first access remain supported.
No public object is wrapped or replaced.

Previously, building every command's parser imported artifact validation,
bag readers, confidence calibration and PnP initializer code merely for choices
and initial values. Shared lightweight definitions now supply the exact same
Livox profile fields and Camera–LiDAR defaults. The original Livox dataclass
stays in its original module; public calibration constants are explicit aliases
to the shared tuples. Already loaded module overrides still take effect, and a
module still being initialized safely uses the declared fallback.

Schema choices resolve when that selected argument is parsed or displayed.
They are attached after argparse's eager metavar check, so constructing the
root parser does not load schema models. Iteration and membership retain the
complete original order and reject unknown kinds. Normal root parser
construction imports no NumPy, SciPy, Pydantic, YAML or bag-processing code.
Selected commands retain all ordinary loading, parsing, validation and fitting.

The complete frozen parser comparison covers **164 routes and 1,257 argument
actions**: every help string, option, action, callback, default, choice and
required field matches after normalizing only the frozen source-root prefix in
packaged resource paths. All **129 public schema exports and their ledger are
byte-identical** across both measured trees.

## Full CLI comparison

The same prepared public KITTI execution control used for the earlier CLI
comparison supplies nine frames and **9,216 projected correspondence rows**.
The image correspondences are generated. This tests performance and complete
result parity, without independent accuracy or leaderboard evidence. All rows,
default iteration limits, train/holdout splits, uncertainty and result saving
are retained. The archive includes exact prepared inputs and their original
scan hashes and construction metadata. It excludes raw scan registration.

Every case runs in three fresh frozen pairs, alternating execution order with
one BLAS thread on an unreserved host. Whole-process time includes Python and
CLI initialization, normal command execution, complete output saving and import
recording. Source freezing, control generation, schema/parser comparisons and
external audits are untimed. Filesystem caches are not flushed.

| Case | Pair/order | Prior s | Selective s |
| --- | --- | ---: | ---: |
| Help | 1 prior, selective | 1.970183 | 0.547647 |
| Help | 2 selective, prior | 1.842589 | 0.579191 |
| Help | 3 prior, selective | 2.043167 | 0.529829 |
| Validate | 1 prior, selective | 1.852106 | 1.293866 |
| Validate | 2 selective, prior | 1.876653 | 1.187515 |
| Validate | 3 prior, selective | 1.856774 | 1.143997 |
| Refine | 1 prior, selective | 2.486135 | 2.327321 |
| Refine | 2 selective, prior | 2.636089 | 2.037895 |
| Refine | 3 prior, selective | 2.468500 | 2.145332 |

Baseline: `08452aa1b5955fc685279b3e664325fa476a943e`.
Measured source: `0073c5c713dc1ccf526969311ecf86babb121718`.
The scoped audit permits exactly the nine package/CLI/default-definition files
changed in this comparison. No numerical solver source changes. All 18 command
outputs and all six complete refinement results, including every trace row,
match. Only creation time, independently checked actual Git identity and the
declared output path are excluded. Input identities and result command arguments
stay compared. Each actual imported path must belong to its frozen source
manifest, and the full source tree is checked before and after each worker.

The package-only v1 cohort improved the three cases by about 1.1x and was
insufficient. v2 preceded the partial-module guard. Their complete timing and
hash inventories remain in the publication receipt; final v3 alone supplies
the reported ratios. The final runtime is unchanged after that cohort.

## Verification and reproduction

Regression coverage exercises original public object identities, concurrent
loading, pickling, optional solver adapters, all command names/help groups,
lightweight parser construction, live public-module overrides, partial imports,
default tuple identity and deferred choice membership. Camera–LiDAR confidence
and initializer calibration, IMU–LiDAR and LiDAR–wheel evaluations retain their
existing regression coverage. The archive audit rejects changed timings,
traces, actual source identities, prepared input bytes and parser snapshots
even after their byte-index digests are rebound.

The [portable archive](../assets/selective_package_initialization/README.md)
retains test sources/logs, strict mypy and Ruff receipts, full parser/schema
comparisons and the exact earlier-cohort inventory. The public source tree stays
ROS-independent and all schemas keep their existing names and versions.

```bash
python -m tools.audit_lazy_runtime_benchmark \
  docs/assets/selective_package_initialization --runtime-profile selective-packages
```

For fresh measurement, decompress `correspondence.json.gz` into a new input
directory as `correspondence.json`, copy `problem.json` beside it, then run:

```bash
python tools/benchmark_lazy_runtime_loading.py \
  --baseline-commit 08452aa1b5955fc685279b3e664325fa476a943e \
  --correspondence outputs/package-inputs/correspondence.json \
  --problem outputs/package-inputs/problem.json \
  --output-dir outputs/package-remeasurement --repetitions 3
```

KITTI data and generated derivatives retain **CC BY-NC-SA 3.0**, attributed to
Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun:
[KITTI](https://www.cvlibs.net/datasets/kitti/),
[data license](https://creativecommons.org/licenses/by-nc-sa/3.0/).
Calibrex source and benchmark code retain Apache-2.0.
