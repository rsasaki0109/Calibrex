# Indexed online-edge execution evidence

The [complete archive](complete-evidence.tar.gz) retains all six workers,
complete results/configs, source and tool snapshots, schema files, import
receipts, logs, protocol and acquisition receipt. Its index binds 445 files by
SHA256 and size. Raw images/scans and compilation caches are excluded.
The [archive checksum](archive.sha256) and [comparison](comparison.json) are
available separately.

| Implementation | Whole-worker median |
| --- | ---: |
| NumPy with parallel preparation | 62.852 s |
| Compiled raw-score CPU with parallel preparation | 31.042 s |
| CUDA with overlapped initialization/preparation | 14.036 s |

This cohort measured 4.478x against NumPy and 2.212x against the compiled
raw-score CPU control. Two observations per implementation ran in the declared
counterbalanced order on an unreserved host. All 48 prepared frames and all
six complete result payloads match exactly. Execution metadata is checked
separately. Earlier development screens are not pooled with this cohort.

This is a **working-copy source snapshot**, with `source_commit: null` and
base revision `cdd95486a0540a91c3d1378e1f8e2ef6b38a2b97`. All 400
package modules and seven benchmark tool/schema files are frozen by content.
The record makes no committed-source or SOTA claim. The
[post-run Git content binding](git-content-binding.json) verifies the frozen
package and toolchain against commit
`4406195f9dd2e6e86a746eeac733b82364c8c90a`, without rewriting the original
lineage or measurements. The later benchmark export correction is not part of
the measured toolchain.
Bindings compare UTF-8 text after universal-newline normalization, matching
the source snapshot and Python's source parsing. Binary content is unchanged.

```bash
python tools/audit_online_edge_cuda_archive.py \
  docs/assets/online_edge_cuda/complete-evidence.tar.gz \
  --output outputs/online-edge-archive-audit.json
```

The auditor uses CPU-only JSON/schema/hash checks. It does not execute archived
Python or import CuPy/Calibrex. It rejects unsafe paths, duplicate files,
unindexed content and external schema references. With `--bind-commit REF`, it
also checks every package module and benchmark tool against that later Git
revision; the output labels this as a post-run content binding.

To reproduce the saved binding, add
`--bind-commit 4406195f9dd2e6e86a746eeac733b82364c8c90a` to the audit command.

Reproduction uses the benchmark driver described in
[the execution guide](../../benchmarks/online_edge_cuda.md). The input is the
same 48 already-spent KITTI image/cloud pairs, with the upstream archive and
all 101 input-file hashes in `receipt.json`. Calibration is an initializer
reference for this performance control, rather than independent accuracy truth.
