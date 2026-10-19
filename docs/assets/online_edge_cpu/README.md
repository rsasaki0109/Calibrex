# Indexed online-edge CPU execution evidence

The [complete archive](complete-evidence.tar.gz) retains every worker from the
final six-process cohort and all complete results/configs, measurement records,
source/tool snapshots, generated schemas, input receipt and stdout/stderr.
Its index binds 449 files by SHA256 and size, including the packaging source
and project license.
Raw images/scans and compiler caches are excluded.

| Implementation | Whole-worker median |
| --- | ---: |
| Production NumPy | 51.174 s |
| Compiled benchmark control | 23.452 s |
| Maintained Numba CPU backend | 23.724 s |

The maintained backend measured **2.157x against production NumPy** and
**0.989x against the compiled benchmark control**. It makes the compiled
control's performance available through supported APIs. All 48 prepared frames
and all six complete calibration result payloads match exactly, excluding only
separately validated execution metadata. Each implementation has two cold
observations on an unreserved host. Default CPU serialization is unchanged.

The measured source commit is
`bb90e12f2fa2b811a10ad92dfe222aaffef7831f`. All 401 package modules and eight
benchmark tool/schema files are frozen. The [comparison](comparison.json) is
also available separately. The [archive checksums](archives.sha256) cover both
cohorts.

The [initial image-JIT archive](initial-image-jit.tar.gz) preserves all six
earlier workers from `ccf3f872663c724dc9bc1d00a4279f35853b6bab`. Its whole-worker
medians were 55.588 s NumPy, 26.107 s compiled control and 26.036 s Numba, with
complete parity. It is a separate development cohort, not pooled with the final
measurement. Its index also binds 449 files.

```bash
python tools/audit_online_edge_cpu_archive.py \
  docs/assets/online_edge_cpu/complete-evidence.tar.gz \
  --bind-commit bb90e12f2fa2b811a10ad92dfe222aaffef7831f \
  --output outputs/online-edge-cpu-archive-audit.json
```

The auditor uses CPU-only JSON/schema/hash checks without importing Numba,
CuPy or Calibrex, or executing archived Python. It applies the shared archive
limits, safe-path checks, complete index validation and local-only schema
references. Optional Git binding checks normalized UTF-8 source/tool content.

The [execution guide](../../benchmarks/online_edge_cpu.md) describes reproduction
from existing inputs. Both cohorts reuse the same 48 already-spent real KITTI
image/cloud pairs. Every input identity is pinned in the acquisition receipt;
the dataset calibration initializes an execution control and supplies no new
extrinsic accuracy or SOTA evidence. One frozen tree serves all workers, with
less than 9 MiB of uncompressed evidence per cohort and a 64 MiB run limit.
