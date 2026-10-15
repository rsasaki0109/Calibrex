# Incremental local-map screening

No candidate is adopted. The warm GPU hash pilot improves local-map preparation
by 2.737x, but the complete fresh-worker median improves only
1.010x. The two whole-worker pairs disagree on the sign of the gain.

| Scope / candidate | Observed ratio |
| --- | ---: |
| CPU retained world-coordinate/key cache | 1.483x |
| CPU retained unique groups and remapping | 0.776x |
| GPU deterministic hash, no retained scan cache | 2.266x |
| GPU deterministic hash and retained scan cache | 2.737x |
| Complete fresh-worker calibration with GPU hash/cache | 1.010x |

Each micro cohort has seven rotated-order rounds over 59 consecutive local-map
windows captured from six seconds of the public GLIM Livox bag. Together they
retain all 2,478 observations. Every variant includes per-scan cache preparation
and matches every reference output hash and point count. The launcher asserts
bit equality while the arrays are present. Captured coordinate payloads are
identified by NPZ SHA-256 but are not redistributed; CPU audit checks the saved
identities, not the actual coordinates. Micro ratios have separate baselines.

CPU medians include rounds zero through six. GPU warm medians use rounds one
through six; round zero is retained. The first uncached GPU sequence takes
approximately 3.538 timed seconds including on-demand operation compilation; module
compilation is outside the micro timing. These are not cold or whole-bag ratios.

The whole-worker cohort runs prior/current then current/prior in four fresh
processes with separate empty CuPy caches. It includes all imports, compilation,
preparation, four map/fit rounds, final covariance, evidence serialization and
source verification. Both sides use the adopted CUDA odometry and spline fitter.
The exploratory candidate is injected by a source-bound launcher into otherwise
identical frozen runtime. All complete calibration/evidence and fit fields match
exactly, allowing only run metadata and fit duration. Each candidate worker
executes 448 local-map calls over 25,427,273 input rows, with zero native fallbacks.
One BLAS thread is used and host CPU/GPU activity is unreserved. This short public
clip provides no independent extrinsic ground truth.

The hash preserves point addition order using a per-voxel sorted list of at most
32 indices, then sorts occupied voxel keys to retain the original output order.
Out-of-domain coordinates and longer lists request native repair. Those guards,
device ownership and cache invalidation are not hardened or boundary-validated
by this screen. This archive contains an experiment, not an optional runtime API.

Two initial micro launches were interrupted after the CuPy cache location was
set too late; no observations were produced. A first whole-worker attempt was
rejected by the empty-cache guard because candidate setup compiled too early.
These incomplete attempts remain in local outputs, outside the published cohorts.
The successful launcher performs candidate compilation inside the worker timing.

All files, source capsules and source/launcher/input identities are bound by the
publication index. Raw bags, coordinate arrays and compiler caches are excluded.

```bash
python -m tools.audit_incremental_local_map_screen docs/assets/incremental_local_map_screen
```
