# CoFiI2P geometry bridge and patch-index audit

This follows the [initializer-only boundary](cofii2p_input_boundary.md). It builds
CPU model-shaped geometry, source lineage and explicit fine-pixel decoding; it
does not run a model, establish official numerical parity or change core defaults.
All code here is independently implemented, with no official code copied into core.

## Pinned-source findings

Reviewed official CoFiI2P commit `ed90edf51cca540ee76152d91133c7e79b63256c`:

| Source | Observed behavior | Adapter consequence |
| --- | --- | --- |
| `data/kitti.py:148–180` | 0.1m voxel averages, then count selection; scarce clouds are repeated | A voxel is a centroid with multiple original contributors, not an original point ID |
| `model/kpconv/preprocess_data.py:56–101` | Five levels, half-count sampling with replacement, 128 nearest neighbors | Preserve duplicate rows and every parent index; neighbor matrices index support rows |
| `data/kitti.py:305–327` | Nominal 0.5 intrinsics, center crop; fine K scales by 0.5, coarse by 0.125 | Feature-grid pixels and source image pixels are different coordinates |
| `model/network.py:156–158,204–225` | Centers are (u,v); patches index top:bottom, left:right, then flatten (h,w) | Flat patch index j means column j%4, row j//4 |
| `evaluation/eval_all.py:102–105` | Adds j//4 to u, j%4 to v | Off-diagonal patch positions are transposed relative to array layout |

The patch discrepancy affects 12 of the 16 possible positions, not an observed
12/16 model error rate. We inspected source; we did not run the external model or
measure its practical accuracy. The bridge requires an explicit decoding choice:
`row_major` follows image-array layout, and `official_eval_legacy` reproduces the
evaluation formula. Neither is selected using calibration reference errors. Freeze
the pilot convention before inference; any legacy comparison must be declared in
advance and must not be reported as additional independent data.

The loader reads already prepared image `.npy` files. `crop_original_top_rows=50`
in options does not itself establish how those files were made. Color order and
upstream top-crop provenance remain unresolved. Existing boundary BGR, top crop
and actual-resize pixel-center mapping are explicit local conventions, not a claim
that those arrays reproduce a particular checkpoint's training distribution.

## Bridge contract

```bash
PYTHONPATH=src:. python tools/cofii2p_model_bridge.py prepared/manifest.json options.json bridge-output
```

`options.json` requires an explicit seed. Defaults are voxel size0.1m, count20480,
five stages and128 neighbors. Input manifest and artifact hashes, options, driver
hash, git commit and NumPy/SciPy versions are recorded in the bridge manifest.
The existing prepared artifact digest is verified before decoding. Output
directories cannot be overwritten.

Voxelization uses a grid anchored at `min(initial-camera points)-voxel_size/2`,
with lexicographically ordered keys. Points, intensity and rotated normals are
averaged; averaged normals are intentionally not renormalized. The stored LiDAR
centroid uses exactly the same contributor rows as the initial-camera centroid.
Thus the two means retain the declared initializer relationship. Never assign an
arbitrary contributor's original ID or position to a centroid correspondence.

Lineage is explicit:

```text
level row -> voxel_index_level -> voxel_member_offsets slice
          -> voxel_member_source_rows -> source_point_id
```

`count_selection` and `parent_level` preserve all count/half-sampling choices,
including duplicates. `points_level`/`feats` are float32. `neighbors_level` queries
the same level, `subsampling_level` queries the next level against the current
one, and `upsampling_level` queries the current against the next. These are KNN
indices, not radius neighborhoods. Use voxel LiDAR centroids for future point–pixel
exports and keep each correspondence identity distinct from source IDs.

Fine decoding returns (u,v) on the256×80 feature grid. To match the declared local
K convention, multiply fine coordinates by2, then invert the stored source-to-input
affine. No additional half-pixel feature offset is silently introduced. Learned
receptive-field alignment still needs checkpoint verification.

This SciPy geometry implementation is not a validated CPU inference port. Open3D
voxel iteration order and equal-distance KNN ordering may differ. The random stream
is locally seeded; it does not reproduce the official dataset-index seed plus
intervening random pose draws. Voxelization in initializer coordinates also cannot
reproduce the official truth-prealigned cloud's grid in general. These differences
require a separate parity experiment before claiming official reproducibility.

## Frozen synthetic smoke protocol

Run the committed driver into a new directory:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/cofii2p_bridge_smoke.py /workspace/work/cofii2p-bridge-smoke-20261030
```

Use4096 synthetic points on a0.3m grid, declared nonidentity initializer and camera,
explicit top crop50, and seed20261030. Run the bridge twice at the default shape
20480/10240/5120/2560/1280, with128 neighbors. Check array equality, complete source
lineage, centroid transform, all neighbor shapes/ranges, and39 selected KNN queries
against exhaustive distances (allowing different indices at equal distance).
Test all16 patch positions and preserve the12 legacy/layout differences. No model,
real frame, calibration fit or reference transform file is involved.

The [plan](../assets/cofii2p_model_bridge_20261030/plan.json) pins implementation,
tests, driver, options and official-source hashes before the smoke run. Unit tests
also cover multi-point voxel means, cancelling normals, exact count rules including
divisible scarce counts, KNN orientation, pixel mapping and digest rejection.

GPU/validated inference runtime, suitable weights and terms, checkpoint image
provenance and training overlap remain prerequisites for the spent-data pilot.

## Outcome

Implementation frozen at `b6d0628e`. The [smoke result](../assets/cofii2p_model_bridge_20261030/result.json)
passed all eight checks; both bridge manifests and prepared-input provenance are
retained beside it. Binary arrays remain under the pinned workspace paths (about49MB
including both outputs); manifests record their SHA256. Repeat-run equality was
checked for every array, not for manifests whose output paths differ.

Stage counts were20480/10240/5120/2560/1280. Their distinct source-voxel counts were
4096/3775/2581/1504/811: replacement sampling retains duplicate rows, as intended.
All39 selected KNN distance checks passed. Pixel layout and legacy decoding differ
at exactly12 of16 patch positions. Related boundary/export/PnP tests passed21 cases.
There were zero real frames, model calls or accuracy fits. The plan's documentation
hash identifies the pre-run protocol at the frozen commit, before this outcome.

Next priority is checkpoint/preprocessing and runtime readiness, followed by an
explicitly frozen inference pilot. Additional geometry refinements alone cannot
resolve the missing model evidence or establish SOTA.
