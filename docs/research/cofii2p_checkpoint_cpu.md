# Pinned CPU checkpoint execution and raw correspondence export

This extends the [random-weight CPU forward probe](cofii2p_cpu_probe.md) into an
explicit checkpoint-loading boundary. The user has no GPU; this path stays on CPU.
Official CoFiI2P weights are still absent locally and their README distribution
hosts remain excluded by the enforced network policy. No repeat denial probes,
unsupported proxy routes or speculative mirror downloads were performed.

## Request and loading contract

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python tools/cofii2p_checkpoint_cpu.py request.json new-output-directory
```

The [request schema](../assets/cofii2p_checkpoint_cpu_20261030/request.schema.json)
requires a pinned bridge manifest, pinned checkpoint, reviewed source checkout,
checkpoint kind/source/training/terms declarations, image-channel policy, patch
layout and seed. Unknown fields, including reference-problem inputs, are rejected.
Metadata is a caller declaration, not automatic verification of official weights,
license or training isolation. Unknown terms/training must remain explicit.

Checkpoint and input digests are verified before model construction. The prepared
artifact/manifest chain is checked and the bridge image must equal the prepared
image. No real problem/reference loader, dataset calibration or labels enter the
forward call. The runner uses the already exercised isolated CPU source patch;
no external model code enters Calibrex core.

Load raw tensor state only with `torch.load(map_location='cpu', weights_only=True)`.
Require exactly the model's keys, shapes and dtypes, and finite CPU tensors, then
load with `strict=True`. No `module.` prefix stripping, wrapper guessing, silent
dtype conversion, unsafe-pickle fallback, partial load or random-weight fallback.
If any tensor fails validation, the model state is not mutated. This matches the
pinned official training code's raw `model.state_dict()` save format; a differently
packaged checkpoint requires an explicit, separately reviewed conversion.

## Correspondence coordinates and source lineage

An observational wrapper records the native `fine_process` selected coarse rows;
it does not modify returned features, scores or indices. Forward receives `None`
for all label arguments. Repeat-forward outputs and selected rows must match
exactly, and every returned point is checked against its selected input row.

Fine-patch cosine matching uses native PyTorch normalization and first argmax.
The request explicitly chooses row-major or legacy evaluation pixel decoding;
the [prior patch audit](cofii2p_model_bridge.md) still applies. Fine-grid pixels
map through the declared local affine to original-image pixels. The input image
policy is either the prepared BGR array or an explicit BGR-to-RGB swap. Neither
choice establishes official checkpoint preprocessing parity.

`raw_pairs.npz` contains LiDAR-frame voxel centroids, original-image pixels, unique
correspondence IDs, selected coarse rows/voxel indices, fine pixels/patch indices
and CSR arrays containing **every contributing original point ID**. Duplicate
centroids are retained as separate predicted pairs. No nearest-source snapping,
invented covariance/confidence, correctness filter or pose fit is added.

The [result schema](../assets/cofii2p_checkpoint_cpu_20261030/result.schema.json)
records checkpoint/input/source hashes, loaded-state signature, runtime, timings,
memory, repeatability, raw pair artifact and all failures. CLI exits2 on a recorded
execution failure; validation failures exit nonzero before inference. Raw exports
require a separately declared provider/uncertainty/quality policy before any
probabilistic calibration claim. Pair count alone is not acceptance or accuracy.

## Frozen integration test

The official learned artifact cannot be tested yet. Instead, the committed fixture
builder constructs the full pinned model once at seed20261030 and saves its raw
state as `SYNTHETIC_RANDOM_STATE.pt`. It is never labeled as trained or official.
The loader refuses fixture execution unless `--allow-fixture` is explicitly given.
Both checkpoint creation and execution occur in separate fresh processes.

Use the existing standard20480-point synthetic bridge and its actual prepared
constant BGR image. Fix row-major decoding and two CPU threads. Run once with a180s
ceiling; retain failure if random features produce insufficient pairs. Do not swap
seeds/images or manufacture matches to make this fixture succeed. Checkpoint tensor
loading is independently unit-tested with rejection of wrong keys/shapes/dtypes,
nonfinite/non-tensor values and nested wrappers, with state immutability on failure.

The [plan](../assets/cofii2p_checkpoint_cpu_20261030/plan.json) pins the implementation,
fixture builder, tests and bridge before the integration run. This is a software
integration check, not model accuracy, a calibration fit or an official-weight run.
No learned dataset or fresh holdout is downloaded. Next obtain an official KITTI
artifact through an authorized accessible location, establish checkpoint/image/
training provenance and run the same CPU loader before freezing the spent-data pilot.
