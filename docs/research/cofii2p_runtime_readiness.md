# CoFiI2P execution and checkpoint readiness

CPU-only follow-up: the user has no GPU. The [CPU execution probe](cofii2p_cpu_probe.md)
subsequently completed the actual test-mode graph at20480 points in10.36s with
random weights. The earlier runtime blocker below is historical; GPU access is
no longer a prerequisite for that exercised path. Checkpoint and image/training
provenance blockers remain, with no learned accuracy claim.

Reviewed2026-10-30 after the [geometry bridge](cofii2p_model_bridge.md).
[Machine review](../assets/cofii2p_runtime_review_20261030/review.json) and
[schema](../assets/cofii2p_runtime_review_20261030/review.schema.json) retain runtime,
network observations, pinned source hashes and hashes/paths of public web snapshots.
No checkpoint, external executable code or new real frame was downloaded or run.
Public GitHub pages were read; no issue or message was posted.

**Decision: defer inference.** The geometry bridge is ready for its tested scope,
but the current runtime and checkpoint evidence do not support a model comparison.
Do not replace the missing model experiment with more geometry polishing.

## Concrete blockers

| Requirement | Observed evidence | What resolves it |
| --- | --- | --- |
| Runtime | Python3.12.14, torch2.4.0+cpu, CUDA unavailable/zero devices, Open3D absent | GPU runtime with verified compatible PyTorch/Open3D, or separately validated CPU inference port |
| Checkpoint bytes | Both official OneDrive/Baidu links fail proxy CONNECT403; no weights acquired | Accessible official KITTI artifact with SHA256 and terms |
| Checkpoint training | Historical translation-augmentation correction; distributed checkpoint lineage unknown | Training commit/date, augmentation recipe and regeneration identity |
| Image preparation | RGB ancestry evidence; exact distributed KITTI generation not established | Pinned prepared example/recipe establishing color, crop and K conventions |
| Evaluation scope | Validation/test not distinguished according to collaborator; our raw-frame overlap unknown | Training/selection/raw-drive disclosure; unknown overlap stays development-only |

The enforced managed network policy allows GitHub and excludes `1drv.ms` and
`pan.baidu.com`. Both README weight URLs were tried once with HEAD and no redirect
following; both returned a proxy CONNECT403. This is an access-policy observation,
not evidence the links or weights are unavailable to everyone. Direct networking
and alternate proxies were not attempted. The GitHub releases page returned200
and explicitly displayed no releases; the reviewed open/closed issue index did not
establish an alternate checkpoint artifact. We did not search every possible mirror.

## Training-history evidence changes the prerequisite

In [issue26, collaborator response](https://github.com/WHU-USI3DV/CoFiI2P/issues/26#issuecomment-3748667059),
the author acknowledged train-only translation suppression and said they would
add training translations and retrain. In the currently pinned source
`ed90edf51cca540ee76152d91133c7e79b63256c`, `data/kitti.py:225` has that suppression
line **commented out**. The current code samples translations in training as well
as evaluation. Do not incorrectly report the historical defect as active in this
source revision. Options activate x/z translations in camera coordinates; issue/
README prose mentions x/y. Record actual axes from code rather than assuming the
terminology is interchangeable.

This correction does not establish which training recipe produced the distributed
`cofii2p_kitti.t7`. The checkpoint link has neither a verified digest nor a pinned
training commit here. Do not infer retraining from the current code, remove test
translations to improve numbers, or report legacy rotation-focused results as
unknown-extrinsic translation recovery.

In [issue25](https://github.com/WHU-USI3DV/CoFiI2P/issues/25#issuecomment-3489897936),
a collaborator wrote: “we did not distinguish validation and test set” and advised
choosing the least validation loss. That limits interpretation of the official
benchmark as an untouched test. It does not prove overlap with Calibrex's six spent
raw drives, and it does not identify a particular checkpoint's selection history.
Retain unknown checkpoint/raw-frame provenance until established independently.

[Issue20](https://github.com/WHU-USI3DV/CoFiI2P/issues/20) also contains a collaborator
report that coarse-only registration can outperform fine registration. This is
context for the prior patch-layout audit, not proof that the axis discrepancy
caused those reported numbers. Do not choose coarse/fine/legacy decoding based on
our reference errors. A comparison must be declared before the bounded pilot.

## Image preparation: evidence and remaining gap

CoFiI2P README points KITTI processing to CorrI2P; a collaborator confirmed the
prepared CorrI2P data could be used directly in
[issue1](https://github.com/WHU-USI3DV/CoFiI2P/issues/1#issuecomment-2147182183).
CorrI2P acknowledges DeepI2P. The pinned DeepI2P `data/kitti_png_to_npy.py:78–86`
explicitly reverses OpenCV BGR to RGB before its optional NPY write. Its loader
crops source top rows and adjusts K before resize. CoFiI2P's nuScenes builder
instead opens images with PIL and crops/adjusts K before writing its prepared
images. Those sources support RGB ancestry but do not prove how the separately
distributed KITTI `img_P2/img_P3` and `K_P2/K_P3` files were generated.

The pinned CoFiI2P KITTI loader reads those NPYs, then resizes/crops using nominal
intrinsics scaling. It does not perform the options' top crop in `__getitem__`.
Applying top crop50 to already cropped files would therefore be wrong. The local
boundary currently accepts original images with explicit top crop and BGR decode;
its actual-scale/pixel-center K is mathematically declared but not checkpoint
parity. Preserve that frozen implementation as geometry evidence. Establish the
checkpoint recipe before adding or selecting a provider-specific RGB/K policy.

## Execution handoff

The collaborator's [hardware answer](https://github.com/WHU-USI3DV/CoFiI2P/issues/7#issuecomment-2308753542)
reports about3GB KITTI inference GPU memory. This is not a measured bound here.
Use a GPU with headroom (e.g.8GB or more as a planning allowance), measure actual
peak memory and lock dependencies. The README example is CUDA11.6/PyTorch1.13.1/
Open3D0.17; it is not a complete tested lockfile for the current CPU/Python3.12
environment. No large environment installation or unverified CPU port was attempted.

To restart useful model work, provide the GPU workspace/access route and the
official KITTI checkpoint location. Then:

1. Record checkpoint SHA256, source/terms, training and validation selection recipe.
   Verify image preparation against a provenance-pinned example; keep any remaining
   uncertainty explicit rather than calling a guessed policy official parity.
2. Run a bounded single-input model smoke through the explicit initializer-only
   boundary; bypass the official reference-aligned/label-producing dataset loader.
   Validate output shape, finite support, point lineage and resource limits. This
   still does not establish calibration accuracy.
3. Freeze the already specified54-call spent-data pilot, its decoding convention,
   aggregation and fit-only acceptance. Retain all failures; reference scoring
   happens after inference. No fresh holdout or new recording is needed.

GPU availability alone does not establish checkpoint readiness, and a checkpoint
alone does not establish a working runtime. The previous MI6/12 baseline and all
negative results remain unchanged; no inference or SOTA result was produced here.
