# Direct RGB–LiDAR provider readiness review

Reviewed2026-10-30. [Machine report](../assets/direct_provider_review_20261030/review.json)
and [schema](../assets/direct_provider_review_20261030/review.schema.json) record
exact repository commits, local source paths/hashes and unresolved prerequisites.
Official source code was inspected in isolated /workspace/work checkouts, not
copied into src/calibrex or executed. No inference, checkpoint download, accuracy
claim or fresh dataset use occurred.

| Candidate | Verified official source | Mechanism | Current decision |
| --- | --- | --- | --- |
| CoFiI2P | [WHU-USI3DV/CoFiI2P](https://github.com/WHU-USI3DV/CoFiI2P/tree/ed90edf51cca540ee76152d91133c7e79b63256c) | Learned coarse/fine point–pixel correspondences, PnP-RANSAC | Conditional first implementation candidate; blocked prerequisites |
| CorrI2P | [rsy6318/CorrI2P](https://github.com/rsy6318/CorrI2P/tree/4f4a7550543ed4fb0e77dff9b0aecad0e42ad227) | Dense cross-modal correspondences | Secondary reference; not runnable as supplied here |
| DeepI2P | [lijx10/DeepI2P](https://github.com/lijx10/DeepI2P/tree/cd21389db9ddcccfd2cea306e54342562681b20f) | Frustum/fine classification with registration | Classification comparator, not selected for next pilot |

Previously rejected SuperGlue/I2PNet correspondence exports are not reopened.
A guessed2D3D-MATR repository URL did not resolve; its failed lookup is retained,
and no official-source or feasibility assertion is made for that method.

## Source evidence and environment

Current runtime is torch2.4.0+cpu with no CUDA and no Open3D, about1.3GB free disk
at review time. CoFiI2P README describes CUDA11.6/PyTorch1.13.1/Open3D0.17; its
evaluation calls model.cuda() and moves inputs to CUDA. Its point pyramid imports
Open3D torch neighbor layers. This blocks the unmodified official evaluation in
the current environment, not proof that a future CPU port is impossible.

CorrI2P evaluation likewise uses CUDA and references a locally trained .t7 path.
README says prepared dataset download is temporarily unavailable. The review did
not establish a usable official weight artifact; this is not a claim none exists.
DeepI2P has a root MIT license, but its index_max setup instantiates CUDAExtension,
and its documented coarse path also uses Ceres. Its README's listed weight links
are Oxford models; a suitable KITTI artifact was not established here.

No root LICENSE/COPYING is tracked in the pinned CoFiI2P/CorrI2P trees. Public
availability is not assumed to authorize redistribution; terms need clarification
before packaging or copying their code. This review makes no broad legal verdict.
Checkpoint bytes/digests and their terms remain unverified for all candidates;
links in READMEs are not reproducible checkpoint identities. No drive/archive or
large prepared dataset was downloaded. CoFiI2P's102GB prepared KITTI download is
not required for a future local single-batch adapter and is not attempted.

All three code paths use KITTI sequence0–8 for training and9/10 for validation
in their source loaders. Exact checkpoint training provenance and mapping to our
raw-frame identities remain unverified; no train/test isolation claim is made.
README-reported accuracy uses different protocols, datasets and gates and is not
compared numerically to Calibrex's .5deg/.20m gate.

## Critical input boundary

CoFiI2P's data/kitti.py applies dataset calibration P_Tr to place the cloud in camera
coordinates, then applies a random transform P. That is its declared benchmark
construction, not evidence of misconduct. It is nevertheless incompatible with
blindly invoking the loader for an unknown-extrinsic calibration comparison.
Its model has separate test inference; an adapter must bypass dataset calibration
and ground-truth correspondence construction entirely, providing only explicit
initial T_camera_lidar, original image/intrinsics and LiDAR/intensity/normals.

Transform points/normals with the initializer, preserve point identities, map
cropped/resized feature pixels back to the source camera, and compose a predicted
camera-from-initial-camera correction with the initializer. Input provenance must
exclude the reference artifact. Ground-truth labels, reference point filtering or
dataset calibration must never enter inference preparation. Synthetic transform/
crop tests and a changed-reference invariance test are required before real input.

## Concrete conditional pilot

Candidate priority is CoFiI2P because it exposes coarse/fine correspondences rather
than the already rejected diffuse I2PNet export. This is a mechanism-based choice,
not a predicted accuracy win. Before any inference, require:

1. Suitable GPU runtime or a separately validated CPU port; lock all dependencies
   and memory/time limits. Do not silently replace CUDA neighbor operations.
2. Official checkpoint bytes with SHA256, source link and applicable terms; resolve
   code redistribution terms if packaging is intended. Keep external code isolated.
3. Camera/point preprocessing adapter with initializer-only boundary and exact
   coordinate/crop/normal/determinism tests; no official reference-aligned loader.
4. Checkpoint training/frame provenance and scope designation. Unknown overlap
   permits only explicitly overlap-unknown development evidence, never holdout proof.

Then freeze one bounded spent-data pilot on the six existing drives/18 original
frames and positive/negative/reference starts. Retain all54 provider invocations,
per-frame support and failure denominators. Use the existing provider-neutral
export/quality/PnP boundaries; do not relabel direct network pose as correspondence
success. Define fit-only acceptance and spatial holdout quality before running;
reference geometry scores afterward. Report source/sampled point counts, crop,
geometry support, exact artifacts and all failed poses. Historical joint MI6/12
is contextual until multi-frame aggregation/input matching is frozen. No fresh
holdouts or new recordings are needed for this pilot.

No candidate is executable unmodified under the present verified resource/checkpoint
conditions. The next concrete work is satisfying these prerequisites, not another
surface-mask variant or a runtime-as-accuracy claim. No installation or execution
was performed just to hide a known dependency blocker.

Follow-up: [execution/checkpoint readiness](cofii2p_runtime_readiness.md) records
actual proxy403 weight access, CPU-only runtime, historical training-translation
correction and unresolved checkpoint/image provenance. Inference remains deferred.
