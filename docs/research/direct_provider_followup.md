# Bounded follow-up: NCLR, FreeReg and Free-FreeReg

Subsequent user constraint: no GPU is available. The [CoFiI2P CPU probe](cofii2p_cpu_probe.md)
completed a full test-mode forward at standard point count with random weights.
This supersedes requesting a GPU as the next action; prioritize accessible official
weights and CPU inference validation. Findings about the three candidates below
remain a record of their own reviewed paths.

Reviewed2026-10-30 after [CoFiI2P execution blockers](cofii2p_runtime_readiness.md).
[Machine review](../assets/direct_provider_followup_20261030/review.json) and
[schema](../assets/direct_provider_followup_20261030/review.schema.json) pin three
official source revisions and18 relevant source files. Selection follows explicit
links in the previously reviewed authors' READMEs, not guessed repository names.

**No newly reviewed provider is ready for inference in the current environment.**
Close this bounded search and prioritize an execution environment plus actual
model artifacts. Do not build another unexercised adapter or continue collecting
candidate names as a substitute for calibration evidence.

| Candidate | Pinned revision | What is actually available | Decision |
| --- | --- | --- | --- |
| [NCLR](https://github.com/Eaphan/NCLR) | `40de8eea555ffce1d47374fdac8c0b2659455a4d` | Self-supervised representation-learning code with neural-calibration components; README weight links are `todo` | No ready pretrained calibration provider established |
| [FreeReg](https://github.com/WHU-USI3DV/FreeReg) | `d13a45f5fa0f1568a311d6e336bcbeaf3c99cd9b` | Diffusion/monocular-depth/geometric feature registration; several external weights and dependencies | Distinct mechanism, no feasible current execution path verified |
| [Free-FreeReg branch](https://github.com/WHU-USI3DV/FreeReg/tree/FFreeReg) | `0b397094feab8060f67afbc78471d4c95477d96d` | DepthPro plus YOHO/FCGF geometric registration without a virtual depth camera | Simpler, still requires weights and a CUDA-oriented implementation |

## NCLR is not a ready calibration checkpoint

CorrI2P's README links NCLR as newer work. NCLR's main instructions concern point
representation pretraining and semantic-segmentation/object-detection transfer.
Its README has placeholder pretrained-model links and an unchecked weight-release
TODO. No checkpoint files are tracked in the reviewed tree.

There **is** calibration code: `bevlab/models.py` constructs soft image matches
and invokes differentiable PnP. Thus it would be incorrect to say NCLR contains no
registration mechanism. But the presence of PnP inside a learning objective does
not establish a callable pretrained unknown-extrinsic inference provider. Its
training dataset reads sensor calibration and builds projections; a no-reference
inference boundary still needs separate establishment. No adaptation or training
was attempted without a usable model artifact.

Root code is Apache2.0; dependencies/checkpoints have their own terms. Requirements
include TorchSparse, torch-scatter/geometric and the sparse-convolution ecosystem.
No compatible runtime or CPU inference port was verified. Reported downstream
segmentation scores are not camera–LiDAR calibration accuracy.

## FreeReg removes task-specific training, not pretrained models

The default branch extracts diffusion and geometric features, with PnP or3D
registration according to configuration. README lists Stable Diffusion1.5, a
fine-tuned depth ControlNet, ZoeDepth and FCGF weights. Which subset is needed
depends on the chosen feature configuration. No weight bytes/digests were acquired.

Some source components explicitly fall back to CPU. Do not claim that the entire
repository intrinsically requires CUDA. However, the complete pipeline has not
been shown to run on this CPU environment with missing dependencies/checkpoints
and roughly1.3GB available disk at the initial check. A device fallback alone does
not establish feasibility. Large checkpoints/installations were not downloaded
merely to reach a predictable resource blocker.

Diffusion features use a virtual depth camera with sufficient overlap; its pose
must be an explicit initializer, never the unknown target extrinsic. The demo uses
identity projection and separately loads evaluation-only depth/pose. Those demo
evaluation paths must be bypassed for a no-reference adapter; their presence is
not a claim that every registration operation requires truth. KITTI-DC benchmark
claims are not directly interchangeable with the existing raw-drive protocol.

No root LICENSE/COPYING is tracked at this revision; this records unresolved
packaging terms, not a blanket legal conclusion. Bundled components and external
weights require separate terms review. No code was copied into Calibrex core.

## Free-FreeReg is a geometric alternative, not a weight-free fallback

The official README recommends the `FFreeReg` branch for simpler use. It removes
diffusion features and the virtual camera, reconstructing image3D with DepthPro,
matching YOHO/FCGF descriptors and estimating pose through combined3D/reprojection
registration. It still requires DepthPro, a YOHO backbone and YOHO PartI weights.
Its demo explicitly requests `device='cuda'` for both learned components, and FCGF
imports MinkowskiEngine. Merely changing that string would not validate a port.

The demo estimates camera intrinsics, rescales reconstructed image geometry from
cloud statistics and performs random subsampling. A Calibrex calibration comparison
must declare measured-versus-estimated intrinsics, scale treatment, seeds and point
lineage before any fit. Replacing DepthPro with our already rejected stereo
surfaces and calling the result Free-FreeReg would be an unvalidated new variant,
not a reproduction. No such substitution was made.

There is no tracked root license; the DepthPro subdirectory has Apple terms, which
do not automatically license the whole repository or all external checkpoint
artifacts. No weights or bundled model code were executed or redistributed here.

## Outcome and next action

The three checkouts occupy about42MB and include their repository-bundled demo
assets. Those images/clouds were not used for calibration or evaluation. No new
KITTI archive/holdout, checkpoint, dependency installation, model call or accuracy
fit was performed. Core/defaults and the MI6/12 baseline remain unchanged.

The practical next task remains the [CoFiI2P execution handoff](cofii2p_runtime_readiness.md#execution-handoff):
provide an accessible GPU workspace and official KITTI checkpoint, establish its
training/image-preparation provenance, then perform one bounded no-reference
inference smoke before freezing the spent-data pilot. Unknown training overlap
stays explicitly development-only. Other candidates can be reopened if concrete
usable artifacts/runtime become available; no further speculative adapters now.
