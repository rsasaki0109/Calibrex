# Published learned stereo runtime

This is a runtime smoke on spent KITTI data, not a calibration accuracy result
or a SOTA claim. It replaces further classical matcher tweaks with a runnable,
externally published learned stereo baseline. The smoke protocol was committed
at `3a9d60e8` before inference.

## Model and exposure

Use [UniMatch](https://github.com/autonomousvision/unimatch/tree/943a99006950ab6c84ea72a72beb147fdd03c19a),
commit `943a99006950ab6c84ea72a72beb147fdd03c19a`, from *Unifying Flow,
Stereo and Depth Estimation* (TPAMI 2023). This is a published baseline, not a
claim that this 2023 model leads the 2026 field.

The official [model zoo](https://github.com/autonomousvision/unimatch/blob/943a99006950ab6c84ea72a72beb147fdd03c19a/MODEL_ZOO.md)
names the selected checkpoint `gmstereo-scale2-regrefine3-sceneflow-2dd12e97.pth`.
Its SHA256 is `2dd12e97b60bb7b91a0a45845a59260f9fdfe1ae9449dc491353c8842080ea6d`.
SceneFlow training is an author declaration, not an independent exposure audit.
We did not select KITTI-finetuned, mixed-data or `resumeflowthings` checkpoints.
All six local comparison recordings remain spent regardless of this choice.

The repository [license](https://github.com/autonomousvision/unimatch/blob/943a99006950ab6c84ea72a72beb147fdd03c19a/LICENSE)
is MIT, copyright 2022 autonomousvision. No separate checkpoint license was
found in the inspected LICENSE, README or model zoo; do not infer an independently
verified weight redistribution grant. Neither third-party source nor weights
are vendored or redistributed here. The adapter uses a local external checkout.

## Frozen smoke

[Plan](../assets/unimatch_runtime_20261029/plan.json): drive 0005, frame 76,
already used in the prior six-drive experiment. Image/calibration hashes are
verified against its existing metadata. Only RGB stereo images and camera
projections enter inference; neither LiDAR nor reference extrinsics enter it.
Images are re-rectified using the existing comparison geometry, padded
symmetrically with replicated pixels to multiples of 32, and cropped back.
No resizing or disparity rescaling. This model consumes RGB, while prior SGBM
consumed grayscale; downstream comparisons must disclose that difference.

Configuration: two scales, upsample factor four, three regression refinements,
attention `self_swin2d_cross_swin1d`, splits `[2,8]`, correlation radii `[-1,4]`,
propagation radii `[-1,1]`; float32 CPU, four PyTorch threads. The checkpoint loads
strictly with `weights_only=True`. Code revision, working-tree cleanliness and
checkpoint hash are checked before loading. The 300-second process budget is
fixed; there is no smaller-resolution retry.

Reproduction (paths refer to local external artifacts):

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 timeout 300 \
  /workspace/work/p0-venv/bin/python tools/unimatch_stereo_smoke.py \
  --source /workspace/work/learned-stereo-20261029/unimatch \
  --checkpoint /workspace/work/learned-stereo-20261029/gmstereo-scale2-regrefine3-sceneflow-2dd12e97.pth \
  --metadata /workspace/work/external-comparison-20261028/preprocessed/matrix-20261028/2011_09_26_0005/frame-76/frontend.json \
  --output /workspace/work/learned-stereo-20261029/smoke
```

CPU PyTorch was installed from `https://download.pytorch.org/whl/cpu` as
`torch==2.4.0+cpu`; torchvision and CUDA packages are unnecessary for this core
model. Runtime and package versions are recorded with the result. The upstream
A100 timing is not a CPU performance expectation. Timing is one cold forward
pass, excluding imports, checkpoint loading and image preparation.

## Next evaluation

Before scoring, freeze a matched six-drive/twelve-seed D2D comparison with the
same central frames, initialization, solver and accuracy gates. Convert learned
disparity into optical depth using the rectified projections, mask out-of-image
correspondences, restore original camera rays, and retain all failures. Compare
with re-rectified SGBM 2/12, not the different three-frame D2D 5/12 denominator.
Finite dense disparity is not evidence of depth accuracy or calibration recovery.

## Observed runtime

[Recorded output](../assets/unimatch_runtime_20261029/runtime.json): strict loading
and inference succeeded on the first attempt. Python 3.12.14, PyTorch 2.4.0+cpu,
NumPy 2.5.4 and OpenCV 4.14.0; CUDA unavailable. The 7,354,416-parameter model
processed 1242×375 pixels, padded to 1248×384, in **17.24 seconds** for one cold
forward. All 465,750 output disparities were finite and positive; this is only a
numerical validity check. Disparity min/median/p90/max were
4.4201/22.1215/51.9678/65.5505 pixels. The output array stays in the local workspace;
its SHA256 is recorded. Upstream emitted a future `torch.meshgrid` indexing
warning, with no execution failure. Three padding/cropping geometry tests pass.
