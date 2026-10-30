# CoFiI2P initializer-only input boundary

Implemented 2026-10-30 after the [provider review](direct_provider_readiness.md).
This is an independent geometry/preprocessing boundary, not an implementation of
CoFiI2P inference. No external model code was copied into the core. No real-data
accuracy, checkpoint parity or SOTA claim follows from these checks.

## Input contract

Run from the repository with its Python environment:

```bash
PYTHONPATH=src:. python tools/cofii2p_input_boundary.py request.json output-directory
```

The request names exactly four SHA256-pinned inputs: original image, cloud NPZ,
camera JSON and explicit initializer JSON. It also requires a frame ID and an
explicit source top-row crop. Unknown request/JSON fields are rejected. See the
[request schema](../assets/cofii2p_input_boundary_20261030/request.schema.json),
[camera schema](../assets/cofii2p_input_boundary_20261030/camera.schema.json) and
[initializer schema](../assets/cofii2p_input_boundary_20261030/initializer.schema.json).
Hashes are checked before image/cloud decoding; existing output directories are
refused. Paths are interpreted relative to the calling working directory unless
absolute. Use absolute paths in reproducible requests.

The initializer supplies a proper rigid `transform_camera_lidar` and a nonempty
provenance declaration. The camera supplies original width/height and a pinhole
intrinsic matrix; rectification must already be established by the caller. The
cloud NPZ contains exactly these arrays:

| Array | Shape | Convention |
| --- | --- | --- |
| `point_lidar_m` | N×3 | Finite LiDAR points in metres |
| `normal_lidar` | N×3 | Finite unit normals in the same LiDAR axes |
| `intensity` | N×1 | Finite intensity; no implicit rescaling |
| `point_id` | N | Unique integer IDs retained in original row order |

There is no problem/reference path or dataset calibration loader. This structural
boundary cannot certify that a caller's declared initializer or supplied normals
were produced without reference access; that upstream provenance remains a
separate requirement. A declaration alone is not training/ground-truth isolation.

## Geometry and image convention

For initializer `(R0,t0)`, output points are `R0 p + t0`, and output normals are
`R0 n`. All original rows/IDs survive; there is no visibility filtering, sampling
or voxelization. A future correction from initial-camera to camera composes on
the left: `T_camera_lidar = Delta @ T_initial_camera_lidar`.

Images decode explicitly as OpenCV BGR, crop the requested top rows, resize to
rounded half dimensions with linear interpolation, then center-crop to 512×160.
Output image is float32 CHW divided by 255. If resize factors are `sx,sy`, top
crop is `h0` and center offsets are `cx,cy`, the source-to-input pixel mapping is:

```text
u_input = sx*u_source + (sx-1)/2 - cx
v_input = sy*v_source + (sy-1)/2 - sy*h0 - cy
K_input = A @ K_source
```

The half-pixel offsets follow OpenCV pixel-center geometry. Actual resize factors
account for rounding. The pinned official loader instead scales intrinsics by a
nominal 0.5 without this pixel-center offset. This difference and BGR decoding
require checkpoint-preprocessing parity review before inference. We do not claim
drop-in official compatibility. Future fine-feature indices also require an
explicit stride/axis conversion before the provided inverse pixel mapping.

The output NPZ, manifest and manifest JSON schema record shapes, input hashes,
initial transform, affine, driver hash and source commit. Schemas validate JSON
structure; runtime checks additionally enforce matrix shapes, rigid transforms,
cloud features/IDs and image dimensions.

## Verification and scope

Four unit tests cover nonidentity point/normal transformation, stable row IDs,
explicit BGR values, reference-file mutation invariance, odd-size/cropped pixel
round-trip and projection, correction composition, reflected-transform rejection,
unknown reference-field rejection and digest mismatch before output creation.
The reference mutation check guards one unrelated reference path; the stronger
structural evidence is the four-file API and absence of a reference loader. It
does not prove arbitrary upstream data provenance.

A frozen synthetic smoke run will retain the request, schemas, prepared manifest
and hashes in `docs/assets/cofii2p_input_boundary_20261030/`; binary inputs/output
remain under `/workspace/work/cofii2p-boundary-smoke-20261030/` with their digests.
This fixture uses three synthetic points and a declared 90-degree rotation plus
translation, not a spent-data calibration fit.

Still required: native voxel/count sampling with point lineage, point-pyramid
and neighborhood construction, fine-feature pixel conversion, verified checkpoint
preprocessing parity, usable GPU or validated CPU port, pinned appropriate weights
and terms, and training overlap disclosure. Only then freeze the planned 54-call
spent-data pilot. The current boundary performs zero model inference.
