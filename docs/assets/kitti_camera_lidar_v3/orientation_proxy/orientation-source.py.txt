"""Experimental image-gradient compatibility with projected LiDAR azimuth motion.

Azimuth motion is a scan-direction proxy, not a recovered boundary normal.
Only organized, undistorted pinhole inputs are supported by this diagnostic.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from calibrex.evaluation import camera_lidar_edge as edge


@dataclass(frozen=True)
class OrientationFeatures:
    raw_response: NDArray[np.float32]
    tensor: NDArray[np.float32]  # xx, xy, yy; trace is normalized gradient confidence.


def prepare_orientation_image(
    gray: NDArray[np.uint8], options: edge.EdgeAlignmentOptions,
) -> tuple[OrientationFeatures, int]:
    """Use the scalar objective's image scale, blur, spill and gradient normalization."""
    response = edge._image_edge_response_with_gradient(
        gray, max_width=options.image_max_width, gamma=options.spill_gamma,
        radius=options.spill_radius_px, background_sigma=None,
    )
    if response is None:
        raise ValueError("orientation input has no image edges")
    raw, factor, gradient_scale = response
    image = gray.astype(np.float32)
    if factor > 1:
        height, width = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
        image = image[:height, :width].reshape(height // factor, factor, width // factor, factor)
        image = image.mean(axis=(1, 3))
    image = ndimage.gaussian_filter(image, 1.0)
    gx, gy = ndimage.sobel(image, axis=1), ndimage.sobel(image, axis=0)
    magnitude2 = gx * gx + gy * gy
    tensor = np.stack([gx * gx, gx * gy, gy * gy], axis=-1)
    np.divide(tensor, magnitude2[..., None], out=tensor, where=magnitude2[..., None] > 0)
    confidence = np.minimum(np.hypot(gx, gy) / gradient_scale, 1.0)
    tensor *= confidence[..., None]
    raw.setflags(write=False)
    tensor.setflags(write=False)
    return OrientationFeatures(raw_response=raw, tensor=tensor), factor


class OrientationObjective(edge.EdgeObjective):
    """Attenuate nonnegative foreground response; keep the original background penalty."""

    def __init__(
        self, frames: Sequence[edge.PreparedFrame], camera: edge.EdgeCamera, start: np.ndarray,
        features: Sequence[OrientationFeatures],
    ) -> None:
        if camera.distortion_model != "none":
            raise ValueError("orientation probe requires undistorted pinhole images")
        if len(features) != len(frames) or not frames or any(f.rings <= 0 for f in frames):
            raise ValueError("orientation probe requires paired organized frames/features")
        for frame, feature in zip(frames, features, strict=True):
            raw, tensor = feature.raw_response, feature.tensor
            if (raw.shape != frame.edge.shape or tensor.shape != (*frame.edge.shape, 3)
                    or not np.all(np.isfinite(raw)) or np.any(raw < 0)
                    or not np.all(np.isfinite(tensor))):
                raise ValueError("invalid orientation image features")
            xx, xy, yy = np.moveaxis(tensor, -1, 0)
            if np.any(xx < 0) or np.any(yy < 0) or np.any(xx * yy - xy * xy < -1e-6):
                raise ValueError("orientation tensor must be positive semidefinite")
            if np.any(xx + yy > 1 + 1e-6):
                raise ValueError("orientation tensor confidence must be at most one")
        super().__init__(frames, camera, start)
        self._orientation = np.concatenate([
            np.concatenate([f.raw_response[..., None], f.tensor], axis=-1).reshape(-1, 4)
            for f in features
        ])
        tangents = []
        for frame in frames:
            tangent = np.zeros_like(frame.points, dtype=np.float64)
            tangent[:, 0], tangent[:, 1] = -frame.points[:, 1], frame.points[:, 0]
            tangents.append(tangent @ start[:3, :3].T)
        self._azimuth = np.concatenate(tangents).astype(np.float32)

    def score(self, theta: np.ndarray) -> float:
        """Score fixed support; unknown image/projected directions are neutral."""
        rotation = edge._rotation_from_vector(np.radians(theta[:3])).astype(np.float32)
        moved = self._points @ rotation.T
        moved += self._start_translation + theta[3:].astype(np.float32)
        u, v, valid = self.camera.project(moved)
        u = (u - self._shift) * self._inverse_scale
        v = (v - self._shift) * self._inverse_scale
        inside = valid & (u >= 0) & (v >= 0) & (u < self._u_limit) & (v < self._v_limit)
        if not np.any(inside):
            return 0.0
        u, v = u[inside], v[inside]
        left, top = u.astype(np.int64), v.astype(np.int64)
        du, dv = u - left, v - top
        stride = self._stride[inside]
        index = self._offset[inside] + top * stride + left
        indices = (index, index + stride, index + 1, index + stride + 1)
        coefficients = ((1 - du) * (1 - dv), (1 - du) * dv, du * (1 - dv), du * dv)
        response = sum(c * self._flat[i] for c, i in zip(coefficients, indices, strict=True))
        values = sum(c[:, None] * self._orientation[i]
                     for c, i in zip(coefficients, indices, strict=True))
        direction = (self._azimuth @ rotation.T)[inside]
        points = moved[inside]
        # The common depth-squared/image-scale factor cancels from the cosine ratio.
        pu = self.camera.fx * (direction[:, 0] * points[:, 2] - points[:, 0] * direction[:, 2])
        pv = self.camera.fy * (direction[:, 1] * points[:, 2] - points[:, 1] * direction[:, 2])
        raw, xx, xy, yy = values.T
        denominator = (pu * pu + pv * pv) * (xx + yy)
        numerator = pu * pu * xx + 2 * pu * pv * xy + pv * pv * yy
        compatibility = np.ones_like(denominator)
        np.divide(numerator, denominator, out=compatibility, where=denominator > 0)
        np.clip(compatibility, 0, 1, out=compatibility)
        response -= raw * np.clip(xx + yy, 0, 1) * (1 - compatibility)
        return float(response @ self._weights[inside]) / self._total_weight
