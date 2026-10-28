"""Research-only bounded nearest-ridge distance field for camera-LiDAR alignment."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from calibrex.evaluation import camera_lidar_edge as edge

SETTINGS = {'gradient_threshold': 0.2, 'distance_sigma_px': 2.0, 'maximum_distance_px': 6.0}


def prepare_nearest_edge(
    gray: NDArray[np.uint8], options: edge.EdgeAlignmentOptions,
) -> tuple[NDArray[np.float32], int, dict[str, int | float]]:
    """Use gradient-normal ridge suppression and a bounded distance reward.

    Distances are measured in downscaled pixels. Unmatched points contribute zero,
    retaining their weight in the objective denominator; there is no inlier filtering.
    """
    factor = max(1, math.ceil(gray.shape[1] / options.image_max_width))
    image = gray.astype(np.float32)
    if factor > 1:
        height, width = image.shape[0] // factor * factor, image.shape[1] // factor * factor
        image = image[:height, :width].reshape(height // factor, factor, width // factor, factor)
        image = image.mean(axis=(1, 3))
    image = ndimage.gaussian_filter(image, 1.0)
    gx, gy = ndimage.sobel(image, axis=1), ndimage.sobel(image, axis=0)
    magnitude = np.hypot(gx, gy)
    scale = float(np.percentile(magnitude, 99))
    if scale < edge._MIN_IMAGE_GRADIENT:
        raise ValueError('nearest-edge input has no image edges')
    ux, uy = np.zeros_like(gx), np.zeros_like(gy)
    np.divide(gx, magnitude, out=ux, where=magnitude > 0)
    np.divide(gy, magnitude, out=uy, where=magnitude > 0)
    y, x = np.indices(image.shape)
    forward = ndimage.map_coordinates(magnitude, [y + uy, x + ux], order=1, mode='nearest')
    backward = ndimage.map_coordinates(magnitude, [y - uy, x - ux], order=1, mode='nearest')
    ridges = ((magnitude >= SETTINGS['gradient_threshold'] * scale)
              & (magnitude >= forward) & (magnitude >= backward))
    ridges[[0, -1], :] = False
    ridges[:, [0, -1]] = False
    if not ridges.any():
        raise ValueError('nearest-edge input has no retained ridges')
    distance = ndimage.distance_transform_edt(~ridges)
    response = np.exp(-0.5 * (distance / SETTINGS['distance_sigma_px']) ** 2)
    response[distance > SETTINGS['maximum_distance_px']] = 0
    response = response.astype(np.float32)
    response.setflags(write=False)
    return response, factor, {'ridge_pixels': int(ridges.sum()), 'pixels': ridges.size,
                              'gradient_scale': scale}
