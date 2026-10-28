"""Fixed graph penalty for distorted direction and spacing of image correspondences."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from tools import kitti_camera_lidar_contour_coherence as contour
from tools import kitti_camera_lidar_match_diagnostics as diagnostics

from calibrex.evaluation import camera_lidar_edge as edge

SETTINGS = {"pixel_tolerance": 1.0, "relative_tolerance": 0.25, "same_component_required": True}


def geometry_map(response: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """Keep the existing labels and additionally retain nearest ridge coordinates."""
    labels, count = contour.contour_map(response)
    indices = ndimage.distance_transform_edt(
        response != 1, return_distances=False, return_indices=True
    )
    return labels, count, np.moveaxis(indices[::-1], 0, -1)


def geometry_factors(
    pixels: np.ndarray, matches: np.ndarray, labels: np.ndarray, links: np.ndarray
) -> np.ndarray:
    """Compare each projected neighbor displacement with its matched ridge displacement."""
    degree = np.bincount(links.ravel(), minlength=len(labels))
    valid = (labels[links[:, 0]] > 0) & (labels[links[:, 0]] == labels[links[:, 1]])
    pair_scores = np.zeros(len(links))
    i, j = links[valid].T
    delta = pixels[j] - pixels[i]
    matched_delta = matches[j] - matches[i]
    tolerance = SETTINGS["pixel_tolerance"] + SETTINGS["relative_tolerance"] * np.linalg.norm(
        delta, axis=1
    )
    pair_scores[valid] = np.exp(-0.5 * np.sum((matched_delta - delta) ** 2, axis=1) / tolerance**2)
    sums = np.bincount(links.ravel(), weights=np.repeat(pair_scores, 2), minlength=len(labels))
    result = np.ones(len(labels))
    np.divide(sums, degree, out=result, where=degree > 0)
    return result


def score_geometry(
    objective: edge.EdgeObjective,
    theta: np.ndarray,
    label_map: np.ndarray,
    links: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[float, float, dict[str, Any]]:
    """Retain the original fixed support and apply the geometry-weighted graph penalty."""
    pixels, values = diagnostics.contributions(objective, theta)
    h, w = label_map.shape
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 0] < w - 1)
        & (pixels[:, 1] < h - 1)
    )
    labels = np.zeros(len(pixels), dtype=np.int64)
    matches = np.zeros_like(pixels)
    rounded = np.rint(pixels[inside]).astype(int)
    labels[inside] = label_map[rounded[:, 1], rounded[:, 0]]
    matches[inside] = coordinates[rounded[:, 1], rounded[:, 0]]
    factors = geometry_factors(pixels, matches, labels, links)
    component = contour.compatibility(labels, links)
    return (
        float(values.sum()),
        float(values @ factors),
        {
            "component_only_score": float(values @ component),
            "matched_points": int(np.count_nonzero(labels)),
            "mean_factor_linked_points": float(factors[np.unique(links)].mean())
            if links.size
            else None,
        },
    )
