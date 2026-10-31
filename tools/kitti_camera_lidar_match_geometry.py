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


class GeometryObjective(edge.EdgeObjective):
    """Batched form of the fixed-pose geometry score, with frame-local graphs."""

    def __init__(
        self,
        frames: list[edge.PreparedFrame],
        camera: edge.EdgeCamera,
        start: np.ndarray,
        maps: list[tuple[np.ndarray, int, np.ndarray]],
        graphs: list[np.ndarray],
    ) -> None:
        super().__init__(frames, camera, start)
        self._labels = np.concatenate([m[0].ravel() for m in maps])
        self._coordinates = np.concatenate([m[2].reshape(-1, 2) for m in maps])
        offsets = np.cumsum([0] + [len(f.points) for f in frames])
        self._links = np.concatenate(
            [g + offset for g, offset in zip(graphs, offsets[:-1], strict=True)]
        )

    def score(self, theta: np.ndarray) -> float:
        pixels, values = diagnostics.contributions(self, theta, verify=False)
        inside = (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 0] < self._u_limit)
            & (pixels[:, 1] < self._v_limit)
        )
        rounded = np.rint(pixels[inside]).astype(int)
        index = self._offset[inside] + rounded[:, 1] * self._stride[inside] + rounded[:, 0]
        labels = np.zeros(len(pixels), dtype=np.int64)
        matches = np.zeros_like(pixels)
        labels[inside] = self._labels[index]
        matches[inside] = self._coordinates[index]
        return float(values @ geometry_factors(pixels, matches, labels, self._links))
