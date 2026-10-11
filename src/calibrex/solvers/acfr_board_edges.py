"""Adapter-side board-boundary extractor for ACFR's VLP-16 feature rows.

ACFR's ``poses.csv`` (``acfr_vlp_poses_v1``) holds 19 rows per capture. Rows 0/1
and 6/7 are the camera and LiDAR board centre and normal (consumed by the plane,
point+plane and Horn rows). Rows 2-5 are the four camera board corners and rows
8-11 the four LiDAR board corners, all in millimetres. Rows 12-18 are not consumed
(they appear to hold corner angles, LiDAR edge lengths, the range and the capture
index).

The corners carry no declared identity, so this adapter derives one and records
every step instead of trusting the row order:

1. **Traversal.** Each corner loop is ordered counter-clockwise about its own
   oriented board normal. The normals are oriented correspondences (preserved
   by the plane rows), so the two loops then share a handedness.
2. **Board symmetry.** A rectangle maps to itself under a 180° rotation about
   its normal, and the remaining cyclic shifts of the LiDAR loop are the only
   labelings left. The shift is chosen by agreement of the oriented edge
   directions under a *prior* rotation. The caller passes the train-split
   plane-only rotation, so the choice uses no line information to fit the prior
   and is a discrete choice between labelings 90° or 180° apart.
3. **Verification.** The chosen pairing must match edge lengths: a paired edge
   whose camera and LiDAR lengths differ by more than
   ``max_edge_length_mismatch_fraction`` rejects the capture (the default,
   0.15, is half the 28 % short/long difference of ACFR's 610 x 850 mm board).

Each boundary line goes through the midpoint of its two corners along the
traversal direction. Rejected captures are reported, never silently dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers.planar_board_lidar_camera_solver import OrientedPlane
from calibrex.solvers.planar_board_line_plane_solver import (
    BoardBoundaryCorrespondence,
    LinePlaneBoardObservation,
    OrientedLine3D,
)

FloatArray: TypeAlias = NDArray[np.float64]

ACFR_BOARD_EDGE_EXTRACTOR = "calibrex.acfr_vlp_board_corner_edges/v0.1"
ACFR_ROWS_PER_CAPTURE = 19
ACFR_CAMERA_CORNER_ROWS = (2, 3, 4, 5)
ACFR_LIDAR_CORNER_ROWS = (8, 9, 10, 11)

AcfrBoardEdgeStatus = Literal["matched", "degenerate_corners", "edge_length_mismatch"]


@dataclass(frozen=True)
class AcfrBoardEdgeCapture:
    """How one capture's corners were turned into boundary correspondences."""

    frame_id: str
    status: AcfrBoardEdgeStatus
    lidar_shift: int | None = None
    symmetric_shift_chosen: bool | None = None
    direction_agreement: float | None = None
    runner_up_direction_agreement: float | None = None
    max_edge_length_mismatch_fraction: float | None = None


@dataclass(frozen=True)
class AcfrBoardEdgeExtraction:
    """Line+plane observations and the per-capture extraction record."""

    observations: tuple[LinePlaneBoardObservation, ...]
    captures: tuple[AcfrBoardEdgeCapture, ...]
    max_edge_length_mismatch_fraction: float

    @property
    def rejected_frame_ids(self) -> tuple[str, ...]:
        return tuple(item.frame_id for item in self.captures if item.status != "matched")

    @property
    def symmetric_frame_ids(self) -> tuple[str, ...]:
        """Captures whose LiDAR labeling is the board's 180° symmetric one."""

        return tuple(item.frame_id for item in self.captures if item.symmetric_shift_chosen)

    def as_dict(self) -> dict[str, object]:
        return {
            "extractor": ACFR_BOARD_EDGE_EXTRACTOR,
            "source_rows": {
                "camera_corners": list(ACFR_CAMERA_CORNER_ROWS),
                "lidar_corners": list(ACFR_LIDAR_CORNER_ROWS),
                "rows_per_capture": ACFR_ROWS_PER_CAPTURE,
            },
            "traversal_policy": "counter-clockwise about each oriented board normal",
            "symmetry_policy": (
                "LiDAR cyclic shift chosen by oriented edge-direction agreement under "
                "the train-split plane-only rotation"
            ),
            "max_edge_length_mismatch_fraction": self.max_edge_length_mismatch_fraction,
            "matched_captures": len(self.observations),
            "rejected_frame_ids": list(self.rejected_frame_ids),
            "symmetric_frame_ids": list(self.symmetric_frame_ids),
            "captures": [item.__dict__ for item in self.captures],
        }


def extract_acfr_vlp_board_edges(
    rows: Sequence[Sequence[float]],
    prior_rotation: SE3,
    *,
    max_edge_length_mismatch_fraction: float = 0.15,
    min_edge_length_m: float = 0.05,
) -> AcfrBoardEdgeExtraction:
    """Build line+plane observations from ACFR rows; see the module docstring.

    Only the rotation of ``prior_rotation`` (``T_camera_lidar``) is used.
    """

    if len(rows) == 0 or len(rows) % ACFR_ROWS_PER_CAPTURE != 0:
        raise ValueError("ACFR poses must contain complete 19-row captures")
    if not 0.0 < max_edge_length_mismatch_fraction < 1.0:
        raise ValueError("max_edge_length_mismatch_fraction must lie in (0, 1)")
    rotation = _rotation_matrix(prior_rotation)
    table = np.asarray(rows, dtype=float).reshape(-1, ACFR_ROWS_PER_CAPTURE, 3)
    observations: list[LinePlaneBoardObservation] = []
    captures: list[AcfrBoardEdgeCapture] = []
    for index, capture in enumerate(table):
        frame_id = f"acfr-vlp-{index + 1:03d}"
        camera_normal = capture[1]
        lidar_normal = capture[7]
        camera_loop = _counter_clockwise(
            capture[list(ACFR_CAMERA_CORNER_ROWS)] / 1000.0, camera_normal
        )
        lidar_loop = _counter_clockwise(
            capture[list(ACFR_LIDAR_CORNER_ROWS)] / 1000.0, lidar_normal
        )
        if camera_loop is None or lidar_loop is None:
            captures.append(AcfrBoardEdgeCapture(frame_id, "degenerate_corners"))
            continue
        camera_edges = _edges(camera_loop)
        lidar_edges = _edges(lidar_loop)
        if min(_lengths(camera_edges) + _lengths(lidar_edges)) < min_edge_length_m:
            captures.append(AcfrBoardEdgeCapture(frame_id, "degenerate_corners"))
            continue
        scores = [
            sum(
                float(
                    _unit(camera_edges[k][1]) @ (rotation @ _unit(lidar_edges[(k + shift) % 4][1]))
                )
                for k in range(4)
            )
            for shift in range(4)
        ]
        shift = int(np.argmax(scores))
        runner_up = max(score for other, score in enumerate(scores) if other != shift)
        paired = [lidar_edges[(k + shift) % 4] for k in range(4)]
        mismatch = max(
            abs(camera - lidar) / camera
            for camera, lidar in zip(_lengths(camera_edges), _lengths(paired), strict=True)
        )
        status: AcfrBoardEdgeStatus = (
            "matched" if mismatch <= max_edge_length_mismatch_fraction else "edge_length_mismatch"
        )
        captures.append(
            AcfrBoardEdgeCapture(
                frame_id,
                status,
                lidar_shift=shift,
                symmetric_shift_chosen=shift == 2,
                direction_agreement=scores[shift],
                runner_up_direction_agreement=runner_up,
                max_edge_length_mismatch_fraction=mismatch,
            )
        )
        if status != "matched":
            continue
        observations.append(
            LinePlaneBoardObservation(
                frame_id=frame_id,
                camera_plane=_plane(capture[1], capture[0] / 1000.0),
                lidar_plane=_plane(capture[7], capture[6] / 1000.0),
                boundaries=tuple(
                    BoardBoundaryCorrespondence(
                        f"edge-{k}",
                        _line(camera_edges[k]),
                        _line(paired[k]),
                    )
                    for k in range(4)
                ),
            )
        )
    return AcfrBoardEdgeExtraction(
        tuple(observations), tuple(captures), max_edge_length_mismatch_fraction
    )


def _counter_clockwise(corners: FloatArray, normal: FloatArray) -> FloatArray | None:
    """Return the corner loop ordered counter-clockwise about ``normal``."""

    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 1.0e-9 or not np.all(np.isfinite(corners)):
        return None
    area = sum(
        (np.cross(corners[k], corners[(k + 1) % 4]) for k in range(4)),
        start=np.zeros(3),
    )
    signed = float(area @ normal) / norm
    if abs(signed) <= 1.0e-9:
        return None
    return corners if signed > 0.0 else corners[::-1].copy()


def _edges(loop: FloatArray) -> list[tuple[FloatArray, FloatArray]]:
    """Midpoint and corner-to-corner vector of each edge of a closed loop."""

    return [(0.5 * (loop[k] + loop[(k + 1) % 4]), loop[(k + 1) % 4] - loop[k]) for k in range(4)]


def _lengths(edges: Sequence[tuple[FloatArray, FloatArray]]) -> list[float]:
    return [float(np.linalg.norm(vector)) for _point, vector in edges]


def _unit(vector: FloatArray) -> FloatArray:
    return vector / float(np.linalg.norm(vector))


def _line(edge: tuple[FloatArray, FloatArray]) -> OrientedLine3D:
    point, vector = edge
    return OrientedLine3D(
        (float(point[0]), float(point[1]), float(point[2])),
        (float(vector[0]), float(vector[1]), float(vector[2])),
    )


def _plane(normal: FloatArray, center_m: FloatArray) -> OrientedPlane:
    return OrientedPlane(
        (float(normal[0]), float(normal[1]), float(normal[2])), -float(normal @ center_m)
    )


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
