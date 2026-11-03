"""Camera-LiDAR extrinsics from a checkerboard: board planes, plane solver, per-axis observability.

A checkerboard held still in front of both sensors gives one plane correspondence per
pose. The camera sees the board's pose from its corners (:func:`detect_board`, PnP on
undistorted corners). The LiDAR sees the board's plane: :func:`extract_board_plane` crops
the scan around the board the camera saw, mapped into the LiDAR frame with the current
extrinsic, and fits a plane with RANSAC. :func:`calibrate_board` solves ``T_camera_lidar``
with the Zhang-Pless plane solver of :mod:`calibrex.solvers.planar_board_lidar_camera_solver`,
re-crops with the solution, and reports each axis with a jackknife (leave-one-pose-out) std.

Planes constrain a translation only along the board normals: boards turned left and right
alone (normals in one plane) leave the translation along the third direction, usually
vertical, unobserved. The translation along such a direction is kept at the prior's value
(so the next crop does not drift off the board) and the axes it touches are reported as not
observed instead of being presented as measured. Tilting the board up and down as well
observes all six axes.

Axes follow ``T_camera_lidar``: roll, pitch and yaw are rotations about the camera's x, y
and z axes (``R = R_estimate R_reference^T`` as a rotation vector), and x, y and z are
the translation in the camera frame.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.solvers.planar_board_lidar_camera_solver import (
    OrientedPlane,
    PlanarBoardLidarCameraSolver,
    PlanarBoardObservation,
    PlanarBoardSolverOptions,
)

if TYPE_CHECKING:
    from calibrex.evaluation.visual_rotation import CameraModel

FloatArray: TypeAlias = NDArray[np.float64]
AXES = ("roll", "pitch", "yaw", "x", "y", "z")
MIN_POSES = 4
"""Fewer board poses than this give no solution (three planes are the bare minimum)."""
TRANSLATION_STD_BOUND_M = 0.02
ROTATION_STD_BOUND_DEG = 0.3
UNOBSERVED_EIGEN_RATIO = 0.01
"""A translation direction with less than this share of the largest normal information."""
UNOBSERVED_AXIS_SHARE = 0.3
"""A translation axis is unobserved when an unobserved direction has this much of it."""
_BOARD_SPEC = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*([0-9]*\.?[0-9]+)\s*$")


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError("board calibration needs OpenCV: pip install 'calibrex[opencv]'") from exc


@dataclass(frozen=True)
class Checkerboard:
    """A checkerboard by its inner corners (``columns`` x ``rows``) and square size."""

    columns: int
    rows: int
    square_m: float

    def __post_init__(self) -> None:
        if self.columns < 2 or self.rows < 2:
            raise ValueError("a checkerboard needs at least 2 x 2 inner corners")
        if not math.isfinite(self.square_m) or self.square_m <= 0.0:
            raise ValueError("the square size must be positive")

    @classmethod
    def parse(cls, text: str) -> Checkerboard:
        """``COLUMNSxROWSxSQUARE_M``, e.g. ``7x5x0.08`` (inner corners, metres)."""

        match = _BOARD_SPEC.match(text)
        if match is None:
            raise ValueError(
                f"board {text!r}: expected COLUMNSxROWSxSQUARE_M with inner corners and the "
                "square size in metres, e.g. 7x5x0.08"
            )
        return cls(int(match.group(1)), int(match.group(2)), float(match.group(3)))

    def object_points(self) -> FloatArray:
        """Inner corners in the board frame, in OpenCV's order (columns fastest), z = 0."""

        return np.array(
            [
                [column * self.square_m, row * self.square_m, 0.0]
                for row in range(self.rows)
                for column in range(self.columns)
            ]
        )

    def squares_extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``((x_min, y_min), (x_max, y_max))`` of the squares in the board frame."""

        s = self.square_m
        return (-s, -s), (self.columns * s, self.rows * s)


@dataclass(frozen=True)
class BoardView:
    """The board's pose in the camera frame: ``p_camera = rotation p_board + translation``."""

    rotation: FloatArray
    translation: FloatArray
    reprojection_px: float

    @property
    def plane(self) -> OrientedPlane:
        normal = self.rotation[:, 2]
        return OrientedPlane(
            (float(normal[0]), float(normal[1]), float(normal[2])),
            -float(normal @ self.translation),
        )


def board_view(
    pixels: NDArray[np.floating], board: Checkerboard, camera: CameraModel
) -> BoardView | None:
    """The board's pose from its inner corners in OpenCV's order (PnP on undistorted corners)."""

    cv2 = _opencv()
    normalized = camera.normalize(np.asarray(pixels, dtype=np.float64).reshape(-1, 2))
    objects = board.object_points()
    if len(normalized) != len(objects):
        raise ValueError(f"{len(normalized)} corners for a {board.columns}x{board.rows} board")
    ok, rvec, tvec = cv2.solvePnP(
        objects, normalized.reshape(-1, 1, 2), np.eye(3), None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        objects, normalized.reshape(-1, 1, 2), np.eye(3), None, rvec, tvec
    )
    rotation, _ = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).ravel()
    projected = objects @ rotation.T + translation
    residual = projected[:, :2] / projected[:, 2:3] - normalized
    reprojection = (
        float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))) * 0.5 * (camera.fx + camera.fy)
    )
    return BoardView(np.asarray(rotation, dtype=np.float64), translation, reprojection)


def detect_board(
    gray: NDArray[np.uint8], board: Checkerboard, camera: CameraModel
) -> BoardView | None:
    """The board's pose from its inner corners, or ``None`` when they are not all found."""

    cv2 = _opencv()
    image = np.ascontiguousarray(gray, dtype=np.uint8)
    pattern = (board.columns, board.rows)
    found, corners = cv2.findChessboardCornersSB(image, pattern)
    if not found:
        found, corners = cv2.findChessboardCorners(image, pattern)
        if not found:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        corners = cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)
    return board_view(np.asarray(corners, dtype=np.float64).reshape(-1, 2), board, camera)


@dataclass(frozen=True)
class BoardCorners:
    """One board found in an image: its inner-corner grid, columns fastest."""

    columns: int
    rows: int
    pixels: FloatArray
    """``(rows * columns, 2)`` pixel coordinates in OpenCV's order (columns fastest)."""

    @property
    def grid(self) -> FloatArray:
        return self.pixels.reshape(self.rows, self.columns, 2)

    def board(self, square_m: float) -> Checkerboard:
        return Checkerboard(self.columns, self.rows, square_m)


def find_boards(
    gray: NDArray[np.uint8],
    *,
    min_pattern: tuple[int, int] = (4, 3),
    max_boards: int = 30,
) -> list[BoardCorners]:
    """Every checkerboard in one image, of any size at least ``min_pattern`` inner corners.

    For a scene with several boards in view at once (boards on walls, a single shot).
    ``findChessboardCornersSBWithMeta`` with ``CALIB_CB_LARGER`` finds one board and its
    full grid; the board and one square around it are then painted over with the image's
    median grey and the search repeats, until no board is found or ``max_boards``. The
    grid can come out turned by 90 deg relative to how the board was printed; its own
    corner order is kept, which describes the same plane.
    """

    cv2 = _opencv()
    work = np.ascontiguousarray(gray, dtype=np.uint8).copy()
    fill = int(np.median(work))
    flags = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found: list[BoardCorners] = []
    for _ in range(max_boards):
        ok, corners, meta = cv2.findChessboardCornersSBWithMeta(work, min_pattern, flags)
        if not ok or corners is None or meta is None:
            break
        rows, columns = (int(v) for v in np.asarray(meta).shape[:2])
        pixels = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if len(pixels) != rows * columns:
            break
        found.append(BoardCorners(columns, rows, pixels))
        # paint the board and one square around it over, through the grid's homography
        units = np.array([[c, r] for r in range(rows) for c in range(columns)], np.float64)
        homography, _ = cv2.findHomography(units, pixels)
        if homography is None:
            break
        outline = np.array(
            [[-1.5, -1.5], [columns + 0.5, -1.5], [columns + 0.5, rows + 0.5], [-1.5, rows + 0.5]]
        )
        polygon = cv2.perspectiveTransform(outline.reshape(-1, 1, 2), homography).reshape(-1, 2)
        cv2.fillConvexPoly(work, np.round(polygon).astype(np.int32), fill)
    return found


@dataclass(frozen=True)
class LidarBoardPlane:
    """The board's plane in the LiDAR frame and the points that fit it."""

    plane: OrientedPlane
    inliers: int
    rms_m: float


def extract_board_plane(
    points_lidar: NDArray[np.floating],
    view: BoardView,
    board: Checkerboard,
    t_camera_lidar: FloatArray,
    *,
    margin_m: float = 0.25,
    gate_m: float = 0.4,
    inlier_m: float = 0.02,
    max_normal_deg: float = 25.0,
    min_inliers: int = 30,
    iterations: int = 300,
    refinements: int = 20,
    seed: int = 0,
) -> LidarBoardPlane | None:
    """The board's plane among the LiDAR points around where the camera saw the board.

    Points are mapped into the camera frame with ``t_camera_lidar`` and then into the
    board frame; those within ``margin_m`` of the squares and ``gate_m`` of the board
    plane are candidates. The plane with most inliers within ``inlier_m`` (RANSAC) is
    refined by least squares to a fixed point, keeping the points within
    ``max(inlier_m, 3 sigma)`` of it (sigma from the median absolute residual). ``None``
    when too few points fit, or when the plane turns more than ``max_normal_deg`` away
    from the board the camera saw.
    """

    points = np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3)
    rotation, translation = t_camera_lidar[:3, :3], t_camera_lidar[:3, 3]
    in_board = ((points @ rotation.T + translation) - view.translation) @ view.rotation
    (x0, y0), (x1, y1) = board.squares_extent()
    keep = (
        (in_board[:, 0] > x0 - margin_m)
        & (in_board[:, 0] < x1 + margin_m)
        & (in_board[:, 1] > y0 - margin_m)
        & (in_board[:, 1] < y1 + margin_m)
        & (np.abs(in_board[:, 2]) < gate_m)
    )
    candidates = points[keep]
    if len(candidates) < min_inliers:
        return None
    rng = np.random.default_rng(seed)
    best_count, best = 0, None
    for _ in range(iterations):
        sample = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length < 1.0e-9:
            continue
        normal /= length
        count = int(np.count_nonzero(np.abs((candidates - sample[0]) @ normal) < inlier_m))
        if count > best_count:
            best_count, best = count, (normal, sample[0])
    if best is None or best_count < min_inliers:
        return None
    normal, origin = best
    # The board's points can spread wider than inlier_m (range noise of a few cm at
    # distance): a fixed band would keep a different slice of them for each RANSAC
    # sample. Refine to a fixed point instead, the band widened to 3 robust sigmas of
    # the residuals, so the plane depends on the points and not on their order.
    selected = np.abs((candidates - origin) @ normal) < inlier_m
    for _ in range(refinements):
        inliers = candidates[selected]
        centroid = inliers.mean(axis=0)
        _, singular, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = vt[2]
        residual = (candidates - centroid) @ normal
        kept = residual[selected]
        sigma = 1.4826 * float(np.median(np.abs(kept - np.median(kept))))
        updated = np.abs(residual) < max(inlier_m, 3.0 * sigma)
        if np.array_equal(updated, selected) or np.count_nonzero(updated) < min_inliers:
            break
        selected = updated
    inliers = candidates[selected]
    centroid = inliers.mean(axis=0)
    _, singular, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[2]
    expected = rotation.T @ view.rotation[:, 2]
    angle = math.degrees(math.acos(min(1.0, abs(float(normal @ expected)))))
    if angle > max_normal_deg:
        return None
    rms = float(singular[2] / math.sqrt(len(inliers)))
    return LidarBoardPlane(
        OrientedPlane(
            (float(normal[0]), float(normal[1]), float(normal[2])), -float(normal @ centroid)
        ),
        len(inliers),
        rms,
    )


@dataclass(frozen=True)
class BoardPose:
    """One board pose: an image of it and the LiDAR points recorded while it stood still."""

    pose_id: str
    view: BoardView
    points_lidar: NDArray[np.floating]
    board: Checkerboard | None = None
    """This pose's board, when the poses do not all share the calibration's board."""


@dataclass(frozen=True)
class BoardAxis:
    """One axis of ``T_camera_lidar`` with its jackknife std."""

    name: str
    unit: str
    value: float
    std: float
    observed: bool


@dataclass(frozen=True)
class BoardCalibration:
    """``T_camera_lidar`` from board planes, per-axis stds, and what was left out."""

    transform_camera_lidar: FloatArray | None
    axes: tuple[BoardAxis, ...]
    poses_used: tuple[str, ...]
    poses_rejected: tuple[tuple[str, str], ...]
    normal_singular_values: tuple[float, float, float] | None
    plane_normal_rmse_deg: float | None
    plane_offset_rmse_m: float | None
    status: str
    reason: str
    unobserved_directions: tuple[tuple[float, float, float], ...] = ()
    """Camera-frame translation directions the board normals did not fix (kept at the prior)."""


def _pose_weight(view: BoardView) -> float:
    """Relative weight of a pose: the camera's board distance error grows with distance.

    A focal-length error scales the PnP distance, so the board-plane offset error grows
    in proportion to the distance; the weight is ``(1 m / distance)^2``.
    """

    distance = max(float(np.linalg.norm(view.translation)), 0.1)
    return 1.0 / (distance * distance)


def _matrix(transform: SE3) -> FloatArray:
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(transform.rotation_quat_xyzw).as_matrix()
    matrix[:3, 3] = transform.translation_m
    return matrix


def _se3(matrix: FloatArray) -> SE3:
    return SE3(
        (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
        quaternion_xyzw_from_rotation_matrix([float(v) for v in matrix[:3, :3].ravel()]),
    )


def unobserved_translation_directions(
    observations: Sequence[PlanarBoardObservation], ratio: float = UNOBSERVED_EIGEN_RATIO
) -> FloatArray:
    """Camera-frame unit directions (rows) along which the board normals fix no translation.

    A plane fixes the translation only along its normal: the weighted information is
    ``sum w n n^T`` over the camera-frame normals, and an eigen-direction whose eigenvalue
    is below ``ratio`` times the largest is unobserved.
    """

    information = np.zeros((3, 3))
    for item in observations:
        normal = np.asarray(item.camera_plane.normal)
        information += item.weight * np.outer(normal, normal)
    values, vectors = np.linalg.eigh(information)
    weak = values < ratio * max(float(values[-1]), 1.0e-300)
    return np.asarray(vectors[:, weak].T, dtype=np.float64)


def _solve(
    observations: Sequence[PlanarBoardObservation],
    initial: FloatArray,
    prior: FloatArray,
) -> tuple[FloatArray | None, Any]:
    """Solve, then keep the translation along unobserved directions at the prior's value."""

    options = PlanarBoardSolverOptions(holdout_ratio=0.0, min_train_observations=3)
    result = PlanarBoardLidarCameraSolver().solve(observations, _se3(initial), options)
    if result.transform_camera_lidar is None:
        return None, result
    matrix = _matrix(result.transform_camera_lidar)
    for direction in unobserved_translation_directions(observations):
        matrix[:3, 3] += direction * float(direction @ (prior[:3, 3] - matrix[:3, 3]))
    return matrix, result


def _axis_values(matrix: FloatArray, reference: FloatArray) -> FloatArray:
    from scipy.spatial.transform import Rotation

    rotvec = Rotation.from_matrix(matrix[:3, :3] @ reference[:3, :3].T).as_rotvec(degrees=True)
    return np.concatenate([rotvec, matrix[:3, 3]])


def calibrate_board(
    poses: Sequence[BoardPose],
    board: Checkerboard | None,
    initial_camera_lidar: FloatArray,
    *,
    passes: int = 3,
    translation_bound_m: float = TRANSLATION_STD_BOUND_M,
    rotation_bound_deg: float = ROTATION_STD_BOUND_DEG,
) -> BoardCalibration:
    """``T_camera_lidar`` from board poses, starting from a rough ``initial_camera_lidar``.

    Each pass extracts the board planes with the current extrinsic and solves; the first
    crops generously around the board, later ones closer to it. Each axis's std is the
    jackknife over poses (one pose left out at a time); an axis is observed when its std
    is within ``translation_bound_m`` or ``rotation_bound_deg``.
    """

    if board is None and any(pose.board is None for pose in poses):
        raise ValueError("every pose needs a board when the calibration has none")
    current = np.asarray(initial_camera_lidar, dtype=np.float64)
    prior = current.copy()
    observations: list[PlanarBoardObservation] = []
    rejected: list[tuple[str, str]] = []
    result: Any = None
    for index in range(max(1, passes)):
        margin, gate = (0.25, 0.4) if index == 0 else (0.15, 0.15)
        observations, rejected = [], []
        for pose in poses:
            pose_board = pose.board or board
            assert pose_board is not None
            lidar = extract_board_plane(
                pose.points_lidar, pose.view, pose_board, current, margin_m=margin, gate_m=gate
            )
            if lidar is None:
                rejected.append((pose.pose_id, "no LiDAR plane near the board the camera saw"))
                continue
            observations.append(
                PlanarBoardObservation(
                    pose.pose_id, pose.view.plane, lidar.plane, weight=_pose_weight(pose.view)
                )
            )
        if len(observations) < MIN_POSES:
            return BoardCalibration(
                None,
                (),
                tuple(o.frame_id for o in observations),
                tuple(rejected),
                None,
                None,
                None,
                "insufficient_poses",
                f"{len(observations)} board pose(s) with a LiDAR plane; "
                f"at least {MIN_POSES} needed",
            )
        solved, result = _solve(observations, current, prior)
        if solved is None:
            return BoardCalibration(
                None,
                (),
                tuple(o.frame_id for o in observations),
                tuple(rejected),
                result.normal_singular_values,
                None,
                None,
                result.status,
                result.reason,
            )
        current = solved

    leave_one_out = []
    for left_out in range(len(observations)):
        subset = [o for i, o in enumerate(observations) if i != left_out]
        matrix, _ = _solve(subset, current, prior)
        if matrix is not None:
            leave_one_out.append(_axis_values(matrix, current))
    count = len(leave_one_out)
    values = _axis_values(current, current)
    values[:3] = 0.0
    spread = np.asarray(leave_one_out) if leave_one_out else np.zeros((0, 6))
    if count >= 2:
        stds = np.sqrt((count - 1) / count * np.sum((spread - spread.mean(axis=0)) ** 2, axis=0))
    else:
        stds = np.full(6, math.inf)
    rotation_vector = _axis_values(current, np.eye(4))[:3]
    unobserved = unobserved_translation_directions(observations)
    share = np.max(np.abs(unobserved), axis=0) if len(unobserved) else np.zeros(3)
    axes = tuple(
        BoardAxis(
            name,
            "deg" if i < 3 else "m",
            float(rotation_vector[i] if i < 3 else current[i - 3, 3]),
            float(stds[i]),
            bool(stds[i] <= rotation_bound_deg)
            if i < 3
            else bool(stds[i] <= translation_bound_m and share[i - 3] < UNOBSERVED_AXIS_SHARE),
        )
        for i, name in enumerate(AXES)
    )
    evaluation = result.train_evaluation
    return BoardCalibration(
        current,
        axes,
        tuple(o.frame_id for o in observations),
        tuple(rejected),
        result.normal_singular_values,
        evaluation.normal_rmse_deg,
        evaluation.offset_rmse_m,
        "converged",
        f"{len(observations)} board poses; jackknife over {count} leave-one-out solves",
        tuple((float(d[0]), float(d[1]), float(d[2])) for d in unobserved),
    )


def plane_residuals(
    pose: BoardPose, board: Checkerboard | None, transform_camera_lidar: FloatArray
) -> tuple[LidarBoardPlane, float, float] | None:
    """The LiDAR plane of ``pose`` under ``transform_camera_lidar`` and its disagreement.

    Returns ``(plane, normal angle in deg, LiDAR offset minus camera offset in m)``, both
    in the camera frame, or ``None`` when no LiDAR plane is found near the board.
    """

    transform = np.asarray(transform_camera_lidar, dtype=np.float64)
    pose_board = pose.board or board
    if pose_board is None:
        raise ValueError("the pose has no board and none was given")
    lidar = extract_board_plane(
        pose.points_lidar, pose.view, pose_board, transform, margin_m=0.15, gate_m=0.15
    )
    if lidar is None:
        return None
    normal = transform[:3, :3] @ np.asarray(lidar.plane.normal)
    offset = lidar.plane.offset_m - float(normal @ transform[:3, 3])
    camera_normal = np.asarray(pose.view.plane.normal)
    if normal @ camera_normal < 0:
        normal, offset = -normal, -offset
    angle = math.degrees(math.acos(min(1.0, float(normal @ camera_normal))))
    return lidar, angle, offset - pose.view.plane.offset_m
