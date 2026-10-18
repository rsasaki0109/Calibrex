"""Private process transport for native TUM factors, never arbitrary callbacks."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from types import FunctionType
from typing import Any, BinaryIO, Literal, TypeAlias, cast

from calibrex.graph._rgbd_batches import _rgbd_binding, _RGBDBinding
from calibrex.graph.joint_factors import (
    JointDepthPointToPlaneMeasurement,
    JointTrilinearDepthPointToPlaneMeasurement,
    JointTrilinearXYZPairMeasurement,
    JointTrilinearXYZPointToPlaneMeasurement,
    make_joint_centered_trilinear_depth_point_to_plane_factor,
    make_joint_depth_point_to_plane_factor,
    make_joint_prior_factor,
    make_joint_trilinear_xyz_pair_factor,
    make_joint_trilinear_xyz_point_to_plane_factor,
)
from calibrex.graph.joint_optimization import (
    JointFactorSplitPolicy,
    JointResidualBlock,
    SquareRootInformation,
)

_Measurement: TypeAlias = (
    JointDepthPointToPlaneMeasurement
    | JointTrilinearDepthPointToPlaneMeasurement
    | JointTrilinearXYZPointToPlaneMeasurement
    | JointTrilinearXYZPairMeasurement
)
_PRIOR_CODE = make_joint_prior_factor(
    factor_id="transport-code", observation_group="transport-code", block="x",
    target=(0.0,), sigma=(1.0,),
).evaluator
assert isinstance(_PRIOR_CODE, FunctionType)
_PRIOR_EVALUATOR_CODE = _PRIOR_CODE.__code__
_FactorKind: TypeAlias = Literal["depth", "centered", "xyz", "pair", "prior"]


@dataclass(frozen=True)
class _FactorSpec:
    kind: _FactorKind
    measurement: _Measurement | None
    names: tuple[str, ...]
    factor_id: str
    observation_group: str
    weight: float
    family: str
    split_policy: JointFactorSplitPolicy
    sqrt_information: SquareRootInformation
    target: tuple[float, ...] = ()


def _factor_spec(factor: JointResidualBlock) -> _FactorSpec:
    binding = _rgbd_binding(factor)
    target: tuple[float, ...] = ()
    measurement: _Measurement | None = None
    kind: _FactorKind
    if isinstance(binding, _RGBDBinding) and binding.kind in ("depth", "centered", "xyz", "pair"):
        kind = binding.kind
        measurement = binding.measurement
        names = binding.names
        if factor.variable_names != names:
            raise ValueError("TUM process factor variable ownership differs")
    elif (
        binding is not None
        and factor.family == "diagonal_prior"
        and isinstance(factor.evaluator, FunctionType)
        and factor.evaluator.__code__ is _PRIOR_EVALUATOR_CODE
    ):
        evaluator = factor.evaluator
        closed = dict(zip(
            evaluator.__code__.co_freevars, evaluator.__closure__ or (), strict=True
        ))
        target = closed["targets"].cell_contents
        names = (closed["block"].cell_contents,)
        if factor.variable_names != names or type(target) is not tuple:
            raise ValueError("TUM process prior ownership differs")
        kind = "prior"
    else:
        raise ValueError("TUM process transport requires an unmodified native data factor or prior")
    return _FactorSpec(
        kind, measurement, names, factor.factor_id, factor.observation_group,
        factor.weight, factor.family, factor.split_policy, factor.sqrt_information, target,
    )


def _restore_factor(spec: _FactorSpec) -> JointResidualBlock:
    names = spec.names
    if spec.kind == "depth":
        factor = make_joint_depth_point_to_plane_factor(
            cast(JointDepthPointToPlaneMeasurement, spec.measurement),
            pose_block=names[0], extrinsic_block=names[1], depth_calibration_block=names[2],
        )
    elif spec.kind == "centered":
        factor = make_joint_centered_trilinear_depth_point_to_plane_factor(
            cast(JointTrilinearDepthPointToPlaneMeasurement, spec.measurement),
            pose_block=names[0], extrinsic_block=names[1],
            depth_bias_block=names[2], centered_lattice_block=names[3],
        )
    elif spec.kind == "xyz":
        factor = make_joint_trilinear_xyz_point_to_plane_factor(
            cast(JointTrilinearXYZPointToPlaneMeasurement, spec.measurement),
            pose_block=names[0], extrinsic_block=names[1], xyz_lattice_block=names[2],
        )
    elif spec.kind == "pair":
        factor = make_joint_trilinear_xyz_pair_factor(
            cast(JointTrilinearXYZPairMeasurement, spec.measurement),
            source_pose_block=names[0], target_pose_block=names[1], xyz_lattice_block=names[2],
        )
    else:
        factor = make_joint_prior_factor(
            factor_id=spec.factor_id, observation_group=spec.observation_group,
            block=names[0], target=spec.target, sigma=(1.0,) * len(spec.target),
        )
    # Retain the stored whitening bits: taking the reciprocal of a stored
    # reciprocal sigma would add an avoidable rounding operation.
    return replace(
        factor, factor_id=spec.factor_id, observation_group=spec.observation_group,
        weight=spec.weight, family=spec.family, split_policy=spec.split_policy,
        sqrt_information=spec.sqrt_information,
    )


class _NativePickler(pickle.Pickler):
    def reducer_override(self, obj: object) -> Any:
        if isinstance(obj, JointResidualBlock):
            return _restore_factor, (_factor_spec(obj),)
        return NotImplemented


def _dump_native(stream: BinaryIO, value: object) -> None:
    # Ordinary dataclass pickle state preserves stored SE3 bits without calling
    # its normalizing constructor. Only our own ephemeral request/response files
    # are read; this is not a result-file or user-supplied pickle interface.
    _NativePickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(value)


def _write_native(path: Path, value: object) -> None:
    with path.open("wb") as stream:
        _dump_native(stream, value)
