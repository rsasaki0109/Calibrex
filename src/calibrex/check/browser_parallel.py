"""Transport and combine browser check shards without changing the pair solvers.

Dependent pairs run together in one Pyodide worker. The coordinator rebuilds
closure and attribution from the original float64 edge inputs, rather than
combining the verdicts of incomplete rigs. This module performs no bag I/O.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import ConfigDict

from calibrex.check.attribution import build_rotation_attribution, observations_from_run
from calibrex.check.cache import code_fingerprint
from calibrex.check.closure import (
    ClosureEdge,
    build_closure_report,
    closure_floor_verdict,
    edge_from_estimates,
    shared_inputs_of,
)
from calibrex.check.estimators import EstimatorRun
from calibrex.check.runner import format_check_table
from calibrex.check.verdict import FOCAL_SCALE_FLOOR, VerdictOptions, worst_verdict
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckPairName,
    CheckPairRecord,
    summarize_pairs,
)
from calibrex.core.exceptions import DatasetError
from calibrex.core.result import StrictModel

Vector3 = tuple[float, float, float]
DEPENDENT_PAIRS = (
    frozenset({"imu-lidar", "gnss-lidar", "gnss-imu"}),
    frozenset({"camera-imu", "camera-focal"}),
)


class _ClosureInput(StrictModel):
    # JSON strings preserve infinite uncertainties on unobserved axes. Pydantic
    # reads them back as floats; the outer worker message remains strict JSON.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    pair: CheckPairName
    frames: tuple[str, ...]
    parent: str
    child: str
    rotation: tuple[Vector3, Vector3, Vector3]
    rot_std_deg: Vector3
    rot_observed: tuple[bool, bool, bool]
    translation: Vector3 | None
    trans_std_m: Vector3
    trans_observed: tuple[bool, bool, bool]
    estimator: str | None
    shared_inputs: tuple[str, ...]

    def edge(self) -> ClosureEdge:
        """Restore the original float64 arrays, without a quaternion round trip."""
        return ClosureEdge(
            pair=self.pair,
            parent=self.parent,
            child=self.child,
            rotation=np.asarray(self.rotation, dtype=np.float64),
            rot_std_deg=self.rot_std_deg,
            rot_observed=self.rot_observed,
            translation=(
                np.asarray(self.translation, dtype=np.float64)
                if self.translation is not None
                else None
            ),
            trans_std_m=self.trans_std_m,
            trans_observed=self.trans_observed,
            estimator=self.estimator,
            shared_inputs=self.shared_inputs,
        )


class _CheckShard(StrictModel):
    artifact: CalibrationCheckArtifact
    closure_inputs: list[_ClosureInput]
    request_key: str
    source_fingerprint: str


def closure_input_from_result(
    record: CheckPairRecord, outcome: EstimatorRun | None
) -> dict[str, Any] | None:
    """The same edge as the serial runner, encoded for a worker message."""
    transform = record.compared_transform
    if outcome is None or transform is None or not outcome.solved:
        return None
    edge = edge_from_estimates(
        record.pair,
        transform.parent_frame,
        transform.child_frame,
        outcome.compared,
        outcome.estimates,
        outcome.estimator,
        shared_inputs_of(record.pair, record.frames),
    )
    value = _ClosureInput(
        pair=record.pair,
        frames=tuple(record.frames),
        parent=edge.parent,
        child=edge.child,
        rotation=edge.rotation.tolist(),
        rot_std_deg=edge.rot_std_deg,
        rot_observed=edge.rot_observed,
        translation=edge.translation.tolist() if edge.translation is not None else None,
        trans_std_m=edge.trans_std_m,
        trans_observed=edge.trans_observed,
        estimator=edge.estimator,
        shared_inputs=edge.shared_inputs,
    )
    return json.loads(value.model_dump_json())  # type: ignore[no-any-return]


def parallel_request_key(request: dict[str, Any]) -> str:
    """Bind shards to one request, excluding their partition and mount paths."""
    keys = (
        "mode",
        "bag_label",
        "tf_labels",
        "vehicle_frame",
        "frame_map",
        "topic_kind",
        "camera",
        "max_duration_s",
        "command_bag",
    )
    data = {key: request.get(key) for key in keys}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def shard_metadata(request: dict[str, Any]) -> dict[str, str]:
    """Identify the request and installed source that produced a shard."""
    return {"request_key": parallel_request_key(request), "source_fingerprint": code_fingerprint()}


def _common(artifact: CalibrationCheckArtifact) -> dict[str, Any]:
    data = artifact.model_dump(mode="json")
    options = data["options"]
    if options is None or artifact.plan_only:
        raise DatasetError("a parallel check shard must be a completed check")
    options = {**options, "pairs": None}
    return {
        **{
            key: data[key]
            for key in (
                "bag",
                "vehicle_frame",
                "candidate_sources",
                "topics",
                "frame_tree",
            )
        },
        "options": options,
        "provenance": {
            key: data["provenance"][key]
            for key in (
                "generator",
                "generator_version",
                "git_commit",
                "input_sha256",
                "input_digest_scope",
            )
        },
        "plan": [
            {
                key: pair[key]
                for key in (
                    "pair",
                    "sensors",
                    "topics",
                    "frames",
                    "candidate_transform",
                )
            }
            for pair in data["pairs"]
        ],
    }


def _combine(request: dict[str, Any], shards: Sequence[_CheckShard]) -> CalibrationCheckArtifact:
    selected = request.get("pairs")
    if request.get("mode", "check") != "check" or not selected or len(shards) != 2:
        raise DatasetError("parallel check requires two shards and explicit selected pairs")
    if len(selected) != len(set(selected)):
        raise DatasetError("duplicate selected pairs")
    first = shards[0]
    workers_used = request.get("workers_used", 2)
    if workers_used not in (1, 2):
        raise DatasetError("a parallel check uses one or two workers")
    common = _common(first.artifact)
    owners: dict[str, _CheckShard] = {}
    request_key = parallel_request_key(request)
    for shard in shards:
        if shard.request_key != request_key or shard.source_fingerprint != code_fingerprint():
            raise DatasetError("parallel check shards have a different request or source")
        if _common(shard.artifact) != common:
            raise DatasetError("parallel check shards have different inputs, plans or options")
        assert shard.artifact.options is not None
        for pair_name in shard.artifact.options.pairs or ():
            if pair_name in owners or pair_name not in selected:
                raise DatasetError("parallel check partitions overlap or contain an extra pair")
            owners[pair_name] = shard
    if set(owners) != set(selected):
        raise DatasetError("parallel check partitions do not cover all selected pairs")
    for group in DEPENDENT_PAIRS:
        # Preserve missing-dependency behavior when a prerequisite was not selected.
        dependent = "gnss-imu" if "gnss-imu" in group else "camera-focal"
        if dependent in owners and any(
            owners[pair] is not owners[dependent] for pair in group if pair in owners
        ):
            raise DatasetError("dependent pairs must run in the same worker")
    by_owner = {
        id(shard): {(pair.pair, tuple(pair.frames)): pair for pair in shard.artifact.pairs}
        for shard in shards
    }
    pairs = [
        by_owner[id(owners.get(pair.pair, first))][(pair.pair, tuple(pair.frames))]
        for pair in first.artifact.pairs
    ]
    edges: dict[tuple[str, tuple[str, ...]], ClosureEdge] = {}
    records = {(pair.pair, tuple(pair.frames)): pair for pair in pairs}
    for shard in shards:
        for item in shard.closure_inputs:
            key = (item.pair, item.frames)
            record = records.get(key)
            if owners.get(item.pair) is not shard or key in edges or record is None:
                raise DatasetError("unexpected or duplicate closure input")
            compared = record.compared_transform
            if compared is None or (item.parent, item.child) != (
                compared.parent_frame,
                compared.child_frame,
            ):
                raise DatasetError("closure input frames do not match the pair result")
            edges[key] = item.edge()
    for pair in pairs:
        if (
            pair.status in {"pass", "warn", "fail"}
            and pair.pair != "camera-focal"
            and (pair.pair, tuple(pair.frames)) not in edges
        ):
            raise DatasetError("missing closure input for a solved pair")
    assert first.artifact.options is not None
    options = first.artifact.options.model_copy(update={"pairs": list(selected)})
    verdict = VerdictOptions(
        sigma_k=options.sigma_k,
        rotation_floor_deg=options.rotation_floor_deg,
        translation_floor_m=options.translation_floor_m,
        detection_probe_deg=options.detection_probe_deg,
        focal_scale_floor=options.focal_scale_floor or FOCAL_SCALE_FLOOR,
    )
    closures = build_closure_report(pairs, edges, verdict)
    attribution = build_rotation_attribution(observations_from_run(pairs, edges), verdict)
    from calibrex.check.browser import run_command

    command = run_command(
        "check",
        request["bag_label"],
        tf_labels=request.get("tf_labels", ()),
        vehicle_frame=request.get("vehicle_frame"),
        frame_map=request.get("frame_map", ()),
        topic_kind=request.get("topic_kind", ()),
        camera=request.get("camera"),
        pairs=selected,
        max_duration_s=request.get("max_duration_s"),
    )
    provenance = first.artifact.provenance.model_copy(
        update={
            "command": shlex.split(command),
            "notes": [
                *first.artifact.provenance.notes,
                f"check computed in {workers_used} browser worker(s) from 2 partitions; "
                "dependent pairs stayed "
                "together; rig closure and rotation attribution rebuilt from all estimates",
            ],
        }
    )
    return CalibrationCheckArtifact.model_validate(
        {
            **first.artifact.model_dump(),
            "pairs": pairs,
            "summary": summarize_pairs(pairs),
            "options": options,
            "closures": closures,
            "rotation_attribution": attribution,
            "overall_verdict": worst_verdict(
                [
                    *(pair.status for pair in pairs),
                    closure_floor_verdict(closures) or "",
                ]
            )
            or "inconclusive",
            "provenance": provenance,
        }
    )


def merge_check_shards_json(request_json: str, shards_json: str) -> str:
    """Validate two disjoint worker results and render one complete rig report."""
    try:
        request = json.loads(request_json)
        replies = json.loads(shards_json)
        shards = []
        for reply in replies:
            if not reply.get("ok") or reply.get("mode") != "check":
                raise DatasetError("a parallel check worker failed")
            shards.append(
                _CheckShard.model_validate(
                    {
                        key: reply[key]
                        for key in (
                            "artifact",
                            "closure_inputs",
                            "request_key",
                            "source_fingerprint",
                        )
                    }
                )
            )
        artifact = _combine(request, shards)
        from calibrex.check.browser import _finite_json, run_command
        from calibrex.visualization.check_report import render_check_html

        command = run_command(
            "check",
            request.get("command_bag") or request["bag_label"],
            tf_labels=request.get("tf_labels", ()),
            vehicle_frame=request.get("vehicle_frame"),
            frame_map=request.get("frame_map", ()),
            topic_kind=request.get("topic_kind", ()),
            camera=request.get("camera"),
            pairs=request["pairs"],
            max_duration_s=request.get("max_duration_s"),
        )
        return json.dumps(
            _finite_json(
                {
                    "ok": True,
                    "mode": "check",
                    "bag": replies[0]["bag"],
                    "command": command,
                    "artifact": artifact.model_dump(mode="json", exclude_none=True),
                    "text": format_check_table(artifact),
                    "html": render_check_html(artifact),
                    "files": {},
                }
            )
        )
    except Exception as error:
        return json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"})
