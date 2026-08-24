"""Downstream Autoware smoke-test adapter.

The calibration core deliberately has no ROS dependency.  This module is the
boundary where a verified :mod:`autoware_promotion` plan may be exercised in a
downstream workspace.  It records enough immutable evidence to make an apply
decision auditable even when ROS is not installed on the machine producing a
plan.  The ``precomputed`` mode is intentionally first class: a CI or field
machine can run the smoke test and hand the signed/digest-bound artifact to a
machine that only performs the guarded promotion.

Commands are always argv vectors.  No command is passed through a shell.  A
container invocation is assembled with read-only workspace and no-network
options, and production mode requires an immutable ``sha256:...`` image digest.
The adapter does not import ROS, xacro, ament, or any GPL implementation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

from calibrex import __version__
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping_atomic
from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel

if TYPE_CHECKING:
    from calibrex.export.autoware_promotion import AutowarePromotionArtifact

AUTOWARE_SMOKE_SCHEMA_VERSION: Literal["slac.autoware_smoke/v0.1"] = "slac.autoware_smoke/v0.1"
AUTOWARE_SMOKE_PROTOCOL_ID = "autoware_sensor_kit_smoke/v0.1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CONTAINER_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"

SmokeMode = Literal["subprocess", "container", "precomputed"]
SmokeProfile = Literal["production", "developer"]
SmokeStatus = Literal["PASS", "WARN", "FAIL", "BLOCKED"]
SmokeStageStatus = Literal["PASS", "FAIL", "TIMEOUT", "SKIPPED", "NOT_RUN"]

# These names are stable contract identifiers.  The adapter accepts a small
# set of human-friendly aliases at its input boundary and emits the canonical
# names below in artifacts.
SMOKE_STAGE_NAMES: tuple[str, ...] = (
    "xacro_urdf",
    "build_test",
    "tf_static",
    "sensor_launch",
    "perception",
)
REQUIRED_SMOKE_STAGES: tuple[str, ...] = SMOKE_STAGE_NAMES[:4]
_STAGE_ALIASES = {
    "xacro_urdf_expansion": "xacro_urdf",
    "xacro_urdf_validation": "xacro_urdf",
    "urdf": "xacro_urdf",
    "build": "build_test",
    "colcon_build_test": "build_test",
    "tf": "tf_static",
    "tf_static_frame": "tf_static",
    "tf_static_frame_validation": "tf_static",
    "sensor_launch_topic_frame": "sensor_launch",
    "launch": "sensor_launch",
}


class AutowareSmokeError(CalibrexError):
    """Raised when smoke evidence is invalid or unsafe to execute."""


class AutowareSmokeCommand(StrictModel):
    """One argv-only command assigned to a smoke stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    argv: list[str] = Field(min_length=1)
    cwd_relative: str = "."

    @field_validator("stage", mode="before")
    @classmethod
    def _canonical_stage(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("smoke stage must be a string")
        name = _STAGE_ALIASES.get(value.strip(), value.strip())
        if name not in SMOKE_STAGE_NAMES:
            raise ValueError(f"unsupported smoke stage: {value!r}")
        return name

    @field_validator("argv")
    @classmethod
    def _argv_only(cls, value: list[str]) -> list[str]:
        if not value or any(not isinstance(item, str) or not item for item in value):
            raise ValueError("smoke command argv must contain non-empty strings")
        if any("\x00" in item for item in value):
            raise ValueError("smoke command argv must not contain NUL bytes")
        return list(value)

    @field_validator("cwd_relative")
    @classmethod
    def _relative_cwd(cls, value: str) -> str:
        try:
            _validate_relative_path_text(value, field_name="cwd_relative")
        except AutowareSmokeError as exc:
            raise ValueError(str(exc)) from exc
        return PurePosixPath(value.replace("\\", "/")).as_posix() or "."


# A descriptive alias used by clients that call this a stage command.
AutowareSmokeStageCommand = AutowareSmokeCommand


class AutowareSmokePolicy(StrictModel):
    """Execution and admission policy captured in every smoke artifact."""

    model_config = ConfigDict(extra="forbid")

    profile: SmokeProfile = "production"
    required_stages: list[str] = Field(default_factory=lambda: list(REQUIRED_SMOKE_STAGES))
    stage_timeout_seconds: float = Field(default=120.0, gt=0.0, le=86_400.0)
    max_output_bytes: int = Field(default=1_048_576, gt=0, le=64 * 1024 * 1024)
    output_tail_chars: int = Field(default=4_096, gt=0, le=1_000_000)
    network: Literal["none", "host"] = "none"
    read_only_workspace: bool = True
    require_immutable_container: bool = True
    resource_cpu_seconds: float | None = Field(default=None, gt=0.0)
    resource_memory_bytes: int | None = Field(default=None, gt=0)

    @field_validator("required_stages")
    @classmethod
    def _valid_stages(cls, value: list[str]) -> list[str]:
        normalized = [_STAGE_ALIASES.get(item.strip(), item.strip()) for item in value]
        if any(item not in SMOKE_STAGE_NAMES for item in normalized):
            raise ValueError(f"required_stages must use {SMOKE_STAGE_NAMES}")
        if len(normalized) != len(set(normalized)):
            raise ValueError("required_stages must be unique")
        if not set(REQUIRED_SMOKE_STAGES).issubset(normalized):
            raise ValueError("xacro_urdf, build_test, tf_static, and sensor_launch are required")
        return normalized

    @model_validator(mode="after")
    def _production_safety(self) -> AutowareSmokePolicy:
        if self.profile == "production":
            if self.network != "none":
                raise ValueError("production smoke policy requires network=none")
            if not self.read_only_workspace:
                raise ValueError("production smoke policy requires read_only_workspace=true")
            if not self.require_immutable_container:
                raise ValueError("production smoke policy requires immutable container images")
        return self


class AutowareSmokeStage(StrictModel):
    """Result and evidence for one smoke stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    required: bool = True
    status: SmokeStageStatus
    command: list[str] = Field(default_factory=list)
    cwd: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    exit_code: int | None = None
    timed_out: bool = False
    stdout_sha256: str = Field(pattern=_SHA256_PATTERN)
    stderr_sha256: str = Field(pattern=_SHA256_PATTERN)
    stdout_tail: str = ""
    stderr_tail: str = ""
    evidence: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    reason: str = ""

    @field_validator("stage", mode="before")
    @classmethod
    def _canonical_stage(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("smoke stage must be a string")
        name = _STAGE_ALIASES.get(value.strip(), value.strip())
        if name not in SMOKE_STAGE_NAMES:
            raise ValueError(f"unsupported smoke stage: {value!r}")
        return name

    @field_validator("command")
    @classmethod
    def _safe_command(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item or "\x00" in item for item in value):
            raise ValueError("stage command must be argv-only non-empty strings")
        return list(value)


# Descriptive compatibility name for evidence-oriented clients.
AutowareSmokeStageEvidence = AutowareSmokeStage


class AutowareSmokeProvenance(StrictModel):
    """Producer and self-digest lineage for smoke evidence."""

    model_config = ConfigDict(extra="forbid")

    producer: Literal["calibrex"] = "calibrex"
    tool_name: str = "calibrex.autoware-smoke"
    tool_version: str = __version__
    protocol_id: str = AUTOWARE_SMOKE_PROTOCOL_ID
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_digest_scope: str = (
        "canonical smoke artifact excluding artifact_sha256, "
        "provenance.artifact_sha256, and provenance.generated_at"
    )


class AutowareSmokeArtifact(StrictModel):
    """Digest-bound downstream smoke evidence for exactly one promotion plan."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["slac.autoware_smoke/v0.1"] = AUTOWARE_SMOKE_SCHEMA_VERSION
    smoke_id: str
    promotion_id: str
    promotion_artifact_sha256: str = Field(
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices(
            "promotion_artifact_sha256", "promotion_plan_sha256", "promotion_self_digest"
        ),
    )
    candidate_sha256: str = Field(
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices("candidate_sha256", "candidate_digest"),
    )
    workspace_baseline_manifest_sha256: str = Field(
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices(
            "workspace_baseline_manifest_sha256", "workspace_baseline_sha256"
        ),
    )
    package_baseline_manifest_sha256: str = Field(
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices(
            "package_baseline_manifest_sha256", "package_baseline_sha256"
        ),
    )
    workspace_root: str
    package_root: str
    profile: SmokeProfile
    mode: SmokeMode
    policy: AutowareSmokePolicy
    allowed_commands: list[list[str]] = Field(default_factory=list)
    environment_digest: str = Field(pattern=_SHA256_PATTERN)
    tool_digest: str = Field(pattern=_SHA256_PATTERN)
    container_digest: str | None = Field(default=None, pattern=_CONTAINER_DIGEST_PATTERN)
    isolation_enforced: bool = True
    status: SmokeStatus
    admission_label: Literal["production-admissible", "developer-only", "not-admissible"]
    reason: str = Field(min_length=1)
    stages: list[AutowareSmokeStage] = Field(default_factory=list)
    provenance: AutowareSmokeProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @property
    def promotion_plan_sha256(self) -> str:
        """Compatibility spelling for callers that call a plan a plan."""

        return self.promotion_artifact_sha256

    @property
    def workspace_manifest_sha256(self) -> str:
        """Compatibility spelling for the workspace baseline digest."""

        return self.workspace_baseline_manifest_sha256

    @property
    def package_baseline_sha256(self) -> str:
        """Compatibility spelling for the package baseline digest."""

        return self.package_baseline_manifest_sha256

    def with_artifact_digest(self) -> AutowareSmokeArtifact:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("artifact_sha256", None)
        provenance.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "provenance": self.provenance.model_copy(update={"artifact_sha256": digest}),
            }
        )

    def verify_artifact_digest(self) -> None:
        expected = self.with_artifact_digest().artifact_sha256
        if expected != self.artifact_sha256:
            raise AutowareSmokeError(
                "smoke artifact self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise AutowareSmokeError("smoke provenance artifact_sha256 mismatch")

    def save(self, path: str | Path) -> None:
        """Write a digest-complete YAML/JSON smoke artifact."""

        write_mapping_atomic(
            Path(path), self.with_artifact_digest().model_dump(mode="json", exclude_none=False)
        )


class AutowareSmokeConfig(StrictModel):
    """Typed request for running or importing smoke evidence."""

    model_config = ConfigDict(extra="forbid")

    mode: SmokeMode = "subprocess"
    profile: SmokeProfile = "production"
    commands: list[AutowareSmokeCommand] = Field(default_factory=list)
    container_runtime_argv: list[str] = Field(default_factory=lambda: ["docker", "run", "--rm"])
    container_digest: str | None = Field(default=None, pattern=_CONTAINER_DIGEST_PATTERN)
    environment_digest: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    tool_digest: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    policy: AutowareSmokePolicy | None = None
    precomputed_path: Path | None = None
    command: list[str] = Field(default_factory=list)

    @field_validator("container_runtime_argv")
    @classmethod
    def _runtime_argv(cls, value: list[str]) -> list[str]:
        if not value or any(
            not isinstance(item, str) or not item or "\x00" in item for item in value
        ):
            raise ValueError("container_runtime_argv must be a non-empty argv")
        return list(value)

    @field_validator("command")
    @classmethod
    def _provenance_command(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or "\x00" in item for item in value):
            raise ValueError("command must contain safe strings")
        return list(value)

    @model_validator(mode="after")
    def _production_runtime_safety(self) -> AutowareSmokeConfig:
        if self.profile == "production" and self.mode == "container":
            # These options can override the isolation flags appended by
            # Calibrex or grant the runtime host-level capabilities that make
            # a read-only/no-network evidence claim meaningless.
            forbidden = (
                "--privileged",
                "--cap-add",
                "--device",
                "--pid=host",
                "--ipc=host",
                "--uts=host",
                "--network=host",
                "--net=host",
                "--read-write",
                "--volume",
                "-v",
                "--mount",
                "--security-opt",
                "--userns=host",
            )
            if any(
                item == option or item.startswith(f"{option}=")
                for item in self.container_runtime_argv
                for option in forbidden
            ):
                raise ValueError(
                    "production container runtime argv contains an unsafe isolation option"
                )
        return self


def autoware_smoke_json_schema() -> dict[str, Any]:
    """Return the generated JSON schema for smoke artifacts."""

    return AutowareSmokeArtifact.model_json_schema()


autoware_smoke_artifact_json_schema = autoware_smoke_json_schema


def load_autoware_smoke(path: str | Path) -> AutowareSmokeArtifact:
    """Load and verify a smoke artifact from YAML or JSON."""

    artifact = AutowareSmokeArtifact.model_validate(read_mapping(Path(path)))
    artifact.verify_artifact_digest()
    return artifact


def verify_autoware_smoke(
    artifact: AutowareSmokeArtifact | str | Path,
    promotion: AutowarePromotionArtifact | None = None,
    *,
    require_pass: bool = False,
) -> AutowareSmokeArtifact:
    """Verify self-digest and (optionally) exact promotion-plan binding.

    This check is deliberately pure.  The promotion adapter performs a fresh
    package inspection immediately after this check and before changing files.
    """

    smoke = (
        artifact if isinstance(artifact, AutowareSmokeArtifact) else load_autoware_smoke(artifact)
    )
    smoke.verify_artifact_digest()
    _verify_stage_contract(smoke)
    if require_pass and (
        smoke.status != "PASS" or smoke.admission_label != "production-admissible"
    ):
        raise AutowareSmokeError("smoke artifact is not a production-admissible PASS")
    if promotion is not None:
        promotion.verify_artifact_digest()
        if smoke.promotion_id != promotion.promotion_id:
            raise AutowareSmokeError("smoke artifact promotion_id does not match the plan")
        if smoke.promotion_artifact_sha256 != promotion.artifact_sha256:
            raise AutowareSmokeError("smoke artifact is bound to a different promotion digest")
        if smoke.candidate_sha256 != promotion.candidate.sha256:
            raise AutowareSmokeError("smoke artifact candidate digest does not match the plan")
        if smoke.workspace_baseline_manifest_sha256 != promotion.baseline_manifest_sha256:
            raise AutowareSmokeError("smoke artifact workspace baseline is stale")
        if smoke.package_baseline_manifest_sha256 != promotion.baseline_manifest_sha256:
            raise AutowareSmokeError("smoke artifact package baseline is stale")
        if Path(smoke.workspace_root).resolve() != promotion.roots.workspace_root:
            raise AutowareSmokeError("smoke artifact workspace root does not match the plan")
        if Path(smoke.package_root).resolve() != promotion.roots.package_root:
            raise AutowareSmokeError("smoke artifact package root does not match the plan")
        if smoke.profile == "production" and promotion.policy.profile != "production":
            raise AutowareSmokeError("production smoke evidence cannot authorize a developer plan")
    return smoke


def import_autoware_smoke(
    artifact: AutowareSmokeArtifact | str | Path,
    *,
    promotion: AutowarePromotionArtifact | None = None,
    output_path: str | Path | None = None,
) -> AutowareSmokeArtifact:
    """Import precomputed evidence and bind it to a promotion plan."""

    smoke = verify_autoware_smoke(artifact, promotion)
    if smoke.mode != "precomputed":
        raise AutowareSmokeError("import requires a smoke artifact with mode=precomputed")
    if output_path is not None:
        smoke.save(output_path)
    return smoke


def run_autoware_smoke(
    promotion: AutowarePromotionArtifact | str | Path,
    *,
    config: AutowareSmokeConfig,
    output_path: str | Path | None = None,
) -> AutowareSmokeArtifact:
    """Run controlled subprocess/container stages or import precomputed evidence."""

    # Import lazily to keep the adapter boundary and avoid a module cycle.
    from calibrex.export.autoware_promotion import load_autoware_promotion

    plan = (
        promotion if not isinstance(promotion, (str, Path)) else load_autoware_promotion(promotion)
    )
    plan.verify_artifact_digest()
    if config.mode == "precomputed":
        if config.precomputed_path is None:
            raise AutowareSmokeError("precomputed mode requires precomputed_path")
        result = import_autoware_smoke(config.precomputed_path, promotion=plan)
        if output_path is not None:
            result.save(output_path)
        return result
    if plan.status != "PASS" or plan.decision != "ADOPT":
        raise AutowareSmokeError("smoke can only be run for a PASS/ADOPT promotion plan")
    policy = config.policy or AutowareSmokePolicy(profile=config.profile)
    if policy.profile != config.profile:
        raise AutowareSmokeError("smoke config profile and policy profile differ")
    if (
        config.mode == "container"
        and policy.profile == "production"
        and config.container_digest is None
    ):
        raise AutowareSmokeError("production container smoke requires an immutable image digest")
    if config.mode == "container" and config.container_digest is None:
        # Developer runs still carry an explicit digest when available.  A
        # mutable tag is never silently recorded as an immutable production
        # evidence source.
        container_digest: str | None = None
    else:
        container_digest = config.container_digest
    command_map: dict[str, AutowareSmokeCommand] = {}
    for command in config.commands:
        if command.stage in command_map:
            raise AutowareSmokeError(f"multiple commands supplied for stage {command.stage}")
        command_map[command.stage] = command
    allowed_commands = [command.argv for command in config.commands]
    stages: list[AutowareSmokeStage] = []
    required = set(policy.required_stages)
    for stage_name in SMOKE_STAGE_NAMES:
        stage_command = command_map.get(stage_name)
        is_required = stage_name in required
        if stage_command is None:
            stages.append(_stage_without_command(stage_name, required=is_required, policy=policy))
            continue
        if config.mode == "subprocess":
            stages.append(
                _run_subprocess_stage(plan, stage_command, required=is_required, policy=policy)
            )
        else:
            stages.append(
                _run_container_stage(
                    plan,
                    stage_command,
                    required=is_required,
                    policy=policy,
                    runtime_argv=config.container_runtime_argv,
                    container_digest=container_digest,
                )
            )
    status = _overall_status(stages, policy)
    admissible = (
        status == "PASS"
        and policy.profile == "production"
        and _isolation_ok(config.mode, policy, container_digest)
    )
    if status == "PASS" and policy.profile == "developer":
        reason = "smoke passed; developer profile is not production-admissible"
    elif status == "PASS" and admissible:
        reason = "all required downstream smoke stages passed"
    elif status == "PASS":
        reason = "smoke passed but execution policy is not production-admissible"
    else:
        reason = "one or more required downstream smoke stages failed or timed out"
    environment_digest = config.environment_digest or _environment_digest(plan.roots.workspace_root)
    tool_digest = config.tool_digest or _canonical_sha256(
        {"mode": config.mode, "commands": allowed_commands, "version": __version__}
    )
    artifact = AutowareSmokeArtifact(
        smoke_id=_smoke_id(plan.promotion_id, config.mode, allowed_commands),
        promotion_id=plan.promotion_id,
        promotion_artifact_sha256=plan.artifact_sha256,
        candidate_sha256=plan.candidate.sha256,
        workspace_baseline_manifest_sha256=plan.baseline_manifest_sha256,
        package_baseline_manifest_sha256=plan.baseline_manifest_sha256,
        workspace_root=str(plan.roots.workspace_root),
        package_root=str(plan.roots.package_root),
        profile=policy.profile,
        mode=config.mode,
        policy=policy,
        allowed_commands=allowed_commands,
        environment_digest=environment_digest,
        tool_digest=tool_digest,
        container_digest=container_digest,
        isolation_enforced=_isolation_ok(config.mode, policy, container_digest),
        status=status,
        admission_label=(
            "production-admissible"
            if admissible
            else "developer-only"
            if policy.profile == "developer"
            else "not-admissible"
        ),
        reason=reason,
        stages=stages,
        provenance=AutowareSmokeProvenance(
            command=list(config.command),
            git_commit=git_commit(),
        ),
        artifact_sha256="0" * 64,
    ).with_artifact_digest()
    if output_path is not None:
        artifact.save(output_path)
    return artifact


def _stage_without_command(
    stage: str, *, required: bool, policy: AutowareSmokePolicy
) -> AutowareSmokeStage:
    status: SmokeStageStatus = "FAIL" if required else "SKIPPED"
    reason = "required stage has no command" if required else "optional stage not configured"
    return AutowareSmokeStage(
        stage=stage,
        required=required,
        status=status,
        stdout_sha256=_sha256_bytes(b""),
        stderr_sha256=_sha256_bytes(reason.encode("utf-8")),
        stderr_tail=reason[-policy.output_tail_chars :],
        evidence={"execution": "not_run", "shell": "false"},
        provenance={
            "producer": "calibrex",
            "tool_name": "calibrex.autoware-smoke",
            "stage": stage,
        },
        reason=reason,
    )


def _run_subprocess_stage(
    plan: AutowarePromotionArtifact,
    command: AutowareSmokeCommand,
    *,
    required: bool,
    policy: AutowareSmokePolicy,
) -> AutowareSmokeStage:
    cwd = _safe_workspace_cwd(plan.roots.workspace_root, command.cwd_relative)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command.argv,
            cwd=cwd,
            shell=False,
            check=False,
            capture_output=True,
            timeout=policy.stage_timeout_seconds,
        )
        full_stdout = completed.stdout or b""
        full_stderr = completed.stderr or b""
        stdout = _bounded_bytes(full_stdout, policy.max_output_bytes)
        stderr = _bounded_bytes(full_stderr, policy.max_output_bytes)
        status: SmokeStageStatus = "PASS" if completed.returncode == 0 else "FAIL"
        reason = "exit code 0" if status == "PASS" else f"exit code {completed.returncode}"
        return _stage_result(
            command,
            required=required,
            status=status,
            cwd=cwd,
            duration=time.monotonic() - started,
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_digest=_sha256_bytes(full_stdout),
            stderr_digest=_sha256_bytes(full_stderr),
            policy=policy,
            reason=reason,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_output(exc.stdout, policy.max_output_bytes)
        stderr = _coerce_output(exc.stderr, policy.max_output_bytes)
        return _stage_result(
            command,
            required=required,
            status="TIMEOUT",
            cwd=cwd,
            duration=time.monotonic() - started,
            exit_code=None,
            stdout=stdout,
            stderr=stderr,
            stdout_digest=_sha256_bytes(stdout),
            stderr_digest=_sha256_bytes(stderr),
            policy=policy,
            reason=f"timeout after {policy.stage_timeout_seconds:.3f}s",
            timed_out=True,
        )
    except OSError as exc:
        message = f"could not execute argv: {exc}"
        return _stage_result(
            command,
            required=required,
            status="FAIL",
            cwd=cwd,
            duration=time.monotonic() - started,
            exit_code=None,
            stdout=b"",
            stderr=message.encode("utf-8", errors="replace"),
            stdout_digest=_sha256_bytes(b""),
            stderr_digest=_sha256_bytes(message.encode("utf-8", errors="replace")),
            policy=policy,
            reason=message,
        )


def _run_container_stage(
    plan: AutowarePromotionArtifact,
    command: AutowareSmokeCommand,
    *,
    required: bool,
    policy: AutowareSmokePolicy,
    runtime_argv: Sequence[str],
    container_digest: str | None,
) -> AutowareSmokeStage:
    if policy.profile == "production" and container_digest is None:
        return _stage_without_command(
            command.stage,
            required=required,
            policy=policy,
        ).model_copy(update={"status": "FAIL", "reason": "immutable container digest required"})
    image = "calibrex-smoke:developer" if container_digest is None else container_digest
    argv = [
        *runtime_argv,
        "--network=none" if policy.network == "none" else "--network=host",
        "--read-only" if policy.read_only_workspace else "--read-write",
        "--mount",
        f"type=bind,src={plan.roots.workspace_root},dst=/workspace,readonly={str(policy.read_only_workspace).lower()}",
        image,
        *command.argv,
    ]
    container_command = command.model_copy(update={"argv": argv, "cwd_relative": "."})
    return _run_subprocess_stage(plan, container_command, required=required, policy=policy)


def _stage_result(
    command: AutowareSmokeCommand,
    *,
    required: bool,
    status: SmokeStageStatus,
    cwd: Path,
    duration: float,
    exit_code: int | None,
    stdout: bytes,
    stderr: bytes,
    stdout_digest: str,
    stderr_digest: str,
    policy: AutowareSmokePolicy,
    reason: str,
    timed_out: bool = False,
) -> AutowareSmokeStage:
    return AutowareSmokeStage(
        stage=command.stage,
        required=required,
        status=status,
        command=list(command.argv),
        cwd=str(cwd),
        duration_seconds=max(duration, 0.0),
        exit_code=exit_code,
        timed_out=timed_out,
        stdout_sha256=stdout_digest,
        stderr_sha256=stderr_digest,
        stdout_tail=_decode_tail(stdout, policy.output_tail_chars),
        stderr_tail=_decode_tail(stderr, policy.output_tail_chars),
        evidence={
            "execution": "subprocess",
            "shell": "false",
            "network": policy.network,
            "workspace_mount": "read-only" if policy.read_only_workspace else "read-write",
        },
        provenance={
            "producer": "calibrex",
            "tool_name": "calibrex.autoware-smoke",
            "stage": command.stage,
            "command_sha256": _canonical_sha256(command.argv),
        },
        reason=reason,
    )


def _verify_stage_contract(artifact: AutowareSmokeArtifact) -> None:
    names = [stage.stage for stage in artifact.stages]
    if len(names) != len(set(names)):
        raise AutowareSmokeError("smoke artifact contains duplicate stages")
    required = set(artifact.policy.required_stages)
    if not required.issubset(names):
        raise AutowareSmokeError("smoke artifact is missing a required stage")
    if artifact.profile != artifact.policy.profile:
        raise AutowareSmokeError("smoke artifact profile and policy profile differ")
    if artifact.profile == "production":
        if artifact.mode == "container" and artifact.container_digest is None:
            raise AutowareSmokeError("production container smoke lacks immutable digest")
        if artifact.admission_label != "production-admissible" and artifact.status == "PASS":
            raise AutowareSmokeError("PASS production smoke must be production-admissible")
        if artifact.admission_label == "production-admissible" and not _isolation_ok(
            artifact.mode, artifact.policy, artifact.container_digest
        ):
            raise AutowareSmokeError("production-admissible smoke has an unsafe execution policy")
    for stage in artifact.stages:
        if stage.status == "PASS" and stage.exit_code not in {0, None}:
            raise AutowareSmokeError(f"PASS stage has non-zero exit code: {stage.stage}")
        for value in stage.command:
            if not value or "\x00" in value:
                raise AutowareSmokeError("smoke artifact contains an unsafe command")
    expected_status = _overall_status(artifact.stages, artifact.policy)
    if artifact.status == "PASS" and expected_status != "PASS":
        raise AutowareSmokeError("smoke artifact status says PASS but a stage did not pass")
    if artifact.status == "FAIL" and expected_status == "PASS":
        raise AutowareSmokeError(
            "smoke artifact status says FAIL despite all required stages passing"
        )


def _overall_status(
    stages: Sequence[AutowareSmokeStage], policy: AutowareSmokePolicy
) -> SmokeStatus:
    required = [
        stage for stage in stages if stage.required or stage.stage in policy.required_stages
    ]
    if any(stage.status in {"FAIL", "TIMEOUT", "NOT_RUN"} for stage in required):
        return "FAIL"
    if any(stage.status == "FAIL" for stage in stages):
        return "WARN"
    return "PASS"


def _isolation_ok(
    mode: SmokeMode, policy: AutowareSmokePolicy, container_digest: str | None
) -> bool:
    if not policy.read_only_workspace or policy.network != "none":
        return False
    return mode != "container" or container_digest is not None


def _safe_workspace_cwd(workspace: Path, relative: str) -> Path:
    _validate_relative_path_text(relative, field_name="smoke cwd")
    workspace = workspace.resolve()
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise AutowareSmokeError(f"smoke cwd escapes workspace: {relative!r}") from exc
    if not candidate.exists() or not candidate.is_dir():
        raise AutowareSmokeError(f"smoke cwd is not an existing directory: {relative!r}")
    # Reject symlink/reparse components so a command cannot silently execute
    # from a path that changed identity after the plan was inspected.
    current = workspace
    for part in candidate.relative_to(workspace).parts:
        current = current / part
        if current.is_symlink():
            raise AutowareSmokeError(f"smoke cwd contains a symlink: {relative!r}")
        stat_result = current.stat()
        if getattr(stat_result, "st_file_attributes", 0) & 0x400:
            raise AutowareSmokeError(f"smoke cwd contains a reparse point: {relative!r}")
    return candidate


def _validate_relative_path_text(value: str, *, field_name: str) -> None:
    """Reject traversal in both POSIX and Windows path grammars."""

    if not isinstance(value, str) or not value.strip():
        raise AutowareSmokeError(f"{field_name} must be a non-empty relative path")
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part == ".." for part in posix.parts)
        or any(part == ".." for part in windows.parts)
    ):
        raise AutowareSmokeError(
            f"{field_name} must not be absolute or traverse parent directories"
        )


def _bounded_bytes(value: bytes, limit: int) -> bytes:
    return value if len(value) <= limit else value[:limit]


def _coerce_output(value: bytes | str | None, limit: int) -> bytes:
    if value is None:
        return b""
    raw = value.encode("utf-8", errors="replace") if isinstance(value, str) else value
    return _bounded_bytes(raw, limit)


def _decode_tail(value: bytes, limit: int) -> str:
    return value.decode("utf-8", errors="replace")[-limit:]


def _environment_digest(workspace: Path) -> str:
    return _canonical_sha256(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": str(Path(sys.executable).resolve()),
            "workspace": str(workspace.resolve()),
            "cwd": str(Path.cwd().resolve()),
        }
    )


def _smoke_id(promotion_id: str, mode: SmokeMode, commands: Sequence[Sequence[str]]) -> str:
    return hashlib.sha256(
        (promotion_id + "\0" + mode + "\0" + _canonical_json(list(commands))).encode("utf-8")
    ).hexdigest()[:24]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "AUTOWARE_SMOKE_PROTOCOL_ID",
    "AUTOWARE_SMOKE_SCHEMA_VERSION",
    "REQUIRED_SMOKE_STAGES",
    "SMOKE_STAGE_NAMES",
    "AutowareSmokeArtifact",
    "AutowareSmokeCommand",
    "AutowareSmokeConfig",
    "AutowareSmokeError",
    "AutowareSmokePolicy",
    "AutowareSmokeProvenance",
    "AutowareSmokeStage",
    "AutowareSmokeStageCommand",
    "AutowareSmokeStageEvidence",
    "autoware_smoke_artifact_json_schema",
    "autoware_smoke_json_schema",
    "import_autoware_smoke",
    "load_autoware_smoke",
    "run_autoware_smoke",
    "verify_autoware_smoke",
]
