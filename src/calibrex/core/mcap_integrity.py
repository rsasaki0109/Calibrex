"""Typed MCAP container-integrity evidence.

The MCAP reader is implemented in :mod:`calibrex.data.mcap`, but the evidence
contract lives in the ROS-independent core.  Keeping the result here makes it
safe for capture manifests and downstream adapters to consume the evidence
without importing ROS, the optional ``mcap`` package, or a compression codec.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from calibrex.core.result import StrictModel

MCAP_INTEGRITY_SCHEMA_VERSION: Literal["slac.mcap_integrity/v0.1"] = (
    "slac.mcap_integrity/v0.1"
)

McapFieldStatus = Literal["known", "unknown", "not_applicable", "blocked"]
McapIntegrityStatus = Literal["pass", "warn", "blocked"]

_McapDiagnostic = Annotated[str, StringConstraints(max_length=512)]
_McapCompression = Annotated[str, StringConstraints(max_length=128)]


class McapIntegrityEvidence(StrictModel):
    """Bounded, schema-valid evidence from an MCAP framing/integrity scan.

    ``unknown`` is intentionally distinct from ``blocked``.  For example, a
    writer may omit the optional DataEnd or summary CRC, or a compressed chunk
    may carry an uncompressed CRC that cannot be checked without the codec.
    Those facts are reported as unknown rather than being treated as a valid
    CRC.  A malformed frame, dangling channel/schema reference, or a mismatch
    in an encoded CRC is a blocking condition.
    """

    schema_version: Literal["slac.mcap_integrity/v0.1"] = MCAP_INTEGRITY_SCHEMA_VERSION
    status: McapIntegrityStatus = "warn"

    magic_status: McapFieldStatus = "unknown"
    framing_status: McapFieldStatus = "unknown"
    header_status: McapFieldStatus = "unknown"
    data_end_status: McapFieldStatus = "unknown"
    footer_status: McapFieldStatus = "unknown"
    links_status: McapFieldStatus = "unknown"
    summary_status: McapFieldStatus = "unknown"
    summary_offset_status: McapFieldStatus = "unknown"
    index_status: McapFieldStatus = "unknown"
    crc_status: McapFieldStatus = "unknown"
    data_crc_status: McapFieldStatus = "unknown"
    summary_crc_status: McapFieldStatus = "unknown"
    chunk_crc_status: McapFieldStatus = "not_applicable"
    compression_status: McapFieldStatus = "not_applicable"

    summary_present: bool = False
    summary_offset_present: bool = False
    index_present: bool = False
    data_crc_encoded: bool = False
    summary_crc_encoded: bool = False

    file_size_bytes: int | None = Field(default=None, ge=0)
    records_checked: int = Field(default=0, ge=0)
    data_records_checked: int = Field(default=0, ge=0)
    summary_records_checked: int = Field(default=0, ge=0)
    schema_count: int = Field(default=0, ge=0)
    channel_count: int = Field(default=0, ge=0)
    message_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    index_count: int = Field(default=0, ge=0)

    unsupported_compressions: list[_McapCompression] = Field(default_factory=list, max_length=16)
    errors: list[_McapDiagnostic] = Field(default_factory=list, max_length=32)
    warnings: list[_McapDiagnostic] = Field(default_factory=list, max_length=32)

    @property
    def overall_status(self) -> McapIntegrityStatus:
        """Compatibility spelling for callers that prefer an explicit name."""

        return self.status

    @property
    def data_section_crc_status(self) -> McapFieldStatus:
        """Return the DataEnd CRC verification state."""

        return self.data_crc_status

    @property
    def summary_section_crc_status(self) -> McapFieldStatus:
        """Return the Footer summary CRC verification state."""

        return self.summary_crc_status


def mcap_integrity_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for :class:`McapIntegrityEvidence`."""

    return McapIntegrityEvidence.model_json_schema()


__all__ = [
    "MCAP_INTEGRITY_SCHEMA_VERSION",
    "McapFieldStatus",
    "McapIntegrityEvidence",
    "McapIntegrityStatus",
    "mcap_integrity_json_schema",
]
