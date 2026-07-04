"""Public dataset catalog helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from slac.core.io import read_mapping

DEFAULT_PUBLIC_DATASET_CATALOG = Path("examples/public_datasets/catalog.yaml")


class PublicDatasetEntry(BaseModel):
    """One public dataset entry known to slac examples."""

    model_config = ConfigDict(extra="forbid")

    name: str
    family: str
    domain: str
    official_url: str
    download_url: str | None = None
    license_note: str
    slac_manifest: str | None = None
    slac_config: str | None = None
    recommended_pipeline: str


class PublicDatasetCatalog(BaseModel):
    """Public dataset catalog schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "slac.public_datasets/v0.1"
    datasets: dict[str, PublicDatasetEntry] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        return self.model_dump(mode="json")


def load_public_dataset_catalog(
    path: str | Path = DEFAULT_PUBLIC_DATASET_CATALOG,
) -> PublicDatasetCatalog:
    """Load the public dataset catalog."""

    return PublicDatasetCatalog.model_validate(read_mapping(Path(path)))
