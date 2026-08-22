"""Pydantic models for the BioCLIP pipeline's data shapes.

Validates external data at its entry points:
  - GBIF TSV rows are constructed into SpeciesRecord
  - iNat JSON responses parse through the Inat* models
  - Snapshot files on disk parse through EvalSnapshot

The internal flow uses list[SpeciesRecord] instead of list[dict].
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SpeciesRecord(BaseModel):
    """One species in the training/label set. Persisted as species_labels.json."""

    model_config = ConfigDict(populate_by_name=True)

    scientific_name: str = Field(alias="scientificName", min_length=1)
    common_name: str | None = Field(default=None, alias="commonName")
    kingdom: str | None = None
    phylum: str | None = None
    # `class` is a Python keyword; field stored as class_, serialized as "class".
    class_: str | None = Field(default=None, alias="class")
    order: str | None = None
    family: str | None = None
    genus: str | None = None

    def bundle_dump(self) -> dict[str, Any]:
        """Alias-keyed dict of only the fields the runtime service reads.

        Used for the copy of species_labels.json that goes into a release
        tarball; see bioclip_models.bundle.
        """
        return self.model_dump(by_alias=True, include=set(BUNDLE_FIELDS))


# Fields `SpeciesLabel` in observ-ing/core actually deserializes
# (crates/observing-species-id/src/embeddings.rs). Everything else in
# species_labels.json is parsed and thrown away on every cold start, so the
# bundled copy carries only these.
BUNDLE_FIELDS = ("scientific_name", "common_name", "kingdom")

# The ranks a bundle-trimmed record loses. export.py needs them to build
# BioCLIP's 7-rank taxonomic prompts, so a label set carrying none of them
# still loads fine but would silently produce taxonomy-free prompts.
TRIMMED_RANKS = ("phylum", "class_", "order", "family", "genus")


def rank_aliases(ranks: tuple[str, ...]) -> list[str]:
    """Field names as their on-disk JSON keys (`class_` -> `class`)."""
    return [SpeciesRecord.model_fields[r].alias or r for r in ranks]


# --- iNaturalist API response boundary ---

class InatTaxon(BaseModel):
    """Partial iNat taxon — only the fields we actually read."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    rank: str | None = None
    ancestor_ids: list[int] = Field(default_factory=list)


class InatSpeciesCountResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    taxon: InatTaxon


class InatSpeciesCountsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[InatSpeciesCountResult] = Field(default_factory=list)


class InatTaxaBatchItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    rank: str


class InatTaxaBatchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[InatTaxaBatchItem] = Field(default_factory=list)


class InatPhoto(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str


class InatObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    taxon: InatTaxon | None = None
    photos: list[InatPhoto] = Field(default_factory=list)


class InatObservationsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[InatObservation] = Field(default_factory=list)


# --- Eval snapshot (on-disk format) ---

class EvalObservation(BaseModel):
    """One row in the eval snapshot — a single iNat observation with a photo."""

    id: int
    inat_taxon_name: str
    image_url: str


class EvalSnapshot(BaseModel):
    """Top-level eval_data/snapshot.json structure."""

    version: Literal["1"] = "1"
    created_at: datetime
    species_count: int
    obs_per_species: int
    total_observations: int
    species: list[str]
    observations: list[EvalObservation]
