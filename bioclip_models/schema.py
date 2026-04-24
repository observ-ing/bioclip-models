"""Pydantic models for the BioCLIP pipeline's data shapes.

Validates external data at its entry points:
  - GBIF TSV rows are constructed into SpeciesRecord
  - iNat JSON responses parse through the Inat* models
  - Snapshot files on disk parse through EvalSnapshot

The internal flow uses list[SpeciesRecord] instead of list[dict].
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

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
