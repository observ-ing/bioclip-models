"""iNaturalist API client for building species lists and evaluation datasets.

Fetches the top-N most observed species and research-grade CC0 observations
for use as a reproducible evaluation snapshot.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import (
    EvalObservation,
    EvalSnapshot,
    InatObservationsResponse,
    InatSpeciesCountsResponse,
    InatTaxaBatchItem,
    InatTaxaBatchResponse,
    InatTaxon,
    SpeciesRecord,
)

INAT_API = "https://api.inaturalist.org/v1"
INAT_API_V2 = "https://api.inaturalist.org/v2"

# Target ranks for species taxonomy (iNat rank field values)
_TAXONOMY_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus")

# Batch size for /v2/taxa ID lookups (iNat v2 taxa endpoint limit)
_TAXA_BATCH_SIZE = 30


def _get(url: str, params: dict[str, str | int]) -> dict[str, Any]:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full_url, headers={"User-Agent": "bioclip-models/eval"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            if attempt == 2:
                raise
            print(f"  Warning: request error (attempt {attempt + 1}/3): {e}, retrying...")
            time.sleep(5)
    raise RuntimeError("unreachable")  # pragma: no cover


def _fetch_taxa_batch(ids: list[int]) -> dict[int, InatTaxaBatchItem]:
    """Batch-fetch taxa by ID from iNat v2 API.

    Returns {id: InatTaxaBatchItem} for all requested IDs. Batches at
    _TAXA_BATCH_SIZE IDs per call, sleeping 1s between batches.
    """
    result: dict[int, InatTaxaBatchItem] = {}

    for i in range(0, len(ids), _TAXA_BATCH_SIZE):
        batch = ids[i: i + _TAXA_BATCH_SIZE]
        id_str = ",".join(str(x) for x in batch)
        url = f"{INAT_API_V2}/taxa/{id_str}?fields=id,name,rank"
        req = urllib.request.Request(url, headers={"User-Agent": "bioclip-models/eval"})

        # Retry up to 3 times on transient network errors
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                break
            except (urllib.error.URLError, OSError) as e:
                if attempt == 2:
                    raise
                print(f"  Warning: taxa batch error (attempt {attempt + 1}/3): {e}, retrying...")
                time.sleep(5)
        else:
            raise RuntimeError("unreachable")  # pragma: no cover

        response = InatTaxaBatchResponse.model_validate(data)
        for taxon in response.results:
            result[taxon.id] = taxon

        if i + _TAXA_BATCH_SIZE < len(ids):
            time.sleep(1)

    return result


def fetch_top_species(count: int) -> list[str]:
    """Fetch the top-N most observed species on iNaturalist (CC0, research grade).

    Returns a list of scientific names ordered by observation count descending.
    """
    print(f"Fetching top {count} most-observed CC0 species from iNaturalist...")
    names: list[str] = []
    page = 1
    per_page = min(500, count)

    while len(names) < count:
        data = _get(
            f"{INAT_API}/observations/species_counts",
            {
                "quality_grade": "research",
                "rank": "species",
                "license": "cc0",
                "per_page": per_page,
                "page": page,
            },
        )
        response = InatSpeciesCountsResponse.model_validate(data)
        if not response.results:
            break

        for r in response.results:
            if r.taxon.name and r.taxon.name not in names:
                names.append(r.taxon.name)
            if len(names) >= count:
                break

        print(f"  {len(names)}/{count} species...")
        if len(response.results) < per_page:
            break
        page += 1
        time.sleep(1)

    print(f"  Fetched {len(names)} species")
    return names[:count]


def build_species_list_inat(count: int) -> list[SpeciesRecord]:
    """Build a species list ordered by iNat observation frequency with full taxonomy.

    Steps:
    1. Fetch top N species with ancestor_ids from /observations/species_counts
       (research-grade, all licenses — no CC0 filter for frequency ranking)
    2. Collect all unique ancestor IDs across all species
    3. Batch-fetch ancestor taxa via /v2/taxa
    4. For each species, resolve kingdom/phylum/class/order/family/genus from ancestor chain
    5. Return list[SpeciesRecord] in observation-frequency order

    Output is compatible with species_labels.json (used by generate_species_embeddings).
    """
    print(f"Fetching top {count} most-observed species from iNaturalist...")
    taxa: list[InatTaxon] = []
    page = 1
    per_page = min(500, count)

    while len(taxa) < count:
        data = _get(
            f"{INAT_API}/observations/species_counts",
            {
                "quality_grade": "research",
                "rank": "species",
                "per_page": per_page,
                "page": page,
            },
        )
        response = InatSpeciesCountsResponse.model_validate(data)
        if not response.results:
            break

        for r in response.results:
            if r.taxon.name:
                taxa.append(r.taxon)
            if len(taxa) >= count:
                break

        print(f"  {len(taxa)}/{count} species fetched...")
        if len(response.results) < per_page:
            break
        page += 1
        time.sleep(1)

    taxa = taxa[:count]
    print(f"  Fetched {len(taxa)} species")

    # Collect all unique ancestor IDs (excluding the species taxon ID itself)
    all_ancestor_ids: set[int] = set()
    for taxon in taxa:
        for aid in taxon.ancestor_ids:
            if aid != taxon.id:
                all_ancestor_ids.add(aid)

    print(f"Resolving {len(all_ancestor_ids)} unique ancestor taxa...")
    ancestor_cache = _fetch_taxa_batch(sorted(all_ancestor_ids))
    print(f"  Resolved {len(ancestor_cache)} ancestors")

    # Build species records with resolved taxonomy
    species_list: list[SpeciesRecord] = []
    for taxon in taxa:
        taxonomy: dict[str, str | None] = {rank: None for rank in _TAXONOMY_RANKS}

        for aid in taxon.ancestor_ids:
            if aid == taxon.id:
                continue
            ancestor = ancestor_cache.get(aid)
            if ancestor and ancestor.rank in taxonomy:
                taxonomy[ancestor.rank] = ancestor.name

        species_list.append(SpeciesRecord.model_validate({
            "scientificName": taxon.name,
            "kingdom": taxonomy["kingdom"],
            "phylum": taxonomy["phylum"],
            "class": taxonomy["class"],
            "order": taxonomy["order"],
            "family": taxonomy["family"],
            "genus": taxonomy["genus"],
        }))

    matched = sum(1 for sp in species_list if sp.kingdom is not None)
    print(f"  Full taxonomy resolved for {matched}/{len(species_list)} species")
    return species_list


def fetch_observations(taxon_name: str, count: int) -> list[EvalObservation]:
    """Fetch research-grade CC0 observations with photos for a species."""
    records: list[EvalObservation] = []
    seen_ids: set[int] = set()
    page = 1

    while len(records) < count:
        try:
            data = _get(
                f"{INAT_API}/observations",
                {
                    "taxon_name": taxon_name,
                    "quality_grade": "research",
                    "photos": "true",
                    "license": "cc0",
                    "photo_license": "cc0",
                    "per_page": min(200, count - len(records)),
                    "page": page,
                },
            )
        except urllib.error.URLError as e:
            print(f"  Warning: API error for {taxon_name}: {e}")
            break

        response = InatObservationsResponse.model_validate(data)
        if not response.results:
            break

        for obs in response.results:
            if obs.id in seen_ids:
                continue
            seen_ids.add(obs.id)

            if not obs.photos:
                continue
            # Replace square thumbnail with medium (better resolution)
            url = obs.photos[0].url.replace("/square.", "/medium.")

            name = obs.taxon.name if obs.taxon and obs.taxon.name else taxon_name

            records.append(EvalObservation(id=obs.id, inat_taxon_name=name, image_url=url))
            if len(records) >= count:
                break

        if len(response.results) < 200:
            break
        page += 1
        time.sleep(1)

    return records


def build_eval_dataset(species_names: list[str], obs_per_species: int) -> list[EvalObservation]:
    """Fetch observations for each species and deduplicate by observation id."""
    all_records: list[EvalObservation] = []
    seen_ids: set[int] = set()

    for i, name in enumerate(species_names, 1):
        print(f"  [{i}/{len(species_names)}] {name}")
        records = fetch_observations(name, obs_per_species)
        for r in records:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                all_records.append(r)
        time.sleep(1)

    return all_records


def save_snapshot(
    species_names: list[str], records: list[EvalObservation], path: Path
) -> None:
    """Save eval snapshot to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = EvalSnapshot(
        version="1",
        created_at=datetime.now(UTC),
        species_count=len(species_names),
        obs_per_species=len(records) // max(len(species_names), 1),
        total_observations=len(records),
        species=species_names,
        observations=records,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(snapshot.model_dump_json(indent=2))
    print(f"Saved snapshot: {path} ({len(records)} observations, {len(species_names)} species)")


def load_snapshot(path: Path) -> EvalSnapshot:
    """Load and validate eval snapshot from JSON."""
    with open(path, encoding="utf-8") as f:
        return EvalSnapshot.model_validate_json(f.read())
