"""GBIF species list builder.

Fetches accepted species from the GBIF backbone taxonomy to build the label set
for BioCLIP zero-shot classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

# GBIF higher taxon keys for major kingdoms
KINGDOMS = {
    "Animalia": 1,
    "Plantae": 6,
    "Fungi": 5,
}

# Use the backbone list endpoint — supports deep pagination unlike /search
GBIF_SPECIES_LIST = "https://api.gbif.org/v1/species"


def fetch_species_for_kingdom(
    taxon_key: int,
    count: int,
    existing_names: set[str] | None = None,
) -> list[dict]:
    """Fetch accepted species from a GBIF higher taxon.

    Uses the backbone species list endpoint which supports deep pagination.

    Args:
        taxon_key: GBIF higher taxon key (e.g. 1 for Animalia).
        count: Maximum number of species to fetch.
        existing_names: Set of scientific names to skip (for deduplication).

    Returns:
        List of species dicts with taxonomy fields.
    """
    if existing_names is None:
        existing_names = set()

    species_list: list[dict] = []
    offset = 0
    limit = 1000

    while len(species_list) < count:
        params = {
            "rank": "SPECIES",
            "status": "ACCEPTED",
            "limit": min(limit, count - len(species_list)),
            "offset": offset,
            "highertaxonKey": str(taxon_key),
        }

        try:
            resp = requests.get(GBIF_SPECIES_LIST, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  GBIF request failed at offset {offset}: {e}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for sp in results:
            name = sp.get("species", sp.get("canonicalName", sp.get("scientificName", "")))
            if not name or name in existing_names:
                continue

            existing_names.add(name)
            species_list.append({
                "scientificName": name,
                "commonName": sp.get("vernacularName"),
                "kingdom": sp.get("kingdom"),
                "phylum": sp.get("phylum"),
                "class": sp.get("class"),
                "order": sp.get("order"),
                "family": sp.get("family"),
                "genus": sp.get("genus"),
            })

        offset += len(results)

        if len(species_list) % 5000 < limit:
            print(f"    {len(species_list)} species collected...")

        if data.get("endOfRecords", False):
            break

    return species_list


def build_species_list(
    total_count: int = 50_000,
    kingdom_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Build a species list from multiple GBIF kingdoms.

    Args:
        total_count: Target number of species.
        kingdom_weights: Fraction of total to allocate per kingdom.
            Defaults to 50% Animalia, 35% Plantae, 15% Fungi.

    Returns:
        Deduplicated list of species dicts.
    """
    if kingdom_weights is None:
        kingdom_weights = {"Animalia": 0.50, "Plantae": 0.35, "Fungi": 0.15}

    all_species: list[dict] = []
    seen_names: set[str] = set()

    for kingdom, weight in kingdom_weights.items():
        taxon_key = KINGDOMS.get(kingdom)
        if taxon_key is None:
            print(f"  Warning: unknown kingdom '{kingdom}', skipping")
            continue

        target = int(total_count * weight)
        print(f"Fetching {target} species from {kingdom} (key={taxon_key})...")

        batch = fetch_species_for_kingdom(taxon_key, target, seen_names)
        all_species.extend(batch)
        print(f"  Got {len(batch)} species from {kingdom} (total: {len(all_species)})")

    return all_species[:total_count]


def save_species_list(species_list: list[dict], path: Path) -> None:
    """Save species list to JSON."""
    with open(path, "w") as f:
        json.dump(species_list, f, indent=2)
    print(f"Saved {len(species_list)} species to {path}")


def load_species_list(path: Path) -> list[dict]:
    """Load species list from JSON."""
    with open(path) as f:
        return json.load(f)
