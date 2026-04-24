"""GBIF species list builder using the backbone taxonomy bulk download.

Downloads the GBIF backbone `simple.txt.gz` (~466 MB) and filters locally,
which is faster and more complete than paginated API calls.

Column layout of simple.txt (tab-delimited, no header):
  0: id                    10: kingdom_fk        20: genus_or_above
  1: parent_fk             11: phylum_fk         21: specific_epithet
  2: basionym_fk           12: class_fk          22: infra_specific_epithet
  3: is_synonym (t/f)      13: order_fk          23: notho_type
  4: status                14: family_fk         24: authorship
  5: rank                  15: genus_fk          25: year
  6: nom_status            16: species_fk        26: bracket_authorship
  7: constituent_key       17: name_id           27: bracket_year
  8: origin                18: scientific_name    28: name_published_in
  9: source_taxon_key      19: canonical_name     29: issues
"""

from __future__ import annotations

import gzip
import json
import random
import urllib.request
from pathlib import Path

from .schema import SpeciesRecord

BACKBONE_URL = "https://hosted-datasets.gbif.org/datasets/backbone/current/simple.txt.gz"

# Column indices
COL_ID = 0
COL_STATUS = 4
COL_RANK = 5
COL_CANONICAL_NAME = 19
COL_GENUS_OR_ABOVE = 20

# Foreign key columns for higher taxonomy
COL_KINGDOM_FK = 10
COL_PHYLUM_FK = 11
COL_CLASS_FK = 12
COL_ORDER_FK = 13
COL_FAMILY_FK = 14
COL_GENUS_FK = 15

KINGDOMS_TO_INCLUDE = {"Animalia", "Plantae", "Fungi"}

NULL = "\\N"


def _download_backbone(cache_path: Path) -> Path:
    """Download the GBIF backbone simple.txt.gz if not already cached."""
    if cache_path.exists():
        print(f"Using cached backbone: {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading GBIF backbone taxonomy (~466 MB)...")
    urllib.request.urlretrieve(BACKBONE_URL, str(cache_path))
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"  Downloaded ({size_mb:.0f} MB)")
    return cache_path


def build_species_list(
    total_count: int = 100_000,
    cache_dir: Path | None = None,
    kingdom_weights: dict[str, float] | None = None,
) -> list[SpeciesRecord]:
    """Build a species list from the GBIF backbone bulk download.

    Two-pass approach:
      1. Read all rows, build id→canonical_name lookup for resolving taxonomy FKs.
         Collect all ACCEPTED SPECIES rows.
      2. Resolve taxonomy names and sample to target count.
    """
    if cache_dir is None:
        cache_dir = Path("cache")
    if kingdom_weights is None:
        kingdom_weights = {"Animalia": 0.50, "Plantae": 0.35, "Fungi": 0.15}

    gz_path = _download_backbone(cache_dir / "simple.txt.gz")

    # Pass 1: build id→name lookup and collect accepted species
    print("Parsing backbone taxonomy...")
    id_to_name: dict[str, str] = {}
    # Each row: (canonical_name, kingdom_fk, phylum_fk, class_fk, order_fk, family_fk, genus_fk)
    species_rows: list[tuple[str, str, str, str, str, str, str]] = []
    line_count = 0

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 21:
                continue

            row_id = cols[COL_ID]
            canonical = cols[COL_CANONICAL_NAME]
            if canonical and canonical != NULL:
                id_to_name[row_id] = canonical

            status = cols[COL_STATUS]
            rank = cols[COL_RANK]
            if status == "ACCEPTED" and rank == "SPECIES":
                name = canonical if canonical and canonical != NULL else None
                if not name:
                    continue
                species_rows.append((
                    name,
                    cols[COL_KINGDOM_FK],
                    cols[COL_PHYLUM_FK],
                    cols[COL_CLASS_FK],
                    cols[COL_ORDER_FK],
                    cols[COL_FAMILY_FK],
                    cols[COL_GENUS_FK],
                ))

            if line_count % 2_000_000 == 0:
                print(f"  {line_count:,} rows parsed, {len(species_rows):,} accepted species...")

    print(f"  Total: {line_count:,} rows, {len(species_rows):,} accepted species, {len(id_to_name):,} names in lookup")

    # Pass 2: resolve taxonomy and filter by kingdom
    def resolve(fk: str) -> str | None:
        if fk == NULL or not fk:
            return None
        return id_to_name.get(fk)

    by_kingdom: dict[str, list[SpeciesRecord]] = {k: [] for k in KINGDOMS_TO_INCLUDE}

    for name, k_fk, p_fk, c_fk, o_fk, f_fk, g_fk in species_rows:
        kingdom = resolve(k_fk)
        if kingdom not in KINGDOMS_TO_INCLUDE:
            continue

        by_kingdom[kingdom].append(SpeciesRecord.model_validate({
            "scientificName": name,
            "kingdom": kingdom,
            "phylum": resolve(p_fk),
            "class": resolve(c_fk),
            "order": resolve(o_fk),
            "family": resolve(f_fk),
            "genus": resolve(g_fk),
        }))

    for k, species in by_kingdom.items():
        print(f"  {k}: {len(species):,} species")

    # Sample to target count per kingdom weight
    result: list[SpeciesRecord] = []
    seen: set[str] = set()

    for kingdom, weight in kingdom_weights.items():
        pool = by_kingdom.get(kingdom, [])
        target = int(total_count * weight)

        if len(pool) > target:
            random.shuffle(pool)
            pool = pool[:target]

        for sp in pool:
            if sp.scientific_name not in seen:
                seen.add(sp.scientific_name)
                result.append(sp)

    # Sort for deterministic output
    result.sort(key=lambda s: s.scientific_name)
    print(f"  Selected {len(result):,} species (target: {total_count:,})")
    return result[:total_count]


def save_species_list(species_list: list[SpeciesRecord], path: Path) -> None:
    """Save species list to JSON, using alias field names (scientificName, class, …)."""
    serialized = [sp.model_dump(by_alias=True) for sp in species_list]
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"Saved {len(species_list)} species to {path}")


def load_species_list(path: Path) -> list[SpeciesRecord]:
    """Load species list from JSON, validating each record against SpeciesRecord."""
    with open(path) as f:
        raw = json.load(f)
    return [SpeciesRecord.model_validate(item) for item in raw]
