"""iNaturalist API client for building evaluation datasets.

Fetches the top-N most observed species and research-grade CC0 observations
for use as a reproducible evaluation snapshot.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

INAT_API = "https://api.inaturalist.org/v1"


def _get(url: str, params: dict) -> dict:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full_url, headers={"User-Agent": "bioclip-models/eval"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_top_species(count: int) -> list[str]:
    """Fetch the top-N most observed species on iNaturalist (CC0, research grade).

    Returns a list of scientific names ordered by observation count descending.
    """
    print(f"Fetching top {count} most-observed species from iNaturalist...")
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
        results = data.get("results", [])
        if not results:
            break

        for r in results:
            taxon = r.get("taxon", {})
            name = taxon.get("name")
            if name and name not in names:
                names.append(name)
            if len(names) >= count:
                break

        print(f"  {len(names)}/{count} species...")
        if len(results) < per_page:
            break
        page += 1
        time.sleep(0.5)

    print(f"  Fetched {len(names)} species")
    return names[:count]


def fetch_observations(taxon_name: str, count: int) -> list[dict]:
    """Fetch research-grade CC0 observations with photos for a species.

    Returns list of {id, inat_taxon_name, image_url} dicts.
    """
    records: list[dict] = []
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

        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            obs_id = obs.get("id")
            if obs_id in seen_ids:
                continue
            seen_ids.add(obs_id)

            photos = obs.get("photos", [])
            if not photos:
                continue
            url = photos[0].get("url", "")
            # Replace square thumbnail with medium (better resolution)
            url = url.replace("/square.", "/medium.")

            taxon = obs.get("taxon", {})
            name = taxon.get("name", taxon_name)

            records.append({"id": obs_id, "inat_taxon_name": name, "image_url": url})
            if len(records) >= count:
                break

        if len(results) < 200:
            break
        page += 1
        time.sleep(0.5)

    return records


def build_eval_dataset(species_names: list[str], obs_per_species: int) -> list[dict]:
    """Fetch observations for each species and deduplicate by observation id."""
    all_records: list[dict] = []
    seen_ids: set[int] = set()

    for i, name in enumerate(species_names, 1):
        print(f"  [{i}/{len(species_names)}] {name}")
        records = fetch_observations(name, obs_per_species)
        for r in records:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_records.append(r)
        time.sleep(0.5)

    return all_records


def save_snapshot(species_names: list[str], records: list[dict], path: Path) -> None:
    """Save eval snapshot to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "version": "1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "species_count": len(species_names),
        "obs_per_species": len(records) // max(len(species_names), 1),
        "total_observations": len(records),
        "species": species_names,
        "observations": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Saved snapshot: {path} ({len(records)} observations, {len(species_names)} species)")


def load_snapshot(path: Path) -> dict:
    """Load eval snapshot from JSON. Returns the full snapshot dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
