"""Geographic range index for species identification reranking.

Builds an H3-4 cell → species index from iNaturalist's Open Range Map
dataset, matching iNat taxa to the BioCLIP species label set by scientific
name. The index is consumed at inference time by the Rust service to boost
visually-similar species that are plausible at a given lat/lon.

Requires the `geo` extra: pip install -e '.[geo]'

## Binary format (species_geo_index.bin)

Little-endian throughout; every field is a natural-alignment u32/u64 so the
Rust side can zero-copy load with bytemuck.

```
Header (32 bytes):
  magic[4]       = b"OGI1"   # Observing Geo Index v1
  version        = u32 = 1
  num_species    = u32         # BioCLIP label count this index targets
  h3_resolution  = u32         # e.g. 4
  num_cells      = u32
  num_entries    = u32         # total species IDs across all cells
  reserved       = u32, u32    # zero

Body:
  cells[num_cells]:      u64   # H3 indices, sorted ascending (binary-searchable)
  offsets[num_cells+1]:  u32   # CSR offsets into species_ids; offsets[0]=0,
                               # offsets[num_cells]=num_entries
  species_ids[num_entries]: u32  # BioCLIP row indices, sorted per cell
```

Lookup: binary-search `cells[]` for the target H3 cell; if found at index `i`,
the in-range species indices are `species_ids[offsets[i]..offsets[i+1]]`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MAGIC = b"OGI1"
VERSION = 1
HEADER_SIZE = 32  # 4 magic + 7 u32

# iNat range maps use `name` for scientific name; keep `scientific_name` as a
# secondary preference in case a consumer renames the column.
_NAME_COLS = ("scientific_name", "name")

# iNaturalist's public S3 bucket publishes monthly-refreshed polygon range
# maps (CC-BY-4.0) sharded by kingdom. The `latest` prefix always points to
# the newest build; the manifest enumerates the per-kingdom archive counts.
INAT_RANGE_MAPS_S3 = (
    "https://inaturalist-open-data.s3.us-east-1.amazonaws.com/geomodel/geopackages/latest"
)


def _manifest_url() -> str:
    return f"{INAT_RANGE_MAPS_S3}/metadata.json"


def _archive_url(kingdom: str, n: int, archives: int) -> str:
    # Kingdoms with only one archive omit the numeric suffix entirely
    # (e.g. iNaturalist_geomodel_Mammalia.gpkg), while multi-shard kingdoms
    # append _1, _2, ... (e.g. iNaturalist_geomodel_Aves_1.gpkg).
    suffix = "" if archives == 1 else f"_{n}"
    return f"{INAT_RANGE_MAPS_S3}/iNaturalist_geomodel_{kingdom}{suffix}.gpkg"


def _http_get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "bioclip-models/geo"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _head_content_length(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "bioclip-models/geo"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            val = resp.headers.get("Content-Length")
            return int(val) if val else None
    except urllib.error.URLError:
        return None


def _download_to(url: str, dest: Path) -> None:
    """Stream `url` to `dest`. Writes via a .partial sidecar so we never leave
    a half-finished file that looks done on re-run."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "bioclip-models/geo"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1024 * 1024):
            f.write(chunk)
    tmp.rename(dest)


def download_range_maps(dest_dir: Path) -> list[Path]:
    """Download the iNaturalist Open Range Map GeoPackage shards from S3.

    Reads the `latest/metadata.json` manifest, enumerates all per-kingdom
    `.gpkg` archives, and fetches any that are missing or size-mismatched in
    `dest_dir`. Returns the list of local paths.

    CC-BY-4.0; attribute "iNaturalist contributors" when redistributing.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching manifest: {_manifest_url()}")
    manifest = _http_get_json(_manifest_url())
    print(f"  version={manifest['version']} total_ranges={manifest['ranges']}")

    files: list[Path] = []
    for kingdom, info in sorted(manifest["collections"].items()):
        archives = info["archives"]
        for n in range(1, archives + 1):
            url = _archive_url(kingdom, n, archives)
            dest = dest_dir / Path(url).name
            files.append(dest)

            remote_size = _head_content_length(url)
            if dest.exists() and (remote_size is None or dest.stat().st_size == remote_size):
                print(f"  [skip] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
                continue

            size_str = f"{remote_size / 1e6:.0f} MB" if remote_size else "?"
            print(f"  [get]  {dest.name} ({size_str})")
            _download_to(url, dest)

    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"Downloaded {len(files)} archives, {total_mb:.0f} MB total")
    return files


def _iter_range_map_frames(path: Path):
    """Yield `(source_label, GeoDataFrame)` for each shard under `path`.

    If `path` is a directory, iterates its `.gpkg` files one at a time (so
    the 3.7 GB iNat sharded release doesn't need to live in memory all at
    once). Otherwise, yields a single frame for the file. Supports GeoPackage,
    GeoParquet, GeoJSON, Shapefile — anything pyogrio/fiona can open.
    """
    import geopandas as gpd

    if path.is_dir():
        gpkgs = sorted(path.glob("*.gpkg"))
        if not gpkgs:
            raise ValueError(f"No .gpkg files found in directory {path}")
        for gp in gpkgs:
            yield gp.name, gpd.read_file(gp)
        return

    name = path.name
    if path.suffix.lower() in (".parquet", ".geoparquet"):
        yield name, gpd.read_parquet(path)
    else:
        yield name, gpd.read_file(path)


def _polygon_to_cells(geom, resolution: int) -> set[int]:
    """Convert a shapely (Multi)Polygon to a set of H3 cell indices (u64).

    Uses h3-py v4's geo_to_h3shape + h3shape_to_cells with the `basic_int`
    API so cells come back as Python ints (bit-for-bit the H3 u64 index the
    Rust h3o crate expects). Centroid containment — cells whose centers fall
    inside the polygon — is the right default for range maps (non-overlapping
    partition of space).

    Returns an empty set on polygons H3 rejects (antimeridian-crossers,
    self-intersecting geometries, etc.). The caller logs skip counts.
    """
    import h3
    from h3.api import basic_int as h3_int

    if geom is None or geom.is_empty:
        return set()

    # MultiPolygons with antimeridian-crossing or self-touching components
    # can make H3's polyfill raise. Try the whole geometry first for speed,
    # but fall back to per-polygon so one bad sub-polygon doesn't drop the
    # entire range.
    try:
        shape = h3_int.geo_to_h3shape(geom.__geo_interface__)
        return set(h3_int.h3shape_to_cells(shape, res=resolution))
    except h3.H3FailedError:
        pass

    out: set[int] = set()
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        try:
            shape = h3_int.geo_to_h3shape(poly.__geo_interface__)
            out.update(h3_int.h3shape_to_cells(shape, res=resolution))
        except h3.H3FailedError:
            continue
    return out


def _species_to_cells(
    path: Path,
    name_to_idx: dict[str, int],
    resolution: int,
) -> dict[int, set[int]]:
    """Stream through all range-map shards under `path`, returning
    {bioclip_species_index → set of H3 cells}.

    Streams one shard at a time so peak memory stays bounded even for the
    full iNat release (~3.7 GB across 30 GeoPackages).
    """
    result: dict[int, set[int]] = {}
    total_records = 0
    total_matched_records = 0
    shards = 0

    for source, gdf in _iter_range_map_frames(path):
        shards += 1
        name_col = next((c for c in _NAME_COLS if c in gdf.columns), None)
        if name_col is None:
            raise ValueError(
                f"{source}: none of {_NAME_COLS} found; got {list(gdf.columns)}"
            )

        shard_matched = 0
        for name, geom in zip(gdf[name_col], gdf.geometry, strict=True):
            total_records += 1
            idx = name_to_idx.get(name)
            if idx is None:
                continue
            cells = _polygon_to_cells(geom, resolution)
            if not cells:
                continue
            # Duplicate taxa (e.g. same species across multiple model versions) union.
            result.setdefault(idx, set()).update(cells)
            shard_matched += 1
            total_matched_records += 1

        print(
            f"  [{shards}] {source}: {len(gdf)} records, {shard_matched} matched BioCLIP species"
        )
        # Release the shard's memory before loading the next one.
        del gdf

    match_rate = len(result) / max(len(name_to_idx), 1)
    print(
        f"  overall: {total_matched_records}/{total_records} records matched → "
        f"{len(result)}/{len(name_to_idx)} BioCLIP species ({match_rate:.1%})"
    )
    return result


def _invert_and_serialize(
    species_to_cells: dict[int, set[int]],
    num_species: int,
    resolution: int,
    output_path: Path,
) -> None:
    """Invert to {cell → species[]} and write the CSR binary via the Rust writer."""
    from species_range_index import (
        SpeciesRangeIndex,  # pyright: ignore[reportAttributeAccessIssue]
    )

    # Invert species → cells into cell → species[]. The Rust writer sorts cells,
    # sorts + dedups ids within each cell, and merges duplicate cells, so we hand
    # it the raw mapping rather than normalizing here.
    cells_to_species: dict[int, list[int]] = {}
    for species_idx, cells in species_to_cells.items():
        for cell in cells:
            cells_to_species.setdefault(cell, []).append(species_idx)

    SpeciesRangeIndex.write(str(output_path), num_species, resolution, cells_to_species)

    num_cells = len(cells_to_species)
    num_entries = sum(len(ids) for ids in cells_to_species.values())
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"  wrote {output_path}: {num_cells} cells, {num_entries} entries, "
        f"{size_mb:.1f} MB"
    )


def build_geo_index(
    range_maps_path: Path,
    species_labels_path: Path,
    output_path: Path,
    resolution: int = 4,
) -> None:
    """Build the species_geo_index.bin artifact.

    Args:
        range_maps_path: iNat Open Range Map dataset — any format geopandas
            can read. Must contain a `name` or `scientific_name` column and a
            polygon/multipolygon geometry.
        species_labels_path: species_labels.json from the bioclip-models
            pipeline. Provides the BioCLIP species list and index ordering.
        output_path: destination for species_geo_index.bin.
        resolution: H3 resolution (default 4, ~26km hex edges).
    """
    from .schema import SpeciesRecord

    print(f"Loading species labels from {species_labels_path}")
    with open(species_labels_path) as f:
        raw_labels = json.load(f)
    labels = [SpeciesRecord.model_validate(item) for item in raw_labels]
    num_species = len(labels)
    name_to_idx = {label.scientific_name: i for i, label in enumerate(labels)}
    print(f"  {num_species} species in label set")

    print(f"Rasterizing range maps under {range_maps_path} to H3-{resolution} cells")
    species_to_cells = _species_to_cells(range_maps_path, name_to_idx, resolution)

    total_cells = sum(len(v) for v in species_to_cells.values())
    print(f"  total species-cell entries: {total_cells}")

    _invert_and_serialize(species_to_cells, num_species, resolution, output_path)
