"""Tests for bioclip_models.geo.

The binary format is the coordination point with the Rust service, so these
tests focus on format invariants (magic, header, CSR structure) and on the
taxon-name matching logic. H3 and geopandas internals are not re-tested here.
"""

from __future__ import annotations

import bisect
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from bioclip_models.geo import (
    HEADER_SIZE,
    MAGIC,
    VERSION,
    _archive_url,
    _polygon_to_cells,
    build_geo_index,
)
from bioclip_models.verify import verify_geo_index


def _unpack_header(data: bytes) -> dict:
    """Parse the 32-byte header into a dict."""
    assert data[:4] == MAGIC, f"bad magic: {data[:4]!r}"
    version, num_species, h3_res, num_cells, num_entries, _, _ = struct.unpack(
        "<IIIIIII", data[4:HEADER_SIZE]
    )
    return {
        "version": version,
        "num_species": num_species,
        "h3_resolution": h3_res,
        "num_cells": num_cells,
        "num_entries": num_entries,
    }


def _unpack_body(data: bytes, num_cells: int, num_entries: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(cells, offsets, species_ids)` from the body."""
    off = HEADER_SIZE
    cells = np.frombuffer(data, dtype=np.uint64, count=num_cells, offset=off)
    off += num_cells * 8
    offsets = np.frombuffer(data, dtype=np.uint32, count=num_cells + 1, offset=off)
    off += (num_cells + 1) * 4
    species_ids = np.frombuffer(data, dtype=np.uint32, count=num_entries, offset=off)
    return cells, offsets, species_ids


# ---------------------------------------------------------------------------
# _archive_url — regression tests for the URL naming convention
# ---------------------------------------------------------------------------


def test_archive_url_single_archive_omits_suffix():
    # Kingdoms with a single archive use the bare name (no `_1` suffix).
    url = _archive_url("Mammalia", 1, archives=1)
    assert url.endswith("/iNaturalist_geomodel_Mammalia.gpkg")


def test_archive_url_multi_archive_appends_number():
    url1 = _archive_url("Aves", 1, archives=2)
    url2 = _archive_url("Aves", 2, archives=2)
    assert url1.endswith("/iNaturalist_geomodel_Aves_1.gpkg")
    assert url2.endswith("/iNaturalist_geomodel_Aves_2.gpkg")


# ---------------------------------------------------------------------------
# _polygon_to_cells — shim over h3-py
# ---------------------------------------------------------------------------


def test_polygon_to_cells_none():
    assert _polygon_to_cells(None, 4) == set()


def test_polygon_to_cells_empty_geometry():
    from shapely.geometry import Polygon

    assert _polygon_to_cells(Polygon(), 4) == set()


def test_polygon_to_cells_contains_known_point():
    """A California-sized polygon must contain the H3-4 cell of San Francisco."""
    from h3.api import basic_int as h3
    from shapely.geometry import Polygon

    poly = Polygon([(-124, 32), (-114, 32), (-114, 42), (-124, 42), (-124, 32)])
    cells = _polygon_to_cells(poly, 4)
    sf_cell = h3.latlng_to_cell(37.77, -122.42, 4)
    assert sf_cell in cells


# ---------------------------------------------------------------------------
# build_geo_index end-to-end
# ---------------------------------------------------------------------------


def test_build_geo_index_header_is_correct(
    sample_labels_file: Path, sample_ranges_geojson: Path, tmp_path: Path
):
    out = tmp_path / "index.bin"
    build_geo_index(sample_ranges_geojson, sample_labels_file, out, resolution=4)

    data = out.read_bytes()
    hdr = _unpack_header(data)
    assert hdr["version"] == VERSION
    assert hdr["num_species"] == 3
    assert hdr["h3_resolution"] == 4
    assert hdr["num_cells"] > 0
    assert hdr["num_entries"] > 0


def test_build_geo_index_body_size_matches_header(
    sample_labels_file: Path, sample_ranges_geojson: Path, tmp_path: Path
):
    out = tmp_path / "index.bin"
    build_geo_index(sample_ranges_geojson, sample_labels_file, out)

    data = out.read_bytes()
    hdr = _unpack_header(data)
    expected = HEADER_SIZE + hdr["num_cells"] * 8 + (hdr["num_cells"] + 1) * 4 + hdr["num_entries"] * 4
    assert len(data) == expected


def test_build_geo_index_csr_invariants(
    sample_labels_file: Path, sample_ranges_geojson: Path, tmp_path: Path
):
    """Cells sorted ascending; offsets monotonic; species_ids in-bounds and
    sorted within each cell."""
    out = tmp_path / "index.bin"
    build_geo_index(sample_ranges_geojson, sample_labels_file, out)

    data = out.read_bytes()
    hdr = _unpack_header(data)
    cells, offsets, species_ids = _unpack_body(data, hdr["num_cells"], hdr["num_entries"])

    assert (np.diff(cells.astype(np.int64)) > 0).all(), "cells must be strictly ascending"
    assert offsets[0] == 0
    assert offsets[-1] == hdr["num_entries"]
    assert (np.diff(offsets.astype(np.int64)) >= 0).all(), "offsets must be monotonic"
    assert species_ids.max() < hdr["num_species"]

    for i in range(hdr["num_cells"]):
        per_cell = species_ids[offsets[i] : offsets[i + 1]]
        assert (np.diff(per_cell.astype(np.int64)) > 0).all(), (
            f"species_ids for cell {i} not sorted: {per_cell.tolist()}"
        )


def test_build_geo_index_lookup_reflects_polygons(
    sample_labels_file: Path, sample_ranges_geojson: Path, tmp_path: Path
):
    """SF falls in species 0's range only; SLC falls in both 0 and 1."""
    from h3.api import basic_int as h3

    out = tmp_path / "index.bin"
    build_geo_index(sample_ranges_geojson, sample_labels_file, out)

    data = out.read_bytes()
    hdr = _unpack_header(data)
    cells, offsets, species_ids = _unpack_body(data, hdr["num_cells"], hdr["num_entries"])
    cells_list = cells.tolist()

    def species_at(lat: float, lon: float) -> list[int]:
        target = h3.latlng_to_cell(lat, lon, 4)
        i = bisect.bisect_left(cells_list, target)
        if i == len(cells_list) or cells_list[i] != target:
            return []
        return species_ids[offsets[i] : offsets[i + 1]].tolist()

    assert species_at(37.77, -122.42) == [0], "SF should be in range A only"
    assert species_at(40.76, -111.89) == [0, 1], "SLC should be in both A and B"


def test_build_geo_index_unmatched_taxa_produce_empty_index(
    sample_labels_file: Path, tmp_path: Path
):
    """iNat taxa that aren't in the BioCLIP label set are silently dropped —
    the result is a valid (empty) binary."""
    ranges = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Definitely Not A Real Species"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-124, 32], [-114, 32], [-114, 42], [-124, 42], [-124, 32]]],
                },
            }
        ],
    }
    range_file = tmp_path / "ranges.geojson"
    range_file.write_text(json.dumps(ranges))

    out = tmp_path / "index.bin"
    build_geo_index(range_file, sample_labels_file, out)

    hdr = _unpack_header(out.read_bytes())
    assert hdr["num_cells"] == 0
    assert hdr["num_entries"] == 0
    # The header still asserts the right label count so a stale consumer fails loudly.
    assert hdr["num_species"] == 3


def test_build_geo_index_accepts_directory_of_shards(
    sample_labels_file: Path, tmp_path: Path
):
    """When `range_maps_path` is a directory, every .gpkg inside is consumed —
    used for iNat's per-kingdom sharded release."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()

    # Two separate shards, each with a single taxon from our label set.
    gpd.GeoDataFrame(
        {"name": ["Anas platyrhynchos"]},
        geometry=[Polygon([(-130, 30), (-110, 30), (-110, 45), (-130, 45), (-130, 30)])],
        crs="EPSG:4326",
    ).to_file(shards_dir / "a.gpkg", driver="GPKG")

    gpd.GeoDataFrame(
        {"name": ["Apis mellifera"]},
        geometry=[Polygon([(-120, 35), (-100, 35), (-100, 50), (-120, 50), (-120, 35)])],
        crs="EPSG:4326",
    ).to_file(shards_dir / "b.gpkg", driver="GPKG")

    out = tmp_path / "index.bin"
    build_geo_index(shards_dir, sample_labels_file, out)

    hdr = _unpack_header(out.read_bytes())
    # Both species matched → non-empty index.
    assert hdr["num_cells"] > 0
    assert hdr["num_entries"] > 0


# ---------------------------------------------------------------------------
# verify_geo_index — catches format drift and staleness
# ---------------------------------------------------------------------------


def test_verify_accepts_well_formed_index(
    sample_labels_file: Path, sample_ranges_geojson: Path, tmp_path: Path
):
    out = tmp_path / "index.bin"
    build_geo_index(sample_ranges_geojson, sample_labels_file, out)
    verify_geo_index(out, sample_labels_file)  # must not raise


def test_verify_rejects_bad_magic(tmp_path: Path):
    bad = tmp_path / "bad.bin"
    # 32 bytes but wrong magic — long enough to reach the magic check.
    bad.write_bytes(b"XXXX" + b"\x00" * 28)
    labels = tmp_path / "labels.json"
    labels.write_text("[]")

    with pytest.raises(ValueError, match="magic"):
        verify_geo_index(bad, labels)


def test_verify_rejects_stale_num_species(tmp_path: Path, sample_labels: list[dict]):
    """A num_species mismatch between index header and labels file is a loud
    failure — otherwise we'd silently mis-index at inference."""
    path = tmp_path / "stale.bin"
    # Header: valid magic/version/resolution, but num_species=42 vs 3 labels.
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIIIII", VERSION, 42, 4, 0, 0, 0, 0))

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(sample_labels))  # 3 entries

    with pytest.raises(ValueError, match="stale"):
        verify_geo_index(path, labels_path)
