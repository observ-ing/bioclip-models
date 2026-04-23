"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Pinned polygons used across tests — covers western US so San Francisco
# (37.77, -122.42) falls inside range A but outside range B, and Salt Lake
# City (40.76, -111.89) falls inside both. This lets tests make lookups that
# exercise "species subset per cell" without needing the real iNat dataset.
_RANGE_A_COORDS = [[-130, 30], [-110, 30], [-110, 45], [-130, 45], [-130, 30]]
_RANGE_B_COORDS = [[-120, 35], [-100, 35], [-100, 50], [-120, 50], [-120, 35]]


@pytest.fixture
def sample_labels() -> list[dict]:
    """A three-species BioCLIP label set.

    Species at index 2 (Ursus arctos) intentionally has no range, so tests
    can confirm that species without range maps stay out of the index.
    """
    return [
        {"scientificName": "Anas platyrhynchos"},
        {"scientificName": "Apis mellifera"},
        {"scientificName": "Ursus arctos"},
    ]


@pytest.fixture
def sample_labels_file(sample_labels: list[dict], tmp_path: Path) -> Path:
    path = tmp_path / "species_labels.json"
    path.write_text(json.dumps(sample_labels))
    return path


@pytest.fixture
def sample_ranges_geojson(tmp_path: Path) -> Path:
    """GeoJSON with ranges for species 0 and 1; species 2 is unmapped."""
    ranges = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Anas platyrhynchos"},
                "geometry": {"type": "Polygon", "coordinates": [_RANGE_A_COORDS]},
            },
            {
                "type": "Feature",
                "properties": {"name": "Apis mellifera"},
                "geometry": {"type": "Polygon", "coordinates": [_RANGE_B_COORDS]},
            },
        ],
    }
    path = tmp_path / "ranges.geojson"
    path.write_text(json.dumps(ranges))
    return path
