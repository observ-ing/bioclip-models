"""Tests for bioclip_models.bundle and the trimmed-labels guard.

The bundle's species_labels.json drops five taxonomy ranks that export.py
needs, so the two things worth pinning down are: the trimmed file really only
carries what the Rust service reads, and a trimmed file can never be fed back
into embedding generation without an error.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from bioclip_models.bundle import (
    BUNDLE_MEMBERS,
    build_bundle,
    write_trimmed_labels,
)
from bioclip_models.gbif import load_species_list, save_species_list
from bioclip_models.schema import SpeciesRecord

_FULL_RECORD = {
    "scientificName": "Turdus migratorius",
    "commonName": "American Robin",
    "kingdom": "Animalia",
    "phylum": "Chordata",
    "class": "Aves",
    "order": "Passeriformes",
    "family": "Turdidae",
    "genus": "Turdus",
}


@pytest.fixture
def full_labels() -> list[dict]:
    """Two fully-populated records plus one with sparse higher taxonomy.

    The sparse record mirrors real GBIF rows where a rank fails to resolve —
    the trim guard must not mistake those for a trimmed file.
    """
    return [
        _FULL_RECORD,
        {**_FULL_RECORD, "scientificName": "Apis mellifera", "commonName": None},
        {"scientificName": "Ursus arctos", "kingdom": "Animalia", "genus": "Ursus"},
    ]


@pytest.fixture
def model_dir(full_labels: list[dict], tmp_path: Path) -> Path:
    """A minimal prepared model directory: all required members, tiny stubs."""
    d = tmp_path / "output"
    d.mkdir()
    (d / "species_labels.json").write_text(json.dumps(full_labels, indent=2))
    (d / "vision_encoder.onnx").write_bytes(b"onnx-stub")
    (d / "species_embeddings.bin").write_bytes(b"\x00" * 48)
    return d


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------


def test_bundle_dump_keeps_only_service_fields() -> None:
    dumped = SpeciesRecord.model_validate(_FULL_RECORD).bundle_dump()
    assert dumped == {
        "scientificName": "Turdus migratorius",
        "commonName": "American Robin",
        "kingdom": "Animalia",
    }


def test_bundle_dump_keeps_null_common_name() -> None:
    """The key stays even when empty — the Rust side deserializes it as a field."""
    record = SpeciesRecord.model_validate({"scientificName": "Apis mellifera"})
    assert record.bundle_dump() == {
        "scientificName": "Apis mellifera",
        "commonName": None,
        "kingdom": None,
    }


def test_write_trimmed_labels_preserves_order_and_count(
    model_dir: Path, full_labels: list[dict], tmp_path: Path
) -> None:
    """Record order is the embeddings' row order — trimming must not disturb it."""
    dest = tmp_path / "trimmed.json"
    count = write_trimmed_labels(model_dir, dest)

    trimmed = json.loads(dest.read_text())
    assert count == len(full_labels) == len(trimmed)
    assert [r["scientificName"] for r in trimmed] == [
        r["scientificName"] for r in full_labels
    ]
    assert all(set(r) == {"scientificName", "commonName", "kingdom"} for r in trimmed)


def test_write_trimmed_labels_is_smaller(model_dir: Path, tmp_path: Path) -> None:
    dest = tmp_path / "trimmed.json"
    write_trimmed_labels(model_dir, dest)
    assert dest.stat().st_size < (model_dir / "species_labels.json").stat().st_size


def test_write_trimmed_labels_rejects_corrupt_records(
    model_dir: Path, tmp_path: Path
) -> None:
    """Validation happens here, not in the service."""
    (model_dir / "species_labels.json").write_text(json.dumps([{"kingdom": "Animalia"}]))
    with pytest.raises(ValidationError):
        write_trimmed_labels(model_dir, tmp_path / "trimmed.json")


# ---------------------------------------------------------------------------
# build_bundle
# ---------------------------------------------------------------------------


def test_build_bundle_contains_flat_required_members(
    model_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "models.tar.gz"
    build_bundle(model_dir, out)

    with tarfile.open(out) as tar:
        names = tar.getnames()
    assert names == [
        "vision_encoder.onnx",
        "species_embeddings.bin",
        "species_labels.json",
    ]


def test_build_bundle_includes_optional_members_when_present(
    model_dir: Path, tmp_path: Path
) -> None:
    (model_dir / "vision_encoder.onnx.data").write_bytes(b"external-data")
    (model_dir / "species_geo_index.bin").write_bytes(b"OGI1")
    out = tmp_path / "models.tar.gz"
    build_bundle(model_dir, out)

    with tarfile.open(out) as tar:
        assert tar.getnames() == list(BUNDLE_MEMBERS)


def test_build_bundle_ships_trimmed_labels(model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "models.tar.gz"
    build_bundle(model_dir, out)

    with tarfile.open(out) as tar:
        member = tar.extractfile("species_labels.json")
        assert member is not None
        bundled = json.loads(member.read())

    assert all(set(r) == {"scientificName", "commonName", "kingdom"} for r in bundled)


def test_build_bundle_leaves_output_dir_untouched(
    model_dir: Path, full_labels: list[dict], tmp_path: Path
) -> None:
    """The pipeline's own labels file keeps its taxonomy — that's the whole point."""
    build_bundle(model_dir, tmp_path / "models.tar.gz")
    assert json.loads((model_dir / "species_labels.json").read_text()) == full_labels


def test_build_bundle_rejects_incomplete_model_dir(
    model_dir: Path, tmp_path: Path
) -> None:
    (model_dir / "species_embeddings.bin").unlink()
    with pytest.raises(FileNotFoundError, match="species_embeddings.bin"):
        build_bundle(model_dir, tmp_path / "models.tar.gz")


# ---------------------------------------------------------------------------
# The regression guard — a trimmed file must never reach embedding generation
# ---------------------------------------------------------------------------


def test_load_species_list_rejects_trimmed_labels(model_dir: Path, tmp_path: Path) -> None:
    trimmed = tmp_path / "species_labels.json"
    write_trimmed_labels(model_dir, trimmed)

    with pytest.raises(ValueError, match="no taxonomy ranks"):
        load_species_list(trimmed, require_taxonomy=True)


def test_load_species_list_accepts_trimmed_labels_without_the_guard(
    model_dir: Path, tmp_path: Path
) -> None:
    """geo.py, verify.py and eval.py read a labels file without needing ranks."""
    trimmed = tmp_path / "species_labels.json"
    write_trimmed_labels(model_dir, trimmed)
    assert len(load_species_list(trimmed)) == 3


def test_load_species_list_accepts_sparse_but_real_taxonomy(
    model_dir: Path, tmp_path: Path
) -> None:
    """One record with only a genus is enough to prove the set isn't trimmed."""
    sparse = tmp_path / "species_labels.json"
    sparse.write_text(json.dumps([
        {"scientificName": "Anas platyrhynchos", "kingdom": "Animalia"},
        {"scientificName": "Ursus arctos", "kingdom": "Animalia", "genus": "Ursus"},
    ]))
    assert len(load_species_list(sparse, require_taxonomy=True)) == 2


def test_load_species_list_guard_ignores_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "species_labels.json"
    empty.write_text("[]")
    assert load_species_list(empty, require_taxonomy=True) == []


def test_saved_species_list_round_trips_through_the_guard(tmp_path: Path) -> None:
    """What save_species_list writes is always accepted by the guard."""
    path = tmp_path / "species_labels.json"
    save_species_list([SpeciesRecord.model_validate(_FULL_RECORD)], path)
    assert len(load_species_list(path, require_taxonomy=True)) == 1
