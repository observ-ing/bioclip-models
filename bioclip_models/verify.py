"""Verification utilities for exported models.

Requires the `verify` extra: pip install -e '.[verify]'
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import SpeciesRecord


def verify_onnx_model(onnx_path: Path) -> int:
    """Verify the ONNX vision encoder produces valid output."""
    import onnxruntime as ort

    print(f"Verifying ONNX model: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path))

    # Check input/output metadata
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"  Inputs:  {[(i.name, i.shape, i.type) for i in inputs]}")
    print(f"  Outputs: {[(o.name, o.shape, o.type) for o in outputs]}")

    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    assert inputs[0].name == "pixel_values"

    # Infer embedding dimension from ONNX output metadata
    embed_dim = outputs[0].shape[1]
    print(f"  Embedding dimension: {embed_dim}")

    # Run inference with random data. ONNX run() is typed as a union with
    # SparseTensor/OrtValue; for this model we always get a plain ndarray.
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = session.run(None, {"pixel_values": dummy})

    embedding = np.asarray(result[0])
    print(f"  Output shape: {embedding.shape}")
    print(f"  Output dtype: {embedding.dtype}")
    assert embedding.shape == (1, embed_dim), f"Unexpected shape: {embedding.shape}"

    # Check that the output is not all zeros or NaN
    assert not np.isnan(embedding).any(), "Output contains NaN"
    assert np.abs(embedding).sum() > 0, "Output is all zeros"

    print("  ONNX verification passed")
    return embed_dim


def verify_embeddings(embeddings_path: Path, labels_path: Path, embed_dim: int | None = None) -> None:
    """Verify the species embeddings and labels are consistent."""
    print(f"Verifying embeddings: {embeddings_path}")

    with open(labels_path) as f:
        raw_labels = json.load(f)
    labels = [SpeciesRecord.model_validate(item) for item in raw_labels]
    num_species = len(labels)

    data = np.fromfile(str(embeddings_path), dtype=np.float32)

    # Infer embedding dimension from file size and label count if not provided
    if embed_dim is None:
        assert data.size % num_species == 0, (
            f"Embeddings size {data.size} is not divisible by species count {num_species}"
        )
        embed_dim = data.size // num_species
        print(f"  Inferred embedding dimension: {embed_dim}")

    expected_size = num_species * embed_dim
    assert data.size == expected_size, (
        f"Embeddings size mismatch: {data.size} != {num_species} * {embed_dim} = {expected_size}"
    )

    embeddings = data.reshape(num_species, embed_dim)
    print(f"  Shape: {embeddings.shape}")
    print(f"  Species count: {num_species}")

    # Check L2 normalization (each row should have norm ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = norms.mean()
    max_deviation = np.abs(norms - 1.0).max()
    print(f"  Mean norm: {mean_norm:.6f} (expected ~1.0)")
    print(f"  Max norm deviation: {max_deviation:.6f}")
    assert max_deviation < 0.01, f"Embeddings are not L2-normalized (max deviation: {max_deviation})"

    # Check no NaN/Inf
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN"
    assert not np.isinf(embeddings).any(), "Embeddings contain Inf"

    # SpeciesRecord validation above already enforced scientific_name presence.
    print(f"  Sample labels: {[s.scientific_name for s in labels[:5]]}")

    print("  Embeddings verification passed")


def verify_geo_index(geo_index_path: Path, labels_path: Path) -> None:
    """Sanity-check the species geo index binary.

    Verifies the header, that cells are sorted, that offsets are monotonic and
    terminate at num_entries, and that every species index is in-bounds for
    the label set.
    """
    from species_range_index import SpeciesRangeIndex

    print(f"Verifying geo index: {geo_index_path}")

    with open(labels_path) as f:
        raw_labels = json.load(f)
    # Validate at the boundary, same pattern as verify_embeddings.
    labels = [SpeciesRecord.model_validate(item) for item in raw_labels]
    num_species = len(labels)

    # Validate magic / version / size / CSR endpoints and staleness through the
    # production Rust reader — the exact code the service loads with — instead of
    # a parallel hand-rolled header parser. A bad/stale/corrupt file (incl. a
    # num_species mismatch) raises ValueError here.
    idx = SpeciesRangeIndex.load(geo_index_path, expected_count=num_species)
    num_cells = idx.num_cells
    num_entries = idx.num_entries
    print(
        f"  version=1 h3_res={idx.resolution} "
        f"num_cells={num_cells} num_entries={num_entries}"
    )

    # The deeper CSR integrity checks below (cells strictly ascending, offsets
    # monotonic, species ids in-bounds, per-cell stats) aren't exposed by the
    # reader's API, so read the raw body directly for those.
    data = geo_index_path.read_bytes()
    off = 32
    cells = np.frombuffer(data, dtype=np.uint64, count=num_cells, offset=off)
    off += num_cells * 8
    offsets = np.frombuffer(data, dtype=np.uint32, count=num_cells + 1, offset=off)
    off += (num_cells + 1) * 4
    species_ids = np.frombuffer(data, dtype=np.uint32, count=num_entries, offset=off)

    if num_cells > 1:
        assert np.all(np.diff(cells.astype(np.int64)) > 0), "cells[] is not strictly ascending"
    assert offsets[0] == 0, "offsets[0] must be 0"
    assert offsets[-1] == num_entries, (
        f"offsets[-1] ({offsets[-1]}) != num_entries ({num_entries})"
    )
    assert np.all(np.diff(offsets.astype(np.int64)) >= 0), "offsets must be monotonic"
    if num_entries:
        assert species_ids.max() < num_species, (
            f"species_ids contains out-of-range value {species_ids.max()} "
            f">= num_species {num_species}"
        )

    avg = num_entries / num_cells if num_cells else 0
    print(f"  avg species per cell: {avg:.0f}; max per cell: "
          f"{int(np.diff(offsets).max()) if num_cells else 0}")
    print("  Geo index verification passed")


def verify_all(model_dir: Path) -> None:
    """Run all verification checks on a model directory."""
    onnx_path = model_dir / "vision_encoder.onnx"
    embeddings_path = model_dir / "species_embeddings.bin"
    labels_path = model_dir / "species_labels.json"
    geo_index_path = model_dir / "species_geo_index.bin"

    for path in [onnx_path, embeddings_path, labels_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    embed_dim = verify_onnx_model(onnx_path)
    verify_embeddings(embeddings_path, labels_path, embed_dim=embed_dim)

    if geo_index_path.exists():
        verify_geo_index(geo_index_path, labels_path)
    else:
        print(f"Geo index not present ({geo_index_path.name}) — skipping")

    # Quick end-to-end test: embed an image and find top matches
    print("\nEnd-to-end test:")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    image_embedding = np.asarray(session.run(None, {"pixel_values": dummy_image})[0])[0]

    # L2 normalize
    image_embedding = image_embedding / np.linalg.norm(image_embedding)

    with open(labels_path) as f:
        raw_labels = json.load(f)
    labels = [SpeciesRecord.model_validate(item) for item in raw_labels]

    embeddings = np.fromfile(str(embeddings_path), dtype=np.float32).reshape(-1, embed_dim)

    # Cosine similarity
    similarities = embeddings @ image_embedding
    top_indices = np.argsort(similarities)[::-1][:5]

    print("  Top 5 matches for random noise (scores should be low and similar):")
    for idx in top_indices:
        print(f"    {labels[idx].scientific_name}: {similarities[idx]:.4f}")

    print("\nAll verifications passed!")
