"""Release bundle packaging.

The pipeline's output directory is the source of truth: its
species_labels.json carries the full 8-field record that export.py needs to
build BioCLIP's 7-rank taxonomic prompts, and that `prepare` reuses as a cache
on re-runs.

The Rust service only deserializes three of those fields, so the copy that
goes into the release tarball is trimmed on the way in — roughly 58% smaller,
which matters because `observing-species-id-live` loads it on a memory-tight
Cloud Run cold start. Trimming happens here rather than in the output dir on
purpose: the trimmed file only ever exists inside the archive, so it can never
be fed back into `prepare` and silently degrade the next set of embeddings.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

from .gbif import load_species_list

# Archive members, in the order the tarball lays them out. `.onnx.data` is
# ONNX external-data storage, present only for models over the 2 GB protobuf
# limit; the geo index is optional (built by `prepare --range-maps`).
BUNDLE_MEMBERS = (
    "vision_encoder.onnx",
    "vision_encoder.onnx.data",
    "species_embeddings.bin",
    "species_labels.json",
    "species_geo_index.bin",
)

REQUIRED_MEMBERS = frozenset({
    "vision_encoder.onnx",
    "species_embeddings.bin",
    "species_labels.json",
})

LABELS_NAME = "species_labels.json"


def write_trimmed_labels(model_dir: Path, dest: Path) -> int:
    """Write model_dir's species_labels.json to `dest` with only bundle fields.

    Returns the record count. Records are validated through SpeciesRecord on
    the way through, so a corrupt labels file fails here rather than in the
    service.
    """
    species_list = load_species_list(model_dir / LABELS_NAME)
    with open(dest, "w") as f:
        json.dump([sp.bundle_dump() for sp in species_list], f, indent=2)
    return len(species_list)


def build_bundle(model_dir: Path, output_path: Path) -> None:
    """Package a model directory into a gzipped release tarball.

    Members are stored flat (no leading directory), matching the layout the
    core repo's image build expects to extract.
    """
    missing = sorted(n for n in REQUIRED_MEMBERS if not (model_dir / n).exists())
    if missing:
        raise FileNotFoundError(
            f"{model_dir} is missing required artifact(s): {', '.join(missing)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_labels_size = (model_dir / LABELS_NAME).stat().st_size

    with tempfile.TemporaryDirectory() as tmp:
        trimmed_labels = Path(tmp) / LABELS_NAME
        count = write_trimmed_labels(model_dir, trimmed_labels)
        trimmed_size = trimmed_labels.stat().st_size
        print(
            f"Trimmed {LABELS_NAME}: {count:,} species, "
            f"{_mib(full_labels_size)} -> {_mib(trimmed_size)} "
            f"({_mib(full_labels_size - trimmed_size)} dropped)"
        )

        print(f"Writing {output_path}...")
        with tarfile.open(output_path, "w:gz") as tar:
            for name in BUNDLE_MEMBERS:
                src = trimmed_labels if name == LABELS_NAME else model_dir / name
                if not src.exists():
                    print(f"  {name}: absent, skipping")
                    continue
                print(f"  {name}: {_mib(src.stat().st_size)}")
                tar.add(src, arcname=name)

    print(f"Done! {output_path} ({_mib(output_path.stat().st_size)} compressed)")


def _mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MiB"
