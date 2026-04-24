"""BioCLIP evaluation: top-1/top-5 accuracy against an iNaturalist snapshot.

Requires the `eval` extra: pip install -e '.[eval]'
"""

from __future__ import annotations

import json
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .schema import EvalObservation, SpeciesRecord

if TYPE_CHECKING:
    import onnxruntime as ort

# CLIP normalization constants — must match preprocessing.rs
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130260, 0.27577710], dtype=np.float32)


def normalize_name(name: str) -> str:
    """Normalize a scientific name for case-insensitive lookup."""
    return name.lower().strip()


def _species_name(name: str) -> str:
    """Return the species-level binomial from a scientific name.

    Drops any subspecies/variety epithet beyond the first two words,
    e.g. 'Anas platyrhynchos domesticus' -> 'Anas platyrhynchos'.
    """
    parts = name.strip().split()
    return " ".join(parts[:2]) if len(parts) > 2 else name


def load_model_artifacts(
    model_dir: Path,
) -> tuple["ort.InferenceSession", np.ndarray, dict[str, str]]:
    """Load ONNX session, embeddings, and labels lookup from a model directory.

    Returns:
        session: ONNX InferenceSession
        embeddings: float32 array of shape [N, embed_dim], L2-normalized
        labels_lookup: dict mapping normalize_name(scientificName) -> scientificName
    """
    import onnxruntime as ort

    onnx_path = model_dir / "vision_encoder.onnx"
    embeddings_path = model_dir / "species_embeddings.bin"
    labels_path = model_dir / "species_labels.json"

    for p in (onnx_path, embeddings_path, labels_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    session = ort.InferenceSession(str(onnx_path))

    with open(labels_path, encoding="utf-8") as f:
        raw_labels = json.load(f)
    labels = [SpeciesRecord.model_validate(item) for item in raw_labels]
    num_species = len(labels)

    data = np.fromfile(str(embeddings_path), dtype=np.float32)
    assert data.size % num_species == 0, (
        f"Embeddings size {data.size} not divisible by species count {num_species}"
    )
    embed_dim = data.size // num_species
    embeddings = data.reshape(num_species, embed_dim)

    labels_lookup = {normalize_name(sp.scientific_name): sp.scientific_name for sp in labels}

    print(
        f"Loaded model: {num_species} species, embed_dim={embed_dim}, "
        f"onnx={onnx_path.name}"
    )
    return session, embeddings, labels_lookup


def preprocess_image_bytes(data: bytes) -> np.ndarray:
    """Decode and preprocess image bytes to CLIP input tensor.

    Returns float32 NCHW array of shape [1, 3, 224, 224].
    """
    from PIL import Image

    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BICUBIC)

    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    arr = (arr - _MEAN) / _STD                      # normalize
    arr = arr.transpose(2, 0, 1)                    # [3, H, W]
    return arr[np.newaxis]                           # [1, 3, H, W]


def preprocess_image_url(url: str) -> np.ndarray:
    """Fetch an image from URL and preprocess it."""
    req = urllib.request.Request(url, headers={"User-Agent": "bioclip-models/eval"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return preprocess_image_bytes(data)


def run_inference(
    session: "ort.InferenceSession",
    image_tensor: np.ndarray,
    embeddings: np.ndarray,
    labels_lookup: dict[str, str],
    top_k: int = 5,
) -> list[str]:
    """Run ONNX inference and return top-k scientific names."""
    # ONNX run() is typed as a union; narrow to ndarray for the known output.
    image_embeds = np.asarray(session.run(None, {"pixel_values": image_tensor})[0])[0]

    # L2 normalize
    norm = np.linalg.norm(image_embeds)
    if norm > 0:
        image_embeds = image_embeds / norm

    similarities = embeddings @ image_embeds
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Reconstruct original name list to index into
    # labels_lookup values in insertion order = species order in embeddings
    names = list(labels_lookup.values())
    return [names[i] for i in top_indices]


def evaluate_snapshot(
    records: list[EvalObservation],
    session: "ort.InferenceSession",
    embeddings: np.ndarray,
    labels_lookup: dict[str, str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate all observations in a snapshot.

    For each observation:
    - If inat_taxon_name not in labels_lookup: label_miss (score 0)
    - Otherwise: run inference, check top-1 and top-k

    Returns aggregate and per-species results dict.
    """
    # Group by species for per-species stats
    species_stats: dict[str, dict[str, Any]] = {}
    total = 0
    error_count = 0
    top1_correct = 0
    topk_correct = 0

    for rec in records:
        inat_name = rec.inat_taxon_name
        image_url = rec.image_url

        if inat_name not in species_stats:
            norm = normalize_name(inat_name)
            # Fall back to species-level binomial for subspecies names
            canonical_key = norm if norm in labels_lookup else normalize_name(_species_name(inat_name))
            matched = canonical_key in labels_lookup
            species_stats[inat_name] = {
                "_canonical_key": canonical_key if matched else None,
                "in_list": matched,
                "n": 0,
                "top1": 0,
                "topk": 0,
                "errors": 0,
            }

        stats = species_stats[inat_name]
        stats["n"] += 1
        total += 1

        if not stats["in_list"]:
            # Label miss — counts as wrong
            continue

        try:
            tensor = preprocess_image_url(image_url)
            predictions = run_inference(session, tensor, embeddings, labels_lookup, top_k)
        except Exception as e:
            print(f"  Error on {inat_name} ({image_url}): {e}")
            stats["errors"] += 1
            error_count += 1
            continue

        canonical = labels_lookup[stats["_canonical_key"]]
        if predictions[0] == canonical:
            stats["top1"] += 1
            top1_correct += 1
        if canonical in predictions:
            stats["topk"] += 1
            topk_correct += 1

    label_misses = sum(1 for s in species_stats.values() if not s["in_list"])
    in_list_species = len(species_stats) - label_misses
    # Label misses and errors score 0; all observations count in denominator
    scored_obs = total - error_count

    return {
        "top_k": top_k,
        "species_in_snapshot": len(species_stats),
        "species_in_model": in_list_species,
        "label_misses": label_misses,
        "total_observations": total,
        "error_count": error_count,
        "scored_observations": scored_obs,
        "top1_correct": top1_correct,
        "topk_correct": topk_correct,
        "top1_accuracy": top1_correct / scored_obs if scored_obs > 0 else 0.0,
        "topk_accuracy": topk_correct / scored_obs if scored_obs > 0 else 0.0,
        "per_species": species_stats,
    }


def print_results(results: dict[str, Any]) -> None:
    """Print a human-readable summary of eval results."""
    k = results["top_k"]
    print("\n=== BioCLIP Eval Results ===")
    print(
        f"Species in snapshot: {results['species_in_snapshot']}  |  "
        f"in model list: {results['species_in_model']}  |  "
        f"label misses: {results['label_misses']}"
    )
    print(
        f"Total observations: {results['total_observations']}  |  "
        f"errors: {results['error_count']}"
    )
    total = results["total_observations"]
    print()
    top1_pct = results["top1_accuracy"] * 100
    topk_pct = results["topk_accuracy"] * 100
    print(f"Top-1: {top1_pct:.1f}%  ({results['top1_correct']} / {total})")
    print(f"Top-{k}: {topk_pct:.1f}%  ({results['topk_correct']} / {total})")

    per_species = results["per_species"]
    if not per_species:
        return

    print("\nPer-species (sorted by top-1 asc):")
    header = f"  {'Species':<35} {'In list?':<10} {'N':>4}  {'Top-1':>6}  {'Top-' + str(k):>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def sort_key(item):
        _, s = item
        if not s["in_list"]:
            return -1.0
        n = s["n"] - s["errors"]
        return s["top1"] / n if n > 0 else 0.0

    for name, s in sorted(per_species.items(), key=sort_key):
        in_list = "yes" if s["in_list"] else "no"
        n = s["n"]
        if not s["in_list"] or n == 0:
            print(f"  {name:<35} {in_list:<10} {n:>4}  {'--':>6}  {'--':>6}")
        else:
            scored_n = n - s["errors"]
            t1 = f"{s['top1'] / scored_n * 100:.0f}%" if scored_n > 0 else "--"
            tk = f"{s['topk'] / scored_n * 100:.0f}%" if scored_n > 0 else "--"
            print(f"  {name:<35} {in_list:<10} {n:>4}  {t1:>6}  {tk:>6}")


def save_results(results: dict[str, Any], path: Path) -> None:
    """Save eval results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {path}")
