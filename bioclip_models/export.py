"""BioCLIP ONNX export and species embedding generation.

Requires the `export` extra: pip install -e '.[export]'
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_bioclip():
    """Load BioCLIP 2.5 (ViT-H/14) model and tokenizer from HuggingFace via open_clip."""
    import open_clip

    model_name = "hf-hub:imageomics/bioclip-2.5-vith14"
    print(f"Loading BioCLIP 2.5 from HuggingFace ({model_name})...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    print("  Model loaded")
    return model, tokenizer, preprocess


def export_vision_encoder(model, output_path: Path) -> None:
    """Export the vision encoder to ONNX format.

    The exported model accepts [batch, 3, 224, 224] float32 input
    and produces [batch, embed_dim] float32 image embeddings.
    """

    class VisionEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.visual = clip_model.visual

        def forward(self, pixel_values):
            return self.visual(pixel_values)

    vision = VisionEncoder(model)
    vision.eval()

    dummy = torch.randn(1, 3, 224, 224)

    print(f"Exporting vision encoder to {output_path}...")
    torch.onnx.export(
        vision,
        dummy,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
        opset_version=17,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved ({size_mb:.1f} MB)")


def generate_species_embeddings(
    model,
    tokenizer,
    species_list: list[dict],
    output_path: Path,
    batch_size: int = 128,
    device: "torch.device | None" = None,
) -> np.ndarray:
    """Generate L2-normalized text embeddings for all species.

    Uses BioCLIP 2.5's text encoder with taxonomic hierarchy as input text,
    matching how the model was trained on TreeOfLife-10M.

    The output is a flat binary file of f32 values, shape [N, embed_dim].
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Generating embeddings for {len(species_list)} species on {device}...")
    model = model.to(device)

    all_embeddings = []

    for i in range(0, len(species_list), batch_size):
        batch = species_list[i : i + batch_size]

        # Build taxonomic text for each species.
        # BioCLIP was trained with space-separated full 7-rank taxonomic names:
        # "Kingdom Phylum Class Order Family Genus epithet"
        # (matching the `taxonomic_name` property in BioCLIP 2's naming_eval.py)
        texts = []
        for sp in batch:
            parts = []
            for rank in ("kingdom", "phylum", "class", "order", "family"):
                val = sp.get(rank)
                if val:
                    parts.append(val.capitalize())
            genus = sp.get("genus") or ""
            if genus:
                parts.append(genus.capitalize())
            # Extract species epithet from scientificName (strip genus prefix)
            scientific = sp["scientificName"]
            if genus and scientific.lower().startswith(genus.lower() + " "):
                epithet = scientific[len(genus) + 1:].strip().lower()
            else:
                epithet = scientific.lower()
            if epithet:
                parts.append(epithet)
            texts.append(" ".join(parts))

        tokens = tokenizer(texts).to(device)

        with torch.no_grad():
            features = model.encode_text(tokens)
            # L2 normalize so cosine similarity = dot product
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())

        processed = min(i + batch_size, len(species_list))
        if processed % 1000 < batch_size:
            print(f"  {processed}/{len(species_list)}")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save as raw little-endian f32
    embeddings.tofile(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.1f} MB)")

    return embeddings
