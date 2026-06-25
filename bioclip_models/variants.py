"""Registry of supported BioCLIP model variants.

Kept dependency-light (no torch) so the CLI can list and resolve variants
without importing the heavy `export` module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    """One exportable BioCLIP checkpoint plus its derived-artifact metadata.

    `version` is the string the `observing-species-id` service reports via
    `MODEL_VERSION` (and the suggested image tag). `embed_dim` is the text/image
    embedding width, used as a post-export sanity check against the generated
    `species_embeddings.bin`.
    """

    key: str
    hf_hub: str
    embed_dim: int
    version: str
    description: str


MODEL_VARIANTS: dict[str, ModelVariant] = {
    "bioclip-2.5-vith14": ModelVariant(
        key="bioclip-2.5-vith14",
        hf_hub="hf-hub:imageomics/bioclip-2.5-vith14",
        embed_dim=1024,
        version="bioclip-2.5-vit-h-14",
        description="Full-accuracy ViT-H/14 — upload/capture path",
    ),
    "bioclip-2": ModelVariant(
        key="bioclip-2",
        hf_hub="hf-hub:imageomics/bioclip-2",
        embed_dim=768,
        version="bioclip-2-vit-l-14",
        description="Faster ViT-L/14 — live camera loop",
    ),
}

DEFAULT_VARIANT = "bioclip-2.5-vith14"


def resolve_variant(key: str) -> ModelVariant:
    """Look up a variant by key, raising a helpful error for unknown keys."""
    try:
        return MODEL_VARIANTS[key]
    except KeyError:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown model variant {key!r}. Choices: {choices}") from None
