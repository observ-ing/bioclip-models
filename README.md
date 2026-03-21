# bioclip-models

Model preparation pipeline for [Observ.ing](https://github.com/observ-ing/core) species identification. Exports [BioCLIP 2.5](https://huggingface.co/imageomics/bioclip-2.5-vith14) (ViT-H/14) to ONNX and generates species text embeddings from GBIF taxonomy for zero-shot classification.

## Output artifacts

| File | Description | Size |
|------|-------------|------|
| `vision_encoder.onnx` | BioCLIP 2.5 ViT-H/14 vision encoder | ~1.2 GB |
| `species_embeddings.bin` | Pre-computed text embeddings, `[N, 1024]` f32 | ~400 MB |
| `species_labels.json` | Species metadata (name, taxonomy, common name) | ~10 MB |

These files are consumed by the `observing-species-id` Rust service in [observ-ing/core](https://github.com/observ-ing/core).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The `export` extra pulls in PyTorch and open_clip (~2 GB). The `verify` extra adds onnxruntime. Install only what you need:

```bash
pip install -e '.[export]'   # ONNX export + embedding generation
pip install -e '.[verify]'   # verify artifacts without PyTorch
```

## Usage

### Full pipeline

```bash
bioclip-prepare prepare --output-dir output/
```

This will:
1. Fetch 100,000 species from GBIF (Animalia, Plantae, Fungi)
2. Download BioCLIP 2.5 from HuggingFace
3. Export the vision encoder to ONNX
4. Generate text embeddings for all species
5. Verify the output

### Just build the species list

```bash
bioclip-prepare labels --species-count 100000 --output output/species_labels.json
```

### Verify existing artifacts

```bash
bioclip-prepare verify output/
```

### Use a custom species list

```bash
bioclip-prepare prepare --species-file my_species.json --output-dir output/
```

The JSON should be an array of objects with at minimum a `scientificName` field:

```json
[
  {
    "scientificName": "Turdus migratorius",
    "commonName": "American Robin",
    "kingdom": "Animalia",
    "family": "Turdidae",
    "genus": "Turdus"
  }
]
```

### Re-run with cached artifacts

```bash
# Skip ONNX export if vision_encoder.onnx already exists
bioclip-prepare prepare --skip-onnx --output-dir output/

# Re-fetch species from GBIF even if species_labels.json exists
bioclip-prepare prepare --rebuild-labels --output-dir output/
```

## Deploying to observ-ing/core

Copy the output to the core repo's model directory:

```bash
cp -r output/ ../core/models/bioclip/
```

Or symlink for development:

```bash
ln -s $(pwd)/output ../core/models/bioclip
```

## How it works

BioCLIP 2.5 is a CLIP-style model (ViT-H/14) trained on TreeOfLife-10M, a dataset of 2.7M images across ~450K taxa. It learns a shared embedding space for images and taxonomic text.

At inference time, the Rust service:
1. Runs the ONNX vision encoder on a user's photo to produce a 1024-dim image embedding
2. Computes cosine similarity against all pre-computed species text embeddings
3. Returns the top-K most similar species

The text embeddings encode the full taxonomic hierarchy ("Animalia, Turdidae, Turdus, Turdus migratorius") which is how BioCLIP was trained -- giving it strong hierarchical understanding even when the exact species is wrong.
