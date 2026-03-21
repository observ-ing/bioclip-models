"""CLI entry point for model preparation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_prepare(args: argparse.Namespace) -> None:
    """Run the full model preparation pipeline."""
    from .export import export_vision_encoder, generate_species_embeddings, load_bioclip
    from .gbif import build_species_list, load_species_list, save_species_list
    from .verify import verify_all

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Species list
    labels_path = output_dir / "species_labels.json"
    if args.species_file:
        print(f"Loading species list from {args.species_file}")
        species_list = load_species_list(args.species_file)
    elif labels_path.exists() and not args.rebuild_labels:
        print(f"Using existing species list: {labels_path}")
        species_list = load_species_list(labels_path)
    else:
        species_list = build_species_list(total_count=args.species_count)
        save_species_list(species_list, labels_path)

    print(f"Species count: {len(species_list)}")

    # Step 2: Load BioCLIP
    model, tokenizer, _ = load_bioclip()

    # Step 3: Export vision encoder
    onnx_path = output_dir / "vision_encoder.onnx"
    if onnx_path.exists() and args.skip_onnx:
        print(f"Skipping ONNX export (existing: {onnx_path})")
    else:
        export_vision_encoder(model, onnx_path)

    # Step 4: Generate embeddings
    embeddings_path = output_dir / "species_embeddings.bin"
    generate_species_embeddings(model, tokenizer, species_list, embeddings_path)

    # Step 5: Verify
    if not args.skip_verify:
        print("\n--- Verification ---")
        verify_all(output_dir)

    print(f"\nDone! Output in {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


def cmd_labels(args: argparse.Namespace) -> None:
    """Build species label set from GBIF (without model export)."""
    from .gbif import build_species_list, save_species_list

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    species_list = build_species_list(
        total_count=args.species_count,
    )
    save_species_list(species_list, output_path)


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify exported model artifacts."""
    from .verify import verify_all

    verify_all(args.model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bioclip-prepare",
        description="BioCLIP 2.5 model preparation pipeline for Observ.ing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- prepare ---
    p_prepare = subparsers.add_parser(
        "prepare",
        help="Run full pipeline: fetch species, export ONNX, generate embeddings",
    )
    p_prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )
    p_prepare.add_argument(
        "--species-count",
        type=int,
        default=100_000,
        help="Target number of species (default: 100000)",
    )
    p_prepare.add_argument(
        "--species-file",
        type=Path,
        default=None,
        help="Use a pre-built species list JSON instead of fetching from GBIF",
    )
    p_prepare.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export if vision_encoder.onnx already exists",
    )
    p_prepare.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step",
    )
    p_prepare.add_argument(
        "--rebuild-labels",
        action="store_true",
        help="Re-fetch species list even if species_labels.json exists",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    # --- labels ---
    p_labels = subparsers.add_parser(
        "labels",
        help="Build species label set from GBIF (no model required)",
    )
    p_labels.add_argument(
        "--output",
        type=Path,
        default=Path("output/species_labels.json"),
        help="Output JSON path",
    )
    p_labels.add_argument(
        "--species-count",
        type=int,
        default=100_000,
        help="Target number of species",
    )
    p_labels.set_defaults(func=cmd_labels)

    # --- verify ---
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify exported model artifacts",
    )
    p_verify.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing vision_encoder.onnx, species_embeddings.bin, species_labels.json",
    )
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
