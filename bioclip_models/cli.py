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


def cmd_fetch_eval_data(args: argparse.Namespace) -> None:
    """Fetch iNaturalist eval snapshot."""
    from .inat import build_eval_dataset, fetch_top_species, save_snapshot

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    species_names = fetch_top_species(args.species_count)
    records = build_eval_dataset(species_names, args.obs_per_species)
    save_snapshot(species_names, records, output)


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate model accuracy against an iNaturalist snapshot."""
    from .eval import (
        evaluate_snapshot,
        load_model_artifacts,
        print_results,
        save_results,
    )
    from .inat import load_snapshot

    snapshot = load_snapshot(args.snapshot)
    records = snapshot["observations"]
    print(
        f"Snapshot: {len(snapshot['species'])} species, "
        f"{len(records)} observations"
    )

    session, embeddings, labels_lookup = load_model_artifacts(args.model_dir)
    results = evaluate_snapshot(records, session, embeddings, labels_lookup, args.top_k)
    print_results(results)

    if args.output:
        save_results(results, args.output)


def main() -> None:
    """Entry point for the bioclip-prepare CLI."""
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
        help="Directory with vision_encoder.onnx, species_embeddings.bin, species_labels.json",
    )
    p_verify.set_defaults(func=cmd_verify)

    # --- fetch-eval-data ---
    p_fetch = subparsers.add_parser(
        "fetch-eval-data",
        help="Fetch iNaturalist eval snapshot (CC0 research-grade observations)",
    )
    p_fetch.add_argument(
        "--species-count",
        type=int,
        default=100,
        help="Top-N most observed iNat species to include (default: 100)",
    )
    p_fetch.add_argument(
        "--obs-per-species",
        type=int,
        default=10,
        help="Max observations per species (default: 10)",
    )
    p_fetch.add_argument(
        "--output",
        type=Path,
        default=Path("eval_data/snapshot.json"),
        help="Output snapshot path (default: eval_data/snapshot.json)",
    )
    p_fetch.set_defaults(func=cmd_fetch_eval_data)

    # --- eval ---
    p_eval = subparsers.add_parser(
        "eval",
        help="Evaluate model top-1/top-5 accuracy against an iNat snapshot",
    )
    p_eval.add_argument(
        "--model-dir",
        type=Path,
        default=Path("output"),
        help="Directory with model artifacts (default: output/)",
    )
    p_eval.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Snapshot JSON from fetch-eval-data",
    )
    p_eval.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K for accuracy metric (default: 5)",
    )
    p_eval.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results JSON",
    )
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
