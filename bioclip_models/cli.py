"""CLI entry point for model preparation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .schema import SpeciesRecord


def _build_species_list(args: argparse.Namespace) -> list[SpeciesRecord]:
    """Build species list according to --strategy."""
    if args.strategy == "inat-ordered":
        from .inat import build_species_list_inat
        return build_species_list_inat(args.species_count)
    from .gbif import build_species_list
    return build_species_list(total_count=args.species_count)


def cmd_prepare(args: argparse.Namespace) -> None:
    """Run the full model preparation pipeline."""
    from .export import export_vision_encoder, generate_species_embeddings, load_bioclip
    from .gbif import load_species_list, save_species_list
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
        species_list = _build_species_list(args)
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

    # Step 5: (optional) Geo index
    if args.range_maps:
        from .geo import build_geo_index

        geo_index_path = output_dir / "species_geo_index.bin"
        build_geo_index(
            range_maps_path=args.range_maps,
            species_labels_path=labels_path,
            output_path=geo_index_path,
            resolution=args.h3_resolution,
        )

    # Step 6: Verify
    if not args.skip_verify:
        print("\n--- Verification ---")
        verify_all(output_dir)

    print(f"\nDone! Output in {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


def cmd_geo(args: argparse.Namespace) -> None:
    """Build the geographic range index as a standalone step."""
    from .geo import build_geo_index

    build_geo_index(
        range_maps_path=args.range_maps,
        species_labels_path=args.species_labels,
        output_path=args.output,
        resolution=args.h3_resolution,
    )


def cmd_download_range_maps(args: argparse.Namespace) -> None:
    """Download iNaturalist Open Range Map shards from their public S3 bucket."""
    from .geo import download_range_maps

    download_range_maps(args.dest)


def cmd_labels(args: argparse.Namespace) -> None:
    """Build species label set (without model export)."""
    from .gbif import save_species_list

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    species_list = _build_species_list(args)
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
    records = snapshot.observations
    print(
        f"Snapshot: {len(snapshot.species)} species, "
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
    p_prepare.add_argument(
        "--strategy",
        choices=["gbif-random", "inat-ordered"],
        default="gbif-random",
        help="Species list strategy (default: gbif-random)",
    )
    p_prepare.add_argument(
        "--range-maps",
        type=Path,
        default=None,
        help="iNat Open Range Map file (gpkg/parquet/geojson). If set, also "
        "builds species_geo_index.bin for geo-prior reranking.",
    )
    p_prepare.add_argument(
        "--h3-resolution",
        type=int,
        default=4,
        help="H3 resolution for the geo index (default: 4, ~26km hex edges)",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    # --- geo ---
    p_geo = subparsers.add_parser(
        "geo",
        help="Build species_geo_index.bin from an iNat range map file",
    )
    p_geo.add_argument(
        "--range-maps",
        type=Path,
        required=True,
        help="iNat Open Range Map file (gpkg/parquet/geojson)",
    )
    p_geo.add_argument(
        "--species-labels",
        type=Path,
        default=Path("output/species_labels.json"),
        help="Path to species_labels.json (default: output/species_labels.json)",
    )
    p_geo.add_argument(
        "--output",
        type=Path,
        default=Path("output/species_geo_index.bin"),
        help="Output binary path (default: output/species_geo_index.bin)",
    )
    p_geo.add_argument(
        "--h3-resolution",
        type=int,
        default=4,
        help="H3 resolution (default: 4, ~26km hex edges)",
    )
    p_geo.set_defaults(func=cmd_geo)

    # --- download-range-maps ---
    p_dl = subparsers.add_parser(
        "download-range-maps",
        help="Download iNat Open Range Map shards from S3 (CC-BY-4.0, ~7 GB)",
    )
    p_dl.add_argument(
        "--dest",
        type=Path,
        default=Path("cache/range_maps"),
        help="Destination directory (default: cache/range_maps)",
    )
    p_dl.set_defaults(func=cmd_download_range_maps)

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
    p_labels.add_argument(
        "--strategy",
        choices=["gbif-random", "inat-ordered"],
        default="gbif-random",
        help="Species list strategy (default: gbif-random)",
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
