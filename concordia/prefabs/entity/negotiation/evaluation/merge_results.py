#!/usr/bin/env python3
"""
Merge Results from Parallel Pod Execution

After running experiments in parallel across multiple pods, use this script
to merge all activation files and train probes on the combined dataset.

Usage:
    python merge_results.py outputs/

This will:
1. Find all activations_*.pt files in subdirectories
2. Merge them into a single dataset
3. Train probes on the combined data
4. Generate final analysis report
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import torch


def find_activation_files(base_dir: str) -> List[str]:
    """Find all activation files in subdirectories."""
    patterns = [
        os.path.join(base_dir, "**/activations_*.pt"),
        os.path.join(base_dir, "activations_*.pt"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    return sorted(set(files))


def merge_activation_files(files: List[str], output_path: str) -> dict:
    """Merge multiple activation files into one."""
    print(f"\nMerging {len(files)} activation files...")

    all_samples = []
    metadata = {
        "source_files": files,
        "merged_at": datetime.now().isoformat(),
        "scenarios": set(),
    }

    for f in files:
        print(f"  Loading: {f}")
        data = torch.load(f, map_location="cpu", weights_only=False)

        if "samples" in data:
            samples = data["samples"]
        elif isinstance(data, list):
            samples = data
        else:
            print(f"    Warning: Unknown format in {f}, skipping")
            continue

        print(f"    Found {len(samples)} samples")
        all_samples.extend(samples)

        # Track scenarios
        for s in samples:
            if hasattr(s, "emergent_scenario") and s.emergent_scenario:
                metadata["scenarios"].add(s.emergent_scenario)

    metadata["scenarios"] = list(metadata["scenarios"])
    metadata["total_samples"] = len(all_samples)

    # Save merged file
    print(f"\nSaving merged dataset to: {output_path}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Scenarios: {metadata['scenarios']}")

    torch.save({
        "samples": all_samples,
        "metadata": metadata,
    }, output_path)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Merge activation files from parallel pod execution"
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing output subdirectories"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for merged file (default: merged_activations.pt)"
    )
    parser.add_argument(
        "--train-probes",
        action="store_true",
        help="Train probes on merged data"
    )

    args = parser.parse_args()

    # Find activation files
    files = find_activation_files(args.base_dir)

    if not files:
        print(f"No activation files found in {args.base_dir}")
        sys.exit(1)

    print(f"Found {len(files)} activation files:")
    for f in files:
        print(f"  - {f}")

    # Merge files
    output_path = args.output or os.path.join(args.base_dir, "merged_activations.pt")
    metadata = merge_activation_files(files, output_path)

    # Optionally train probes
    if args.train_probes:
        print("\n" + "=" * 60)
        print("TRAINING PROBES ON MERGED DATA")
        print("=" * 60)

        try:
            from concordia.prefabs.entity.negotiation.evaluation import run_full_analysis
            results = run_full_analysis(output_path)

            print("\nProbe Results:")
            if results.get("best_probe"):
                print(f"  Best layer: {results['best_probe']['layer']}")
                print(f"  Best R²: {results['best_probe']['r2']:.3f}")

            if results.get("gm_vs_agent"):
                gm_vs = results["gm_vs_agent"]
                print(f"\n  GM R²: {gm_vs['gm_ridge_r2']:.3f}")
                print(f"  Agent R²: {gm_vs['agent_ridge_r2']:.3f}")
                if gm_vs["gm_wins"]:
                    print("  >> GM labels more predictable")

        except ImportError as e:
            print(f"Could not import analysis tools: {e}")
            print("Run manually with: python run_deception_experiment.py --train-only --data merged_activations.pt")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Merged file: {output_path}")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Scenarios: {metadata['scenarios']}")

    if not args.train_probes:
        print("\nTo train probes on merged data:")
        print(f"  python run_deception_experiment.py --train-only --data {output_path}")


if __name__ == "__main__":
    main()
