#!/usr/bin/env python3
"""
Plotting script for RNA statistical potential visualization.

Usage example:
    python plot.py --potentials-dir data/potentials --output-dir plots
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot RNA distance-based statistical potentials."
    )
    parser.add_argument(
        "--potentials-dir",
        type=str,
        default="data/potentials",
        help="Directory containing potential files (*.txt). Default: data/potentials"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots. Default: plots"
    )
    return parser.parse_args()


def load_potential(filepath: str) -> Tuple[List[float], List[float]]:
    """
    Load distances and scores from a potential file.

    Args:
        filepath: Path to the potential file (e.g., AU.txt)

    Returns:
        Tuple of (distances, scores) - both as lists
    """
    distances = []
    scores = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            try:
                # Parse "distance  score" format
                parts = line.split()
                if len(parts) >= 2:
                    distances.append(float(parts[0]))
                    scores.append(float(parts[1]))
                else:
                    print(f"[WARN] Invalid format (expected 2 columns): {line}", file=sys.stderr)
            except (ValueError, IndexError):
                print(f"[WARN] Could not parse line: {line}", file=sys.stderr)
    return distances, scores


def load_all_potentials(potentials_dir: str) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Load all potential files from directory.

    Args:
        potentials_dir: Directory containing *.txt files

    Returns:
        Dictionary mapping base pair (e.g., 'AU') to (distances, scores) tuple
    """
    potentials = {}

    # Discover all .txt files in directory
    for filename in sorted(os.listdir(potentials_dir)):
        if not filename.endswith('.txt'):
            continue

        base_pair = filename.replace('.txt', '')
        filepath = os.path.join(potentials_dir, filename)

        distances, scores = load_potential(filepath)
        if distances and scores:
            potentials[base_pair] = (distances, scores)
            print(f"[INFO] Loaded {len(scores)} points for {base_pair}")
        else:
            print(f"[WARN] No data loaded from {filepath}", file=sys.stderr)

    return potentials


def plot_potential(
    base_pair: str,
    distances: List[float],
    scores: List[float],
    output_path: str
) -> None:
    """
    Plot a single potential curve.

    Args:
        base_pair: Base pair label (e.g., 'AU')
        distances: List of distances (Angstroms)
        scores: List of scores (one per distance)
        output_path: Path to save the plot
    """
    # Convert to numpy arrays
    distances = np.array(distances)
    scores_array = np.array(scores)

    # Create line plot
    plt.figure(figsize=(10, 6))

    # Plot main curve
    plt.plot(distances, scores_array, 'b-', linewidth=2, marker='o', markersize=4, label=f'{base_pair} potential')

    # Fill favorable (< 0) and unfavorable (> 0) regions
    plt.fill_between(distances, scores_array, 0, where=(scores_array < 0),
                     alpha=0.3, color='green', label='Favorable (< 0)', interpolate=True)
    plt.fill_between(distances, scores_array, 0, where=(scores_array > 0),
                     alpha=0.3, color='red', label='Unfavorable (> 0)', interpolate=True)

    plt.axhline(y=0, color='k', linestyle='-', linewidth=2, alpha=0.8, label='Zero energy')

    # Formatting
    plt.xlabel('Distance (Å)', fontsize=12)
    plt.ylabel('Pseudo-energy Score', fontsize=12)
    plt.title(f'Statistical Potential: {base_pair} Pair', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved plot: {output_path}")


def main() -> None:
    """Main function to generate all plots."""
    args = parse_args()

    print(f"[INFO] Potentials directory: {args.potentials_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")

    # Check if potentials directory exists
    if not os.path.exists(args.potentials_dir):
        print(f"[ERROR] Potentials directory not found: {args.potentials_dir}",
              file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all potentials
    print("\n[INFO] Loading potentials...")
    potentials = load_all_potentials(args.potentials_dir)

    if not potentials:
        print("[ERROR] No potentials loaded. Check your potentials directory.",
              file=sys.stderr)
        sys.exit(1)

    # Generate plots
    print(f"\n[INFO] Generating plots...")
    for base_pair, (distances, scores) in potentials.items():
        output_path = os.path.join(args.output_dir, f"{base_pair}_potential.png")
        plot_potential(
            base_pair=base_pair,
            distances=distances,
            scores=scores,
            output_path=output_path
        )

    print(f"\n[SUCCESS] Generated {len(potentials)} plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
