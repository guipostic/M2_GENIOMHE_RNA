#!/usr/bin/env python3
"""
Training script for RNA statistical potential (C3'–C3' distances).

Usage example:
    python training.py data/pdbs/*.pdb --out-dir potentials --round-decimals 3
"""

import os
import sys
import argparse
from math import log
from typing import Dict, List, Tuple

import numpy as np
from Bio import PDB

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Canonical base-pair labels required by the assignment
BASE_PAIRS = ['AA', 'AU', 'AC', 'AG', 'UU', 'UC', 'UG', 'CC', 'CG', 'GG']

# Distance bin edges: 0–1, 1–2, ..., 19–20 Å  → 20 intervals
DISTANCE_BINS = np.linspace(0.0, 20.0, 21)
N_BINS = len(DISTANCE_BINS) - 1

MAX_SCORE = 10.0

# Map residue names in PDB to canonical A/U/C/G labels
RESIDUE_NAME_MAP = {
    'A': 'A', 'ADE': 'A', 'DA': 'A',
    'C': 'C', 'CYT': 'C', 'DC': 'C',
    'G': 'G', 'GUA': 'G', 'DG': 'G',
    'U': 'U', 'URA': 'U', 'DU': 'U',
}

# Mapping from sorted pair (e.g. "CU") to canonical label (e.g. "UC")
SORTED_PAIR_TO_CANONICAL = {
    'AA': 'AA',
    'AC': 'AC',
    'AG': 'AG',
    'AU': 'AU',
    'CC': 'CC',
    'CG': 'CG',
    'CU': 'UC',  # C,U → UC
    'GG': 'GG',
    'GU': 'UG',  # G,U → UG
    'UU': 'UU',
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RNA distance-based statistical potential "
                    "from a dataset of RNA PDB structures."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDB files and/or directories containing PDB files."
    )
    parser.add_argument(
        "--out-dir",
        default="potentials",
        help="Output directory for the 10 base-pair potential files (default: potentials)."
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=3,
        help="Number of decimal places to round scores (default: 3)."
    )
    return parser.parse_args()


def collect_pdb_paths(inputs: List[str]) -> List[str]:
    """
    Expand a list of paths (files or directories) into a flat list of PDB file paths.
    """
    pdb_paths: List[str] = []

    for path in inputs:
        if os.path.isdir(path):
            # all files in directory that look like PDBs
            for name in os.listdir(path):
                full = os.path.join(path, name)
                if os.path.isfile(full) and (
                    name.lower().endswith(".pdb") or name.lower().endswith(".ent")
                ):
                    pdb_paths.append(full)
        elif os.path.isfile(path):
            pdb_paths.append(path)
        else:
            print(f"[WARN] Input path not found: {path}", file=sys.stderr)

    return sorted(set(pdb_paths))


def calculate_ED(atom1: PDB.Atom.Atom, atom2: PDB.Atom.Atom) -> float:
    """Euclidean distance between two Biopython Atom objects."""
    v = atom1.coord - atom2.coord
    return float(np.linalg.norm(v))


def extract_c3_atoms(chain: PDB.Chain.Chain) -> List[Tuple[int, PDB.Atom.Atom, str]]:
    """
    Extract C3' atoms and their residue types (A/U/C/G) from a chain.

    Returns a list of tuples: (sequence_index, atom, base_code)
    """
    c3_atoms: List[Tuple[int, PDB.Atom.Atom, str]] = []
    residues = list(chain)  # chain is iterable over residues

    for i, residue in enumerate(residues):
        res_name = residue.resname.strip()
        base_code = RESIDUE_NAME_MAP.get(res_name, '')
        if not base_code:
            continue

        try:
            c3_atom = residue["C3'"]
        except KeyError:
            # No C3' atom; skip
            continue

        c3_atoms.append((i, c3_atom, base_code))

    return c3_atoms


def canonical_pair(res1: str, res2: str) -> str:
    """
    Map unordered pair (res1, res2) to one of the 10 canonical base-pair labels.
    Returns "" if the pair is not one of the 10 we care about.
    """
    s = ''.join(sorted((res1, res2)))  # e.g. "CU", "AU", "GG"
    return SORTED_PAIR_TO_CANONICAL.get(s, "")


def update_distance_counts(
    c3_atoms: List[Tuple[int, PDB.Atom.Atom, str]],
    observed_counts: Dict[str, np.ndarray],
    reference_counts: np.ndarray,
) -> None:
    """
    Given the C3' atoms for one chain, update the global observed and reference
    distance counts.

    - Only pairs with |j - i| >= 4 are considered.
    - Only distances < 20.0 Å are considered.
    - Binning: DISTANCE_BINS defines 20 bins between 0 and 20 Å.
    """
    n = len(c3_atoms)
    for idx1 in range(n):
        seq_idx1, atom1, res1 = c3_atoms[idx1]

        for idx2 in range(idx1 + 1, n):
            seq_idx2, atom2, res2 = c3_atoms[idx2]

            # Sequence separation: |j - i| >= 4
            if seq_idx2 - seq_idx1 < 4:
                continue

            dist = calculate_ED(atom1, atom2)

            # We restrict to [0,20) to stay inside 20 bins cleanly.
            if dist < 0.0 or dist >= 20.0:
                continue

            # Find bin index: 0..19
            bin_index = int(np.searchsorted(DISTANCE_BINS, dist, side="right") - 1)
            if bin_index < 0 or bin_index >= N_BINS:
                continue

            bp = canonical_pair(res1, res2)
            if bp in observed_counts:
                observed_counts[bp][bin_index] += 1

            # Reference counts: all pairs (X,X) regardless of type
            reference_counts[bin_index] += 1


def compute_scores(
    observed_counts: Dict[str, np.ndarray],
    reference_counts: np.ndarray,
    round_decimals: int,
) -> Dict[str, List[float]]:
    """
    Compute pseudo-energy scores for each base pair and distance bin:

        u_bar(i,j,r) = -log( f_obs(i,j,r) / f_ref(r) )

    where f_obs and f_ref are normalized frequencies in that bin.

    Returns a dict: {bp: [score_bin_0, ..., score_bin_19]}
    """
    scores_dict: Dict[str, List[float]] = {}

    ref_total = float(reference_counts.sum())
    if ref_total <= 0.0:
        print("[WARN] Reference counts are all zero; something is wrong.",
              file=sys.stderr)
        ref_total = 1.0

    # Reference frequencies per bin
    f_ref_bins = reference_counts.astype(float) / ref_total
    # Avoid log( / 0)
    f_ref_bins = np.clip(f_ref_bins, 1e-12, None)

    for bp, counts in observed_counts.items():
        total_bp = float(counts.sum())
        if total_bp <= 0.0:
            # No observations at all for this pair
            scores_dict[bp] = [MAX_SCORE] * N_BINS
            continue

        f_obs_bins = counts.astype(float) / total_bp
        f_obs_bins = np.clip(f_obs_bins, 0.0, None)

        bp_scores: List[float] = []
        for r in range(N_BINS):
            f_obs = f_obs_bins[r]
            f_ref = f_ref_bins[r]

            if f_obs > 0.0 and f_ref > 0.0:
                val = -log(f_obs / f_ref)
                if val > MAX_SCORE:
                    val = MAX_SCORE
            else:
                val = MAX_SCORE

            bp_scores.append(round(val, round_decimals))

        scores_dict[bp] = bp_scores

    return scores_dict


def save_scores(scores_dict: Dict[str, List[float]], out_dir: str) -> None:
    """
    Save one file per base pair, each containing 20 lines:
        line r = score for bin r (0..19).
    """
    os.makedirs(out_dir, exist_ok=True)

    for bp, scores in scores_dict.items():
        out_path = os.path.join(out_dir, f"{bp}.txt")
        with open(out_path, "w") as f:
            for s in scores:
                f.write(f"{s}\n")
        print(f"[INFO] Wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    pdb_files = collect_pdb_paths(args.inputs)
    if not pdb_files:
        print("[ERROR] No valid PDB files found.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(pdb_files)} PDB files.")

    # Initialize global counts
    observed_counts = {
        bp: np.zeros(N_BINS, dtype=np.int64) for bp in BASE_PAIRS
    }
    reference_counts = np.zeros(N_BINS, dtype=np.int64)

    parser = PDB.PDBParser(QUIET=True)

    for pdb_path in pdb_files:
        print(f"[INFO] Processing {pdb_path}")
        structure = parser.get_structure("RNA", pdb_path)

        for model in structure:
            for chain in model:
                c3_atoms = extract_c3_atoms(chain)
                if not c3_atoms:
                    continue
                update_distance_counts(c3_atoms, observed_counts, reference_counts)

    # Compute scores and save
    scores_dict = compute_scores(observed_counts, reference_counts, args.round_decimals)
    save_scores(scores_dict, args.out_dir)


if __name__ == "__main__":
    main()
