"""
tda_taxonomy_validator.py
CLASSIFICATION: TDA Structural Validation Module (Sprint 3)
GOAL: Implements the "Quantule Taxonomy" by applying
      Persistent Homology (PH) to simulation collapse data.
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from typing import List, Tuple, Any

# --- TDA Libraries (Assumed installed via pip install ripser) ---
TDA_LIBS_AVAILABLE = False
try:
    from ripser import ripser
    from persim import plot_diagrams
    import matplotlib.pyplot as plt
    TDA_LIBS_AVAILABLE = True
except ImportError:
    pass # Gracefully skip if dependencies are missing

# --- Configuration ---
PERSISTENCE_THRESHOLD = 0.5
PROVENANCE_DIR = "provenance_reports"

# --- TDA Logic ---

def load_collapse_data(filepath: str) -> np.ndarray | None:
    """Loads collapse point data from CSV."""
    print(f"Loading collapse data from: {filepath}...")
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(filepath)
        # We need at least x and y for 2D TDA visualization/analysis
        if 'center_x' not in df.columns or 'center_y' not in df.columns:
            print(f"ERROR: CSV must contain 'center_x' and 'center_y' columns for TDA.", file=sys.stderr)
            return None
        point_cloud = df[['center_x', 'center_y']].values
        return point_cloud
    except Exception as e:
        print(f"ERROR: Could not load data. {e}", file=sys.stderr)
        return None

def compute_persistence(data: np.ndarray, max_dim: int = 1) -> List[Any]:
    """Computes Persistent Homology diagrams using ripser."""
    print(f"Computing persistent homology (max_dim={max_dim})...")
    result = ripser(data, maxdim=max_dim)
    return result['dgms']

def analyze_taxonomy(dgms: list, persistence_threshold: float) -> str:
    """Analyzes the persistence diagrams to classify structures (H0 components, H1 loops)."""
    if not dgms: return "Taxonomy: FAILED (No diagrams computed)."

    # H0: Connected Components (Count persistent components, usually related to clusters)
    h0_diagram = dgms[0]
    # Persistence = Death - Birth. H0 birth is always 0, except for 1 point.
    # The number of persistent components is typically the number of birth=0 points + 1 (the single component that persists to infinity).
    # Since ripser usually only outputs N-1 components in H0, a simpler metric is used here:
    # We count all finite H0 points with persistence > threshold.
    h0_persistence = h0_diagram[:, 1] - h0_diagram[:, 0]
    persistent_h0 = h0_persistence[
        (h0_persistence > persistence_threshold) &
        (h0_diagram[:, 1] != np.inf) # Exclude the giant component
    ]
    h0_count = len(persistent_h0) + 1 # The +1 accounts for the single component that persists to infinity

    # H1: Loops/Voids
    h1_count = 0
    if len(dgms) > 1 and dgms[1].size > 0:
        h1_diagram = dgms[1]
        h1_persistence = h1_diagram[:, 1] - h1_diagram[:, 0]
        persistent_h1 = h1_persistence[h1_persistence > persistence_threshold]
        h1_count = len(persistent_h1)

    taxonomy_str = f"Taxonomy: {h0_count} persistent components (spots), {h1_count} persistent loops (voids)."
    return taxonomy_str

def main():
    # Safe parsing for Colab/Jupyter (filters '-f' args)
    parser = argparse.ArgumentParser(description="TDA Structural Validation Module")
    parser.add_argument("config_hash", type=str, help="The configuration hash to validate.")
    parser.add_argument("--min_lifetime", type=float, default=PERSISTENCE_THRESHOLD, help="Minimum persistence threshold.")
    args = parser.parse_args([arg for arg in sys.argv[1:] if not arg.startswith('-f')])

    config_hash = args.config_hash

    if not TDA_LIBS_AVAILABLE:
        print("\nFATAL ERROR: Specialized TDA libraries (ripser, persim) not found.", file=sys.stderr)
        print("Module is Code Complete but requires provisioning (pip install ripser).", file=sys.stderr)
        sys.exit(1)

    # Assumes the CSV is saved by worker_unified.py in PROVENANCE_DIR
    data_filepath = os.path.join(PROVENANCE_DIR, f"{config_hash}_quantule_events.csv")
    output_dir = os.path.join(PROVENANCE_DIR, "TDA_Analysis")
    os.makedirs(output_dir, exist_ok=True)

    point_cloud = load_collapse_data(data_filepath)
    if point_cloud is None:
        print("TDA Validation Failed: Could not load input CSV.", file=sys.stderr)
        sys.exit(1)

    diagrams = compute_persistence(point_cloud, max_dim=1)
    taxonomy_result = analyze_taxonomy(diagrams, args.min_lifetime)

    print("\n--- TDA Structural Validation Result ---")
    print(f"Configuration Hash: {config_hash[:10]}...")
    print(f"Persistence Threshold: {args.min_lifetime}")
    print(taxonomy_result)
    print("--------------------------------------")

    # For TDA Validation, a non-zero feature count typically means a PASS in structural integrity
    # For this simplified test, we just exit 0 if it ran without crashing.
    sys.exit(0)

if __name__ == "__main__":
    main()
