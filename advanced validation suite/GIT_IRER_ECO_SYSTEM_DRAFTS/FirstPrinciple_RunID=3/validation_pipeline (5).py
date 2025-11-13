#!/usr/bin/env python3

"""
validation_pipeline.py
ASSET: A6 (Spectral Fidelity & Provenance Module)
VERSION: 2.0 (Phase 3 Scientific Mandate)
CLASSIFICATION: Final Implementation Blueprint / Governance Instrument
GOAL: Serves as the immutable source of truth that cryptographically binds
      experimental intent (parameters) to scientific fact (spectral fidelity)
      and Aletheia cognitive coherence.
"""

import json
import hashlib
import sys
import os
import argparse
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import tempfile # Added for temporary file handling

# --- V2.0 DEPENDENCIES ---
# Import the core analysis engine (CEPP v1.0 / Quantule Profiler)
# This file (quantulemapper.py) must be in the same directory.
try:
    import quantulemapper_real as cep_profiler
except ImportError:
    print("FATAL: Could not import 'quantulemapper.py'.", file=sys.stderr)
    print("This file is the core Quantule Profiler (CEPP v1.0).", file=sys.stderr)
    sys.exit(1)

# Import Scipy for new Aletheia Metrics
try:
    from scipy.signal import coherence as scipy_coherence
    from scipy.stats import entropy as scipy_entropy
except ImportError:
    print("FATAL: Missing 'scipy'. Please install: pip install scipy", file=sys.stderr)
    sys.exit(1)


# --- MODULE CONSTANTS ---
SCHEMA_VERSION = "SFP-v2.0-ARCS" # Upgraded schema version

# ---
# SECTION 1: PROVENANCE KERNEL (EVIDENTIAL INTEGRITY)
# ---

def generate_canonical_hash(params_dict: Dict[str, Any]) -> str:
    """
    Generates a canonical, deterministic SHA-256 hash from a parameter dict.
    This function now explicitly filters out non-canonical metadata like 'run_uuid' and 'config_hash'
    to ensure consistency across components.
    """
    try:
        # Create a filtered dictionary for hashing, excluding non-canonical keys
        filtered_params = {k: v for k, v in params_dict.items() if k not in ["run_uuid", "config_hash", "param_hash_legacy"]}

        canonical_string = json.dumps(
            filtered_params,
            sort_keys=True,
            separators=(
                ',', ':'
            )
        )
        string_bytes = canonical_string.encode('utf-8')
        hash_object = hashlib.sha256(string_bytes)
        config_hash = hash_object.hexdigest()
        return config_hash
    except Exception as e:
        print(f"[ProvenanceKernel Error] Failed to generate hash: {e}", file=sys.stderr)
        raise

# ---
# SECTION 2: FIDELITY KERNEL (SCIENTIFIC VALIDATION)
# ---

def run_quantule_profiler(rho_history_path: str) -> Dict[str, Any]:
    """
    Orchestrates the core scientific analysis by calling the
    Quantule Profiler (CEPP v1.0 / quantulemapper.py).

    This function replaces the v1.0 mock logic. It loads the HDF5 artifact,
    saves it as a temporary .npy file (as required by the profiler's API),
    and runs the full analysis.
    """
    temp_npy_file = None
    try:
        # 1. Load HDF5 data (as required by Orchestrator)
        with h5py.File(rho_history_path, 'r') as f:
            # Load the full 4D stack
            rho_history = f['rho_history'][:]

        if rho_history.ndim != 4:
            raise ValueError(f"Input HDF5 'rho_history' is not 4D (t,x,y,z). Shape: {rho_history.shape}")

        # 2. Convert to .npy (as required by quantulemapper.py API)
        # We create a temporary .npy file for the profiler to consume
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp, rho_history)
            temp_npy_file = tmp.name

        # 3. Run the Quantule Profiler (CEPP v1.0)
        # This performs: Multi-Ray Sampling, FFT, Peak Find, Calibration,
        # SSE Calculation, and Quantule Classification.
        print(f"[FidelityKernel] Calling Quantule Profiler (CEPP v1.0) on {temp_npy_file}")

        # The mapper.py analyze_4d function runs on the *last time step*
        profiler_results = cep_profiler.analyze_4d(temp_npy_file)

        # 4. Extract key results for the SFP artifact
        # The profiler already calculates SSE, which we now use as the
        # definitive "log_prime_sse"

        spectral_fidelity = {
            "validation_status": profiler_results.get("validation_status", "FAIL: PROFILER"),
            "log_prime_sse": profiler_results.get("total_sse", 999.0),
            "scaling_factor_S": profiler_results.get("scaling_factor_S", 0.0),
            "dominant_peak_k": profiler_results.get("dominant_peak_k", 0.0),
            "analysis_protocol": "CEPP v1.0",
            "log_prime_targets": cep_profiler.LOG_PRIME_VALUES.tolist()
        }

        # Return the full set of results for the Aletheia Metrics
        return {
            "spectral_fidelity": spectral_fidelity,
            "classification_results": profiler_results.get("csv_files", {}),
            "raw_rho_final_state": rho_history[-1, :, :, :] # Pass final state
        }

    except Exception as e:
        print(f"[FidelityKernel Error] Failed during Quantule Profiler execution: {e}", file=sys.stderr)
        return {
            "spectral_fidelity": {"validation_status": "FAIL: KERNEL_ERROR", "log_prime_sse": 999.9},
            "classification_results": {},
            "raw_rho_final_state": None
        }
    finally:
        # Clean up the temporary .npy file
        if temp_npy_file and os.path.exists(temp_npy_file):
            os.remove(temp_npy_file)

# ---
# SECTION 3: ALETHEIA COHERENCE METRICS (PHASE 3)
# ---

def calculate_pcs(rho_final_state: np.ndarray) -> float:
    """
    [Phase 3] Calculates the Phase Coherence Score (PCS).
    Analogue: Superfluid order parameter.
    Implementation: Magnitude-squared coherence function.

    We sample two different, parallel 1D rays from the final state
    and measure their coherence.
    """
    try:
        # Sample two 1D rays from the middle of the state
        center_idx = rho_final_state.shape[0] // 2
        ray_1 = rho_final_state[center_idx, center_idx, :]
        ray_2 = rho_final_state[center_idx + 1, center_idx + 1, :] # Offset ray

        # Calculate coherence
        f, Cxy = scipy_coherence(ray_1, ray_2)

        # PCS is the mean coherence across all frequencies
        pcs_score = np.mean(Cxy)

        if np.isnan(pcs_score):
            return 0.0
        return float(pcs_score)

    except Exception as e:
        print(f"[AletheiaMetrics] WARNING: PCS calculation failed: {e}", file=sys.stderr)
        return 0.0 # Failed coherence is 0

def calculate_pli(rho_final_state: np.ndarray) -> float:
    """
    [Phase 3] Calculates the Principled Localization Index (PLI).
    Analogue: Mott Insulator phase.
    Implementation: Inverse Participation Ratio (IPR).

    IPR = sum(psi^4) / (sum(psi^2))^2
    A value of 1.0 is perfectly localized (Mott), 1/N is perfectly delocalized (Superfluid).
    We use the density field `rho` as our `psi^2` equivalent.
    """
    try:
        # Normalize the density field (rho is already > 0)
        # Changed jnp.sum to np.sum and jnp.array to np.array
        rho_norm = rho_final_state / np.sum(rho_final_state)

        # Calculate IPR on the normalized density
        # IPR = sum(p_i^2)
        pli_score = np.sum(rho_norm**2)

        # Scale by N to get a value between (0, 1)
        N_cells = rho_final_state.size
        pli_score_normalized = float(pli_score * N_cells)

        if np.isnan(pli_score_normalized):
            return 0.0
        return pli_score_normalized

    except Exception as e:
        print(f"[AletheiaMetrics] WARNING: PLI calculation failed: {e}", file=sys.stderr)
        return 0.0

def calculate_ic(rho_final_state: np.ndarray) -> float:
    """
    [Phase 3] Calculates the Informational Compressibility (IC).
    Analogue: Thermodynamic compressibility.
    Implementation: K_I = dS / dE (numerical estimation).
    """
    try:
        # 1. Proxy for System Energy (E):
        # We use the L2 norm of the field (sum of squares) as a simple energy proxy.
        # Changed jnp.sum to np.sum
        proxy_E = np.sum(rho_final_state**2)

        # 2. Proxy for System Entropy (S):
        # We treat the normalized field as a probability distribution
        # and calculate its Shannon entropy.
        rho_flat = rho_final_state.flatten()
        rho_prob = rho_flat / np.sum(rho_flat)
        # Add epsilon to avoid log(0)
        proxy_S = scipy_entropy(rho_prob + 1e-9)

        # 3. Calculate IC = dS / dE
        # We perturb the system slightly to estimate the derivative

        # Create a tiny perturbation (add 0.1% energy)
        epsilon = 0.001
        rho_perturbed = rho_final_state * (1.0 + epsilon)

        # Calculate new E and S
        # Changed jnp.sum to np.sum
        proxy_E_p = np.sum(rho_perturbed**2)

        rho_p_flat = rho_perturbed.flatten()
        rho_p_prob = rho_p_flat / np.sum(rho_p_flat)
        proxy_S_p = scipy_entropy(rho_p_prob + 1e-9)

        # Numerical derivative
        dE = proxy_E_p - proxy_E
        dS = proxy_S_p - proxy_S

        if dE == 0:
            return 0.0 # Incompressible

        ic_score = float(dS / dE)

        if np.isnan(ic_score):
            return 0.0
        return ic_score

    except Exception as e:
        print(f"[AletheiaMetrics] WARNING: IC calculation failed: {e}", file=sys.stderr)
        return 0.0

# ---
# SECTION 4: MAIN ORCHESTRATION (DRIVER HOOK)
# ---

def main():
    """
    Main execution entry point for the SFP Module (v2.0).
    Orchestrates the Quantule Profiler (CEPP), Provenance Kernel,
    and Aletheia Metrics calculations.
    """
    parser = argparse.ArgumentParser(
        description="Spectral Fidelity & Provenance (SFP) Module (Asset A6, v2.0)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input rho_history.h5 data artifact."
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to the parameters.json file for this run."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the provenance.json and atlas CSVs."
    )
    args = parser.parse_args()

    print(f"--- SFP Module (Asset A6, v2.0) Initiating Validation ---")
    print(f"  Input Artifact: {args.input}")
    print(f"  Params File:    {args.params}")

    # --- 1. Provenance Kernel (Hashing) ---
    print("\n[1. Provenance Kernel]")
    try:
        with open(args.params, 'r') as f:
            params_dict = json.load(f)
    except Exception as e:
        print(f"CRITICAL_FAIL: Could not load params file: {e}", file=sys.stderr)
        sys.exit(1)

    config_hash = generate_canonical_hash(params_dict)
    print(f"  Generated Canonical config_hash: {config_hash}")
    param_hash_legacy = params_dict.get("param_hash_legacy", None)

    # --- 2. Fidelity Kernel (Quantule Profiler) ---
    print("\n[2. Fidelity Kernel (CEPP v1.0)]")

    # Check for mock input file from previous tests
    if args.input == "rho_history_mock.h5":
        print("WARNING: Using 'rho_history_mock.h5'. This file is empty.")
        print("Fidelity and Aletheia Metrics will be 0 or FAIL.")
        # Create a dummy structure to allow the script to complete
        profiler_run_results = {
            "spectral_fidelity": {"validation_status": "FAIL: MOCK_INPUT", "log_prime_sse": 999.0},
            "classification_results": {},
            "raw_rho_final_state": np.zeros((16,16,16)) # Dummy shape
        }
    else:
        # This is the normal execution path
        if not os.path.exists(args.input):
            print(f"CRITICAL_FAIL: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        profiler_run_results = run_quantule_profiler(args.input)

    spectral_fidelity_results = profiler_run_results["spectral_fidelity"]
    classification_data = profiler_run_results["classification_results"]
    rho_final = profiler_run_results["raw_rho_final_state"]

    print(f"  Validation Status: {spectral_fidelity_results['validation_status']}")
    print(f"  Calculated SSE:    {spectral_fidelity_results['log_prime_sse']:.6f}")

    # --- 3. Aletheia Metrics (Phase 3 Implementation) ---
    print("\n[3. Aletheia Coherence Metrics (Phase 3)]")
    if rho_final is None or rho_final.size == 0:
        print("  SKIPPING: No final state data to analyze.")
        metrics_pcs, metrics_pli, metrics_ic = 0.0, 0.0, 0.0
    else:
        metrics_pcs = calculate_pcs(rho_final)
        metrics_pli = calculate_pli(rho_final)
        metrics_ic = calculate_ic(rho_final)

    print(f"  Phase Coherence Score (PCS): {metrics_pcs:.6f}")
    print(f"  Principled Localization (PLI): {metrics_pli:.6f}")
    print(f"  Informational Compressibility (IC): {metrics_ic:.6f}")

    # --- 4. Assemble & Save Canonical Artifacts ---
    print("\n[4. Assembling Canonical Artifacts]")

    # A. Save Quantule Atlas CSV files
    # The profiler returns a dict of {'filename': 'csv_content_string'}
    atlas_paths = {}
    for csv_name, csv_content in classification_data.items():
        try:
            # Save the CSV file, prefixed with the config_hash
            csv_filename = f"{config_hash}_{csv_name}"
            csv_path = os.path.join(args.output_dir, csv_filename)
            with open(csv_path, 'w') as f:
                f.write(csv_content)
            atlas_paths[csv_name] = csv_path
            print(f"  Saved Quantule Atlas artifact: {csv_path}")
        except Exception as e:
            print(f"WARNING: Could not save Atlas CSV {csv_name}: {e}", file=sys.stderr)

    # B. Save the primary provenance.json artifact
    provenance_artifact = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash,
        "param_hash_legacy": param_hash_legacy,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_artifact_path": args.input,

        "spectral_fidelity": spectral_fidelity_results,

        "aletheia_metrics": {
            "pcs": metrics_pcs,
            "pli": metrics_pli,
            "ic": metrics_ic
        },

        "quantule_atlas_artifacts": atlas_paths,

        "secondary_metrics": {
            "full_spectral_sse_tda": None # Deprecated
        }
    }

    output_filename = os.path.join(
        args.output_dir,
        f"provenance_{config_hash}.json"
    )

    try:
        with open(output_filename, 'w') as f:
            json.dump(provenance_artifact, f, indent=2, sort_keys=True)
        print(f"  SUCCESS: Saved primary artifact to {output_filename}")
    except Exception as e:
        print(f"CRITICAL_FAIL: Could not save artifact: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
