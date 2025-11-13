#!/usr/bin/env python3

"""
adaptive_hunt_orchestrator.py
CLASSIFICATION: Master Driver (ASTE V1.0)
GOAL: Manages the entire end-to-end simulation lifecycle. This script
       bootstraps the system, calls the Hunter for parameters, launches
      the Worker to simulate, and initiates the Validator (SFP module)
      to certify the results, closing the adaptive loop.
"""

import os
import json
import subprocess
import sys
import uuid
from typing import Dict, Any, List

# --- Import Shared Components ---
# We import the Provenance Kernel from the SFP module to generate
# the canonical hash. This is a critical architectural link.
try:
    from validation_pipeline import generate_canonical_hash
except ImportError:
    print("Error: Could not import 'generate_canonical_hash'.", file=sys.stderr)
    print("Please ensure 'validation_pipeline.py' is in the same directory.", file=sys.stderr)
    sys.exit(1)

# We also import the "Brain" of the operation
try:
    import aste_hunter
except ImportError:
    print("Error: Could not import 'aste_hunter'.", file=sys.stderr)
    print("Please ensure 'aste_hunter.py' is in the same directory.", file=sys.stderr)
    sys.exit(1)


# --- Configuration ---
# These paths define the ecosystem's file structure
CONFIG_DIR = "input_configs"
DATA_DIR = "simulation_data"
PROVENANCE_DIR = "provenance_reports"
WORKER_SCRIPT = "worker_v7.py" # The JAX-based worker
VALIDATOR_SCRIPT = "validation_pipeline.py" # The SFP Module

# --- Test Parameters ---
# Use small numbers for a quick test run
NUM_GENERATIONS = 2     # Run 2 full loops (Gen 0, Gen 1)
POPULATION_SIZE = 4    # Run 4 simulations per generation


def setup_directories():
    """Ensures all required I/O directories exist."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROVENANCE_DIR, exist_ok=True)
    print(f"Orchestrator: I/O directories ensured: {CONFIG_DIR}, {DATA_DIR}, {PROVENANCE_DIR}")

def run_simulation_job(config_hash: str, params_filepath: str) -> bool:
    """
    Executes a single end-to-end simulation job (Worker + Validator).
    This function enforces the mandated workflow.
    """
    print(f"\n--- ORCHESTRATOR: STARTING JOB {config_hash[:10]}... ---")

    # Define file paths based on the canonical hash
    # This enforces the "unbreakable cryptographic link"
    rho_history_path = os.path.join(DATA_DIR, f"rho_history_{config_hash}.h5")

    try:
        # --- 3. Execution Step (Simulation) ---
        print(f"[Orchestrator] -> Calling Worker: {WORKER_SCRIPT}")
        worker_command = [
            "python", WORKER_SCRIPT,
            "--params", params_filepath,
            "--output", rho_history_path
        ]

        # We use subprocess.run() which waits for the command to complete.
        # This is where the JAX compilation will happen on the first run.
        worker_process = subprocess.run(worker_command, check=False, capture_output=True, text=True)
        
        if worker_process.returncode != 0:
            print(f"ERROR: [JOB {config_hash[:10]}] WORKER FAILED.", file=sys.stderr)
            print(f"COMMAND: {' '.join(worker_process.args)}", file=sys.stderr)
            print(f"STDOUT: {worker_process.stdout}", file=sys.stderr)
            print(f"STDERR: {worker_process.stderr}", file=sys.stderr)
            return False

        print(f"[Orchestrator] <- Worker {config_hash[:10]} OK.")

        # --- 4. Fidelity Step (Validation) ---
        print(f"[Orchestrator] -> Calling Validator: {VALIDATOR_SCRIPT}")
        validator_command = [
            "python", VALIDATOR_SCRIPT,
            "--input", rho_history_path,
            "--params", params_filepath,
            "--output_dir", PROVENANCE_DIR
        ]
        validator_process = subprocess.run(validator_command, check=False, capture_output=True, text=True)

        if validator_process.returncode != 0:
            print(f"ERROR: [JOB {config_hash[:10]}] VALIDATOR FAILED.", file=sys.stderr)
            print(f"COMMAND: {' '.join(validator_process.args)}", file=sys.stderr)
            print(f"STDOUT: {validator_process.stdout}", file=sys.stderr)
            print(f"STDERR: {validator_process.stderr}", file=sys.stderr)
            return False

        print(f"[Orchestrator] <- Validator {config_hash[:10]} OK.")

        print(f"--- ORCHESTRATOR: JOB {config_hash[:10]} SUCCEEDED ---")
        return True

    except FileNotFoundError as e:
        print(f"ERROR: [JOB {config_hash[:10]}] Script not found: {e.filename}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: [JOB {config_hash[:10]}] An unexpected error occurred: {e}", file=sys.stderr)
        return False

def main():
    """
    Main entry point for the Adaptive Simulation Steering Engine (ASTE).
    """
    print("--- ASTE ORCHESTRATOR V1.0 [BOOTSTRAP] ---")
    setup_directories()

    # 1. Bootstrap: Initialize the Hunter "Brain"
    hunter = aste_hunter.Hunter(ledger_file="simulation_ledger.csv")

    # --- MAIN ORCHESTRATION LOOP ---
    for gen in range(NUM_GENERATIONS):
        print(f"\n========================================================")
        print(f"    ASTE ORCHESTRATOR: STARTING GENERATION {gen}")
        print(f"========================================================")

        # 2. Get Tasks: Hunter breeds the next generation of parameters
        parameter_batch = hunter.get_next_generation(POPULATION_SIZE)

        jobs_to_run = []

        # --- 2a. Provenance & Registration Step ---
        print(f"[Orchestrator] Registering {len(parameter_batch)} new jobs for Gen {gen}...")
        for params_dict in parameter_batch:

            # Create a temporary dictionary for hashing that does NOT include run_uuid or config_hash
            # These are metadata about the run, not the parameters of the simulation itself.
            params_for_hashing = params_dict.copy()

            # Generate the canonical hash (Primary Key) from the core parameters
            config_hash = generate_canonical_hash(params_for_hashing)

            # Now add metadata to the original params_dict
            params_dict['config_hash'] = config_hash # Add hash to params_dict. This line was intentionally added here based on previous user conversation.
            params_dict['run_uuid'] = str(uuid.uuid4()) # Add a unique ID to distinguish identical parameter sets

            # --- 2b. Save Config ---
            params_filepath = os.path.join(CONFIG_DIR, f"config_{config_hash}.json")
            try:
                with open(params_filepath, 'w') as f:
                    json.dump(params_dict, f, indent=2, sort_keys=True)
            except Exception as e:
                print(f"ERROR: Could not write config file {params_filepath}. {e}", file=sys.stderr)
                continue # Skip this job

            # --- 2c. Register Job with Hunter ---
            job_entry = {
                aste_hunter.HASH_KEY: config_hash,
                "generation": gen,
                "param_kappa": params_dict["param_kappa"],
                "param_sigma_k": params_dict["param_sigma_k"],
                "params_filepath": params_filepath
            }
            jobs_to_run.append(job_entry)

        # Register the *full* batch with the Hunter's ledger
        hunter.register_new_jobs(jobs_to_run)

        # --- 3 & 4. Execute Batch Loop (Worker + Validator) ---
        job_hashes_completed = []
        for job in jobs_to_run:
            success = run_simulation_job(
                config_hash=job[aste_hunter.HASH_KEY],
                params_filepath=job["params_filepath"]
            )
            if success:
                job_hashes_completed.append(job[aste_hunter.HASH_KEY])

        # --- 5. Ledger Step (Cycle Completion) ---
        print(f"\n[Orchestrator] GENERATION {gen} COMPLETE.")
        print("[Orchestrator] Notifying Hunter to process results...")
        hunter.process_generation_results(
            provenance_dir=PROVENANCE_DIR,
            job_hashes=job_hashes_completed
        )

        best_run = hunter.get_best_run()
        if best_run:
            print(f"[Orch] Best Run So Far: {best_run[aste_hunter.HASH_KEY][:10]}... (SSE: {best_run[aste_hunter.SSE_METRIC_KEY]:.6f})")

    print("\n========================================================")
    print("--- ASTE ORCHESTRATOR: ALL GENERATIONS COMPLETE ---")
    print("========================================================")

if __name__ == "__main__":
    main()
