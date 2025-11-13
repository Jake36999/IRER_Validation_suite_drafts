"""
adaptive_hunt_orchestrator.py
CLASSIFICATION: Master Driver (ASTE V10.0 - Run 3 Focused Hunt)
GOAL: Manages the hunt lifecycle using settings.py. Loads a seed config
      if available to focus the evolutionary search (Run 3 Mandate).
"""

import sys
import os

# Ensure current directory is in sys.path for local module imports
sys.path.insert(0, os.getcwd())
print(f"DEBUG_ORCH_IMPORT: Current working directory: {os.getcwd()}")
print(f"DEBUG_ORCH_IMPORT: sys.path: {sys.path}")

import argparse
import json
import subprocess
import uuid
import random
from typing import Dict, Any, List, Optional
import time

# --- Import Shared Components ---
import settings
import aste_hunter
print(f"DEBUG_ORCH_IMPORT: Attempting to import validation_pipeline...")
from validation_pipeline import generate_canonical_hash
print(f"DEBUG_ORCH_IMPORT: Successfully imported validation_pipeline.")

# Configuration from centralized settings
NUM_GENERATIONS = settings.NUM_GENERATIONS
POPULATION_SIZE = settings.POPULATION_SIZE
CONFIG_DIR = settings.CONFIG_DIR
DATA_DIR = settings.DATA_DIR
PROVENANCE_DIR = settings.PROVENANCE_DIR
WORKER_SCRIPT = "worker_unified.py"
VALIDATOR_SCRIPT = "validation_pipeline.py"


def run_simulation_job(config_hash: str, params_filepath: str) -> bool:
    """Executes the worker and the validator sequentially."""
    # Note: We use the .h5 extension for output path consistency, even though
    # the lite worker writes a JSON file in this minimal setup.
    data_filepath = os.path.join(DATA_DIR, f"rho_history_{config_hash}.h5")

    # 1. Execute Worker (Generates synthetic data + quantule_events.csv)
    worker_cmd = [
        sys.executable, WORKER_SCRIPT,
        "--params", params_filepath,
        "--output", data_filepath
    ]
    try:
        worker_result = subprocess.run(worker_cmd, capture_output=True, text=True, check=True)
        print(f"Worker STDOUT: {worker_result.stdout}", file=sys.stdout) # DEBUG: Show worker output
    except subprocess.CalledProcessError as e:
        print(f"ERROR: [JOB {config_hash[:10]}] WORKER FAILED.", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return False

    # 2. Execute Validator (Reads H5/JSON, runs PPN check, creates provenance.json)
    validator_cmd = [
        sys.executable, VALIDATOR_SCRIPT,
        config_hash # The validator reads the H5 from DATA_DIR and config from CONFIG_DIR
    ]
    try:
        validator_result = subprocess.run(validator_cmd, capture_output=True, text=True, check=True)
        print(f"Validator STDOUT: {validator_result.stdout}", file=sys.stdout) # DEBUG: Show validator output
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: [JOB {config_hash[:10]}] VALIDATOR FAILED.", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return False


def load_seed_config() -> Optional[Dict[str, float]]:
    """Loads a seed configuration from a well-known ENV file for focused hunts."""
    seed_path = os.path.join(settings.BASE_DIR, "best_config_seed.env")
    if not os.path.exists(seed_path):
        return None

    try:
        with open(seed_path, 'r') as f:
            for line in f:
                if line.startswith("SEED_HASH="):
                    seed_hash = line.strip().split('=')[1]
                    config_path = os.path.join(CONFIG_DIR, f"config_{seed_hash}.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as cf:
                            config = json.load(cf)
                            # Extract only the physics parameters
                            return {k: v for k, v in config.items() if k.startswith("param_")}
        return None
    except Exception as e:
        print(f"Warning: Failed to load seed config: {e}", file=sys.stderr)
        return None

def main():
    # Safe parsing for Colab/Jupyter (filters '-f' args)
    parser = argparse.ArgumentParser(description="ASTE Orchestrator")
    parser.parse_args([arg for arg in sys.argv[1:] if not arg.startswith('-f')])

    print("--- ASTE ORCHESTRATOR V10.0 [RUN ID 3] ---")

    # Ensure directories exist
    for d in [CONFIG_DIR, DATA_DIR, PROVENANCE_DIR, os.path.dirname(settings.LEDGER_FILE)]:
        os.makedirs(d, exist_ok=True)
    print("Orchestrator: I/O directories ensured.")

    hunter = aste_hunter.Hunter(ledger_file=settings.LEDGER_FILE)

    # --- Check for Seed (RUN 3 Mandate) ---
    seed_config = load_seed_config()
    if seed_config:
        print(f"Orchestrator: Loaded seed config for focused hunt.")

    # Main Evolutionary Loop
    for gen in range(hunter.get_current_generation(), NUM_GENERATIONS):
        print(f"\n========================================================")
        print(f"    ASTE ORCHESTRATOR: STARTING GENERATION {gen}")
        print(f"========================================================")

        # 1. Get next batch of parameters from the Hunter
        parameter_batch = hunter.get_next_generation(POPULATION_SIZE, seed_config=seed_config)

        jobs_to_run = []

        # 2. Prepare/Save Job Configurations
        print(f"[Orchestrator] Registering {len(parameter_batch)} new jobs for Gen {gen}...")
        for params_dict in parameter_batch:
            # Add metadata for worker and hashing consistency
            params_dict['run_uuid'] = str(uuid.uuid4())
            params_dict['simulation'] = {"N_grid": 16, "T_steps": 50}

            # --- Deterministic Seeding ---
            seed_64_bit_int = int(params_dict['run_uuid'].replace('-','')[:16], 16)
            params_dict['global_seed'] = seed_64_bit_int % (2**32)

            # Generate canonical hash, which includes the new metadata/seed
            config_hash = generate_canonical_hash(params_dict)
            params_dict['config_hash'] = config_hash # Inject config_hash into the dict

            params_filepath = os.path.join(CONFIG_DIR, f"config_{config_hash}.json")

            with open(params_filepath, 'w') as f:
                json.dump(params_dict, f, indent=2)

            job_entry = {
                aste_hunter.HASH_KEY: config_hash,
                "generation": gen,
                "param_D": params_dict["param_D"],
                "param_eta": params_dict["param_eta"],
                "param_rho_vac": params_dict["param_rho_vac"],
                "param_a_coupling": params_dict["param_a_coupling"],
                "params_filepath": params_filepath
            }
            jobs_to_run.append(job_entry)

        # Register the *full* batch with the Hunter's ledger
        hunter.register_new_jobs(jobs_to_run)

        # 3 & 4. Execute Batch Loop (Worker + Validator)
        job_hashes_completed = []
        for job in jobs_to_run:
            # We call run_simulation_job, which calls the worker and validator sequentially
            success = run_simulation_job(
                config_hash=job[aste_hunter.HASH_KEY],
                params_filepath=job["params_filepath"]
            )
            if success:
                job_hashes_completed.append(job[aste_hunter.HASH_KEY])

        # 5. Ledger Step (Cycle Completion)
        print(f"\n[Orchestrator] GENERATION {gen} COMPLETE.")
        print("[Orchestrator] Notifying Hunter to process results...")
        hunter.process_generation_results(
            provenance_dir=PROVENANCE_DIR,
            job_hashes=job_hashes_completed
        )

        best_run = hunter.get_best_run()
        if best_run:
            print(f"[Orch] Best Run So Far: {best_run[aste_hunter.HASH_KEY][:10]}... (SSE: {best_run[aste_hunter.SSE_METRIC_KEY]:.6f}, Fitness: {best_run['fitness']:.4f})")

    print("\n========================================================")
    print("--- ASTE ORCHESTRATOR: ALL GENERATIONS COMPLETE ---")
    print("========================================================")

if __name__ == "__main__":
    main()
