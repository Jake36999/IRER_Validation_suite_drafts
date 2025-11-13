#!/usr/bin/env python3

"""
aste_hunter.py
CLASSIFICATION: Adaptive Learning Engine (ASTE V10.0 - Falsifiability Bonus)
GOAL: Acts as the "Brain" of the ASTE. It reads validation reports
      (provenance.json), calculates a falsifiability-driven fitness,
      and breeds new generations to minimize SSE while maximizing
      the gap between signal and null-test noise.
"""

import os
import json
import csv
import random
import numpy as np
from typing import Dict, Any, List, Optional
import sys
import uuid

# --- Configuration ---
LEDGER_FILENAME = "simulation_ledger.csv"
PROVENANCE_DIR = "provenance_reports"
SSE_METRIC_KEY = "log_prime_sse"
HASH_KEY = "config_hash"

# Evolutionary Algorithm Parameters
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.05

# --- PATCH APPLIED ---
# Reward weight for falsifiability gap (null SSEs >> main SSE)
# Tune: 0.05–0.2 are sensible. Start at 0.1.
LAMBDA_FALSIFIABILITY = 0.1
# --- END PATCH ---

class Hunter:
    """
    Implements the core evolutionary "hunt" logic.
    Manages a population of parameters stored in a ledger
    and breeds new generations to minimize SSE.
    """

    def __init__(self, ledger_file: str = LEDGER_FILENAME):
        self.ledger_file = ledger_file
        
        # --- PATCHED FIELDNAMES ---
        # (This matches your aste_hunter (9).py version)
        self.fieldnames = [
            HASH_KEY,
            SSE_METRIC_KEY,
            "fitness",
            "generation",
            "param_D",
            "param_eta",
            "param_rho_vac",
            "param_a_coupling",
            "sse_null_phase_scramble",
            "sse_null_target_shuffle",
            "n_peaks_found_main",
            "failure_reason_main",
            "n_peaks_found_null_a",
            "failure_reason_null_a",
            "n_peaks_found_null_b",
            "failure_reason_null_b"
        ]
        # --- END PATCH ---
        
        self.population = self._load_ledger()
        if self.population:
            print(f"[Hunter] Initialized. Loaded {len(self.population)} runs from {ledger_file}")
        else:
            print(f"[Hunter] Initialized. No prior runs found in {ledger_file}")

    def _load_ledger(self) -> List[Dict[str, Any]]:
        """
        Loads the existing population from the ledger CSV.
        Handles type conversion and missing files.
        """
        population = []
        if not os.path.exists(self.ledger_file):
            return population

        try:
            with open(self.ledger_file, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Ensure all fieldnames are present
                if not all(field in reader.fieldnames for field in self.fieldnames):
                     print(f"[Hunter Warning] Ledger {self.ledger_file} has mismatched columns. Re-init may be needed.", file=sys.stderr)
                     # Use the file's fieldnames as a fallback
                     self.fieldnames = reader.fieldnames
                
                for row in reader:
                    try:
                        # Convert numeric types
                        for key in [SSE_METRIC_KEY, "fitness", "generation",
                                    "param_D", "param_eta", "param_rho_vac",
                                    "param_a_coupling", "sse_null_phase_scramble",
                                    "sse_null_target_shuffle", "n_peaks_found_main",
                                    "n_peaks_found_null_a", "n_peaks_found_null_b"]:
                            if row.get(key) is not None and row[key] != '':
                                row[key] = float(row[key])
                            else:
                                row[key] = None # Use None for missing numeric data
                        population.append(row)
                    except (ValueError, TypeError) as e:
                        print(f"[Hunter Warning] Skipping malformed row: {row}. Error: {e}", file=sys.stderr)
            
            # Sort population by fitness, best first (if fitness exists)
            population.sort(key=lambda x: x.get('fitness', 0.0) or 0.0, reverse=True)
            return population
        except Exception as e:
            print(f"[Hunter Error] Failed to load ledger {self.ledger_file}: {e}", file=sys.stderr)
            return []

    def _save_ledger(self):
        """Saves the entire population back to the ledger CSV."""
        try:
            with open(self.ledger_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                for row in self.population:
                    # Ensure all rows have all fields to avoid write errors
                    complete_row = {field: row.get(field) for field in self.fieldnames}
                    writer.writerow(complete_row)
        except Exception as e:
            print(f"[Hunter Error] Failed to save ledger {self.ledger_file}: {e}", file=sys.stderr)

    def _get_random_parent(self) -> Dict[str, Any]:
        """Selects a parent using tournament selection."""
        tournament = random.sample(self.population, TOURNAMENT_SIZE)
        # Handle runs that may not have fitness yet
        best = max(tournament, key=lambda x: x.get("fitness") or 0.0)
        return best

    def _breed(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a child by crossover and mutation."""
        child = {}
        # Crossover
        for key in ["param_D", "param_eta", "param_rho_vac", "param_a_coupling"]:
            child[key] = random.choice([parent1[key], parent2[key]])
        
        # Mutation
        if random.random() < MUTATION_RATE:
            key_to_mutate = random.choice(["param_D", "param_eta", "param_rho_vac", "param_a_coupling"])
            mutation = random.gauss(0, MUTATION_STRENGTH)
            child[key_to_mutate] = child[key_to_mutate] * (1 + mutation)
            # Add clipping/clamping if necessary
            child[key_to_mutate] = max(0.01, min(child[key_to_mutate], 5.0)) # Simple clamp
            
        return child

    def get_next_generation(self, n_population: int) -> List[Dict[str, Any]]:
        """
        Breeds a new generation of parameters.
        Returns a list of parameter dicts for the Orchestrator.
        """
        new_generation_params = []
        current_gen = self.get_current_generation()
        
        if not self.population:
            # Generation 0: Random search
            print(f"[Hunter] No population found. Generating random Generation {current_gen}.")
            for _ in range(n_population):
                new_generation_params.append({
                    "param_D": random.uniform(0.01, 5.0),
                    "param_eta": random.uniform(0.001, 1.0),
                    "param_rho_vac": random.uniform(0.1, 2.0),
                    "param_a_coupling": random.uniform(0.1, 3.0),
                })
        else:
            # Subsequent Generations: Evolve
            print(f"[Hunter] Breeding Generation {current_gen}...")
            # Elitism: Carry over the best run
            best_run = self.get_best_run()
            if best_run:
                new_generation_params.append({k: best_run[k] for k in ["param_D", "param_eta", "param_rho_vac", "param_a_coupling"]})
            
            # Fill the rest with children
            while len(new_generation_params) < n_population:
                parent1 = self._get_random_parent()
                parent2 = self._get_random_parent()
                child = self._breed(parent1, parent2)
                new_generation_params.append(child)
        
        # Prepare job entries for registration
        self.last_generation_jobs = []
        for params in new_generation_params:
            job_entry = {
                "generation": current_gen,
                "param_D": params["param_D"],
                "param_eta": params["param_eta"],
                "param_rho_vac": params["param_rho_vac"],
                "param_a_coupling": params["param_a_coupling"]
            }
            self.last_generation_jobs.append(job_entry)

        return new_generation_params

    def register_new_jobs(self, job_list: List[Dict[str, Any]]):
        """
        Called by the Orchestrator *after* it has generated
        canonical hashes for the new jobs.
        """
        for job in job_list:
            job["n_peaks_found_main"] = None
            job["failure_reason_main"] = None
            job["n_peaks_found_null_a"] = None
            job["failure_reason_null_a"] = None
            job["n_peaks_found_null_b"] = None
            job["failure_reason_null_b"] = None
        
        self.population.extend(job_list)
        print(f"[Hunter] Registered {len(job_list)} new jobs in ledger.")

    def get_best_run(self) -> Optional[Dict[str, Any]]:
        """Utility to get the best-performing run from the ledger."""
        if not self.population:
            return None
        valid_runs = [r for r in self.population if r.get("fitness") is not None]
        if not valid_runs:
            return None
        return max(valid_runs, key=lambda x: x["fitness"])

    def get_current_generation(self) -> int:
        """Determines the next generation number to breed."""
        if not self.population:
            return 0
        valid_generations = [run['generation'] for run in self.population if 'generation' in run and run['generation'] is not None]
        if not valid_generations:
            return 0
        return max(valid_generations) + 1

    # ---
    # --- PATCH APPLIED: New Falsifiability-Reward Fitness Function ---
    # ---
    
    def process_generation_results(self, provenance_dir: str, job_hashes: List[str]):
        """
        Processes all provenance reports from a completed generation.
        Reads metrics, calculates FALSIFIABILITY-REWARD fitness,
        and updates the ledger.
        """
        print(f"[Hunter] Processing {len(job_hashes)} new results from {provenance_dir}...")
        processed_count = 0

        pop_lookup = {run[HASH_KEY]: run for run in self.population}

        for config_hash in job_hashes:
            prov_file = os.path.join(provenance_dir, f"provenance_{config_hash}.json")
            if not os.path.exists(prov_file):
                print(f"[Hunter Warning] Missing provenance for {config_hash[:10]}...", file=sys.stderr)
                continue
            try:
                with open(prov_file, 'r') as f:
                    provenance = json.load(f)
                run_to_update = pop_lookup.get(config_hash)
                if not run_to_update:
                    print(f"[Hunter Warning] {config_hash[:10]} not in population ledger.", file=sys.stderr)
                    continue

                spec = provenance.get("spectral_fidelity", {})
                sse = float(spec.get("log_prime_sse", 1002.0))
                sse_null_a = float(spec.get("sse_null_phase_scramble", 1002.0))
                sse_null_b = float(spec.get("sse_null_target_shuffle", 1002.0))

                # Cap nulls at 1000 to avoid runaway bonus from profiler error codes
                sse_null_a = min(sse_null_a, 1000.0)
                sse_null_b = min(sse_null_b, 1000.0)

                if not (np.isfinite(sse) and sse < 900.0):
                    fitness = 0.0  # failed or sentinel main SSE
                else:
                    base_fitness = 1.0 / max(sse, 1e-12)
                    delta_a = max(0.0, sse_null_a - sse)
                    delta_b = max(0.0, sse_null_b - sse)
                    bonus = LAMBDA_FALSIFIABILITY * (delta_a + delta_b)
                    fitness = base_fitness + bonus

                # Update run fields
                run_to_update[SSE_METRIC_KEY] = sse
                run_to_update["fitness"] = fitness
                run_to_update["sse_null_phase_scramble"] = sse_null_a
                run_to_update["sse_null_target_shuffle"] = sse_null_b
                run_to_update["n_peaks_found_main"] = spec.get("n_peaks_found_main")
                run_to_update["failure_reason_main"] = spec.get("failure_reason_main")
                run_to_update["n_peaks_found_null_a"] = spec.get("n_peaks_found_null_a")
                run_to_update["failure_reason_null_a"] = spec.get("failure_reason_null_a")
                run_to_update["n_peaks_found_null_b"] = spec.get("n_peaks_found_null_b")
                run_to_update["failure_reason_null_b"] = spec.get("failure_reason_null_b")
                processed_count += 1
            except Exception as e:
                print(f"[Hunter Error] Failed to process {prov_file}: {e}", file=sys.stderr)

        self._save_ledger()
        print(f"[Hunter] Successfully processed and updated {processed_count} runs.")

    # ---
    # --- END OF PATCH ---
    # ---

if __name__ == '__main__':
    print("--- ASTE Hunter (Self-Test) ---")
    
    # Simple test logic
    TEST_LEDGER = "test_ledger.csv"
    if os.path.exists(TEST_LEDGER):
        os.remove(TEST_LEDGER)
        
    hunter = Hunter(ledger_file=TEST_LEDGER)
    print(f"\n1. Current Generation (should be 0): {hunter.get_current_generation()}")
    
    print("\n2. Breeding Generation 0...")
    gen_0_params = hunter.get_next_generation(n_population=4)
    print(f"  -> Bred {len(gen_0_params)} param sets.")
    
    # Mock registration
    mock_jobs = []
    for i, params in enumerate(gen_0_params):
        job = params.copy()
        job[HASH_KEY] = f"hash_gen0_{i}"
        job["generation"] = 0
        mock_jobs.append(job)
    hunter.register_new_jobs(mock_jobs)
    
    print(f"\n3. Population after registration (should be 4): {len(hunter.population)}")
    
    # Mock results processing
    print("\n4. Mocking provenance and processing results...")
    mock_prov_dir = "mock_provenance"
    os.makedirs(mock_prov_dir, exist_ok=True)
    
    # Mock the "Golden Run"
    golden_hash = "hash_gen0_0"
    golden_prov = {
        "config_hash": golden_hash,
        "spectral_fidelity": {
            "log_prime_sse": 0.129,
            "sse_null_phase_scramble": 999.0,
            "sse_null_target_shuffle": 996.0,
            "n_peaks_found_main": 1, "failure_reason_main": None,
            "n_peaks_found_null_a": 0, "failure_reason_null_a": "No peaks",
            "n_peaks_found_null_b": 0, "failure_reason_null_b": "No peaks"
        }
    }
    with open(os.path.join(mock_prov_dir, f"provenance_{golden_hash}.json"), 'w') as f:
        json.dump(golden_prov, f)

    # Mock a "Failed Run"
    failed_hash = "hash_gen0_1"
    failed_prov = {
        "config_hash": failed_hash,
        "spectral_fidelity": {
            "log_prime_sse": 999.0, "failure_reason_main": "No peaks",
            # ... (other fields)
        }
    }
    with open(os.path.join(mock_prov_dir, f"provenance_{failed_hash}.json"), 'w') as f:
        json.dump(failed_prov, f)
    
    # Process
    hunter.process_generation_results(
        provenance_dir=mock_prov_dir,
        job_hashes=["hash_gen0_0", "hash_gen0_1", "hash_gen0_2"] # 2 found, 1 missing
    )
    
    print("\n5. Checking ledger for fitness...")
    best_run = hunter.get_best_run()
    
    if best_run and best_run[HASH_KEY] == golden_hash:
        print(f"  -> SUCCESS: Best run is {best_run[HASH_KEY]}")
        print(f"  -> Fitness (should be ~207): {best_run['fitness']:.4f}")
        expected_fitness = (1.0 / 0.129) + LAMBDA_FALSIFIABILITY * ( (999.0-0.129) + (996.0-0.129) )
        print(f"  -> Expected Fitness: {expected_fitness:.4f}")
        if not np.isclose(best_run['fitness'], expected_fitness): 
             print("  -> TEST FAILED: Fitness mismatch!")
    else:
        print(f"  -> TEST FAILED: Did not find best run. Found: {best_run}")
        
    print(f"\n6. Current Generation (should be 1): {hunter.get_current_generation()}")

    # Cleanup
    if os.path.exists(TEST_LEDGER): os.remove(TEST_LEDGER)
    if os.path.exists(os.path.join(mock_prov_dir, f"provenance_{golden_hash}.json")): os.remove(os.path.join(mock_prov_dir, f"provenance_{golden_hash}.json"))
    if os.path.exists(os.path.join(mock_prov_dir, f"provenance_{failed_hash}.json")): os.remove(os.path.join(mock_prov_dir, f"provenance_{failed_hash}.json"))
    if os.path.exists(mock_prov_dir): os.rmdir(mock_prov_dir)
