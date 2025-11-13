"""
aste_hunter.py
CLASSIFICATION: Adaptive Learning Engine (ASTE V10.0 - Falsifiability Bonus)
GOAL: Acts as the "Brain" of the ASTE. Calculates fitness based on the
      Falsifiability Bonus and breeds new generations.
"""

import os
import json
import csv
import random
import numpy as np
from typing import Dict, Any, List, Optional
import sys
import uuid

# --- Import Centralized Settings ---
import settings

# Configuration from centralized settings
LEDGER_FILENAME = settings.LEDGER_FILE
PROVENANCE_DIR = settings.PROVENANCE_DIR
SSE_METRIC_KEY = "log_prime_sse"
HASH_KEY = "config_hash"

# Evolutionary Algorithm Parameters
TOURNAMENT_SIZE = 3
MUTATION_RATE = settings.MUTATION_RATE
MUTATION_STRENGTH = settings.MUTATION_STRENGTH
LAMBDA_FALSIFIABILITY = settings.LAMBDA_FALSIFIABILITY

class Hunter:
    """
    Manages population, calculates fitness, and breeds new generations.
    """

    def __init__(self, ledger_file: str = LEDGER_FILENAME):
        self.ledger_file = ledger_file
        self.fieldnames = [
            HASH_KEY, SSE_METRIC_KEY, "fitness", "generation",
            "param_D", "param_eta", "param_rho_vac", "param_a_coupling",
            "sse_null_phase_scramble", "sse_null_target_shuffle",
            "n_peaks_found_main", "failure_reason_main",
            "n_peaks_found_null_a", "failure_reason_null_a",
            "n_peaks_found_null_b", "failure_reason_null_b"
        ]
        self.population = self._load_ledger()
        if self.population:
            print(f"[Hunter] Initialized. Loaded {len(self.population)} runs from {os.path.basename(ledger_file)}")
        else:
            print(f"[Hunter] Initialized. No prior runs found in {os.path.basename(ledger_file)}")

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

                # Ensure fieldnames exist before accessing them
                if reader.fieldnames:
                    # Update fieldnames if ledger has new columns
                    if len(reader.fieldnames) > len(self.fieldnames):
                         self.fieldnames = reader.fieldnames
                else:
                    # If no fieldnames (empty file), still proceed with default ones
                    print(f"[Hunter Warning] Ledger file {self.ledger_file} is empty or has no header. Using default fieldnames.", file=sys.stderr)

                for row in reader:
                    # Conversion logic for safety
                    try:
                        for key in self.fieldnames:
                            if key in row and row[key] not in ('', 'None', 'NaN'):
                                # Explicitly handle 'generation' and peak counts as integers
                                if key in ["generation", "n_peaks_found_main", "n_peaks_found_null_a", "n_peaks_found_null_b"]:
                                     # Cast to float first to handle decimals from csv, then to int
                                     row[key] = int(float(row[key]))
                                else:
                                    row[key] = float(row[key])
                        population.append(row)
                    except Exception as e:
                        print(f"[Hunter Warning] Skipping malformed row: {row}. Error: {e}", file=sys.stderr)

            # Sort population by fitness, best first
            population.sort(key=lambda x: x.get('fitness', 0.0) or 0.0, reverse=True)
            return population
        except Exception as e:
            print(f"[Hunter Error] Failed to load ledger: {e}", file=sys.stderr)
            return []

    def _save_ledger(self):
        """Saves the entire population back to the ledger CSV."""
        os.makedirs(os.path.dirname(self.ledger_file), exist_ok=True)
        try:
            with open(self.ledger_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                for row in self.population:
                    complete_row = {field: row.get(field) for field in self.fieldnames}
                    writer.writerow(complete_row)
        except Exception as e:
            print(f"[Hunter Error] Failed to save ledger: {e}", file=sys.stderr)

    def _get_random_parent(self) -> Dict[str, Any]:
        """Selects a parent using tournament selection."""
        valid_runs = [r for r in self.population if r.get("fitness") is not None and r["fitness"] > 0]
        if len(valid_runs) < TOURNAMENT_SIZE:
            # Fallback to general population if not enough fit runs exist
            return random.choice(self.population) if self.population else {}

        tournament = random.sample(valid_runs, min(len(valid_runs), TOURNAMENT_SIZE))
        best = max(tournament, key=lambda x: x.get("fitness") or 0.0)
        return best

    def _breed(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a child by crossover and mutation."""
        child = {}
        param_keys = ["param_D", "param_eta", "param_rho_vac", "param_a_coupling"]

        # Crossover
        for key in param_keys:
            child[key] = random.choice([parent1.get(key, 1.0), parent2.get(key, 1.0)])

        # Mutation
        if random.random() < MUTATION_RATE:
            key_to_mutate = random.choice(param_keys)
            if child.get(key_to_mutate) is not None:
                mutation = random.gauss(0, MUTATION_STRENGTH)
                child[key_to_mutate] = child[key_to_mutate] * (1 + mutation)
                # Add clipping/clamping
                child[key_to_mutate] = max(0.01, min(child[key_to_mutate], 5.0))

        return child

    def get_next_generation(self, n_population: int, seed_config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Breeds a new generation of parameters.
        Returns a list of parameter dicts for the Orchestrator.
        """
        new_generation_params = []
        current_gen = self.get_current_generation()

        # Check if population is valid for breeding
        valid_parents = [r for r in self.population if r.get("fitness") is not None and r["fitness"] > 0]

        if not valid_parents and current_gen == 0:
            # Generation 0: Random search
            print(f"[Hunter] Not enough fit parents. Generating random Generation {current_gen}.")
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
                new_generation_params.append({k: best_run.get(k, 1.0) for k in ["param_D", "param_eta", "param_rho_vac", "param_a_coupling"]})

            # Fill the rest with children
            while len(new_generation_params) < n_population:
                parent1 = self._get_random_parent()
                parent2 = self._get_random_parent()
                child = self._breed(parent1, parent2)
                new_generation_params.append(child)

        # Prepare job entries for registration
        job_list = []
        for params in new_generation_params:
            job_entry = {
                "generation": current_gen,
                "param_D": params["param_D"],
                "param_eta": params["param_eta"],
                "param_rho_vac": params["param_rho_vac"],
                "param_a_coupling": params["param_a_coupling"]
            }
            job_list.append(job_entry)
        return job_list

    def register_new_jobs(self, job_list: List[Dict[str, Any]]):
        """Called by the Orchestrator to register jobs and initialize fields."""
        # Initialize new diagnostic fields for newly registered jobs to None
        for job in job_list:
            for field in self.fieldnames:
                 if field not in job: job[field] = None

        self.population.extend(job_list)
        print(f"[Hunter] Registered {len(job_list)} new jobs in ledger.")

    def get_best_run(self) -> Optional[Dict[str, Any]]:
        if not self.population: return None
        valid_runs = [r for r in self.population if r.get("fitness") is not None]
        if not valid_runs: return None
        return max(valid_runs, key=lambda x: x.get("fitness") or 0.0)

    def get_current_generation(self) -> int:
        """Determines the next generation number to breed."""
        if not self.population: return 0
        valid_generations = [run['generation'] for run in self.population if 'generation' in run and run['generation'] is not None]
        if not valid_generations: return 0
        # Explicitly cast to int to prevent TypeError with range()
        return int(max(valid_generations)) + 1


    def process_generation_results(self, provenance_dir: str, job_hashes: List[str]):
        """
        Processes all provenance reports, calculates FALSIFIABILITY-REWARD fitness,
        and updates the ledger.
        """
        print(f"[Hunter] Processing {len(job_hashes)} new results from {provenance_dir}...")
        processed_count = 0

        pop_lookup = {run[HASH_KEY]: run for run in self.population}

        for config_hash in job_hashes:
            prov_file = os.path.join(provenance_dir, f"provenance_{config_hash}.json")
            if not os.path.exists(prov_file):
                print(f"[Hunter Warning] Missing provenance file for {config_hash[:10]}...", file=sys.stderr)
                continue

            try:
                with open(prov_file, 'r') as f:
                    provenance = json.load(f)
                run_to_update = pop_lookup.get(config_hash)
                if not run_to_update:
                    print(f"[Hunter Warning] Hash {config_hash[:10]} not in population ledger.", file=sys.stderr)
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

                # Ensure diagnostic fields are handled, converting back to string if necessary for the CSV writer
                run_to_update["n_peaks_found_main"] = spec.get("n_peaks_found_main", 0)
                run_to_update["failure_reason_main"] = str(spec.get("failure_reason_main", "None"))
                run_to_update["n_peaks_found_null_a"] = spec.get("n_peaks_found_null_a", 0)
                run_to_update["failure_reason_null_a"] = str(spec.get("failure_reason_null_a", "None"))
                run_to_update["n_peaks_found_null_b"] = spec.get("n_peaks_found_null_b", 0)
                run_to_update["failure_reason_null_b"] = str(spec.get("failure_reason_null_b", "None"))

                processed_count += 1
            except Exception as e:
                print(f"[Hunter Error] Failed to process {prov_file}: {e}", file=sys.stderr)

        self._save_ledger()
        print(f"[Hunter] Successfully processed and updated {processed_count} runs.")
