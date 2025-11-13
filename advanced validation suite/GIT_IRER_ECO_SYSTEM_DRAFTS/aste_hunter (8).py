#!/usr/bin/env python3

"""
aste_hunter.py
CLASSIFICATION: Adaptive Learning Engine (ASTE V1.0)
GOAL: Acts as the "Brain" of the ASTE. It reads validation reports
      (provenance.json) from the SFP module, updates its internal
      ledger, and applies an evolutionary algorithm to breed a
      new generation of parameters that minimize the log_prime_sse.
"""

import os
import json
import csv
import random
import numpy as np
from typing import Dict, Any, List, Optional
import sys # Added for stderr
import uuid # Added for temporary hash generation

# --- Configuration ---
LEDGER_FILENAME = "simulation_ledger.csv"
PROVENANCE_DIR = "provenance_reports" # Where the Validator saves reports
SSE_METRIC_KEY = "log_prime_sse"
HASH_KEY = "config_hash"

# Evolutionary Algorithm Parameters
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.05

class Hunter:
    """
    Implements the core evolutionary "hunt" logic.
    Manages a population of parameters stored in a ledger
    and breeds new generations to minimize SSE.
    """

    def __init__(self, ledger_file: str = LEDGER_FILENAME):
        self.ledger_file = ledger_file
        self.fieldnames = [
            HASH_KEY,
            SSE_METRIC_KEY,
            "fitness",
            "generation",
            "param_D",          # Metric-Aware Diffusion
            "param_eta",        # Damping
            "param_rho_vac",    # Derived metric param
            "param_a_coupling",  # Derived metric param
            "sse_null_phase_scramble", # SPRINT 2
            "sse_null_target_shuffle",  # SPRINT 2
            "n_peaks_found_main", # New Diagnostic Field
            "failure_reason_main", # New Diagnostic Field
            "n_peaks_found_null_a", # New Diagnostic Field
            "failure_reason_null_a", # New Diagnostic Field
            "n_peaks_found_null_b", # New Diagnostic Field
            "failure_reason_null_b"  # New Diagnostic Field
        ]
        self.population = self._load_ledger()
        print(f"[Hunter] Initialized. Loaded {len(self.population)} runs from {self.ledger_file}")

    def _load_ledger(self) -> List[Dict[str, Any]]:
        """Loads the historical population from the CSV ledger."""
        if not os.path.exists(self.ledger_file):
            # Create an empty ledger if it doesn't exist
            with open(self.ledger_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            return []

        population = []
        try:
            with open(self.ledger_file, 'r') as f:
                reader = csv.DictReader(f)
                # Filter out fields from the CSV that are not in self.fieldnames (e.g. older schema)
                # This ensures forward compatibility if fields are added.
                current_fieldnames_set = set(self.fieldnames)
                for row in reader:
                    filtered_row = {k: v for k, v in row.items() if k in current_fieldnames_set}
                    # Convert numerical strings back to floats/ints
                    for key in self.fieldnames:
                        if key in filtered_row and filtered_row[key] is not None and filtered_row[key] != '':
                            try:
                                if key == "generation":
                                    filtered_row[key] = int(float(filtered_row[key])) # Ensure generation is an integer
                                elif key.startswith('param_') or key.startswith('sse_') or key == SSE_METRIC_KEY or key == 'fitness':
                                    filtered_row[key] = float(filtered_row[key])
                                elif key.startswith('n_peaks_found'):
                                    filtered_row[key] = int(float(filtered_row[key]))
                            except ValueError:
                                filtered_row[key] = None # Keep as None if conversion fails
                        elif key not in filtered_row: # Ensure all expected fields are present, even if None
                            filtered_row[key] = None

                    population.append(filtered_row)
        except Exception as e:
            print(f"[Hunter Error] Failed to load ledger: {e}", file=sys.stderr)
        return population

    def _save_ledger(self):
        """Saves the entire population back to the CSV ledger."""
        try:
            with open(self.ledger_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.population)
        except Exception as e:
            print(f"[Hunter Error] Failed to save ledger: {e}", file=sys.stderr)

    def process_generation_results(self, provenance_dir: str, job_hashes: List[str]):
        """
        MANDATE: Reads new provenance.json files, calculates fitness,
        and updates the internal ledger.
        """
        print(f"[Hunter] Processing {len(job_hashes)} new results from {provenance_dir}...")
        new_runs_processed = 0
        for config_hash in job_hashes:
            report_path = os.path.join(provenance_dir, f"provenance_{config_hash}.json")

            try:
                with open(report_path, 'r') as f:
                    data = json.load(f)

                # Extract metrics
                sse = data["spectral_fidelity"][SSE_METRIC_KEY]
                sse_null_phase_scramble = data["spectral_fidelity"].get("sse_null_phase_scramble", np.nan)
                sse_null_target_shuffle = data["spectral_fidelity"].get("sse_null_target_shuffle", np.nan)

                # Extract new diagnostic fields
                n_peaks_found_main = data["spectral_fidelity"].get("n_peaks_found_main", 0)
                failure_reason_main = data["spectral_fidelity"].get("failure_reason_main", None)
                n_peaks_found_null_a = data["spectral_fidelity"].get("n_peaks_found_null_a", 0)
                failure_reason_null_a = data["spectral_fidelity"].get("failure_reason_null_a", None)
                n_peaks_found_null_b = data["spectral_fidelity"].get("n_peaks_found_null_b", 0)
                failure_reason_null_b = data["spectral_fidelity"].get("failure_reason_null_b", None)

                # --- New Falsifiability-Driven Fitness Calculation ---
                base_fitness = 1.0 / (sse + 1e-9)
                falsifiability_penalty = 1.0 # Start with no penalty

                # Penalize if main run fails to find peaks, or other issues
                if failure_reason_main is not None and sse >= 998.0: # 998.0/999.0 are failure codes
                    falsifiability_penalty *= 0.1 # Significant penalty for outright failure

                # If the main SSE is 'good' (low), but the nulls are not significantly higher
                # This indicates a lack of falsifiability (i.e., the effect isn't real)
                if sse < 1.0: # Check if main SSE is low enough to be considered 'good'
                    # Null A check: if null A SSE is NOT significantly higher, apply penalty
                    # (997.0 is specific code from profiler for Null A not differentiating)
                    if sse_null_phase_scramble == 997.0:
                         falsifiability_penalty *= 0.2 # Penalty for Null A not differentiating

                    # Null B check: if null B SSE is NOT significantly higher, apply penalty
                    # (996.0 is specific code from profiler for Null B not differentiating)
                    if sse_null_target_shuffle == 996.0:
                         falsifiability_penalty *= 0.2 # Penalty for Null B not differentiating

                fitness = base_fitness * falsifiability_penalty

                # Find the run in our population and update it
                found = False
                for run in self.population:
                    if run[HASH_KEY] == config_hash:
                        run[SSE_METRIC_KEY] = sse
                        run["fitness"] = fitness
                        run["sse_null_phase_scramble"] = sse_null_phase_scramble
                        run["sse_null_target_shuffle"] = sse_null_target_shuffle
                        # Store new diagnostic fields
                        run["n_peaks_found_main"] = n_peaks_found_main
                        run["failure_reason_main"] = failure_reason_main
                        run["n_peaks_found_null_a"] = n_peaks_found_null_a
                        run["failure_reason_null_a"] = failure_reason_null_a
                        run["n_peaks_found_null_b"] = n_peaks_found_null_b
                        run["failure_reason_null_b"] = failure_reason_null_b
                        found = True
                        break

                if found:
                    new_runs_processed += 1
                else:
                    print(f"[Hunter Warning] Hash {config_hash} found in JSON but not in population ledger.", file=sys.stderr)

            except FileNotFoundError:
                print(f"[Hunter Warning] Provenance file not found: {report_path}", file=sys.stderr)
            except Exception as e:
                print(f"[Hunter Error] Failed to parse {report_path}: {e}", file=sys.stderr)

        print(f"[Hunter] Successfully processed and updated {new_runs_processed} runs.")
        self._save_ledger()

    def _select_parent(self) -> Dict[str, Any]:
        """Selects one parent using tournament selection."""
        # Filter out runs with no fitness (i.e., failed runs that weren't processed)
        eligible_population = [r for r in self.population if r.get("fitness") is not None]
        if not eligible_population:
            raise RuntimeError("Cannot select parent: no runs with valid fitness in population.")

        # Pick N random individuals from the eligible population
        tournament = random.sample(eligible_population, min(TOURNAMENT_SIZE, len(eligible_population)))

        # The winner is the one with the highest fitness
        winner = max(tournament, key=lambda x: x.get("fitness", 0.0))
        return winner

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Performs simple average crossover on parameters."""
        child_params = {
            "param_D": (parent1["param_D"] + parent2["param_D"]) / 2.0,
            "param_eta": (parent1["param_eta"] + parent2["param_eta"]) / 2.0,
            "param_rho_vac": (parent1["param_rho_vac"] + parent2["param_rho_vac"]) / 2.0,
            "param_a_coupling": (parent1["param_a_coupling"] + parent2["param_a_coupling"]) / 2.0
        }
        return child_params

    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Applies random mutation to parameters."""
        if random.random() < MUTATION_RATE:
            params["param_D"] += np.random.normal(0, MUTATION_STRENGTH)
            params["param_D"] = max(0.001, params["param_D"])

        if random.random() < MUTATION_RATE:
            params["param_eta"] += np.random.normal(0, MUTATION_STRENGTH)
            params["param_eta"] = max(0.001, params["param_eta"])

        if random.random() < MUTATION_RATE:
            params["param_rho_vac"] += np.random.normal(0, MUTATION_STRENGTH)
            params["param_rho_vac"] = max(0.001, params["param_rho_vac"])

        if random.random() < MUTATION_RATE:
            params["param_a_coupling"] += np.random.normal(0, MUTATION_STRENGTH)
            params["param_a_coupling"] = max(0.001, params["param_a_coupling"])

        return params

    def get_next_generation(self, population_size: int) -> List[Dict[str, Any]]:
        """
        Breeds a new generation of parameters.
        This is the main function called by the Orchestrator.
        """
        new_generation_params = []

        # Filter out runs with no fitness for breeding
        eligible_population = [r for r in self.population if r.get("fitness") is not None]

        if not eligible_population:
            # Generation 0 or all previous runs failed: Create a random population
            print("[Hunter] No eligible population found. Generating random Generation 0.")
            for _ in range(population_size):
                params = {
                    "param_D": np.random.uniform(0.01, 5.0), # Adjusted range
                    "param_eta": np.random.uniform(0.001, 1.0), # Adjusted range
                    "param_rho_vac": np.random.uniform(0.1, 2.0), # Adjusted range
                    "param_a_coupling": np.random.uniform(0.1, 3.0) # Adjusted range
                }
                new_generation_params.append(params)
        else:
            # Breed a new generation from the existing population
            current_gen = self.get_current_generation()
            print(f"[Hunter] Breeding Generation {current_gen}...")
            # Sort by fitness (highest first)
            sorted_population = sorted(eligible_population, key=lambda x: x.get("fitness", 0.0), reverse=True)

            # Elitism: Keep the top 2 best individuals
            # Ensure there are at least 2 individuals in sorted_population for elitism
            if len(sorted_population) >= 2:
                new_generation_params.append({
                    "param_D": sorted_population[0]["param_D"],
                    "param_eta": sorted_population[0]["param_eta"],
                    "param_rho_vac": sorted_population[0]["param_rho_vac"],
                    "param_a_coupling": sorted_population[0]["param_a_coupling"]
                })
                new_generation_params.append({
                    "param_D": sorted_population[1]["param_D"],
                    "param_eta": sorted_population[1]["param_eta"],
                    "param_rho_vac": sorted_population[1]["param_rho_vac"],
                    "param_a_coupling": sorted_population[1]["param_a_coupling"]
                })
            elif len(sorted_population) == 1: # If only one eligible, just add it once
                new_generation_params.append({
                    "param_D": sorted_population[0]["param_D"],
                    "param_eta": sorted_population[0]["param_eta"],
                    "param_rho_vac": sorted_population[0]["param_rho_vac"],
                    "param_a_coupling": sorted_population[0]["param_a_coupling"]
                })

            # Breed the rest
            for _ in range(population_size - len(new_generation_params)):
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child_params = self._crossover(parent1, parent2)
                mutated_child_params = self._mutate(child_params)
                new_generation_params.append(mutated_child_params)

        return new_generation_params # Return raw params to Orchestrator

    def register_new_jobs(self, job_list: List[Dict[str, Any]]):
        """
        Called by the Orchestrator *after* it has generated
        canonical hashes for the new jobs.
        """
        # Initialize new diagnostic fields for newly registered jobs to None
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
        """
        Utility to get the best-performing run from the ledger.
        """
        if not self.population:
            return None
        valid_runs = [r for r in self.population if r.get("fitness") is not None]
        if not valid_runs:
            return None
        return max(valid_runs, key=lambda x: x["fitness"])

    def get_current_generation(self) -> int:
        """
        Determines the next generation number to breed.
        """
        if not self.population:
            return 0
        # Filter out runs where 'generation' might be None or invalid
        valid_generations = [run['generation'] for run in self.population if 'generation' in run and run['generation'] is not None]
        if not valid_generations:
            return 0
        return max(valid_generations) + 1
