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
            "param_kappa", # Example parameter 1
            "param_sigma_k"  # Example parameter 2
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
                for row in reader:
                    # Convert numerical strings back to floats/ints
                    for key in [SSE_METRIC_KEY, "fitness", "generation", "param_kappa", "param_sigma_k"]:
                        if key in row and row[key]:
                            try:
                                row[key] = float(row[key])
                            except ValueError:
                                row[key] = None
                    population.append(row)
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
                
                # Extract the critical SSE metric
                sse = data["spectral_fidelity"][SSE_METRIC_KEY]
                
                # Mandated Fitness Formula: fitness = 1 / SSE
                # Add a small epsilon to prevent division by zero
                fitness = 1.0 / (sse + 1e-9)
                
                # Find the run in our population and update it
                found = False
                for run in self.population:
                    if run[HASH_KEY] == config_hash:
                        run[SSE_METRIC_KEY] = sse
                        run["fitness"] = fitness
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
        # Pick N random individuals from the population
        tournament = random.sample(self.population, TOURNAMENT_SIZE)
        
        # The winner is the one with the highest fitness (lowest SSE)
        winner = max(tournament, key=lambda x: x.get("fitness", 0.0))
        return winner

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Performs simple average crossover on parameters."""
        child_params = {
            "param_kappa": (parent1["param_kappa"] + parent2["param_kappa"]) / 2.0,
            "param_sigma_k": (parent1["param_sigma_k"] + parent2["param_sigma_k"]) / 2.0,
        }
        return child_params

    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Applies random mutation to parameters."""
        if random.random() < MUTATION_RATE:
            params["param_kappa"] += np.random.normal(0, MUTATION_STRENGTH)
            # Ensure parameters stay within reasonable bounds
            params["param_kappa"] = max(0.001, params["param_kappa"])
            
        if random.random() < MUTATION_RATE:
            params["param_sigma_k"] += np.random.normal(0, MUTATION_STRENGTH)
            params["param_sigma_k"] = max(0.1, params["param_sigma_k"])
            
        return params

    def get_next_generation(self, population_size: int) -> List[Dict[str, Any]]:
        """
        Breeds a new generation of parameters.
        This is the main function called by the Orchestrator.
        """
        new_generation_params = []
        
        if not self.population:
            # Generation 0: Create a random population
            print("[Hunter] No population found. Generating random Generation 0.")
            for _ in range(population_size):
                params = {
                    "param_kappa": np.random.uniform(0.01, 0.1),
                    "param_sigma_k": np.random.uniform(0.1, 1.0)
                }
                new_generation_params.append(params)
        else:
            # Breed a new generation from the existing population
            print(f"[Hunter] Breeding Generation {self.population[-1]['generation'] + 1}...")
            # Sort by fitness (highest first)
            sorted_population = sorted(self.population, key=lambda x: x.get("fitness", 0.0), reverse=True)
            
            # Elitism: Keep the top 2 best individuals
            new_generation_params.append({"param_kappa": sorted_population[0]["param_kappa"], "param_sigma_k": sorted_population[0]["param_sigma_k"]})
            new_generation_params.append({"param_kappa": sorted_population[1]["param_kappa"], "param_sigma_k": sorted_population[1]["param_sigma_k"]})
            
            # Breed the rest
            for _ in range(population_size - 2):
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child_params = self._crossover(parent1, parent2)
                mutated_child_params = self._mutate(child_params)
                new_generation_params.append(mutated_child_params)

        # --- Update Internal Ledger ---
        # Add these new jobs to our internal population *before* they run
        # They will be updated with SSE/fitness later.
        current_gen = self.population[-1]['generation'] + 1 if self.population else 0
        new_jobs = []
        for params in new_generation_params:
            # This is a temporary hash, as the Orchestrator will add a UUID
            # The *real* hash will be in the provenance.json file
            temp_hash = f"temp_job_{uuid.uuid4().hex[:10]}"
            job_entry = {
                HASH_KEY: temp_hash,
                SSE_METRIC_KEY: None,
                "fitness": None,
                "generation": current_gen,
                "param_kappa": params["param_kappa"],
                "param_sigma_k": params["param_sigma_k"]
            }
            new_jobs.append(job_entry)
        
        # We will add these to the population *after* the Orchestrator
        # provides the real hashes. This function just returns the raw params.
        
        # We need to return the parameters *and* update our internal state
        # The orchestrator will give us the *real* hashes
        
        # Let's simplify: The Hunter only returns params.
        # The Orchestrator adds them to the ledger with the *real* hashes.
        # This is a cleaner separation of concerns.
        # --> We will adjust this in the Orchestrator test.
        
        # For this test, we'll keep the logic simple as above.
        # The Orchestrator will call `process_generation_results`
        # We need a way to link the new jobs.
        
        # Let's stick to the prompt's architecture:
        # 1. Hunter returns new params.
        # 2. Orchestrator creates hashes, saves configs, runs jobs, saves provenance.
        # 3. Orchestrator calls `process_generation_results` with the *list of hashes* it created.
        
        # This means the Hunter *doesn't* add them to the ledger.
        # The Orchestrator must pass the params *back* to the hunter.
        # Let's create a new function for that.

        self.last_generation_jobs = [] # Clear last batch
        for params in new_generation_params:
            # This job entry is temporary, to be confirmed by the orchestrator
            job_entry = {
                SSE_METRIC_KEY: None,
                "fitness": None,
                "generation": current_gen,
                "param_kappa": params["param_kappa"],
                "param_sigma_k": params["param_sigma_k"]
            }
            self.last_generation_jobs.append(job_entry)

        return new_generation_params # Return raw params to Orchestrator

    def register_new_jobs(self, job_list: List[Dict[str, Any]]):
        """
        Called by the Orchestrator *after* it has generated
        canonical hashes for the new jobs.
        """
        self.population.extend(job_list)
        print(f"[Hunter] Registered {len(job_list)} new jobs in ledger.")

    def get_best_run(self) -> Optional[Dict[str, Any]]:
        """Utility to get the best-performing run from the ledger."""
        if not self.population:
            return None
        valid_runs = [r for r in self.population if r.get("fitness")]
        if not valid_runs:
            return None
        return max(valid_runs, key=lambda x: x["fitness"])
