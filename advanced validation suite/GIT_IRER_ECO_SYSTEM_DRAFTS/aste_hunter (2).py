#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASTE Hunter (v7.0) — targets worker_v7.py

Usage:
  python aste_hunter.py <HUNT_ID> <TODO_FILE>

Environment overrides (optional):
  ASTE_POP_SIZE, ASTE_ELITE_K, ASTE_MUT_SCALE, ASTE_RESEED_FRAC, ASTE_STAG_GENS
"""

import os, sys, json, glob, time, math, random
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np # Added for math safety and compatibility

# ---- Defaults ---------------------------------------------------------------
TARGET_WORKER        = "worker_v7.py" # <-- CRITICAL: Targets the 3D worker
MASTER_DIR           = "sweep_runs"
DEFAULT_POP_SIZE     = 100 # Resetting to 100 to match your standard batch size
DEFAULT_ELITE_K      = 10  # 10% of 100
DEFAULT_MUTATION_SCALE = 0.15 # Aggressive mutation for wide 3D space
DEFAULT_MUTATION_MIN   = 1e-4
DEFAULT_RESEED_FRAC    = 0.35
DEFAULT_STAG_GENS      = 5

# Fallback param space (Use the established 5D range from V6/V7 project docs)
FALLBACK_PARAM_SPACE = {
    # Match the ranges used in worker_v6.py/v7.py for consistency
    "alpha":         {"min": 0.01,  "max": 1.0,   "scale": "linear"},
    "sigma_k":       {"min": 0.1,   "max": 10.0,  "scale": "linear"},
    "nu":            {"min": 0.1,   "max": 5.0,   "scale": "linear"},
    "OMEGA_PARAM_A": {"min": 0.1,   "max": 2.5,   "scale": "linear"},
    "KAPPA":         {"min": 0.001, "max": 5.0,   "scale": "linear"},
}

# ---- Small utils ------------------------------------------------------------
def _hunt_dir(hunt_id: str) -> str:
    return os.path.join(MASTER_DIR, hunt_id)

def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _dump_json(path: str, obj: Any):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def _ledger_candidates(hunt_id: str) -> List[str]:
    hd = _hunt_dir(hunt_id)
    patt = [
        os.path.join(hd, f"ledger_{hunt_id}*.csv"),
        os.path.join(MASTER_DIR, f"ledger_{hunt_id}*.csv"),
    ]
    files: List[str] = []
    for p in patt:
        files.extend(glob.glob(p))
    return files

def _latest_scored_ledger(hunt_id: str) -> Tuple[str, pd.DataFrame]:
    best_path, best_mtime = "", -1.0
    for f in _ledger_candidates(hunt_id):
        try:
            # Use low_memory=False to handle potential mixed dtypes correctly
            d = pd.read_csv(f, low_memory=False)
            if "final_sse" in d.columns and (d["final_sse"] < 90000).any():
                mt = os.path.getmtime(f)
                if mt > best_mtime:
                    best_mtime, best_path = mt, f
        except Exception:
            pass
    if not best_path:
        return "", pd.DataFrame()
    return best_path, pd.read_csv(best_path, low_memory=False)

def _resolve_param_space(hunt_id: str, todo_file: str) -> Dict[str, Dict[str, Any]]:
    # priority: hunt-local param_space.json > existing TODO -> fallback
    ps_local = _load_json(os.path.join(_hunt_dir(hunt_id), "param_space.json"))
    if isinstance(ps_local, dict) and ps_local:
        return ps_local
    # NOTE: The V6/V7 worker bootstrap generates the initial jobs list, not the hunter,
    # so we rely mainly on the fallback/local config.
    return FALLBACK_PARAM_SPACE

def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))

def _mutate_param(v: float, spec: Dict[str, Any], scale: float) -> float:
    lo, hi = float(spec["min"]), float(spec["max"])
    span = max(hi - lo, 1e-12)
    step = max(span * scale, DEFAULT_MUTATION_MIN)
    nv = v + random.gauss(0.0, step)

    # Reflect & clip logic for boundary constraints
    if nv < lo:
        nv = lo + (lo - nv)
    if nv > hi:
        nv = hi - (nv - hi)

    return _clip(nv, lo, hi)

def _random_params(pspace: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, spec in pspace.items():
        lo, hi = float(spec["min"]), float(spec["max"])
        if spec.get("scale", "linear") == "log":
            loL, hiL = math.log(max(lo, 1e-12)), math.log(max(hi, 1e-11))
            out[k] = float(math.exp(random.uniform(loL, hiL)))
        else:
            out[k] = float(random.uniform(lo, hi))
    return out

def _params_from_row(row: pd.Series) -> Dict[str, float]:
    params = {}
    # Handles both 'alpha' and 'params.alpha' style columns
    for k in FALLBACK_PARAM_SPACE.keys():
        if k in row.index and row[k] < 90000.0:
            params[k] = float(row[k])
        elif f"params.{k}" in row.index and row[f"params.{k}"] < 90000.0:
            params[k] = float(row[f"params.{k}"])
    return params

def _best_elites(df: pd.DataFrame, k: int) -> List[Dict[str, float]]:
    # Filter out failed runs (SSE > 90000.0)
    df_ok = df[df["final_sse"] < 90000.0].copy()

    # Fill NaN columns for sorting compatibility, assuming NaN implies bad data or
    # the column was added later (using 99999.0 as a safe worst-case value for sorting)
    df_ok = df_ok.fillna(99999.0)

    if df_ok.empty: return []
    df_ok.sort_values("final_sse", ascending=True, inplace=True)

    elites: List[Dict[str, float]] = []
    for _, r in df_ok.head(k).iterrows():
        pr = _params_from_row(r)
        if pr and len(pr) == len(FALLBACK_PARAM_SPACE): # Ensure we get all 5 parameters
            elites.append(pr)
    return elites

def _resolve_generation(hunt_id: str, df: pd.DataFrame) -> int:
    state = _load_json(os.path.join(_hunt_dir(hunt_id), "hunter_state.json")) or {}
    if "generation" in df.columns and not df.empty:
        try: return int(df["generation"].max()) + 1
        except Exception: pass
    if isinstance(state.get("generation"), int):
        return state["generation"] + 1
    return 0

def _update_state(hunt_id: str, gen: int, best_sse: float, stagnant_gens: int) -> None:
    _dump_json(os.path.join(_hunt_dir(hunt_id), "hunter_state.json"), {
        "generation": gen,
        "best_sse": best_sse,
        "stagnant_gens": stagnant_gens,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    })

# ---- Core evolve -------------------------------------------------------------
def evolve_next_population(hunt_id: str, todo_file: str,
                           pop_size: int, elite_k: int,
                           mutation_scale: float,
                           reseed_frac: float, stagnation_gens: int) -> Dict[str, Any]:

    # Use Hunt ID and time for a more unique seed
    random.seed(int(time.time() * 1000) ^ hash(hunt_id))

    os.makedirs(_hunt_dir(hunt_id), exist_ok=True)

    param_space = _resolve_param_space(hunt_id, todo_file)
    latest_path, df = _latest_scored_ledger(hunt_id)

    best_sse = float("inf")
    if not df.empty and "final_sse" in df.columns:
        try:
            best_sse = float(df.loc[df["final_sse"].idxmin(), "final_sse"])
        except Exception:
            try: best_sse = float(df["final_sse"].min())
            except Exception: pass

    next_gen = _resolve_generation(hunt_id, df)

    # Stagnation tracking logic
    prev = _load_json(os.path.join(_hunt_dir(hunt_id), "hunter_state.json")) or {}
    prev_best = prev.get("best_sse", float("inf"))
    prev_stag = int(prev.get("stagnant_gens", 0))
    stagnant = 0 if best_sse < prev_best - 1e-12 else prev_stag + 1
    mut_scale = mutation_scale * (2.0 if stagnant >= stagnation_gens else 1.0)

    # Elite selection
    elites = _best_elites(df, elite_k) if not df.empty else []
    if not elites:
        print("[HUNTER] WARNING: No valid elites found. Generating random parents.")
        elites = [_random_params(param_space) for _ in range(elite_k)]

    # Determine population composition
    reseed_count = int(max(0, round(pop_size * reseed_frac))) if stagnant >= stagnation_gens else 0
    # Reserve space for existing elites (they are cloned to the next generation)
    elite_clone_count = len(elites)
    breed_count  = max(0, pop_size - elite_clone_count - reseed_count)

    # Breed Children
    children: List[Dict[str, float]] = []
    for _ in range(breed_count):
        # Select parents, must ensure minimum of 1 elite is selected (handled by logic above)
        if elite_clone_count >= 2:
            pa, pb = random.sample(elites, k=2)
        else:
            pa = pb = elites[0]

        child = {}
        for k in param_space.keys():
            # Crossover: Average with random weighting
            w = random.random()
            child[k] = w * pa[k] + (1.0 - w) * pb[k]

            # Mutate
            child[k] = _mutate_param(child[k], param_space[k], mut_scale)
        children.append(child)

    # Reseed (Immigrants)
    reseeds = [_random_params(param_space) for _ in range(reseed_count)]

    # Next generation composition: Cloned Elites + Children + Reseeds
    params_list = elites + children + reseeds

    # Final cleanup (padding/truncating)
    while len(params_list) < pop_size:
        params_list.append(_random_params(param_space))
    if len(params_list) > pop_size:
        params_list = params_list[:pop_size]

    # Create final payload structure
    population = [{"id": f"gen{next_gen:04d}_{i:03d}",
                   "params": {k: float(v) for k, v in p.items()}}
                  for i, p in enumerate(params_list)]

    _update_state(hunt_id, next_gen, best_sse, stagnant)

    return {
        "worker": TARGET_WORKER,
        "hunt_id": hunt_id,
        "generation": next_gen,
        "param_space": param_space,
        "population": population,
        "notes": (
            f"ASTE Hunter v7.0 | elites={elite_clone_count} breed={breed_count} reseed={reseed_count} "
            f"| stagnant={stagnant} (threshold={stagnation_gens}) "
            f"| best_sse={best_sse:.10f}"
        ),
    }

# ---- CLI --------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python aste_hunter.py <HUNT_ID> <TODO_FILE>")
        sys.exit(2)

    hunt_id, todo_file = sys.argv[1], sys.argv[2]

    # Resolve environment overrides or use defaults
    pop_size       = int(os.getenv("ASTE_POP_SIZE", str(DEFAULT_POP_SIZE)))
    elite_k        = int(os.getenv("ASTE_ELITE_K", str(DEFAULT_ELITE_K)))
    mutation_scale = float(os.getenv("ASTE_MUT_SCALE", str(DEFAULT_MUTATION_SCALE)))
    reseed_frac    = float(os.getenv("ASTE_RESEED_FRAC", str(DEFAULT_RESEED_FRAC)))
    stag_gens      = int(os.getenv("ASTE_STAG_GENS", str(DEFAULT_STAG_GENS)))

    if not os.path.exists(TARGET_WORKER):
        print(f"[HUNTER] WARNING: '{TARGET_WORKER}' not found in CWD ({os.getcwd()}). Ensure worker_v7.py is saved.")

    print(f"[HUNTER] Starting Evolution for Gen {int(_resolve_generation(hunt_id, pd.DataFrame()))}...")

    payload = evolve_next_population(
        hunt_id=hunt_id,
        todo_file=todo_file,
        pop_size=pop_size,
        elite_k=elite_k,
        mutation_scale=mutation_scale,
        reseed_frac=reseed_frac,
        stagnation_gens=stag_gens,
    )

    _dump_json(todo_file, payload)

    print(f"[HUNTER] Wrote next generation TODO \u2192 {todo_file}")
    print(f"[HUNTER] worker: {payload['worker']} | generation: {payload['generation']} | pop: {len(payload['population'])}")

if __name__ == "__main__":
    main()
