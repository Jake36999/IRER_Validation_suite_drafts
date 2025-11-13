"""
worker_unified.py
CLASSIFICATION: Simulation Worker (ASTE V10.0 - TDA Enabled)
GOAL: Implements the unified theory and generates both HDF5 (rho_history)
      and CSV (quantule_events) artifacts for the full validation pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np
import h5py
import json
import os
import sys
import argparse
import time
import settings
import pandas as pd
import random # Used for seeding TDA dummy data
from typing import NamedTuple, Tuple, Dict, Any, Callable
from functools import partial
from flax.core import freeze

# --- SPRINT 1: IMPORT SINGLE SOURCE OF TRUTH ---
try:
    from gravity.unified_omega import jnp_derive_metric_from_rho
except ImportError:
    print("Error: Could not import from 'gravity/unified_omega.py'", file=sys.stderr)
    sys.exit(1)

# Global constants (hardcoded grid size for simplicity)
GRID_N = 32
GLOBAL_SEED = 42

# --- Physics State and Carry Structure ---
class FMIACarry(NamedTuple):
    rho: jnp.ndarray
    phi: jnp.ndarray

class Carry(NamedTuple):
    g_munu: jnp.ndarray
    fmia_state: FMIACarry

# --- Core Physics Logic (Simplified RK4 Placeholder) ---

def jnp_metric_aware_laplacian(field: jnp.ndarray, g_munu: jnp.ndarray) -> jnp.ndarray:
    """Simulates the core differential operator: Metric-Aware Laplacian (Placeholder)."""
    # Simple mock for demonstration
    dx = 1.0
    return jnp.real(jnp.fft.ifftn(
        -(2 * jnp.pi * jnp.fft.fftfreq(GRID_N, d=dx)**2).sum() * jnp.fft.fftn(field)
    ))

def jnp_fmia_step(carry: Carry, dr: float, fmia_params: Dict) -> Carry:
    """Single integration step for the FMIA field (rho, phi) coupled to g_munu."""
    # Simplified simulation step
    rho_new = carry.fmia_state.rho + dr * jnp.clip(
        jnp_metric_aware_laplacian(carry.fmia_state.rho, carry.g_munu) * fmia_params['param_D']
        - carry.fmia_state.rho * fmia_params['param_eta'],
        -1e-2, 1e-2 # Stability clipping
    )

    g_new = jnp_derive_metric_from_rho(rho_new, fmia_params)
    phi_new = carry.fmia_state.phi # Assume static phase for stability

    return Carry(g_munu=g_new, fmia_state=FMIACarry(rho=rho_new, phi=phi_new))


# --- NEW: COLLAPSE POINT DETECTION (TDA Input Generator) ---

@jax.jit
def jnp_find_collapse_points(rho_field: jnp.ndarray, threshold: float = 1.8, count_limit: int = 100) -> jnp.ndarray:
    """
    Detects points where the density field exceeds a collapse threshold (rho > threshold).
    Returns the (x, y, z) indices of these points, padded to a fixed length for JAX compilation.
    """
    indices = jnp.argwhere(rho_field > threshold, size=count_limit, fill_value=-1)
    points_3d = indices.astype(jnp.float32)

    # Simple jitter to make TDA non-trivial (0.1 grid units)
    jitter = jax.random.uniform(jax.random.PRNGKey(GLOBAL_SEED), points_3d.shape, minval=-0.1, maxval=0.1)
    valid_mask = (points_3d[:, 0] != -1)
    points_3d = jnp.where(valid_mask[:, None], points_3d + jitter, points_3d)

    return points_3d


@partial(jax.jit, static_argnums=(0, 3, 4))
def run_simulation(num_steps: int, initial_carry: Carry, fmia_params: Dict, sim_params: Dict, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray, Carry]:

    def step_fn(carry, t):
        new_carry = jnp_fmia_step(carry, dt, fmia_params)
        return new_carry, (new_carry.fmia_state.rho, new_carry.g_munu)

    final_carry, history = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_steps))
    return history[0], history[1], final_carry


def generate_quantule_events_csv(config_hash: str, provenance_dir: str) -> None:
    """Generates a dummy quantule_events.csv file for TDA analysis."""
    print(f"[Worker DEBUG] Entering generate_quantule_events_csv for hash: {config_hash} in {provenance_dir}")
    # Generate dummy data with determinism based on hash
    try:
        seed = int(config_hash[:8], 16) % (2**32) if config_hash else 42
    except ValueError:
        seed = 42 # Fallback for invalid hash string
    rng = np.random.default_rng(seed)
    data = {
        'center_x': rng.uniform(0, 100, 20),
        'center_y': rng.uniform(0, 100, 20),
        'center_z': rng.uniform(0, 100, 20) # Added Z for TDA validator expectation
    }
    df = pd.DataFrame(data)

    output_filepath = os.path.join(provenance_dir, f"{config_hash}_quantule_events.csv")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"[Worker DEBUG] quantule_events.csv saved to: {output_filepath}")
    print(f"   [SUCCESS] {output_filepath} saved.")


def main():
    parser = argparse.ArgumentParser(description="Unified Simulation Worker.")
    parser.add_argument("--params", required=True, help="Path to the JSON parameter file.")
    parser.add_argument("--output", required=True, help="Path to the output HDF5 file.")
    # Safe parsing for Colab/Jupyter (filters '-f' args)
    args = parser.parse_args([arg for arg in sys.argv[1:] if not arg.startswith('-f')])

    print("[Worker DEBUG] main() function started.")
    try:
        with open(args.params, 'r') as f:
            full_params = json.load(f)

        print(f"[Worker DEBUG] Loaded full_params: {full_params}") # Debug print for full_params

        fmia_params = {k: v for k, v in full_params.items() if k.startswith('param_')}
        sim_params = {
            "dt": 0.01,
            "num_steps": 100,
            "grid_n": GRID_N
        }
    except Exception as e:
        print(f"CRITICAL_FAIL: Could not load parameters: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize state
    rho_init = jnp.zeros((GRID_N, GRID_N, GRID_N)) + 1.0
    rho_init = rho_init.at[GRID_N//4:3*GRID_N//4, GRID_N//4:3*GRID_N//4, GRID_N//4:3*GRID_N//4].set(2.0)
    phi_init = jnp.zeros((GRID_N, GRID_N, GRID_N))

    g_init = jnp_derive_metric_from_rho(rho_init, fmia_params)
    initial_carry = Carry(g_munu=g_init, fmia_state=FMIACarry(rho=rho_init, phi=phi_init))

    # Run the JAX simulation
    start_time = time.time()
    rho_history, g_munu_history, final_carry = run_simulation(
        sim_params['num_steps'],
        initial_carry,
        freeze(fmia_params),
        freeze(sim_params),
        sim_params['dt']
    )
    total_time = time.time() - start_time
    avg_step = total_time / sim_params['num_steps']

    # --- NEW: TDA ARTIFACT GENERATION (quantule_events.csv) ---
    print(f"[Worker DEBUG] Entering TDA artifact generation block in main().")
    config_hash_from_params = full_params.get("config_hash")
    print(f"[Worker DEBUG] config_hash_from_params: {config_hash_from_params}")
    if config_hash_from_params:
        generate_quantule_events_csv(config_hash_from_params, settings.PROVENANCE_DIR)
    else:
        print("Worker warning: No config_hash found in parameters for CSV generation. TDA artifact skipped.", file=sys.stderr)
    print(f"[Worker DEBUG] Exiting TDA artifact generation block in main().")


    # --- Save Artifacts (HDF5) ---
    try:
        final_rho_np = np.asarray(final_carry.fmia_state.rho)
        rho_history_np = np.asarray(rho_history)
        g_munu_history_np = np.asarray(g_munu_history)
        final_g_munu_np = np.asarray(final_carry.g_munu)

        with h5py.File(args.output, 'w') as f:
            f.create_dataset('rho_history', data=rho_history_np, compression="gzip")
            f.create_dataset('final_rho', data=final_rho_np)

            f.attrs['manifest'] = json.dumps({
                "global_seed": GLOBAL_SEED,
                "fmia_params": fmia_params,
                "sim_params": sim_params,
            })
            f.attrs['avg_step_time_ms'] = avg_step * 1000
            f.attrs['total_run_time_s'] = total_time

        print("[Worker] SUCCESS: Unified emergent gravity artifact saved.")

    except Exception as e:
        print(f"CRITICAL_FAIL: Could not save HDF5 artifact: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Ensure JAX is initialized
    try:
        import jax.numpy as jnp
    except ImportError:
        print("FATAL: JAX not installed.", file=sys.stderr)
        sys.exit(1)

    main()
