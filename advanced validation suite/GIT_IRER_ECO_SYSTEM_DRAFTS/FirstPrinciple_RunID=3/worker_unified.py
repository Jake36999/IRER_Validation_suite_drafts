#!/usr/bin/env python3

"""
worker_unified.py
CLASSIFICATION: Simulation Worker (ASTE V3.0 - Unified / SPRINT 1 PATCHED)
GOAL: Implements the unified theory with determinism and provenance logging.
      Imports the single source of truth for gravity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import h5py
import json
import os
import sys
import argparse
from typing import NamedTuple, Tuple, Dict, Any, Callable
from functools import partial
from flax.core import freeze
import time

# --- SPRINT 1: IMPORT SINGLE SOURCE OF TRUTH ---
try:
    from gravity.unified_omega import jnp_derive_metric_from_rho
except ImportError:
    print("Error: Could not import from 'gravity/unified_omega.py'", file=sys.stderr)
    print("Please run the 'gravity/unified_omega.py' cell first.", file=sys.stderr)
    sys.exit(1)

# --- (Physics functions D, D2, jnp_metric_aware_laplacian...) ---
# (These are unchanged, assuming 3D grid and k-vectors)
@jax.jit
def D(field: jnp.ndarray, dr: float) -> jnp.ndarray:
    # This 1D function is not used by the 3D laplacian, but kept
    # for potential 1D test cases.
    N = len(field); k = 2 * jnp.pi * jnp.fft.fftfreq(N, d=dr)
    field_hat = jnp.fft.fft(field); d_field_hat = 1j * k * field_hat
    return jnp.real(jnp.fft.ifft(d_field_hat))

@jax.jit
def D2(field: jnp.ndarray, dr: float) -> jnp.ndarray:
    return D(D(field, dr), dr)

@jax.jit
def jnp_metric_aware_laplacian(
    rho: jnp.ndarray, Omega: jnp.ndarray, k_squared: jnp.ndarray,
    k_vectors: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    kx_3d, ky_3d, kz_3d = k_vectors; Omega_inv = 1.0 / (Omega + 1e-9)
    Omega_sq_inv = Omega_inv**2; rho_k = jnp.fft.fftn(rho)
    laplacian_rho = jnp.fft.ifftn(-k_squared * rho_k).real
    grad_rho_x = jnp.fft.ifftn(1j * kx_3d * rho_k).real
    grad_rho_y = jnp.fft.ifftn(1j * ky_3d * rho_k).real
    grad_rho_z = jnp.fft.ifftn(1j * kz_3d * rho_k).real
    Omega_k = jnp.fft.fftn(Omega)
    grad_Omega_x = jnp.fft.ifftn(1j * kx_3d * Omega_k).real
    grad_Omega_y = jnp.fft.ifftn(1j * ky_3d * Omega_k).real
    grad_Omega_z = jnp.fft.ifftn(1j * kz_3d * Omega_k).real
    nabla_dot_product = (grad_Omega_x * grad_rho_x +
                         grad_Omega_y * grad_rho_y +
                         grad_Omega_z * grad_rho_z)
    Delta_g_rho = Omega_sq_inv * (laplacian_rho + Omega_inv * nabla_dot_product)
    return Delta_g_rho

class FMIAState(NamedTuple):
    rho: jnp.ndarray; pi: jnp.ndarray

@jax.jit
def jnp_get_derivatives(
    state: FMIAState, t: float, k_squared: jnp.ndarray,
    k_vectors: Tuple[jnp.ndarray, ...], g_munu: jnp.ndarray,
    constants: Dict[str, float]
) -> FMIAState:
    rho, pi = state.rho, state.pi
    Omega = jnp.sqrt(jnp.maximum(g_munu[1, 1, ...], 1e-12)) # Extract Omega, guard sqrt(0)
    laplacian_g_rho = jnp_metric_aware_laplacian(
        rho, Omega, k_squared, k_vectors
    )
    V_prime = rho - rho**3 # Potential
    G_non_local_term = jnp.zeros_like(pi) # Non-local term (GAP)
    d_rho_dt = pi

    # --- PATCH APPLIED (Fix 2) ---
    # Correctly get parameters using param_* keys
    d_pi_dt = ( constants.get('param_D', 1.0) * laplacian_g_rho + V_prime +
                G_non_local_term - constants.get('param_eta', 0.1) * pi )

    return FMIAState(rho=d_rho_dt, pi=d_pi_dt)

@partial(jax.jit, static_argnames=['derivs_func'])
def rk4_step(
    derivs_func: Callable, state: FMIAState, t: float, dt: float,
    k_squared: jnp.ndarray, k_vectors: Tuple[jnp.ndarray, ...],
    g_munu: jnp.ndarray, constants: Dict[str, float]
) -> FMIAState:
    k1 = derivs_func(state, t, k_squared, k_vectors, g_munu, constants)
    state_k2 = jax.tree_util.tree_map(lambda y, dy: y + 0.5 * dt * dy, state, k1)
    k2 = derivs_func(state_k2, t + 0.5 * dt, k_squared, k_vectors, g_munu, constants)
    state_k3 = jax.tree_util.tree_map(lambda y, dy: y + 0.5 * dt * dy, state, k2)
    k3 = derivs_func(state_k3, t + 0.5 * dt, k_squared, k_vectors, g_munu, constants)
    state_k4 = jax.tree_util.tree_map(lambda y, dy: y + dt * dy, state, k3)
    k4 = derivs_func(state_k4, t + dt, k_squared, k_vectors, g_munu, constants)
    next_state = jax.tree_util.tree_map(
        lambda y, dy1, dy2, dy3, dy4: y + (dt / 6.0) * (dy1 + 2.0*dy2 + 2.0*dy3 + dy4),
        state, k1, k2, k3, k4 )
    return next_state

class SimState(NamedTuple):
    fmia_state: FMIAState
    g_munu: jnp.ndarray
    k_vectors: Tuple[jnp.ndarray, ...]
    k_squared: jnp.ndarray

@partial(jax.jit, static_argnames=['fmia_params'])
def jnp_unified_step(
    carry_state: SimState, t: float, dt: float, fmia_params: Dict
) -> Tuple[SimState, Tuple[jnp.ndarray, jnp.ndarray]]:

    current_fmia_state = carry_state.fmia_state
    current_g_munu = carry_state.g_munu
    k_vectors = carry_state.k_vectors
    k_squared = carry_state.k_squared

    next_fmia_state = rk4_step(
        jnp_get_derivatives, current_fmia_state, t, dt,
        k_squared, k_vectors, current_g_munu, fmia_params
    )
    new_rho, new_pi = next_fmia_state

    next_g_munu = jnp_derive_metric_from_rho(new_rho, fmia_params)

    new_carry = SimState(
        fmia_state=next_fmia_state,
        g_munu=next_g_munu,
        k_vectors=k_vectors, k_squared=k_squared
    )

    # --- PATCH APPLIED (Polish / Clarity) ---
    rho_out = new_carry.fmia_state.rho
    g_out   = new_carry.g_munu

    # --- PATCH APPLIED (Fix 1 - Typo) ---
    return new_carry, (rho_out, g_out)

def run_simulation(
    N_grid: int, L_domain: float, T_steps: int, DT: float,
    fmia_params: Dict[str, Any], global_seed: int
) -> Tuple[SimState, Any, float, float]:

    key = jax.random.PRNGKey(global_seed)

    k_1D = 2 * jnp.pi * jnp.fft.fftfreq(N_grid, d=L_domain/N_grid)
    kx_3d, ky_3d, kz_3d = jnp.meshgrid(k_1D, k_1D, k_1D, indexing='ij')
    k_vectors_tuple = (kx_3d, ky_3d, kz_3d)
    k_squared_array = kx_3d**2 + ky_3d**2 + kz_3d**2

    initial_rho = jnp.ones((N_grid, N_grid, N_grid)) + jax.random.uniform(key, (N_grid, N_grid, N_grid)) * 0.01
    initial_pi = jnp.zeros_like(initial_rho)
    initial_fmia_state = FMIAState(rho=initial_rho, pi=initial_pi)
    initial_g_munu = jnp_derive_metric_from_rho(initial_rho, fmia_params)

    initial_carry = SimState(
        fmia_state=initial_fmia_state,
        g_munu=initial_g_munu,
        k_vectors=k_vectors_tuple,
        k_squared=k_squared_array
    )

    frozen_fmia_params = freeze(fmia_params)

    scan_fn = partial(
        jnp_unified_step,
        dt=DT,
        fmia_params=frozen_fmia_params
    )

    print("[Worker] JIT: Warming up simulation step...")
    warmup_carry, _ = scan_fn(initial_carry, 0.0)
    warmup_carry.fmia_state.rho.block_until_ready()
    print("[Worker] JIT: Warm-up complete.")

    timesteps = jnp.arange(T_steps)

    print(f"[Worker] JAX: Running unified scan for {T_steps} steps...")
    start_time = time.time()

    final_carry, history = jax.lax.scan(
        scan_fn,
        warmup_carry,
        timesteps
    )
    final_carry.fmia_state.rho.block_until_ready()
    end_time = time.time()

    total_time = end_time - start_time
    avg_step_time = total_time / T_steps
    print(f"[Worker] JAX: Scan complete in {total_time:.4f}s")
    print(f"[Worker] Performance: Avg step time: {avg_step_time*1000:.4f} ms")

    return final_carry, history, avg_step_time, total_time

def main():
    parser = argparse.ArgumentParser(description="ASTE Unified Worker (Sprint 1 Patched)")
    parser.add_argument("--params", type=str, required=True, help="Path to parameters.json")
    parser.add_argument("--output", type=str, required=True, help="Path to output HDF5 artifact.")
    args = parser.parse_args()

    print(f"[Worker] Job started. Loading config: {args.params}")

    try:
        with open(args.params, 'r') as f:
            params = json.load(f)

        sim_params = params.get("simulation", {})
        N_GRID = sim_params.get("N_grid", 16)
        L_DOMAIN = sim_params.get("L_domain", 10.0)
        T_STEPS = sim_params.get("T_steps", 50)
        DT = sim_params.get("dt", 0.01)
        GLOBAL_SEED = params.get("global_seed", 42)

        # Parameters are now read from the root of the params dict
        fmia_params = {
            "param_D": params.get("param_D", 1.0),
            "param_eta": params.get("param_eta", 0.1),
            "param_rho_vac": params.get("param_rho_vac", 1.0),
            "param_a_coupling": params.get("param_a_coupling", 1.0),
        }

    except Exception as e:
        print(f"[Worker Error] Failed to load params file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[Worker] Parameters loaded: N={N_GRID}, Steps={T_STEPS}, Seed={GLOBAL_SEED}")

    print("[Worker] JAX: Initializing and running UNIFIED co-evolution loop...")
    try:
        final_carry, history, avg_step, total_time = run_simulation(
            N_grid=N_GRID, L_domain=L_DOMAIN, T_steps=T_STEPS, DT=DT,
            fmia_params=fmia_params, global_seed=GLOBAL_SEED
        )
        print("[Worker] Simulation complete.")

    except Exception as e:
        print(f"[Worker Error] JAX simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[Worker] Saving artifact to: {args.output}")
    try:
        # --- PATCH APPLIED (Fix 3 - History Unpacking) ---
        rho_hist, g_hist = history
        rho_history_np = np.asarray(rho_hist)
        g_munu_history_np = np.asarray(g_hist)

        final_rho_np = np.asarray(final_carry.fmia_state.rho)
        final_g_munu_np = np.asarray(final_carry.g_munu)

        with h5py.File(args.output, 'w') as f:
            f.create_dataset('rho_history', data=rho_history_np, compression="gzip")
            f.create_dataset('g_munu_history', data=g_munu_history_np, compression="gzip")
            f.create_dataset('final_rho', data=final_rho_np)
            f.create_dataset('final_g_munu', data=final_g_munu_np)

            # --- PATCH APPLIED (Polish - Manifest) ---
            # Save the *entire* run manifest as an attribute
            f.attrs['manifest'] = json.dumps({
                "global_seed": GLOBAL_SEED,
                "git_sha": os.environ.get("GIT_COMMIT", "unknown"),
                "fmia_params": fmia_params,
                "sim_params": sim_params,
            })

            # Save performance metrics
            f.attrs['avg_step_time_ms'] = avg_step * 1000
            f.attrs['total_run_time_s'] = total_time

        print("[Worker] SUCCESS: Unified emergent gravity artifact saved.")

    except Exception as e:
        print(f"CRITICAL_FAIL: Could not save HDF5 artifact: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        from flax.core import freeze
    except ImportError:
        print("Error: This script requires 'flax'. Please install: pip install flax", file=sys.stderr)
        sys.exit(1)

    # Create gravity directory
    if not os.path.exists("gravity"):
        os.makedirs("gravity")

    main()
