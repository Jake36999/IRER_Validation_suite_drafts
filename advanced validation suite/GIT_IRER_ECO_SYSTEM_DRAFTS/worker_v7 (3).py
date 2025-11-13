#!/usr/bin/env python3

"""
worker_v7.py
CLASSIFICATION: Simulation Worker (ASTE V1.0)
GOAL: Executes a single, JAX-based, JIT-compiled simulation run.
      This component is architected to be called by the
      adaptive_hunt_orchestrator.py and is the core physics engine.
      It adheres to the jax.lax.scan HPC mandate.
"""

import jax
import jax.numpy as jnp
import numpy as np # For initial setup
import h5py
import json
import os
import sys
import argparse
from typing import NamedTuple, Tuple, Dict, Any
from functools import partial

# ---
# SECTION 1: JAX STATE AND PHYSICS DEFINITIONS
# ---

class FMIAState(NamedTuple):
    """
    JAX Pytree for the core FMIA state (Field Dynamics).
    This is the state evolved by the RK4 integrator.
    """
    rho: jnp.ndarray  # Resonance Density
    pi: jnp.ndarray   # Conjugate Momentum (d_rho_dt)

class SimState(NamedTuple):
    """
    The unified "carry" state for the jax.lax.scan loop.
    MANDATE: Includes k_vectors as dynamic state to ensure JIT integrity.
    """
    fmia_state: FMIAState     # The evolving physics fields
    g_munu: jnp.ndarray       # The evolving metric tensor
    Omega: jnp.ndarray        # The evolving conformal factor
    k_vectors: Tuple[jnp.ndarray, ...] # (kx, ky, kz) grids
    k_squared: jnp.ndarray    # |k|^2 grid


# ---
# SECTION 2: PHYSICS KERNELS (GAPS & PROXIES)
# ---

@jax.jit
def jnp_effective_conformal_factor(
    rho: jnp.ndarray,
    coupling_alpha: float,
    epsilon: float = 1e-9
) -> jnp.ndarray:
    """
    [ECM Model Core] Computes the Effective Conformal Factor Omega(rho).
    Model: Omega = exp[ alpha * (rho - 1) ].
    This is the computable proxy that solves the "Gravity Gap".
    """
    alpha = jnp.maximum(coupling_alpha, epsilon)
    rho_fluctuation = rho - 1.0 # Fluctuation from vacuum (rho=1)
    Omega = jnp.exp(alpha * rho_fluctuation)
    return Omega

@jax.jit
def jnp_construct_conformal_metric(
    rho: jnp.ndarray,
    coupling_alpha: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Constructs the 4x4xNxNxN spacetime metric g_munu = Omega^2 * eta_munu.
    This function closes the geometric loop using the ECM proxy model.
    """
    # 1. Calculate the Effective Conformal Factor
    Omega = jnp_effective_conformal_factor(rho, coupling_alpha)
    Omega_sq = jnp.square(Omega)

    # 2. Construct the Conformal Metric: g_munu = Omega^2 * eta_munu
    grid_shape = rho.shape
    g_munu = jnp.zeros((4, 4) + grid_shape)

    # g_00 = -Omega^2 (Time component)
    g_munu = g_munu.at[0, 0, ...].set(-Omega_sq)

    # g_ii = Omega^2 (Spatial components)
    g_munu = g_munu.at[1, 1, ...].set(Omega_sq) # g_xx
    g_munu = g_munu.at[2, 2, ...].set(Omega_sq) # g_yy
    g_munu = g_munu.at[3, 3, ...].set(Omega_sq) # g_zz

    return g_munu, Omega

@jax.jit
def jnp_metric_aware_laplacian(
    rho: jnp.ndarray,
    Omega: jnp.ndarray,
    k_squared: jnp.ndarray,
    k_vectors: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    PLACEHOLDER for the Covariant Laplace-Beltrami operator (square_g).
    Implements the 3D formula: Delta_g(rho) = O^-2 * [ (nabla^2 rho) + O^-1 * (nabla O . nabla rho) ]
    """
    kx_3d, ky_3d, kz_3d = k_vectors
    Omega_inv = 1.0 / (Omega + 1e-9)
    Omega_sq_inv = Omega_inv**2

    # --- 1. Compute flat-space Laplacian: nabla^2(rho) ---
    rho_k = jnp.fft.fftn(rho)
    laplacian_rho_k = -k_squared * rho_k
    laplacian_rho = jnp.fft.ifftn(laplacian_rho_k).real

    # --- 2. Compute flat-space gradients: nabla(rho) and nabla(Omega) ---
    grad_rho_x = jnp.fft.ifftn(1j * kx_3d * rho_k).real
    grad_rho_y = jnp.fft.ifftn(1j * ky_3d * rho_k).real
    grad_rho_z = jnp.fft.ifftn(1j * kz_3d * rho_k).real

    Omega_k = jnp.fft.fftn(Omega)
    grad_Omega_x = jnp.fft.ifftn(1j * kx_3d * Omega_k).real
    grad_Omega_y = jnp.fft.ifftn(1j * ky_3d * Omega_k).real
    grad_Omega_z = jnp.fft.ifftn(1j * kz_3d * Omega_k).real

    # --- 3. Compute dot product: (nabla O . nabla rho) ---
    nabla_dot_product = (grad_Omega_x * grad_rho_x +
                         grad_Omega_y * grad_rho_y +
                         grad_Omega_z * grad_rho_z)

    # --- 4. Assemble the final formula ---
    term1 = laplacian_rho
    term2 = Omega_inv * nabla_dot_product
    Delta_g_rho = Omega_sq_inv * (term1 + term2)

    return Delta_g_rho

@jax.jit
def jnp_get_derivatives(
    state: FMIAState,
    t: float,
    k_squared: jnp.ndarray,
    k_vectors: Tuple[jnp.ndarray, ...],
    g_munu: jnp.ndarray,
    Omega: jnp.ndarray,
    constants: Dict[str, float]
) -> FMIAState:
    """
    Calculates the time derivatives (d_rho_dt, d_pi_dt) for the
    Metric-Aware FMIA EOM (Equation of Motion).
    """
    rho, pi = state.rho, state.pi

    # --- 1. Physics Calculations ---

    # CRITICAL: Replace flat Laplacian with Metric-Aware Laplacian
    laplacian_g_rho = jnp_metric_aware_laplacian(
        rho, Omega, k_squared, k_vectors
    )

    # Placeholder for Potential Term (V'(rho))
    V_prime = rho - rho**3 # Example: phi^4 potential derivative

    # GAP: Non-Local "Splash" Term (G_non_local)
    # This is a known physics gap (Part V.B) and is zeroed out.
    G_non_local_term = jnp.zeros_like(pi)

    # --- 2. Calculate Time Derivatives ---
    d_rho_dt = pi

    d_pi_dt = (
        constants.get('D', 1.0) * laplacian_g_rho +  # <-- METRIC-AWARE
        V_prime +
        G_non_local_term +                         # <-- GAP
        -constants.get('eta', 0.1) * pi            # Damping
    )

    return FMIAState(rho=d_rho_dt, pi=d_pi_dt)

@partial(jax.jit, static_argnames=['derivs_func'])
def rk4_step(
    derivs_func: callable,
    state: FMIAState,
    t: float,
    dt: float,
    k_squared: jnp.ndarray,
    k_vectors: Tuple[jnp.ndarray, ...],
    g_munu: jnp.ndarray,
    Omega: jnp.ndarray,
    constants: Dict[str, float]
) -> FMIAState:
    """
    Performs a single Runge-Kutta 4th Order (RK4) step.
    """

    k1 = derivs_func(state, t, k_squared, k_vectors, g_munu, Omega, constants)

    state_k2 = jax.tree_util.tree_map(lambda y, dy: y + 0.5 * dt * dy, state, k1)
    k2 = derivs_func(state_k2, t + 0.5 * dt, k_squared, k_vectors, g_munu, Omega, constants)

    state_k3 = jax.tree_util.tree_map(lambda y, dy: y + 0.5 * dt * dy, state, k2)
    k3 = derivs_func(state_k3, t + 0.5 * dt, k_squared, k_vectors, g_munu, Omega, constants)

    state_k4 = jax.tree_util.tree_map(lambda y, dy: y + dt * dy, state, k3)
    k4 = derivs_func(state_k4, t + dt, k_squared, k_vectors, g_munu, Omega, constants)

    next_state = jax.tree_util.tree_map(
        lambda y, dy1, dy2, dy3, dy4: y + (dt / 6.0) * (dy1 + 2.0*dy2 + 2.0*dy3 + dy4),
        state, k1, k2, k3, k4
    )

    return next_state

# ---
# SECTION 3: JAX.LAX.SCAN ORCHESTRATOR (HPC MANDATE)
# ---

@partial(jax.jit, static_argnames=['dt', 'D_const', 'eta_const', 'kappa_const', 'sigma_k_const', 'alpha_coupling'])
def simulation_step(
    carry_state: SimState,
    t: float,
    dt: float,
    D_const: float,
    eta_const: float,
    kappa_const: float,
    sigma_k_const: float,
    alpha_coupling: float
) -> Tuple[SimState, jnp.ndarray]:
    """
    Executes one full, JIT-compiled step of the co-evolutionary loop.
    This is the function body for jax.lax.scan.
    """
    # 1. Unpack all state variables from the carry
    current_fmia_state = carry_state.fmia_state
    current_g_munu = carry_state.g_munu
    current_Omega = carry_state.Omega
    k_vectors = carry_state.k_vectors
    k_squared = carry_state.k_squared

    # Reconstruct constants dictionary for functions expecting it
    constants_dict = {
        'D': D_const,
        'eta': eta_const,
        'kappa': kappa_const,
        'sigma_k': sigma_k_const
    }

    # --- STAGE 1: FIELD EVOLUTION (rho_n -> rho_n+1) ---
    # Field evolves on the *current* geometry
    next_fmia_state = rk4_step(
        jnp_get_derivatives,
        current_fmia_state,
        t, dt,
        k_squared, k_vectors,
        current_g_munu, current_Omega,
        constants_dict
    )

    # --- STAGE 2 & 3: SOURCE & GEOMETRY (rho_n+1 -> g_munu_n+1) ---
    # New geometry is calculated from the *new* field state
    next_g_munu, next_Omega = jnp_construct_conformal_metric(
        next_fmia_state.rho,
        alpha_coupling
    )

    # 4. Assemble NEW Carry State (Closing the Loop)
    new_carry = SimState(
        fmia_state=next_fmia_state,
        g_munu=next_g_munu,
        Omega=next_Omega,
        k_vectors=k_vectors,  # Constants are passed through
        k_squared=k_squared
    )

    # Return (new_carry, data_to_log)
    # We log the rho field for this step
    return new_carry, new_carry.fmia_state.rho

def run_simulation(N_grid: int, L_domain: float, T_steps: int, ALPHA: float, KAPPA: float, SIGMA_K: float, D: float, ETA: float) -> Tuple[SimState, jnp.ndarray]:
    """
    Main JAX driver function. Sets up and runs the jax.lax.scan loop.
    """

    # 1. Precompute JAX k-vectors (Non-hashable)
    k_1D = 2 * jnp.pi * jnp.fft.fftfreq(N_grid, d=L_domain/N_grid)
    kx_3d, ky_3d, kz_3d = jnp.meshgrid(k_1D, k_1D, k_1D, indexing='ij')
    k_vectors_tuple = (kx_3d, ky_3d, kz_3d)
    k_squared_array = kx_3d**2 + ky_3d**2 + kz_3d**2

    # 2. Initialize the SimState (Bundling k-vectors into the 'carry')
    # Use a small random noise to break symmetry
    key = jax.random.PRNGKey(42)
    initial_rho = jnp.ones((N_grid, N_grid, N_grid)) + jax.random.uniform(key, (N_grid, N_grid, N_grid)) * 0.01
    initial_pi = jnp.zeros_like(initial_rho)

    initial_fmia_state = FMIAState(rho=initial_rho, pi=initial_pi)

    initial_g_munu, initial_Omega = jnp_construct_conformal_metric(
        initial_rho, ALPHA
    )

    initial_carry = SimState(
        fmia_state=initial_fmia_state,
        g_munu=initial_g_munu,
        Omega=initial_Omega,
        k_vectors=k_vectors_tuple,
        k_squared=k_squared_array
    )

    # 3. Define the main scan body function
    scan_fn = partial(
        simulation_step,
        dt=0.01, # DT is fixed for now
        D_const=D,
        eta_const=ETA,
        kappa_const=KAPPA,
        sigma_k_const=SIGMA_K,
        alpha_coupling=ALPHA
    )

    # 4. Execute the fully JIT-compiled loop
    timesteps = jnp.arange(T_steps) * 0.01 # DT is fixed for now

    final_carry, history = jax.lax.scan(
        scan_fn,
        initial_carry,
        timesteps,
        length=T_steps
    )

    return final_carry, history # history is rho_history

# ---
# SECTION 4: MAIN ORCHESTRATOR (DRIVER HOOK)
# ---

def main():
    """
    Main entry point for the Worker.
    Called by adaptive_hunt_orchestrator.py.
    """
    parser = argparse.ArgumentParser(
        description="ASTE Simulation Worker (worker_v7.py)"
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to the parameters.json file for this job."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output rho_history.h5 data artifact."
    )
    args = parser.parse_args()

    print(f"[Worker] Job started. Loading config: {args.params}")

    # --- 1. Load Parameters ---
    try:
        with open(args.params, 'r') as f:
            params = json.load(f)

        # Extract simulation and physics parameters
        # Using .get() for safety, with defaults
        sim_params = params.get("simulation", {})
        phys_params = params.get("physics", {})

        N_GRID = sim_params.get("N_grid", 32)
        L_DOMAIN = sim_params.get("L_domain", 10.0)
        T_STEPS = sim_params.get("T_steps", 100)
        # DT is fixed in run_simulation now, not passed from params
        # DT = sim_params.get("dt", 0.01)

        ALPHA = phys_params.get("alpha", 0.5)
        # Parameters from Hunter (Part III.B)
        KAPPA = params.get("param_kappa", 0.007) # Default to a known good value
        SIGMA_K = params.get("param_sigma_k", 0.55) # Default

        D = phys_params.get("D", 1.0)
        ETA = phys_params.get("eta", 0.1)

    except Exception as e:
        print(f"[Worker Error] Failed to load or parse params file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[Worker] Parameters loaded: N={N_GRID}, Steps={T_STEPS}, K_NonLocal={KAPPA}")

    # --- 2. Run Simulation ---
    print("[Worker] JAX: Compiling and running simulation loop...")
    try:
        final_carry, rho_history = run_simulation(
            N_grid=N_GRID,
            L_domain=L_DOMAIN,
            T_steps=T_STEPS,
            # DT is fixed in run_simulation now
            ALPHA=ALPHA,
            KAPPA=KAPPA,
            SIGMA_K=SIGMA_K,
            D=D,
            ETA=ETA
        )
        print("[Worker] Simulation complete.")

    except Exception as e:
        print(f"[Worker Error] JAX simulation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Save Artifact ---
    print(f"[Worker] Saving artifact to: {args.output}")
    try:
        # Convert JAX array back to NumPy for HDF5 I/O
        rho_history_np = np.asarray(rho_history)

        with h5py.File(args.output, 'w') as f:
            f.create_dataset(
                'rho_history',
                data=rho_history_np,
                chunks=(1, N_GRID, N_GRID, N_GRID), # Chunked by timestep
                compression="gzip"
            )
            # Save parameters as metadata attributes
            f.attrs['config_hash'] = params.get('config_hash', 'unknown')
            f.attrs['param_kappa'] = KAPPA
            f.attrs['param_sigma_k'] = SIGMA_K

        print("[Worker] SUCCESS: Artifact saved.")

    except Exception as e:
        print(f"[Worker Error] Failed to save HDF5 artifact: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
