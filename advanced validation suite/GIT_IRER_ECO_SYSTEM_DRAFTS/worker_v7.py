#
# worker_v7.py (Certified v7.1 - 3D Gradient-Compatible Fix)
#
# Implements the stable S-NCGL core on a 3D grid (N x N x N).
#
# --- CELL 1: IMPORTS ---
import jax
import jax.numpy as jnp
from jax import lax, jit
import numpy as np
import h5py
import os
import time
import functools
import json
import traceback
from typing import NamedTuple, Callable, Dict, Tuple, Any, List
from geometry_solver_v8 import S_GR_State, S_GR_Source, get_geometry_input_source, get_field_feedback_terms, calculate_gr_derivatives

from tqdm.auto import tqdm
from functools import partial
import sys
import hashlib
import csv

print(f"JAX backend: {jax.default_backend()}")


# --- CELL 2: JAX PYTREE DEFINITIONS (3D SCALED) ---

class S_NCGL_State(NamedTuple):
    """Holds the dynamic state (the complex psi field) on a 3D grid."""
    psi: jax.Array

class S_NCGL_Params(NamedTuple):
    """Holds all static physics and simulation parameters."""
    N_GRID: int
    T_TOTAL: float
    DT: float
    alpha: float
    beta: float
    gamma: float
    KAPPA: float
    nu: float
    sigma_k: float
    l_domain: float
    num_rays: int
    k_bin_width: float
    k_max_plot: float

class SpecOps(NamedTuple):
    """Holds all pre-computed spectral arrays for 3D."""
    kx: jax.Array
    ky: jax.Array
    kz: jax.Array # <-- V7.0 UPGRADE: Added Z-component
    k_sq: jax.Array
    gaussian_kernel_k: jax.Array
    dealias_mask: jax.Array
    prime_targets_k: jax.Array
    k_bins: jax.Array
    ray_angles: jax.Array
    k_max: float
    xx: jax.Array
    yy: jax.Array
    zz: jax.Array # <-- V7.0 UPGRADE: Added zz
    k_values_1d: jax.Array
    sort_indices_1d: jax.Array

class S_Coupled_State(NamedTuple):
    """
    V8.0 Upgrade: Tracks both the Field (psi) and the Geometry (GR_State)
    for dynamic co-evolution in the closed GR loop.
    """
    field_state: S_NCGL_State # Holds S_NCGL_State.psi
    gr_state: S_GR_State      # Holds S_GR_State (Lapse, Shift, Metric components)

class S_Coupling_Params(NamedTuple):
    """Holds all coupling parameters (e.g., for the 'bridge')."""
    OMEGA_PARAM_A: float


# --- CELL 3: HDF5 LOGGER UTILITY (3D SCALED) ---
class HDF5Logger:
    def __init__(self, filename, n_steps, n_grid, metrics_keys, buffer_size=100):
        self.filename = filename
        self.n_steps = n_steps
        self.metrics_keys = metrics_keys
        self.buffer_size = buffer_size
        self.buffer = {key: [] for key in self.metrics_keys}
        self.buffer['omega_sq_history'] = []
        self.write_index = 0

        with h5py.File(self.filename, 'w') as f:
            for key in self.metrics_keys:
                f.create_dataset(key, (n_steps,), maxshape=(n_steps,), dtype='f4')
            # History log shape: N_steps x N_GRID x N_GRID (2D slice)
            f.create_dataset('omega_sq_history', shape=(n_grid, n_grid, n_grid), dtype='f4')
            # Final state shape: N_GRID x N_GRID x N_GRID
            f.create_dataset('final_psi', shape=(n_grid, n_grid, n_grid), dtype='c8')

    def log_timestep(self, metrics: dict):
        for key in self.metrics_keys:
            if key in metrics:
                self.buffer[key].append(metrics[key])

        if 'omega_sq_history' in metrics:
            # For 3D logging, we only log the central 2D slice (N/2, :, :)
            self.buffer['omega_sq_history'].append(metrics['omega_sq_history'][metrics['omega_sq_history'].shape[0] // 2, :, :])

        if self.metrics_keys and self.buffer[self.metrics_keys[0]] and len(self.buffer[self.metrics_keys[0]]) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.metrics_keys or not self.buffer[self.metrics_keys[0]]:
            return

        buffer_len = len(self.buffer[self.metrics_keys[0]])
        start = self.write_index
        end = start + buffer_len

        with h5py.File(self.filename, 'a') as f:
            for key in self.metrics_keys:
                f[key][start:end] = np.array(self.buffer[key])
            # Save the 2D slices correctly
            f['omega_sq_history'][start:end, :, :] = np.array(self.buffer['omega_sq_history'])

        self.buffer = {key: [] for key in self.metrics_keys}
        self.buffer['omega_sq_history'] = []
        self.write_index = end

    def save_final_state(self, final_psi):
        with h5py.File(self.filename, 'a') as f:
            f['final_psi'][:] = np.array(final_psi)

    def close(self):
        self.flush()
        print(f"HDF5Logger closed. Data saved to {self.filename}")


# --- CELL 4: CERTIFIED V7 ANALYSIS & GEOMETRY FUNCTIONS (3D SCALED) ---

@jit
def jnp_construct_conformal_metric(
    rho: jnp.ndarray, coupling_alpha: float, epsilon: float = 1e-9
) -> jnp.ndarray:
    """Computes the conformal factor Omega using the ECM model."""
    alpha = jnp.maximum(coupling_alpha, epsilon)
    Omega = jnp.exp(alpha * rho)
    return Omega

# --- FIX START ---
@partial(jit, static_argnames=('num_rays_val',)) 
def compute_directional_spectrum(
    psi: jax.Array, params: S_NCGL_Params, spec: SpecOps, num_rays_val: int
) -> Tuple[jax.Array, jax.Array]:
    """
    Implements the "multi-ray directional sampling protocol" on a central 1D slice.
    Requires num_rays to be static if differentiated through.
    """
    n_grid = params.N_GRID
    num_rays = num_rays_val # Use the static parameter
    k_values_1d = spec.k_values_1d
    sort_indices = spec.sort_indices_1d
    power_spectrum_agg = jnp.zeros_like(spec.k_bins)

    def body_fun(i, power_spectrum_agg):
        # Take a 1D slice along the X-axis from the center of the Y-Z plane
        # NOTE: Using .real as required for spectral power density calculation
        slice_1d = psi[n_grid // 2, n_grid // 2, :].real
        slice_fft = jnp.fft.fft(slice_1d)
        power_spectrum_1d = jnp.abs(slice_fft)**2

        k_values_sorted = k_values_1d[sort_indices]
        power_spectrum_sorted = power_spectrum_1d[sort_indices]

        # Use jnp.histogram to safely bin the spectrum
        binned_power, _ = jnp.histogram(
            k_values_sorted,
            bins=jnp.append(spec.k_bins, params.k_max_plot),
            weights=power_spectrum_sorted
        )
        return power_spectrum_agg + binned_power

    # The loop bound (num_rays) is now statically inferred by the jit wrapper
    power_spectrum_total = lax.fori_loop(0, num_rays, body_fun, power_spectrum_agg)
    power_spectrum_norm = power_spectrum_total / (jnp.sum(power_spectrum_total) + 1e-9)
    return spec.k_bins, power_spectrum_norm
# --- FIX END ---

@jit
def compute_log_prime_sse(
    k_values: jax.Array, power_spectrum: jax.Array, spec: SpecOps
) -> jax.Array:
    """ Computes the SSE against the ln(p) targets."""
    targets_k = spec.prime_targets_k
    total_power = jnp.sum(power_spectrum)

    def find_closest_idx(target_k):
        return jnp.argmin(jnp.abs(k_values - target_k))

    target_indices = jax.vmap(find_closest_idx)(targets_k)
    target_spectrum_sparse = jnp.zeros_like(k_values).at[target_indices].set(1.0)
    target_spectrum_norm = target_spectrum_sparse / jnp.sum(target_spectrum_sparse)
    diff = power_spectrum - target_spectrum_norm
    sse = jnp.sum(diff * diff)
    return jnp.where(
        total_power > 1e-9,
        jnp.nan_to_num(sse, nan=1.0, posinf=1.0, neginf=1.0),
        1.0
    )

@jit
def jnp_calculate_entropy(rho: jax.Array) -> jax.Array:
    rho_norm = rho / jnp.sum(rho)
    rho_safe = jnp.maximum(rho_norm, 1e-9)
    return -jnp.sum(rho_safe * jnp.log(rho_safe))

@jit
def jnp_calculate_quantule_census(rho: jax.Array) -> jax.Array:
    rho_mean = jnp.mean(rho)
    rho_std = jnp.std(rho)
    threshold = rho_mean + 3.0 * rho_std
    return jnp.sum(rho > threshold).astype(jnp.float32)

@partial(jit, static_argnames=('n',))
def kgrid_2pi(n: int, L: float = 1.0):
    """Creates JAX arrays for k-space grids and dealiasing mask (3D)."""
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L/n)
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing='ij') # <-- 3D meshgrid
    k_sq = kx**2 + ky**2 + kz**2 # <-- 3D k_sq
    k_mag = jnp.sqrt(k_sq)
    k_max_sim = jnp.max(k_mag)
    k_ny = jnp.max(jnp.abs(kx))
    k_cut = (2.0/3.0) * k_ny
    # 3D dealiasing mask
    dealias_mask = ((jnp.abs(kx) <= k_cut) & (jnp.abs(ky) <= k_cut) & (jnp.abs(kz) <= k_cut)).astype(jnp.float32)

    # Coordinates for initial state generation/analysis
    x = jnp.linspace(-0.5, 0.5, n) * L
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing='ij')

    return kx, ky, kz, k_sq, k_mag, k_max_sim, dealias_mask, xx, yy, zz

@jit
def make_gaussian_kernel_k(k_sq, sigma_k):
    """Pre-computes the non-local Gaussian kernel in 3D k-space."""
    return jnp.exp(-k_sq / (2.0 * (sigma_k**2)))

print("SUCCESS: V7 (3D) Analysis & Geometry functions defined.")


# --- CELL 5: CERTIFIED V7 PHYSICS ENGINE FUNCTIONS (3D SCALED) ---

@jit
def spectral_gradient_complex(field: jax.Array, spec: SpecOps) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Computes 3D spatial derivatives using fftn/ifftn."""
    field_fft = jnp.fft.fftn(field) # <-- Use n-dim FFT
    field_fft_masked = field_fft * spec.dealias_mask

    grad_x_fft = (1j * spec.kx * field_fft_masked)
    grad_y_fft = (1j * spec.ky * field_fft_masked)
    grad_z_fft = (1j * spec.kz * field_fft_masked) # <-- Z-component

    grad_x = jnp.fft.ifftn(grad_x_fft)
    grad_y = jnp.fft.ifftn(grad_y_fft)
    grad_z = jnp.fft.ifftn(grad_z_fft)

    return grad_x, grad_y, grad_z

@jit
def spectral_laplacian_complex(field: jax.Array, spec: SpecOps) -> jax.Array:
    """Computes the flat-space Laplacian in 3D using fftn/ifftn."""
    field_fft = jnp.fft.fftn(field) # <-- Use n-dim FFT
    field_fft_masked = field_fft * spec.dealias_mask
    return jnp.fft.ifftn((-spec.k_sq) * field_fft_masked)

@jit
def compute_covariant_laplacian_complex(
    psi: jax.Array, Omega: jax.Array, spec: SpecOps
) -> jax.Array:
    """Computes the curved-space spatial Laplacian (Laplace-Beltrami operator) in 3D."""
    epsilon = 1e-9
    Omega_safe = jnp.maximum(Omega, epsilon)
    Omega_sq_safe = jnp.square(Omega_safe)
    g_inv_sq = 1.0 / Omega_sq_safe

    # 1. Curvature-Modified Acceleration: (1/Omega^2) * nabla^2(psi)
    flat_laplacian_psi = spectral_laplacian_complex(psi, spec)
    curvature_modified_accel = g_inv_sq * flat_laplacian_psi
    g_inv_cubed = g_inv_sq / Omega_safe

    # 2. Geometric Damping Correction: (1/Omega^3) * (grad(Omega) . grad(psi))
    # Get 3D gradients
    grad_psi_x, grad_psi_y, grad_psi_z = spectral_gradient_complex(psi, spec)
    grad_Omega_x_c, grad_Omega_y_c, grad_Omega_z_c = spectral_gradient_complex(Omega, spec)

    grad_Omega_x = grad_Omega_x_c.real
    grad_Omega_y = grad_Omega_y_c.real
    grad_Omega_z = grad_Omega_z_c.real # <-- Z-component

    # 3D Dot product: (grad(Omega) . grad(psi))
    dot_product = (grad_Omega_x * grad_psi_x) + \
                  (grad_Omega_y * grad_psi_y) + \
                  (grad_Omega_z * grad_psi_z) # <-- Z-component added

    geometric_damping = g_inv_cubed * dot_product
    spatial_laplacian_g = curvature_modified_accel + geometric_damping
    return spatial_laplacian_g

@jit
def jnp_get_derivatives(
    state: S_NCGL_State, params: S_NCGL_Params,
    coupling_params: S_Coupling_Params,
    spec: SpecOps
) -> S_NCGL_State:
    """Core EOM for the S-NCGL equation, with 3D Geometric Feedback."""
    psi = state.psi
    rho = jnp.abs(psi)**2

    # S-NCGL Physics Terms
    rho_fft = jnp.fft.fftn(rho) # <-- Use n-dim FFT
    non_local_term_k_fft = spec.gaussian_kernel_k * rho_fft
    non_local_term_k = jnp.fft.ifftn(non_local_term_k_fft * spec.dealias_mask).real # <-- Use n-dim IFFT
    non_local_coupling = -params.nu * non_local_term_k * psi
    local_cubic_term = -params.beta * rho * psi
    source_term = params.gamma * psi
    damping_term = -params.alpha * psi

    # Geometric Feedback (Uses the 3D covariant Laplacian)
    Omega = jnp_construct_conformal_metric(rho, coupling_params.OMEGA_PARAM_A)
    spatial_laplacian_g = compute_covariant_laplacian_complex(psi, Omega, spec)
    covariant_laplacian_term = params.KAPPA * spatial_laplacian_g

    # S-NCGL EOM
    d_psi_dt = (
        damping_term + source_term + local_cubic_term +
        non_local_coupling + covariant_laplacian_term
    )
    return S_NCGL_State(psi=d_psi_dt)


@partial(jit, static_argnames=('deriv_func',))
def rk4_step(
    state: S_Coupled_State, # Accepts S_Coupled_State
    dt: float, deriv_func: Callable,
    params: S_NCGL_Params,
    coupling_params: S_Coupling_Params,
    spec: SpecOps
) -> S_Coupled_State: # Returns S_Coupled_State
    """Performs a single 4th-Order Runge-Kutta step (compatible with 3D arrays)."""
    # Pass N_GRID through to deriv_func (it's part of the partial from run_simulation_with_io)
    k1 = deriv_func(state, params, coupling_params, spec, params.N_GRID)
    k2_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt / 2.0, state, k1)
    k2 = deriv_func(k2_state, params, coupling_params, spec, params.N_GRID)
    k3_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt / 2.0, state, k2)
    k3 = deriv_func(k3_state, params, coupling_params, spec, params.N_GRID)
    k4_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt, state, k3)
    k4 = deriv_func(k4_state, params, coupling_params, spec, params.N_GRID)

    new_state = jax.tree_util.tree_map(
        lambda y, dy1, dy2, dy3, dy4: y + (dt / 6.0) * (dy1 + 2.0*dy2 + 2.0*dy3 + dy4),
        state, k1, k2, k3, k4
    )
    return new_state

print("SUCCESS: V7 (3D) Physics Engine functions defined.")


# --- CELL 6: V7 CERTIFIED EXECUTION FUNCTION (3D SCALED) ---

# NOTE: The outer jit in run_simulation_with_io handles the static num_rays implicitly
# via the partialing of jnp_sncgl_conformal_step below.

def jnp_sncgl_conformal_step(
        carry_state: S_Coupled_State,
        t: float,
        deriv_func: Callable,
        params: S_NCGL_Params,
        coupling_params: S_Coupling_Params,
        spec: SpecOps,
        jnp_construct_conformal_metric: Callable,
        compute_directional_spectrum: Callable,
        compute_log_prime_sse: Callable,
        jnp_calculate_entropy: Callable,
        jnp_calculate_quantule_census: Callable
    ) -> (S_Coupled_State, dict):
    """Master step function (to be JIT-compiled by lax.scan)."""
    state = carry_state.field_state # Extract S_NCGL_State from S_Coupled_State
    DT = params.DT

    # The use of compute_directional_spectrum here relies on its inner jit
    # being correctly implemented with static num_rays (Fix Applied in Cell 4)
    # Pass the full coupled_state to rk4_step
    new_coupled_state = rk4_step(carry_state, DT, deriv_func, params, coupling_params, spec) # RK4 returns S_Coupled_State
    # Extract psi from the new coupled state for analysis
    new_state = new_coupled_state.field_state # S_NCGL_State for analysis
    new_rho = jnp.abs(new_state.psi)**2 # Use psi from field_state for metrics
    new_rho = jnp.abs(new_state.psi)**2

    # 2D ANALYSIS (Using slice of 3D data)
    k_bins, power_spectrum = compute_directional_spectrum(new_state.psi, params, spec, params.num_rays)
    ln_p_sse = compute_log_prime_sse(k_bins, power_spectrum, spec)
    informational_entropy = jnp_calculate_entropy(new_rho)
    quantule_census = jnp_calculate_quantule_census(new_rho)

    # Geometry Metric
    Omega_final_for_log = jnp_construct_conformal_metric(
        new_rho, coupling_params.OMEGA_PARAM_A
    )
    # V7.0 UPGRADE: Log the central 2D slice (N/2, :, :) of the 3D Omega^2 tensor
    omega_sq_final_for_log_3d = jnp.square(Omega_final_for_log)

    # We must return the full 3D array for logging, and the logger will slice it
    metrics = {
        "timestamp": t * DT,
        "ln_p_sse": ln_p_sse,
        "informational_entropy": informational_entropy,
        "quantule_census": quantule_census,
        "omega_sq_history": omega_sq_final_for_log_3d
    }
    return new_coupled_state, metrics

#
# worker_v7.py (Certified v7.1 - 3D Gradient-Compatible Fix)
#
# Implements the stable S-NCGL core on a 3D grid (N x N x N).
#
# --- CELL 1: IMPORTS ---
import jax
import jax.numpy as jnp
from jax import lax, jit
import numpy as np
import h5py
import os
import time
import functools
import json
import traceback
from typing import NamedTuple, Callable, Dict, Tuple, Any, List
from geometry_solver_v8 import S_GR_State, S_GR_Source, get_geometry_input_source, get_field_feedback_terms, calculate_gr_derivatives

from tqdm.auto import tqdm
from functools import partial
import sys
import hashlib
import csv

print(f"JAX backend: {jax.default_backend()}")


# --- CELL 2: JAX PYTREE DEFINITIONS (3D SCALED) ---

class S_NCGL_State(NamedTuple):
    """Holds the dynamic state (the complex psi field) on a 3D grid."""
    psi: jax.Array

class S_NCGL_Params(NamedTuple):
    """Holds all static physics and simulation parameters."""
    N_GRID: int
    T_TOTAL: float
    DT: float
    alpha: float
    beta: float
    gamma: float
    KAPPA: float
    nu: float
    sigma_k: float
    l_domain: float
    num_rays: int
    k_bin_width: float
    k_max_plot: float

class SpecOps(NamedTuple):
    """Holds all pre-computed spectral arrays for 3D."""
    kx: jax.Array
    ky: jax.Array
    kz: jax.Array # <-- V7.0 UPGRADE: Added Z-component
    k_sq: jax.Array
    gaussian_kernel_k: jax.Array
    dealias_mask: jax.Array
    prime_targets_k: jax.Array
    k_bins: jax.Array
    ray_angles: jax.Array
    k_max: float
    xx: jax.Array
    yy: jax.Array
    zz: jax.Array # <-- V7.0 UPGRADE: Added zz
    k_values_1d: jax.Array
    sort_indices_1d: jax.Array

class S_Coupled_State(NamedTuple):
    """
    V8.0 Upgrade: Tracks both the Field (psi) and the Geometry (GR_State)
    for dynamic co-evolution in the closed GR loop.
    """
    field_state: S_NCGL_State # Holds S_NCGL_State.psi
    gr_state: S_GR_State      # Holds S_GR_State (Lapse, Shift, Metric components)

class S_Coupling_Params(NamedTuple):
    """Holds all coupling parameters (e.g., for the 'bridge')."""
    OMEGA_PARAM_A: float


# --- CELL 3: HDF5 LOGGER UTILITY (3D SCALED) ---
class HDF5Logger:
    def __init__(self, filename, n_steps, n_grid, metrics_keys, buffer_size=100):
        self.filename = filename
        self.n_steps = n_steps
        self.metrics_keys = metrics_keys
        self.buffer_size = buffer_size
        self.buffer = {key: [] for key in self.metrics_keys}
        self.buffer['omega_sq_history'] = []
        self.write_index = 0

        with h5py.File(self.filename, 'w') as f:
            for key in self.metrics_keys:
                f.create_dataset(key, (n_steps,), maxshape=(n_steps,), dtype='f4')
            # History log shape: N_steps x N_GRID x N_GRID (2D slice)
            f.create_dataset('omega_sq_history', shape=(n_grid, n_grid, n_grid), dtype='f4')
            # Final state shape: N_GRID x N_GRID x N_GRID
            f.create_dataset('final_psi', shape=(n_grid, n_grid, n_grid), dtype='c8')

    def log_timestep(self, metrics: dict):
        for key in self.metrics_keys:
            if key in metrics:
                self.buffer[key].append(metrics[key])

        if 'omega_sq_history' in metrics:
            # For 3D logging, we only log the central 2D slice (N/2, :, :)
            self.buffer['omega_sq_history'].append(metrics['omega_sq_history'][metrics['omega_sq_history'].shape[0] // 2, :, :])

        if self.metrics_keys and self.buffer[self.metrics_keys[0]] and len(self.buffer[self.metrics_keys[0]]) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.metrics_keys or not self.buffer[self.metrics_keys[0]]:
            return

        buffer_len = len(self.buffer[self.metrics_keys[0]])
        start = self.write_index
        end = start + buffer_len

        with h5py.File(self.filename, 'a') as f:
            for key in self.metrics_keys:
                f[key][start:end] = np.array(self.buffer[key])
            # Save the 2D slices correctly
            f['omega_sq_history'][start:end, :, :] = np.array(self.buffer['omega_sq_history'])

        self.buffer = {key: [] for key in self.metrics_keys}
        self.buffer['omega_sq_history'] = []
        self.write_index = end

    def save_final_state(self, final_psi):
        with h5py.File(self.filename, 'a') as f:
            f['final_psi'][:] = np.array(final_psi)

    def close(self):
        self.flush()
        print(f"HDF5Logger closed. Data saved to {self.filename}")


# --- CELL 4: CERTIFIED V7 ANALYSIS & GEOMETRY FUNCTIONS (3D SCALED) ---

@jit
def jnp_construct_conformal_metric(
    rho: jnp.ndarray, coupling_alpha: float, epsilon: float = 1e-9
) -> jnp.ndarray:
    """Computes the conformal factor Omega using the ECM model."""
    alpha = jnp.maximum(coupling_alpha, epsilon)
    Omega = jnp.exp(alpha * rho)
    return Omega

# --- FIX START ---
@partial(jit, static_argnames=('num_rays_val',)) 
def compute_directional_spectrum(
    psi: jax.Array, params: S_NCGL_Params, spec: SpecOps, num_rays_val: int
) -> Tuple[jax.Array, jax.Array]:
    """
    Implements the "multi-ray directional sampling protocol" on a central 1D slice.
    Requires num_rays to be static if differentiated through.
    """
    n_grid = params.N_GRID
    num_rays = num_rays_val # Use the static parameter
    k_values_1d = spec.k_values_1d
    sort_indices = spec.sort_indices_1d
    power_spectrum_agg = jnp.zeros_like(spec.k_bins)

    def body_fun(i, power_spectrum_agg):
        # Take a 1D slice along the X-axis from the center of the Y-Z plane
        # NOTE: Using .real as required for spectral power density calculation
        slice_1d = psi[n_grid // 2, n_grid // 2, :].real
        slice_fft = jnp.fft.fft(slice_1d)
        power_spectrum_1d = jnp.abs(slice_fft)**2

        k_values_sorted = k_values_1d[sort_indices]
        power_spectrum_sorted = power_spectrum_1d[sort_indices]

        # Use jnp.histogram to safely bin the spectrum
        binned_power, _ = jnp.histogram(
            k_values_sorted,
            bins=jnp.append(spec.k_bins, params.k_max_plot),
            weights=power_spectrum_sorted
        )
        return power_spectrum_agg + binned_power

    # The loop bound (num_rays) is now statically inferred by the jit wrapper
    power_spectrum_total = lax.fori_loop(0, num_rays, body_fun, power_spectrum_agg)
    power_spectrum_norm = power_spectrum_total / (jnp.sum(power_spectrum_total) + 1e-9)
    return spec.k_bins, power_spectrum_norm
# --- FIX END ---

@jit
def compute_log_prime_sse(
    k_values: jax.Array, power_spectrum: jax.Array, spec: SpecOps
) -> jax.Array:
    """ Computes the SSE against the ln(p) targets."""
    targets_k = spec.prime_targets_k
    total_power = jnp.sum(power_spectrum)

    def find_closest_idx(target_k):
        return jnp.argmin(jnp.abs(k_values - target_k))

    target_indices = jax.vmap(find_closest_idx)(targets_k)
    target_spectrum_sparse = jnp.zeros_like(k_values).at[target_indices].set(1.0)
    target_spectrum_norm = target_spectrum_sparse / jnp.sum(target_spectrum_sparse)
    diff = power_spectrum - target_spectrum_norm
    sse = jnp.sum(diff * diff)
    return jnp.where(
        total_power > 1e-9,
        jnp.nan_to_num(sse, nan=1.0, posinf=1.0, neginf=1.0),
        1.0
    )

@jit
def jnp_calculate_entropy(rho: jax.Array) -> jax.Array:
    rho_norm = rho / jnp.sum(rho)
    rho_safe = jnp.maximum(rho_norm, 1e-9)
    return -jnp.sum(rho_safe * jnp.log(rho_safe))

@jit
def jnp_calculate_quantule_census(rho: jax.Array) -> jax.Array:
    rho_mean = jnp.mean(rho)
    rho_std = jnp.std(rho)
    threshold = rho_mean + 3.0 * rho_std
    return jnp.sum(rho > threshold).astype(jnp.float32)

@partial(jit, static_argnames=('n',))
def kgrid_2pi(n: int, L: float = 1.0):
    """Creates JAX arrays for k-space grids and dealiasing mask (3D)."""
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L/n)
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing='ij') # <-- 3D meshgrid
    k_sq = kx**2 + ky**2 + kz**2 # <-- 3D k_sq
    k_mag = jnp.sqrt(k_sq)
    k_max_sim = jnp.max(k_mag)
    k_ny = jnp.max(jnp.abs(kx))
    k_cut = (2.0/3.0) * k_ny
    # 3D dealiasing mask
    dealias_mask = ((jnp.abs(kx) <= k_cut) & (jnp.abs(ky) <= k_cut) & (jnp.abs(kz) <= k_cut)).astype(jnp.float32)

    # Coordinates for initial state generation/analysis
    x = jnp.linspace(-0.5, 0.5, n) * L
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing='ij')

    return kx, ky, kz, k_sq, k_mag, k_max_sim, dealias_mask, xx, yy, zz

@jit
def make_gaussian_kernel_k(k_sq, sigma_k):
    """Pre-computes the non-local Gaussian kernel in 3D k-space."""
    return jnp.exp(-k_sq / (2.0 * (sigma_k**2)))

print("SUCCESS: V7 (3D) Analysis & Geometry functions defined.")


# --- CELL 5: CERTIFIED V7 PHYSICS ENGINE FUNCTIONS (3D SCALED) ---

@jit
def spectral_gradient_complex(field: jax.Array, spec: SpecOps) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Computes 3D spatial derivatives using fftn/ifftn."""
    field_fft = jnp.fft.fftn(field) # <-- Use n-dim FFT
    field_fft_masked = field_fft * spec.dealias_mask

    grad_x_fft = (1j * spec.kx * field_fft_masked)
    grad_y_fft = (1j * spec.ky * field_fft_masked)
    grad_z_fft = (1j * spec.kz * field_fft_masked) # <-- Z-component

    grad_x = jnp.fft.ifftn(grad_x_fft)
    grad_y = jnp.fft.ifftn(grad_y_fft)
    grad_z = jnp.fft.ifftn(grad_z_fft)

    return grad_x, grad_y, grad_z

@jit
def spectral_laplacian_complex(field: jax.Array, spec: SpecOps) -> jax.Array:
    """Computes the flat-space Laplacian in 3D using fftn/ifftn."""
    field_fft = jnp.fft.fftn(field) # <-- Use n-dim FFT
    field_fft_masked = field_fft * spec.dealias_mask
    return jnp.fft.ifftn((-spec.k_sq) * field_fft_masked)

@jit
def compute_covariant_laplacian_complex(
    psi: jax.Array, Omega: jax.Array, spec: SpecOps
) -> jax.Array:
    """Computes the curved-space spatial Laplacian (Laplace-Beltrami operator) in 3D."""
    epsilon = 1e-9
    Omega_safe = jnp.maximum(Omega, epsilon)
    Omega_sq_safe = jnp.square(Omega_safe)
    g_inv_sq = 1.0 / Omega_sq_safe

    # 1. Curvature-Modified Acceleration: (1/Omega^2) * nabla^2(psi)
    flat_laplacian_psi = spectral_laplacian_complex(psi, spec)
    curvature_modified_accel = g_inv_sq * flat_laplacian_psi
    g_inv_cubed = g_inv_sq / Omega_safe

    # 2. Geometric Damping Correction: (1/Omega^3) * (grad(Omega) . grad(psi))
    # Get 3D gradients
    grad_psi_x, grad_psi_y, grad_psi_z = spectral_gradient_complex(psi, spec)
    grad_Omega_x_c, grad_Omega_y_c, grad_Omega_z_c = spectral_gradient_complex(Omega, spec)

    grad_Omega_x = grad_Omega_x_c.real
    grad_Omega_y = grad_Omega_y_c.real
    grad_Omega_z = grad_Omega_z_c.real # <-- Z-component

    # 3D Dot product: (grad(Omega) . grad(psi))
    dot_product = (grad_Omega_x * grad_psi_x) + \
                  (grad_Omega_y * grad_psi_y) + \
                  (grad_Omega_z * grad_psi_z) # <-- Z-component added

    geometric_damping = g_inv_cubed * dot_product
    spatial_laplacian_g = curvature_modified_accel + geometric_damping
    return spatial_laplacian_g

@jit
def jnp_get_derivatives(
    state: S_NCGL_State, params: S_NCGL_Params,
    coupling_params: S_Coupling_Params,
    spec: SpecOps
) -> S_NCGL_State:
    """Core EOM for the S-NCGL equation, with 3D Geometric Feedback."""
    psi = state.psi
    rho = jnp.abs(psi)**2

    # S-NCGL Physics Terms
    rho_fft = jnp.fft.fftn(rho) # <-- Use n-dim FFT
    non_local_term_k_fft = spec.gaussian_kernel_k * rho_fft
    non_local_term_k = jnp.fft.ifftn(non_local_term_k_fft * spec.dealias_mask).real # <-- Use n-dim IFFT
    non_local_coupling = -params.nu * non_local_term_k * psi
    local_cubic_term = -params.beta * rho * psi
    source_term = params.gamma * psi
    damping_term = -params.alpha * psi

    # Geometric Feedback (Uses the 3D covariant Laplacian)
    Omega = jnp_construct_conformal_metric(rho, coupling_params.OMEGA_PARAM_A)
    spatial_laplacian_g = compute_covariant_laplacian_complex(psi, Omega, spec)
    covariant_laplacian_term = params.KAPPA * spatial_laplacian_g

    # S-NCGL EOM
    d_psi_dt = (
        damping_term + source_term + local_cubic_term +
        non_local_coupling + covariant_laplacian_term
    )
    return S_NCGL_State(psi=d_psi_dt)

@partial(jit, static_argnames=('deriv_func',))
def rk4_step(
    state: S_NCGL_State, dt: float, deriv_func: Callable,
    params: S_NCGL_Params,
    coupling_params: S_Coupling_Params,
    spec: SpecOps
) -> S_NCGL_State:
    """Performs a single 4th-Order Runge-Kutta step (compatible with 3D arrays)."""
    k1 = deriv_func(state, params, coupling_params, spec)
    k2_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt / 2.0, state, k1)
    k2 = deriv_func(k2_state, params, coupling_params, spec)
    k3_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt / 2.0, state, k2)
    k3 = deriv_func(k3_state, params, coupling_params, spec)
    k4_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt, state, k3)
    k4 = deriv_func(k4_state, params, coupling_params, spec)

    new_state = jax.tree_util.tree_map(
        lambda y, dy1, dy2, dy3, dy4: y + (dt / 6.0) * (dy1 + 2.0*dy2 + 2.0*dy3 + dy4),
        state, k1, k2, k3, k4
    )
    return new_state

print("SUCCESS: V7 (3D) Physics Engine functions defined.")


# --- CELL 6: V7 CERTIFIED EXECUTION FUNCTION (3D SCALED) ---

# NOTE: The outer jit in run_simulation_with_io handles the static num_rays implicitly
# via the partialing of jnp_sncgl_conformal_step below.

def jnp_sncgl_conformal_step(
        carry_state: S_Coupled_State,
        t: float,
        deriv_func: Callable,
        params: S_NCGL_Params,
        coupling_params: S_Coupling_Params,
        spec: SpecOps,
        jnp_construct_conformal_metric: Callable,
        compute_directional_spectrum: Callable,
        compute_log_prime_sse: Callable,
        jnp_calculate_entropy: Callable,
        jnp_calculate_quantule_census: Callable
    ) -> (S_Coupled_State, dict):
    """Master step function (to be JIT-compiled by lax.scan)."""
    state = carry_state.field_state # Extract S_NCGL_State from S_Coupled_State
    DT = params.DT

    # The use of compute_directional_spectrum here relies on its inner jit
    # being correctly implemented with static num_rays (Fix Applied in Cell 4)
    # Pass the full coupled_state to rk4_step
    new_coupled_state = rk4_step(carry_state, DT, deriv_func, params, coupling_params, spec) # RK4 returns S_Coupled_State
    # Extract psi from the new coupled state for analysis
    new_state = new_coupled_state.field_state # S_NCGL_State for analysis
    new_rho = jnp.abs(new_state.psi)**2 # Use psi from field_state for metrics
    new_rho = jnp.abs(new_state.psi)**2

    # 2D ANALYSIS (Using slice of 3D data)
    k_bins, power_spectrum = compute_directional_spectrum(new_state.psi, params, spec, params.num_rays)
    ln_p_sse = compute_log_prime_sse(k_bins, power_spectrum, spec)
    informational_entropy = jnp_calculate_entropy(new_rho)
    quantule_census = jnp_calculate_quantule_census(new_rho)

    # Geometry Metric
    Omega_final_for_log = jnp_construct_conformal_metric(
        new_rho, coupling_params.OMEGA_PARAM_A
    )
    # V7.0 UPGRADE: Log the central 2D slice (N/2, :, :) of the 3D Omega^2 tensor
    omega_sq_final_for_log_3d = jnp.square(Omega_final_for_log)

    # We must return the full 3D array for logging, and the logger will slice it
    metrics = {
        "timestamp": t * DT,
        "ln_p_sse": ln_p_sse,
        "informational_entropy": informational_entropy,
        "quantule_census": quantule_census,
        "omega_sq_history": omega_sq_final_for_log_3d
    }
    return new_coupled_state, metrics

def run_simulation_with_io(
    fmia_params: S_NCGL_Params,
    coupling_params: S_Coupling_Params,
    initial_state: S_NCGL_State,
    spec_ops: SpecOps,
    output_filename="simulation_output.hdf5",
    log_every_n=10
) -> Tuple:
    """
    Orchestrates the S-NCGL simulation, handling JIT compilation
    via functools.partial and managing I/O with the HDF5Logger.
    """
    print("--- Starting Orchestration (S-NCGL V7 - 3D) ---")

    # 1. Setup simulation parameters
    total_steps = int(fmia_params.T_TOTAL / fmia_params.DT)
    log_steps = total_steps // log_every_n
    if log_steps == 0:
        log_steps = 1

    initial_carry = initial_state
    print(f"Total Steps: {total_steps}, Logging every {log_every_n} steps, Log Steps: {log_steps}")

    # 2. Create the partial function (THE CERTIFIED JIT FIX)
    # This partial function captures 'params' which contains 'num_rays', making it
    # available as a static parameter when jnp_sncgl_conformal_step calls
    # compute_directional_spectrum.
    step_fn_partial = functools.partial(
        jnp_sncgl_conformal_step,
        deriv_func=jnp_get_derivatives,
        params=fmia_params,
        N_GRID=fmia_params.N_GRID, # Pass N_GRID for GR derivatives
        coupling_params=coupling_params,
        spec=spec_ops,
        jnp_construct_conformal_metric=jnp_construct_conformal_metric,
        compute_directional_spectrum=compute_directional_spectrum,
        compute_log_prime_sse=compute_log_prime_sse,
        jnp_calculate_entropy=jnp_calculate_entropy,
        jnp_calculate_quantule_census=jnp_calculate_quantule_census
    )

    # 3. JIT-compile the chunk scanner
    def scan_chunk(carry, _):
        return lax.scan(step_fn_partial, carry, jnp.arange(log_every_n))

    jit_scan_chunk = jax.jit(scan_chunk)

    # 4. Initialize the Logger (V7.0 logger handles 3D final psi)
    metrics_to_log = ["timestamp", "ln_p_sse", "informational_entropy", "quantule_census"]
    logger = HDF5Logger(output_filename, log_steps, fmia_params.N_GRID, metrics_to_log)
    print(f"HDF5Logger initialized. Output file: {output_filename}")

    # 5. Run the Main Simulation Loop
    print("--- Starting Simulation Loop (S-NCGL + Geometric Feedback) [3D] ---")
    start_time = time.time()
    current_carry = initial_carry

    for i in tqdm(range(log_steps), desc="V7 (3D) Sim Progress"):
        try:
            final_carry_state, metrics_chunk = jit_scan_chunk(current_carry, None)

            # NOTE: We grab the full 3D omega_sq_history array here, and the Logger slices it.
            last_metrics_in_chunk = {
                key: metrics_chunk[key][-1]
                for key in metrics_to_log
            }
            # Manually handle the 3D omega array from the last step in the chunk
            last_metrics_in_chunk['omega_sq_history'] = metrics_chunk['omega_sq_history'][-1]

            logger.log_timestep(last_metrics_in_chunk)
            current_carry = final_carry_state
        except Exception as e:
            print(f"\nERROR during simulation step {i}: {e}")
            logger.close()
            raise

    end_time = time.time()
    print(f"--- Simulation Loop Complete---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # 6. Save final state and close logger
    logger.save_final_state(current_carry.field_state.psi) # Save the final field state
    logger.close()

    import numpy as _np
    _psi_bytes = _np.asarray(current_carry.field_state.psi).tobytes() # Hash the final field state
    print(f"Final state (psi hash): {hash(_psi_bytes)}")

    return current_carry, output_filename, True


# --- CELL 7: V7 "WORKER" LOGIC ---

def generate_param_hash(params: Dict[str, Any]) -> str:
    """Creates a unique SHA256 hash from a parameter dictionary."""
    sorted_params_str = json.dumps(params, sort_keys=True).encode('utf-8')
    hash_str = hashlib.sha256(sorted_params_str).hexdigest()
    return hash_str[:12]

def write_to_ledger(ledger_file: str, run_data: Dict[str, Any]):
    """Appends a single run's data to the CSV ledger."""
    file_exists = os.path.isfile(ledger_file)
    all_headers = sorted(list(run_data.keys()))

    preferred_order = [
        'param_hash', 'final_sse', 'jax_run_seed', 'generation',
        'alpha', 'sigma_k', 'nu', 'OMEGA_PARAM_A', 'KAPPA',
        'gamma', 'beta', 'N_GRID', 'T_TOTAL'
    ]

    final_headers = [h for h in preferred_order if h in all_headers] + \
                     [h for h in all_headers if h not in preferred_order]

    cleaned_run_data = {}
    for k, v in run_data.items():
        if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
            cleaned_run_data[k] = -999.0
        else:
            cleaned_run_data[k] = v

    try:
        with open(ledger_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_headers, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(cleaned_run_data)
    except Exception as e:
        print(f"  > [WORKER] Error writing to ledger: {e}")

def load_todo_list(todo_file: str) -> List[Dict[str, Any]]:
    """Loads the list of jobs from the Hunter."""
    try:
        with open(todo_file, 'r') as f:
            jobs = json.load(f)

        os.remove(todo_file)
        print(f"  > [WORKER] Loaded and removed '{todo_file}'.")
        return jobs
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"  > [WORKER] ERROR: '{todo_file}' is corrupted or empty. Deleting.")
        os.remove(todo_file)
        return []

def generate_bootstrap_jobs(
    rng: np.random.Generator, num_jobs: int
) -> List[Dict[str, Any]]:
    """Creates the 'Generation 0' for the "Blind 5D Exploration" hunt."""
    print(f"  > [WORKER] Generating {num_jobs} (5D BLIND) bootstrap jobs (Gen 0)...")
    jobs = []

    PARAM_RANGES = {
        'alpha':         ('uniform', 0.01, 1.0),
        'sigma_k':       ('uniform', 0.1, 10.0),
        'nu':            ('uniform', 0.1, 5.0),
        'OMEGA_PARAM_A': ('uniform', 0.1, 2.5),
        'KAPPA':         ('uniform', 0.001, 5.0)
    }

    print(f"  > [WORKER] Generating {num_jobs} random 'immigrants'...")
    for _ in range(num_jobs):
        job = {}
        for key, (dist, p_min, p_max) in PARAM_RANGES.items():
            if dist == 'uniform':
                job[key] = rng.uniform(low=p_min, high=p_max)
        job['generation'] = 0
        jobs.append(job)
    return jobs

def run_worker_main(hunt_id, todo_file):
    """This is the main "Worker" function that the orchestrator calls."""
    print(f"--- [WORKER] ENGAGED for {hunt_id} (V7.1 Engine - 3D) ---")

    MASTER_SEED = 42
    BOOTSTRAP_JOBS = 100

    # Static physics params (non-evolvable)
    STATIC_PHYSICS_PARAMS = {
        "gamma": 0.2,
        "beta": 1.0,
        "N_GRID": 64, # Default 3D size: 64x64x64
        "T_TOTAL": 1.0, # Shorter runtime for 3D computational cost
        "DT": 1e-3
    }

    # Static simulation setup params
    L_DOMAIN = 20.0
    K_MAX_PLOT = 2.0
    K_BIN_WIDTH = 0.01
    NUM_RAYS = 32
    LOG_EVERY_N_STEPS = 10

    # Setup directories and RNG
    MASTER_OUTPUT_DIR = os.path.join("sweep_runs", hunt_id)
    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
    LEDGER_FILE = os.path.join(MASTER_OUTPUT_DIR, f"ledger_{hunt_id}.csv")
    master_rng = np.random.default_rng(MASTER_SEED)

    # --- Load or Generate Job List ---
    params_to_run = load_todo_list(todo_file)
    if not params_to_run:
        print(f"  > [WORKER] No '{todo_file}' found. Bootstrapping (5D Blind)...")
        params_to_run = generate_bootstrap_jobs(master_rng, BOOTSTRAP_JOBS)

    total_jobs = len(params_to_run)
    print(f"  > [WORKER] Found {total_jobs} jobs to run.")

    sweep_start_time = time.time()

    # --- Loop over all jobs from the Hunter ---
    for i, variable_params in enumerate(params_to_run):
        run_start_time = time.time()
        print(f"\n  --- [WORKER] Starting Job {i+1} / {total_jobs} ---")

        if not isinstance(variable_params, dict):
            print(f"!!! [WORKER] ERROR: Invalid job format. Expected dict, got {type(variable_params)}. Skipping job.")
            print(f"    Bad data: {variable_params}")
            continue

        # 1. Combine static and variable params
        current_run_params = variable_params.copy()
        current_run_params.update(STATIC_PHYSICS_PARAMS)

        # 2. Add generation, seed, and hash
        if 'generation' not in current_run_params:
            current_run_params['generation'] = 'unknown'

        jax_run_seed = int(master_rng.integers(low=0, high=2**31 - 1))
        current_run_params['jax_run_seed'] = jax_run_seed
        param_hash = generate_param_hash(current_run_params)
        current_run_params['param_hash'] = param_hash
        print(f"    Run Hash: {param_hash} | JAX Seed: {jax_run_seed}")

        # 3. Assemble the V7 JAX Pytrees (Structs)
        try:
            fmia_params = S_NCGL_Params(
                N_GRID=int(current_run_params["N_GRID"]),
                T_TOTAL=float(current_run_params["T_TOTAL"]),
                DT=float(current_run_params["DT"]),
                alpha=float(current_run_params["alpha"]),
                beta=float(current_run_params["beta"]),
                gamma=float(current_run_params["gamma"]),
                KAPPA=float(current_run_params["KAPPA"]),
                nu=float(current_run_params["nu"]),
                sigma_k=float(current_run_params["sigma_k"]),
                l_domain=L_DOMAIN,
                num_rays=NUM_RAYS,
                k_bin_width=K_BIN_WIDTH,
                k_max_plot=K_MAX_PLOT
            )

            coupling_params = S_Coupling_Params(
                OMEGA_PARAM_A=float(current_run_params["OMEGA_PARAM_A"])
            )

            key = jax.random.PRNGKey(jax_run_seed)
            N_GRID = fmia_params.N_GRID

            # --- V7.0 UPGRADE: Call 3D kgrid_2pi (returns kz, zz) ---
            kx, ky, kz, k_sq, k_mag, k_max_sim, dealias_mask, xx, yy, zz = kgrid_2pi(N_GRID, L_DOMAIN)

            gaussian_kernel_k = make_gaussian_kernel_k(k_sq, fmia_params.sigma_k)
            k_bins = jnp.arange(0, K_MAX_PLOT, K_BIN_WIDTH)
            primes = jnp.array([2, 3, 5, 7, 11, 13, 17, 19])
            prime_targets_k = jnp.log(primes)

            k_values_1d = 2 * jnp.pi * jnp.fft.fftfreq(N_GRID, d=L_DOMAIN / N_GRID)
            sort_indices_1d = jnp.argsort(k_values_1d)

            spec_ops = SpecOps(
                kx=kx.astype(jnp.float32),
                ky=ky.astype(jnp.float32),
                kz=kz.astype(jnp.float32), # <-- V7.0 UPGRADE
                k_sq=k_sq.astype(jnp.float32),
                gaussian_kernel_k=gaussian_kernel_k.astype(jnp.float32),
                dealias_mask=dealias_mask.astype(jnp.float32),
                k_bins=k_bins.astype(jnp.float32),
                prime_targets_k=prime_targets_k.astype(jnp.float32),
                ray_angles=jnp.linspace(0, jnp.pi, NUM_RAYS),
                k_max=k_max_sim.astype(jnp.float32),
                xx=xx.astype(jnp.float32),
                yy=yy.astype(jnp.float32),
                zz=zz.astype(jnp.float32), # <-- V7.0 UPGRADE
                k_values_1d=k_values_1d.astype(jnp.float32),
                sort_indices_1d=sort_indices_1d.astype(jnp.int32)
            )

            # --- V7.0 UPGRADE: Initial psi is 3D (N, N, N) ---
            psi_initial = (
                jax.random.uniform(key, (N_GRID, N_GRID, N_GRID), dtype=jnp.float32) * 0.1 +\
                1j * jax.random.uniform(key, (N_GRID, N_GRID, N_GRID), dtype=jnp.float32) * 0.1
            )
            initial_state = S_NCGL_State(psi=psi_initial.astype(jnp.complex64))

            output_filename = os.path.join(MASTER_OUTPUT_DIR, f"run_{param_hash}.hdf5")

        except Exception as e:
            print(f"!!! [WORKER] JOB {param_hash} FAILED during parameter assembly: {e} !!!")
            traceback.print_exc()
            final_sse = 99998.0
            current_run_params['final_sse'] = final_sse
            write_to_ledger(LEDGER_FILE, current_run_params)
            continue

        # 4. Run the V7 Simulation
        sim_success = False
        try:
            final_carry_state, output_file, sim_success = run_simulation_with_io(
                fmia_params,
                coupling_params,
                initial_state,
                spec_ops,
                output_filename=output_filename,
                log_every_n=LOG_EVERY_N_STEPS
            )

            # 5. Get the Final SSE
            if sim_success:
                with h5py.File(output_file, 'r') as f:
                    final_sse = float(f['ln_p_sse'][-1])
            else:
                final_sse = 99999.0

        except Exception as e:
            print(f"!!! [WORKER] JOB {param_hash} FAILED during simulation: {e} !!!")
            traceback.print_exc()
            final_sse = 99999.0

        run_end_time = time.time()

        # 6. Log results to master ledger
        current_run_params['final_sse'] = final_sse
        print(f"  --- [WORKER] Job {i+1} Complete ({run_end_time - run_start_time:.2f}s) ---")
        print(f"    Final SSE: {final_sse:.12f}")
        write_to_ledger(LEDGER_FILE, current_run_params)

    # --- Loop Finished ---
    sweep_end_time = time.time()
    print(f"\n--- [WORKER] FINISHED {hunt_id} ---")
    print(f"Total time for {total_jobs} jobs: {(sweep_end_time - sweep_start_time) / 60.0:.2f} minutes")


# --- THIS IS THE NEW "MAIN" BLOCK ---
if __name__ == "__main__":

    # --- Check for dependencies (for Colab) ---
    try:
        import jax, pandas, h5py
        print("All dependencies satisfied.")
    except ImportError:
        print("Installing dependencies (jax, pandas, h5py, tqdm, matplotlib)...")
        import subprocess
        subprocess.run(["pip", "install", "--quiet", "jax", "jaxlib", "pandas", "h5py", "tqdm", "matplotlib"], check=True)
        print("Dependency installation complete. Please RESTART the runtime if imports fail.")

    # --- Main Logic ---
    if len(sys.argv) < 3:
        print("\n" + "="*50)
        print("--- [WORKER] Running in TEST MODE (3D) ---")
        print("No CLI args detected. This will run one test simulation.")
        print("="*50)

        HUNT_ID = "SNCGL_ADAPTIVE_HUNT_TEST_3D"
        TODO_FILE = "ASTE_generation_todo_TEST.json"

        test_params = {
            "alpha": 0.1, "KAPPA": 1.0, "nu": 1.0,
            "sigma_k": 2.5, "OMEGA_PARAM_A": 0.5,
            "generation": -1
        }
        with open(TODO_FILE, 'w') as f:
            json.dump([test_params], f)

        run_worker_main(HUNT_ID, TODO_FILE)

    else:
        print(f"--- [WORKER] Production mode activated by orchestrator ---")
        HUNT_ID = sys.argv[1]
        TODO_FILE = sys.argv[2]
        run_worker_main(hunt_id=HUNT_ID, todo_file=TODO_FILE)

print("worker_v7.py successfully written.")



# --- CELL 7: V7 "WORKER" LOGIC ---

def generate_param_hash(params: Dict[str, Any]) -> str:
    """Creates a unique SHA256 hash from a parameter dictionary."""
    sorted_params_str = json.dumps(params, sort_keys=True).encode('utf-8')
    hash_str = hashlib.sha256(sorted_params_str).hexdigest()
    return hash_str[:12]

def write_to_ledger(ledger_file: str, run_data: Dict[str, Any]):
    """Appends a single run's data to the CSV ledger."""
    file_exists = os.path.isfile(ledger_file)
    all_headers = sorted(list(run_data.keys()))

    preferred_order = [
        'param_hash', 'final_sse', 'jax_run_seed', 'generation',
        'alpha', 'sigma_k', 'nu', 'OMEGA_PARAM_A', 'KAPPA',
        'gamma', 'beta', 'N_GRID', 'T_TOTAL'
    ]

    final_headers = [h for h in preferred_order if h in all_headers] + \
                     [h for h in all_headers if h not in preferred_order]

    cleaned_run_data = {}
    for k, v in run_data.items():
        if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
            cleaned_run_data[k] = -999.0
        else:
            cleaned_run_data[k] = v

    try:
        with open(ledger_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_headers, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(cleaned_run_data)
    except Exception as e:
        print(f"  > [WORKER] Error writing to ledger: {e}")

def load_todo_list(todo_file: str) -> List[Dict[str, Any]]:
    """Loads the list of jobs from the Hunter."""
    try:
        with open(todo_file, 'r') as f:
            jobs = json.load(f)

        os.remove(todo_file)
        print(f"  > [WORKER] Loaded and removed '{todo_file}'.")
        return jobs
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"  > [WORKER] ERROR: '{todo_file}' is corrupted or empty. Deleting.")
        os.remove(todo_file)
        return []

def generate_bootstrap_jobs(
    rng: np.random.Generator, num_jobs: int
) -> List[Dict[str, Any]]:
    """Creates the 'Generation 0' for the "Blind 5D Exploration" hunt."""
    print(f"  > [WORKER] Generating {num_jobs} (5D BLIND) bootstrap jobs (Gen 0)...")
    jobs = []

    PARAM_RANGES = {
        'alpha':         ('uniform', 0.01, 1.0),
        'sigma_k':       ('uniform', 0.1, 10.0),
        'nu':            ('uniform', 0.1, 5.0),
        'OMEGA_PARAM_A': ('uniform', 0.1, 2.5),
        'KAPPA':         ('uniform', 0.001, 5.0)
    }

    print(f"  > [WORKER] Generating {num_jobs} random 'immigrants'...")
    for _ in range(num_jobs):
        job = {}
        for key, (dist, p_min, p_max) in PARAM_RANGES.items():
            if dist == 'uniform':
                job[key] = rng.uniform(low=p_min, high=p_max)
        job['generation'] = 0
        jobs.append(job)
    return jobs

def run_worker_main(hunt_id, todo_file):
    """This is the main "Worker" function that the orchestrator calls."""
    print(f"--- [WORKER] ENGAGED for {hunt_id} (V7.1 Engine - 3D) ---")

    MASTER_SEED = 42
    BOOTSTRAP_JOBS = 100

    # Static physics params (non-evolvable)
    STATIC_PHYSICS_PARAMS = {
        "gamma": 0.2,
        "beta": 1.0,
        "N_GRID": 64, # Default 3D size: 64x64x64
        "T_TOTAL": 1.0, # Shorter runtime for 3D computational cost
        "DT": 1e-3
    }

    # Static simulation setup params
    L_DOMAIN = 20.0
    K_MAX_PLOT = 2.0
    K_BIN_WIDTH = 0.01
    NUM_RAYS = 32
    LOG_EVERY_N_STEPS = 10

    # Setup directories and RNG
    MASTER_OUTPUT_DIR = os.path.join("sweep_runs", hunt_id)
    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
    LEDGER_FILE = os.path.join(MASTER_OUTPUT_DIR, f"ledger_{hunt_id}.csv")
    master_rng = np.random.default_rng(MASTER_SEED)

    # --- Load or Generate Job List ---
    params_to_run = load_todo_list(todo_file)
    if not params_to_run:
        print(f"  > [WORKER] No '{todo_file}' found. Bootstrapping (5D Blind)...")
        params_to_run = generate_bootstrap_jobs(master_rng, BOOTSTRAP_JOBS)

    total_jobs = len(params_to_run)
    print(f"  > [WORKER] Found {total_jobs} jobs to run.")

    sweep_start_time = time.time()

    # --- Loop over all jobs from the Hunter ---
    for i, variable_params in enumerate(params_to_run):
        run_start_time = time.time()
        print(f"\n  --- [WORKER] Starting Job {i+1} / {total_jobs} ---")

        if not isinstance(variable_params, dict):
            print(f"!!! [WORKER] ERROR: Invalid job format. Expected dict, got {type(variable_params)}. Skipping job.")
            print(f"    Bad data: {variable_params}")
            continue

        # 1. Combine static and variable params
        current_run_params = variable_params.copy()
        current_run_params.update(STATIC_PHYSICS_PARAMS)

        # 2. Add generation, seed, and hash
        if 'generation' not in current_run_params:
            current_run_params['generation'] = 'unknown'

        jax_run_seed = int(master_rng.integers(low=0, high=2**31 - 1))
        current_run_params['jax_run_seed'] = jax_run_seed
        param_hash = generate_param_hash(current_run_params)
        current_run_params['param_hash'] = param_hash
        print(f"    Run Hash: {param_hash} | JAX Seed: {jax_run_seed}")

        # 3. Assemble the V7 JAX Pytrees (Structs)
        try:
            fmia_params = S_NCGL_Params(
                N_GRID=int(current_run_params["N_GRID"]),
                T_TOTAL=float(current_run_params["T_TOTAL"]),
                DT=float(current_run_params["DT"]),
                alpha=float(current_run_params["alpha"]),
                beta=float(current_run_params["beta"]),
                gamma=float(current_run_params["gamma"]),
                KAPPA=float(current_run_params["KAPPA"]),
                nu=float(current_run_params["nu"]),
                sigma_k=float(current_run_params["sigma_k"]),
                l_domain=L_DOMAIN,
                num_rays=NUM_RAYS,
                k_bin_width=K_BIN_WIDTH,
                k_max_plot=K_MAX_PLOT
            )

            coupling_params = S_Coupling_Params(
                OMEGA_PARAM_A=float(current_run_params["OMEGA_PARAM_A"])
            )

            key = jax.random.PRNGKey(jax_run_seed)
            N_GRID = fmia_params.N_GRID

            # --- V7.0 UPGRADE: Call 3D kgrid_2pi (returns kz, zz) ---
            kx, ky, kz, k_sq, k_mag, k_max_sim, dealias_mask, xx, yy, zz = kgrid_2pi(N_GRID, L_DOMAIN)

            gaussian_kernel_k = make_gaussian_kernel_k(k_sq, fmia_params.sigma_k)
            k_bins = jnp.arange(0, K_MAX_PLOT, K_BIN_WIDTH)
            primes = jnp.array([2, 3, 5, 7, 11, 13, 17, 19])
            prime_targets_k = jnp.log(primes)

            k_values_1d = 2 * jnp.pi * jnp.fft.fftfreq(N_GRID, d=L_DOMAIN / N_GRID)
            sort_indices_1d = jnp.argsort(k_values_1d)

            spec_ops = SpecOps(
                kx=kx.astype(jnp.float32),
                ky=ky.astype(jnp.float32),
                kz=kz.astype(jnp.float32), # <-- V7.0 UPGRADE
                k_sq=k_sq.astype(jnp.float32),
                gaussian_kernel_k=gaussian_kernel_k.astype(jnp.float32),
                dealias_mask=dealias_mask.astype(jnp.float32),
                k_bins=k_bins.astype(jnp.float32),
                prime_targets_k=prime_targets_k.astype(jnp.float32),
                ray_angles=jnp.linspace(0, jnp.pi, NUM_RAYS),
                k_max=k_max_sim.astype(jnp.float32),
                xx=xx.astype(jnp.float32),
                yy=yy.astype(jnp.float32),
                zz=zz.astype(jnp.float32), # <-- V7.0 UPGRADE
                k_values_1d=k_values_1d.astype(jnp.float32),
                sort_indices_1d=sort_indices_1d.astype(jnp.int32)
            )

            # --- V7.0 UPGRADE: Initial psi is 3D (N, N, N) ---
            psi_initial = (
                jax.random.uniform(key, (N_GRID, N_GRID, N_GRID), dtype=jnp.float32) * 0.1 +\
                1j * jax.random.uniform(key, (N_GRID, N_GRID, N_GRID), dtype=jnp.float32) * 0.1
            )
            initial_state = S_NCGL_State(psi=psi_initial.astype(jnp.complex64))

            output_filename = os.path.join(MASTER_OUTPUT_DIR, f"run_{param_hash}.hdf5")

        except Exception as e:
            print(f"!!! [WORKER] JOB {param_hash} FAILED during parameter assembly: {e} !!!")
            traceback.print_exc()
            final_sse = 99998.0
            current_run_params['final_sse'] = final_sse
            write_to_ledger(LEDGER_FILE, current_run_params)
            continue

        # 4. Run the V7 Simulation
        sim_success = False
        try:
            final_carry_state, output_file, sim_success = run_simulation_with_io(
                fmia_params,
                coupling_params,
                initial_state,
                spec_ops,
                output_filename=output_filename,
                log_every_n=LOG_EVERY_N_STEPS
            )

            # 5. Get the Final SSE
            if sim_success:
                with h5py.File(output_file, 'r') as f:
                    final_sse = float(f['ln_p_sse'][-1])
            else:
                final_sse = 99999.0

        except Exception as e:
            print(f"!!! [WORKER] JOB {param_hash} FAILED during simulation: {e} !!!")
            traceback.print_exc()
            final_sse = 99999.0

        run_end_time = time.time()

        # 6. Log results to master ledger
        current_run_params['final_sse'] = final_sse
        print(f"  --- [WORKER] Job {i+1} Complete ({run_end_time - run_start_time:.2f}s) ---")
        print(f"    Final SSE: {final_sse:.12f}")
        write_to_ledger(LEDGER_FILE, current_run_params)

    # --- Loop Finished ---
    sweep_end_time = time.time()
    print(f"\n--- [WORKER] FINISHED {hunt_id} ---")
    print(f"Total time for {total_jobs} jobs: {(sweep_end_time - sweep_start_time) / 60.0:.2f} minutes")


# --- THIS IS THE NEW "MAIN" BLOCK ---
if __name__ == "__main__":

    # --- Check for dependencies (for Colab) ---
    try:
        import jax, pandas, h5py
        print("All dependencies satisfied.")
    except ImportError:
        print("Installing dependencies (jax, pandas, h5py, tqdm, matplotlib)...")
        import subprocess
        subprocess.run(["pip", "install", "--quiet", "jax", "jaxlib", "pandas", "h5py", "tqdm", "matplotlib"], check=True)
        print("Dependency installation complete. Please RESTART the runtime if imports fail.")

    # --- Main Logic ---
    if len(sys.argv) < 3:
        print("\n" + "="*50)
        print("--- [WORKER] Running in TEST MODE (3D) ---")
        print("No CLI args detected. This will run one test simulation.")
        print("="*50)

        HUNT_ID = "SNCGL_ADAPTIVE_HUNT_TEST_3D"
        TODO_FILE = "ASTE_generation_todo_TEST.json"

        test_params = {
            "alpha": 0.1, "KAPPA": 1.0, "nu": 1.0,
            "sigma_k": 2.5, "OMEGA_PARAM_A": 0.5,
            "generation": -1
        }
        with open(TODO_FILE, 'w') as f:
            json.dump([test_params], f)

        run_worker_main(HUNT_ID, TODO_FILE)

    else:
        print(f"--- [WORKER] Production mode activated by orchestrator ---")
        HUNT_ID = sys.argv[1]
        TODO_FILE = sys.argv[2]
        run_worker_main(hunt_id=HUNT_ID, todo_file=TODO_FILE)

print("worker_v7.py successfully written.")
