#
# worker_v7.py (Certified V8.0 - 3D+1 GR-Coupled System)
#
# Implements the co-evolution of the S-NCGL field (psi) and the dynamic 3+1 metric.
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
from tqdm.auto import tqdm
from functools import partial
import sys
import hashlib
import csv

# --- V8.0 UPGRADE: IMPORT GR COMPONENTS ---
from geometry_solver_v8 import (
    S_GR_State, S_GR_Source,
    get_geometry_input_source,
    gr_solver_step, # gr_solver_step is now just returning derivatives from calculate_gr_derivatives_internal
    get_field_feedback_terms,
    apply_stabilization_mandates # Explicitly imported
)

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
    kz: jax.Array
    k_sq: jax.Array
    gaussian_kernel_k: jax.Array
    dealias_mask: jax.Array
    prime_targets_k: jax.Array
    k_bins: jax.Array
    ray_angles: jax.Array
    k_max: float
    xx: jax.Array
    yy: jax.Array
    zz: jax.Array
    k_values_1d: jax.Array
    sort_indices_1d: jax.Array

class S_Coupling_Params(NamedTuple):
    """Holds all coupling parameters (e.g., for the 'bridge')."""
    OMEGA_PARAM_A: float

# --- V8.0 UPGRADE: NEW COUPLED STATE DEFINITION ---
class S_Coupled_State(NamedTuple):
    """V8.0: Tracks both the Field (psi) and the Geometry (GR_State) for co-evolution."""
    field_state: S_NCGL_State # Holds S_NCGL_State.psi
    gr_state: S_GR_State      # Holds S_GR_State (Lapse, Shift, Metric components)


# --- CELL 3: HDF5 LOGGER UTILITY (Updated for V8.0) ---

class HDF5Logger:
    def __init__(self, filename, n_steps, n_grid, metrics_keys, buffer_size=100):
        # Logger needs to be updated to log GR metrics as well
        self.filename = filename
        self.n_steps = n_steps
        self.metrics_keys = metrics_keys
        self.buffer_size = buffer_size
        self.buffer = {key: [] for key in self.metrics_keys}
        self.buffer['lapse_center'] = [] # New GR metric to track
        self.write_index = 0

        with h5py.File(self.filename, 'w') as f:
            for key in self.metrics_keys:
                f.create_dataset(key, (n_steps,), maxshape=(n_steps,), dtype='f4')
            f.create_dataset('lapse_center', (n_steps,), maxshape=(n_steps,), dtype='f4')
            # Final state now stores the combined state
            f.create_dataset('final_psi', shape=(n_grid, n_grid, n_grid), dtype='c8')
            f.create_dataset('final_lapse', shape=(n_grid, n_grid, n_grid), dtype='f4')


    def log_timestep(self, metrics: dict):
        for key in self.metrics_keys:
            if key in metrics:
                self.buffer[key].append(metrics[key])

        if 'lapse_center' in metrics:
            self.buffer['lapse_center'].append(metrics['lapse_center'])

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
            f['lapse_center'][start:end] = np.array(self.buffer['lapse_center'])

        self.buffer = {key: [] for key in self.metrics_keys}
        self.buffer['lapse_center'] = []
        self.write_index = end

    def save_final_state(self, final_coupled_state: S_Coupled_State, N_GRID: int):
        final_psi_np = np.asarray(final_coupled_state.field_state.psi)
        final_lapse_np = np.asarray(final_coupled_state.gr_state.lapse)

        # Ensure arrays are correctly shaped before saving
        final_psi_np = final_psi_np.reshape(N_GRID, N_GRID, N_GRID)
        final_lapse_np = final_lapse_np.reshape(N_GRID, N_GRID, N_GRID)

        with h5py.File(self.filename, 'a') as f:
            f['final_psi'][:] = final_psi_np
            f['final_lapse'][:] = final_lapse_np

    def close(self):
        self.flush()
        print(f"HDF5Logger closed. Data saved to {self.filename}")


# --- CELL 4: V7 ANALYSIS & GEOMETRY FUNCTIONS (Unchanged from V7.1) ---

@jit
def jnp_construct_conformal_metric(
    rho: jnp.ndarray, coupling_alpha: float, epsilon: float = 1e-9
) -> jnp.ndarray:
    """Computes the conformal factor Omega using the ECM model (Unchanged V7 function)."""
    alpha = jnp.maximum(coupling_alpha, epsilon)
    Omega = jnp.exp(alpha * rho)
    return Omega

@partial(jit, static_argnames=('num_rays',))
def compute_directional_spectrum(
    psi: jax.Array, params: S_NCGL_Params, spec: SpecOps
) -> Tuple[jax.Array, jax.Array]:
    """ Implements the "multi-ray directional sampling protocol" (V7.1 Fixed)."""
    n_grid = params.N_GRID
    num_rays = params.num_rays
    k_values_1d = spec.k_values_1d
    sort_indices = spec.sort_indices_1d
    power_spectrum_agg = jnp.zeros_like(spec.k_bins)

    def body_fun(i, power_spectrum_agg):
        slice_1d = psi[n_grid // 2, n_grid // 2, :].real
        slice_fft = jnp.fft.fft(slice_1d)
        power_spectrum_1d = jnp.abs(slice_fft)**2

        k_values_sorted = k_values_1d[sort_indices]
        power_spectrum_sorted = power_spectrum_1d[sort_indices]

        binned_power, _ = jnp.histogram(
            k_values_sorted,
            bins=jnp.append(spec.k_bins, params.k_max_plot),
            weights=power_spectrum_sorted
        )
        return power_spectrum_agg + binned_power

    power_spectrum_total = lax.fori_loop(0, num_rays, body_fun, power_spectrum_agg)

    power_spectrum_norm = power_spectrum_total / (jnp.sum(power_spectrum_total) + 1e-9)
    return spec.k_bins, power_spectrum_norm


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
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2
    k_mag = jnp.sqrt(k_sq)
    k_max_sim = jnp.max(k_mag)
    k_ny = jnp.max(jnp.abs(kx))
    k_cut = (2.0/3.0) * k_ny
    dealias_mask = ((jnp.abs(kx) <= k_cut) & (jnp.abs(ky) <= k_cut) & (jnp.abs(kz) <= k_cut)).astype(jnp.float32)

    x = jnp.linspace(-0.5, 0.5, n) * L
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing='ij')

    return kx, ky, kz, k_sq, k_mag, k_max_sim, dealias_mask, xx, yy, zz

@jit
def make_gaussian_kernel_k(k_sq, sigma_k):
    """Pre-computes the non-local Gaussian kernel in 3D k-space."""
    return jnp.exp(-k_sq / (2.0 * (sigma_k**2)))

print("SUCCESS: V7 (3D) Analysis & Geometry functions defined.")


# --- CELL 5: CERTIFIED V8.0 PHYSICS ENGINE FUNCTIONS (Coupled EOMs) ---

@jit
def spectral_gradient_complex(field: jax.Array, spec: SpecOps) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Computes 3D spatial derivatives using fftn/ifftn."""
    field_fft = jnp.fft.fftn(field)
    field_fft_masked = field_fft * spec.dealias_mask

    grad_x_fft = (1j * spec.kx * field_fft_masked)
    grad_y_fft = (1j * spec.ky * field_fft_masked)
    grad_z_fft = (1j * spec.kz * field_fft_masked)

    grad_x = jnp.fft.ifftn(grad_x_fft)
    grad_y = jnp.fft.ifftn(grad_y_fft)
    grad_z = jnp.fft.ifftn(grad_z_fft)

    return grad_x, grad_y, grad_z

# --- V8.0 UPGRADE: Modified jnp_get_derivatives ---
@jit
def jnp_get_derivatives(
    coupled_state: S_Coupled_State,
    params: S_NCGL_Params, coupling_params: S_Coupling_Params,
    spec: SpecOps, N_GRID: int) -> S_Coupled_State:

    # --- 1. Extract States and Compute Source (Field -> Source) ---
    field_state = coupled_state.field_state
    gr_state = coupled_state.gr_state
    psi = field_state.psi
    rho = jnp.abs(psi)**2

    # Generate GR source term (T_mu_nu^info) from the current field state
    gr_source = get_geometry_input_source(psi)

    # --- 2. Calculate d(Geometry)/dt (Source -> Geometry) ---
    # The GR Solver EOM requires the current metric state and the field source.
    # We call the core derivative function from the geometry solver module.
    # gr_solver_step now directly returns the derivatives from internal functions.
    d_gr_state_dt_raw = gr_solver_step(gr_state, gr_source, N_GRID)

    # --- 3. Geometric Feedback (Geometry -> Feedback) ---
    # Get the connection terms (Γ) derived from the Evolved Metric State
    connection_coefficients, modified_laplacian_factor = get_field_feedback_terms(gr_state, N_GRID)

    # --- 4. Calculate d(Field)/dt (S-NCGL EOM) ---

    # (Re-calculating V7 S-NCGL Physics Terms)
    rho_fft = jnp.fft.fftn(rho)
    non_local_term_k_fft = spec.gaussian_kernel_k * rho_fft
    non_local_term_k = jnp.fft.ifftn(non_local_term_k_fft * spec.dealias_mask).real

    damping_term = -params.alpha * psi
    source_term = params.gamma * psi
    local_cubic_term = -params.beta * rho * psi
    non_local_coupling = -params.nu * non_local_term_k * psi

    # Calculate Dynamic Covariant Laplacian (∇_g^2 ψ)
    dynamic_geometry_feedback = params.KAPPA * modified_laplacian_factor * (connection_coefficients * psi)

    d_psi_dt = (
        damping_term + source_term + local_cubic_term +
        non_local_coupling + dynamic_geometry_feedback
    )

    # --- 5. Return Coupled Derivative State ---
    return S_Coupled_State(
        field_state=S_NCGL_State(psi=d_psi_dt),
        gr_state=d_gr_state_dt_raw # Use the raw derivative for RK4
    )


# --- V8.0 UPGRADE: Modified rk4_step ---
@partial(jit, static_argnames=('deriv_func', 'N_GRID'))
def rk4_step(
    state: S_Coupled_State, dt: float, deriv_func: Callable, # <- Now S_Coupled_State
    params: S_NCGL_Params,
    coupling_params: S_Coupling_Params,
    spec: SpecOps,
    N_GRID: int # <- N_GRID added for explicit passing to gr_solver_step/stabilization
) -> S_Coupled_State: # <- Returns S_Coupled_State
    """Performs a single 4th-Order Runge-Kutta step for the coupled state."""

    # K1
    k1_d_state = deriv_func(state, params, coupling_params, spec, N_GRID)
    stabilized_k1_gr = apply_stabilization_mandates(k1_d_state.gr_state, dt, N_GRID)
    k1 = S_Coupled_State(field_state=k1_d_state.field_state, gr_state=stabilized_k1_gr)

    # K2
    k2_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt / 2.0, state, k1)
    k2_d_state = deriv_func(k2_state, params, coupling_params, spec, N_GRID)
    stabilized_k2_gr = apply_stabilization_mandates(k2_d_state.gr_state, dt, N_GRID)
    k2 = S_Coupled_State(field_state=k2_d_state.field_state, gr_state=stabilized_k2_gr)

    # K3
    k3_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt / 2.0, state, k2)
    k3_d_state = deriv_func(k3_state, params, coupling_params, spec, N_GRID)
    stabilized_k3_gr = apply_stabilization_mandates(k3_d_state.gr_state, dt, N_GRID)
    k3 = S_Coupled_State(field_state=k3_d_state.field_state, gr_state=stabilized_k3_gr)

    # K4
    k4_state = jax.tree_util.tree_map(lambda y, dy: y + dy * dt, state, k3)
    k4_d_state = deriv_func(k4_state, params, coupling_params, spec, N_GRID)
    stabilized_k4_gr = apply_stabilization_mandates(k4_d_state.gr_state, dt, N_GRID)
    k4 = S_Coupled_State(field_state=k4_d_state.field_state, gr_state=stabilized_k4_gr)

    # Combined Update
    new_state = jax.tree_util.tree_map(
        lambda y, dy1, dy2, dy3, dy4: y + (dt / 6.0) * (dy1 + 2.0*dy2 + 2.0*dy3 + dy4),
        state, k1, k2, k3, k4
    )
    return new_state


print("SUCCESS: V8.0 Physics Engine functions defined (GR-Coupled EOMs).")


# --- CELL 6: V8.0 CERTIFIED EXECUTION FUNCTION (Coupled State Management) ---

# NOTE: The outer jit in run_simulation_with_io handles the static num_rays implicitly
def jnp_sncgl_conformal_step(
    carry_state: S_Coupled_State, # <- Now S_Coupled_State
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
) -> (S_Coupled_State, dict): # <- Returns S_Coupled_State
    """Master step function for V8.0 (to be JIT-compiled by lax.scan)."""
    state = carry_state
    DT = params.DT
    N_GRID = params.N_GRID # Used for stabilization/RK4

    # The RK4 step now operates on the S_Coupled_State
    new_state = rk4_step(state, DT, deriv_func, params, coupling_params, spec, N_GRID)
    new_psi = new_state.field_state.psi
    new_rho = jnp.abs(new_psi)**2

    # 2D ANALYSIS (Field Metrics)
    k_bins, power_spectrum = compute_directional_spectrum(new_psi, params, spec)
    ln_p_sse = compute_log_prime_sse(k_bins, power_spectrum, spec)
    informational_entropy = jnp_calculate_entropy(new_rho)
    quantule_census = jnp_calculate_quantule_census(new_rho)

    # GR Metric Tracking
    lapse_center = new_state.gr_state.lapse[N_GRID // 2, N_GRID // 2, N_GRID // 2]

    metrics = {
        "timestamp": t * DT,
        "ln_p_sse": ln_p_sse,
        "informational_entropy": informational_entropy,
        "quantule_census": quantule_census,
        "lapse_center": lapse_center
    }
    return new_state, metrics

def run_simulation_with_io(
    fmia_params: S_NCGL_Params,
    coupling_params: S_Coupling_Params,
    initial_coupled_state: S_Coupled_State, # <- New initial state type
    spec_ops: SpecOps,
    output_filename="simulation_output.hdf5",
    log_every_n=10
) -> Tuple:
    """
    Orchestrates the S-NCGL + GR simulation (V8.0), managing JIT compilation
    and I/O with the updated HDF5Logger.
    """
    print("--- Starting Orchestration (V8.0 - GR-Coupled 3D) ---")

    # 1. Setup simulation parameters
    total_steps = int(fmia_params.T_TOTAL / fmia_params.DT)
    log_steps = total_steps // log_every_n
    if log_steps == 0:
        log_steps = 1

    initial_carry = initial_coupled_state # <- Use coupled state
    print(f"Total Steps: {total_steps}, Logging every {log_every_n} steps, Log Steps: {log_steps}")

    # 2. Create the partial function
    step_fn_partial = functools.partial(
        jnp_sncgl_conformal_step,
        deriv_func=jnp_get_derivatives,
        params=fmia_params,
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

    # 4. Initialize the Logger (V8.0 logger handles GR metrics)
    metrics_to_log = ["timestamp", "ln_p_sse", "informational_entropy", "quantule_census"]
    logger = HDF5Logger(output_filename, log_steps, fmia_params.N_GRID, metrics_to_log)
    print(f"HDF5Logger initialized. Output file: {output_filename}")

    # 5. Run the Main Simulation Loop
    print("--- Starting Simulation Loop (V8.0: S-NCGL + GR Co-Evolution) [3D] ---")
    start_time = time.time()
    current_carry = initial_carry

    for i in tqdm(range(log_steps), desc="V8.0 Sim Progress"):
        try:
            final_carry_state, metrics_chunk = jit_scan_chunk(current_carry, None)

            last_metrics_in_chunk = {
                key: metrics_chunk[key][-1]
                for key in metrics_to_log
            }
            last_metrics_in_chunk['lapse_center'] = metrics_chunk['lapse_center'][-1]

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
    logger.save_final_state(current_carry, fmia_params.N_GRID) # <- Pass coupled state
    logger.close()

    import numpy as _np
    _psi_bytes = _np.asarray(current_carry.field_state.psi).tobytes()
    print(f"Final field state (psi hash): {hash(_psi_bytes})")

    return current_carry, output_filename, True


# --- CELL 7: V8.0 "WORKER" LOGIC ---
# ... (Worker logic for param management remains the same, but now initializes
# and returns S_Coupled_State instead of S_NCGL_State)

def generate_param_hash(params: Dict[str, Any]) -> str:
    """Generates a unique hash for a given set of parameters."""
    # Ensure consistent order by sorting items before hashing
    param_string = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_string.encode('utf-8')).hexdigest()

def write_to_ledger(ledger_file: str, run_data: Dict[str, Any]):
    """Appends a single simulation run's data to a CSV ledger."""
    # Check if file exists to write header
    file_exists = os.path.exists(ledger_file)

    with open(ledger_file, 'a', newline='') as f:
        fieldnames = run_data.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(run_data)

def load_todo_list(todo_file: str) -> List[Dict[str, Any]]:
    """Loads the list of parameter sets (jobs) from the todo JSON file."""
    try:
        with open(todo_file, 'r') as f:
            todo_data = json.load(f)
        if isinstance(todo_data, dict) and 'population' in todo_data:
            return todo_data['population']
        elif isinstance(todo_data, list):
            return todo_data
        else:
            print(f"Warning: Unexpected structure in todo file: {todo_file}")
            return []
    except FileNotFoundError:
        print(f"Todo file not found: {todo_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from todo file: {todo_file}")
        return []

def generate_bootstrap_jobs(
    rng: np.random.Generator, num_jobs: int
) -> List[Dict[str, Any]]:
    """Generates a list of random parameter sets for initial exploration."""
    # This function is not used by the orchestrator, which generates its own todo list.
    # It's here for completeness if a standalone worker needed bootstrapping.
    pass

def run_worker_main(hunt_id, todo_file):
    """This is the main "Worker" function that the orchestrator calls (V8.0)."""
    MASTER_OUTPUT_DIR = os.path.join("sweep_runs", hunt_id)
    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
    LEDGER_FILE = os.path.join(MASTER_OUTPUT_DIR, f"ledger_{hunt_id}.csv")
    LOG_EVERY_N_STEPS = 100 # Logging frequency

    print(f"[WORKER] Starting for Hunt ID: {hunt_id}")
    print(f"[WORKER] Output directory: {MASTER_OUTPUT_DIR}")
    print(f"[WORKER] Ledger file: {LEDGER_FILE}")

    params_to_run = load_todo_list(todo_file)
    if not params_to_run:
        print("[WORKER] No jobs in TODO list. Exiting.")
        return

    # Basic parameters for all runs (can be overridden by variable_params)
    N_GRID = 32
    T_TOTAL = 10.0
    DT = 0.01
    L_DOMAIN = 1.0
    NUM_RAYS = 1  # For compute_directional_spectrum
    K_BIN_WIDTH = 0.1
    K_MAX_PLOT = 5.0

    # Generate spectral operators once
    kx, ky, kz, k_sq, k_mag, k_max_sim, dealias_mask, xx, yy, zz = kgrid_2pi(N_GRID, L_DOMAIN)
    k_values_1d = jnp.fft.fftfreq(N_GRID, d=L_DOMAIN/N_GRID) * 2.0 * jnp.pi
    sort_indices_1d = jnp.argsort(k_values_1d)
    prime_targets_k = jnp.array([1.0, 2.0, 3.0]) # Example targets
    k_bins = jnp.arange(0, K_MAX_PLOT, K_BIN_WIDTH)
    ray_angles = jnp.array([0.0]) # Not used for 3D slice, but to satisfy SpecOps

    spec_ops = SpecOps(
        kx=kx, ky=ky, kz=kz, k_sq=k_sq, gaussian_kernel_k=jnp.zeros_like(k_sq),
        dealias_mask=dealias_mask, prime_targets_k=prime_targets_k, k_bins=k_bins,
        ray_angles=ray_angles, k_max=k_max_sim, xx=xx, yy=yy, zz=zz,
        k_values_1d=k_values_1d, sort_indices_1d=sort_indices_1d
    )

    print(f"[WORKER] Processing {len(params_to_run)} jobs...")

    # Setup JAX RNG key
    key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32 - 1))

    # --- Job Loop ---
    for i, job_data in enumerate(tqdm(params_to_run, desc="Worker Jobs")):
        job_id = job_data['id']
        variable_params = job_data['params']

        run_params = {
            "N_GRID": N_GRID,
            "T_TOTAL": T_TOTAL,
            "DT": DT,
            "l_domain": L_DOMAIN,
            "num_rays": NUM_RAYS,
            "k_bin_width": K_BIN_WIDTH,
            "k_max_plot": K_MAX_PLOT,
            **variable_params # Overwrite defaults with job-specific params
        }

        # Assemble the V8.0 JAX Pytrees (Structs) and Initial State
        try:
            fmia_params = S_NCGL_Params(
                N_GRID=run_params["N_GRID"], T_TOTAL=run_params["T_TOTAL"], DT=run_params["DT"],
                alpha=run_params["alpha"], beta=1.0, gamma=1.0,
                KAPPA=run_params["KAPPA"], nu=run_params["nu"], sigma_k=run_params["sigma_k"],
                l_domain=run_params["l_domain"], num_rays=run_params["num_rays"],
                k_bin_width=run_params["k_bin_width"], k_max_plot=run_params["k_max_plot"]
            )
            coupling_params = S_Coupling_Params(OMEGA_PARAM_A=run_params["OMEGA_PARAM_A"])

            # Update spec_ops for gaussian_kernel_k based on current sigma_k
            current_gaussian_kernel_k = make_gaussian_kernel_k(spec_ops.k_sq, fmia_params.sigma_k)
            spec_ops_for_run = spec_ops._replace(gaussian_kernel_k=current_gaussian_kernel_k)

            # V8.0: Initial Field State (S_NCGL_State)
            key, subkey = jax.random.split(key)
            psi_initial = (
                jax.random.uniform(subkey, (N_GRID, N_GRID, N_GRID), dtype=jnp.float32) * 0.1 +
                1j * jax.random.uniform(subkey, (N_GRID, N_GRID, N_GRID), dtype=jnp.float32) * 0.1
            )
            initial_field_state = S_NCGL_State(psi=psi_initial.astype(jnp.complex64))

            # V8.0: Initial Geometry State (S_GR_State - Start in Minkowski)
            lapse_initial = jnp.ones((N_GRID, N_GRID, N_GRID), dtype=jnp.float32) # alpha = 1
            shift_initial = jnp.zeros((N_GRID, N_GRID, N_GRID, 3), dtype=jnp.float32) # beta^i = 0
            # Conformal metric is flat 3-space, stored as 6 unique components
            conformal_metric_initial = jnp.zeros((N_GRID, N_GRID, N_GRID, 6), dtype=jnp.float32)

            initial_gr_state = S_GR_State(
                lapse=lapse_initial,
                shift_vec=shift_initial,
                conformal_metric=conformal_metric_initial
            )

            # V8.0: Final Coupled Initial State
            initial_coupled_state = S_Coupled_State(
                field_state=initial_field_state,
                gr_state=initial_gr_state
            )

            param_hash = generate_param_hash(run_params)
            output_filename = os.path.join(MASTER_OUTPUT_DIR, f"run_{param_hash}.hdf5")

        except Exception as e:
            print(f"[WORKER] Error assembling params for job {job_id}: {e}")
            run_data = {
                'job_id': job_id,
                'param_hash': "ERROR",
                **run_params,
                'final_sse': 99999.0,
                'informational_entropy': 0.0,
                'quantule_census': 0.0,
                'lapse_center_final': 0.0,
                'success': False,
                'error_message': str(e)
            }
            write_to_ledger(LEDGER_FILE, run_data)
            continue

        # 4. Run the V8.0 Simulation
        final_sse = 99999.0
        informational_entropy = 0.0
        quantule_census = 0.0
        lapse_center_final = 0.0
        sim_success = False
        error_message = ""

        try:
            final_carry_state, output_file, sim_success = run_simulation_with_io(
                fmia_params,
                coupling_params,
                initial_coupled_state, # <- Pass the coupled state
                spec_ops_for_run,
                output_filename=output_filename,
                log_every_n=LOG_EVERY_N_STEPS
            )

            # 5. Get the Final Metrics (from HDF5 if necessary, or from final_carry_state)
            # For simplicity, let's re-calculate some final metrics or load from output.
            # Here we'll take them directly from the last state for efficiency.
            final_psi = final_carry_state.field_state.psi
            final_rho = jnp.abs(final_psi)**2
            
            k_bins, power_spectrum = compute_directional_spectrum(final_psi, fmia_params, spec_ops_for_run)
            final_sse = float(compute_log_prime_sse(k_bins, power_spectrum, spec_ops_for_run))
            informational_entropy = float(jnp_calculate_entropy(final_rho))
            quantule_census = float(jnp_calculate_quantule_census(final_rho))
            lapse_center_final = float(final_carry_state.gr_state.lapse[N_GRID // 2, N_GRID // 2, N_GRID // 2])

        except Exception as e:
            error_message = f"Simulation failed: {e}"
            sim_success = False
            print(f"[WORKER] Job {job_id} failed: {error_message}")
            # traceback.print_exc() # Uncomment for detailed debug

        # 6. Log results to master ledger
        run_data = {
            'job_id': job_id,
            'param_hash': param_hash,
            **run_params,
            'final_sse': final_sse,
            'informational_entropy': informational_entropy,
            'quantule_census': quantule_census,
            'lapse_center_final': lapse_center_final,
            'success': sim_success,
            'error_message': error_message
        }
        write_to_ledger(LEDGER_FILE, run_data)

    print(f"[WORKER] All jobs processed for Hunt ID: {hunt_id}")

# --- THIS IS THE NEW "MAIN" BLOCK ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python worker_v7.py <HUNT_ID> <TODO_FILE>")
        sys.exit(1)

    hunt_id_arg = sys.argv[1]
    todo_file_arg = sys.argv[2]

    # Set JAX to use 64-bit floats for higher precision
    jax.config.update("jax_enable_x64", True)

    try:
        run_worker_main(hunt_id_arg, todo_file_arg)
    except Exception as e:
        print(f"[WORKER] Unhandled exception in main: {e}")
        traceback.print_exc()
        sys.exit(1)

print("worker_v7.py successfully written as V8.0.")
