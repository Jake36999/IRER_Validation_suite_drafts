"""
gravity/unified_omega.py (Sprint 1 - Patched)
Single source of truth for the IRER Unified Gravity derivation.
Implements the analytical solution for the conformal factor Omega(rho)
and the emergent metric g_munu.
"""

import jax
import jax.numpy as jnp
from typing import Dict

@jax.jit
def jnp_derive_metric_from_rho(
    rho: jnp.ndarray,
    fmia_params: Dict,
    epsilon: float = 1e-10
) -> jnp.ndarray:
    """
    [THEORETICAL BRIDGE] Derives the emergent spacetime metric g_munu directly
    from the Resonance Density (rho) field.

    Implements the analytical solution: g_munu = Omega^2 * eta_munu
    Where Omega(rho) = (rho_val / rho)^(a/2)
    As derived in the Declaration of Intellectual Provenance (Section 5.3).
    """

    # Get parameters from the derivation using the correct param_* keys
    rho_vac = fmia_params.get('param_rho_vac', 1.0)
    a_coupling = fmia_params.get('param_a_coupling', 1.0)

    # Add stabilization (mask rho <= 0)
    rho_safe = jnp.maximum(rho, epsilon)

    # 1. Calculate Omega^2 = (rho_vac / rho)^a
    omega_squared = (rho_vac / rho_safe)**a_coupling

    # Clip the result to prevent NaN/Inf propagation
    omega_squared = jnp.clip(omega_squared, 1e-12, 1e12)

    # 2. Construct the 4x4xNxNxN metric
    grid_shape = rho.shape
    g_munu = jnp.zeros((4, 4) + grid_shape)

    # We assume eta_munu = diag(-1, 1, 1, 1)
    g_munu = g_munu.at[0, 0, ...].set(-omega_squared) # g_00
    g_munu = g_munu.at[1, 1, ...].set(omega_squared)  # g_xx
    g_munu = g_munu.at[2, 2, ...].set(omega_squared)  # g_yy
    g_munu = g_munu.at[3, 3, ...].set(omega_squared)  # g_zz

    return g_munu
