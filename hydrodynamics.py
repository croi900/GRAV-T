"""
Hydrodynamic mass transfer model (arXiv:2505.10616v2 Section 4.2).

Computes Ṁ dynamically from orbital separation and stellar properties.
State vector: [a, e, M1] - includes donor mass for proper evolution.
"""

from dataclasses import dataclass
import numpy as np
from numba import njit

from constants import G, c
from roche_geometry import (
    roche_lobe_radius_eggleton, 
    compute_L1_position, 
    roche_coefficients,
    eddington_factor
)


# Cache for hydro params (set by integration_run before ODE solving)
_HYDRO_CACHE = {
    'R_donor': 1e7,
    'rho_ph': 1e-6,
    'P_gas_ph': 1e4,
    'Gamma_Edd': 0.5,
    'M2_initial': 0.0,  # Initial accretor mass (for scaling)
    'T_ph': 10000.0,  # Photosphere temperature (K)
    'mu': 0.6,  # Mean molecular weight
    'use_full_bvp': False,  # Use full BVP solver
    'beta': 1.0,  # Accretion efficiency (1.0 = fully conservative)
}


def set_hydro_params(R_donor, rho_ph, P_gas_ph, L_rad, M_donor, kappa_R, 
                     Gamma_Edd_fixed=0.0, M2=0.0, T_ph=10000.0, mu=0.6, 
                     use_full_bvp=False, beta=1.0):
    """Set global hydro parameters before ODE integration.
    
    Args:
        beta: Accretion efficiency (0-1). 1.0 = fully conservative transfer.
    """
    _HYDRO_CACHE['R_donor'] = R_donor
    _HYDRO_CACHE['rho_ph'] = rho_ph
    _HYDRO_CACHE['P_gas_ph'] = P_gas_ph
    _HYDRO_CACHE['M2_initial'] = M2  # Store initial M2 for scaling
    _HYDRO_CACHE['T_ph'] = T_ph
    _HYDRO_CACHE['mu'] = mu
    _HYDRO_CACHE['use_full_bvp'] = use_full_bvp
    _HYDRO_CACHE['beta'] = beta  # Accretion efficiency
    
    if Gamma_Edd_fixed > 0:
        _HYDRO_CACHE['Gamma_Edd'] = Gamma_Edd_fixed
    else:
        _HYDRO_CACHE['Gamma_Edd'] = eddington_factor(L_rad, M_donor, kappa_R)


@njit(cache=True)
def _compute_sound_speed_sq(P_gas: float, rho: float) -> float:
    """Isothermal sound speed squared: c²_T = P_gas / ρ."""
    return P_gas / rho


@njit(cache=True)
def _compute_nozzle_cross_section(c_T_sq: float, Gamma_Edd: float, B: float, C: float) -> float:
    """Nozzle cross-section Q at L1 (Eq 31): Q = π*c²_T / [(1-Γ)*B*C]."""
    factor = 1.0 - Gamma_Edd
    if factor <= 0.01:
        factor = 0.01
    return np.pi * c_T_sq / (factor * B * C)


@njit(cache=True)
def _compute_M_dot_rlof(rho_ph: float, P_gas_ph: float, Gamma_Edd: float, 
                         B: float, C: float) -> float:
    """Mass transfer rate for Roche lobe overflow (Eq 32)."""
    c_T_sq = P_gas_ph / rho_ph
    c_T = np.sqrt(c_T_sq)
    Q = _compute_nozzle_cross_section(c_T_sq, Gamma_Edd, B, C)
    rho_crit = rho_ph / np.sqrt(np.e)
    v_crit = c_T
    return -Q * rho_crit * v_crit


@njit(cache=True)
def _compute_M_dot_underfilling(rho_ph: float, P_gas_ph: float, Gamma_Edd: float,
                                  B: float, C: float, M_donor: float, 
                                  R_donor: float, R_L: float) -> float:
    """Mass transfer rate for underfilling donors (Eq 36)."""
    if R_donor >= R_L:
        return _compute_M_dot_rlof(rho_ph, P_gas_ph, Gamma_Edd, B, C)
    
    c_T_sq = P_gas_ph / rho_ph
    c_T = np.sqrt(c_T_sq)
    Q = _compute_nozzle_cross_section(c_T_sq, Gamma_Edd, B, C)
    M_dot_0 = -Q * rho_ph * c_T
    
    G_val = 6.674e-11
    delta_phi = G_val * M_donor * (1.0 / R_donor - 1.0 / R_L)
    exponent = (1.0 - Gamma_Edd) * delta_phi / c_T_sq
    
    return M_dot_0 * (1.0 - Gamma_Edd) * np.exp(-exponent)


def compute_dynamic_M_dot(a: float, M1: float, M2: float) -> float:
    """Compute instantaneous mass transfer rate from orbital separation."""
    if M1 <= 0 or M2 <= 0 or a <= 0 or not np.isfinite(a) or not np.isfinite(M1):
        return 0.0
    
    R_donor = _HYDRO_CACHE['R_donor']
    rho_ph = _HYDRO_CACHE['rho_ph']
    P_gas_ph = _HYDRO_CACHE['P_gas_ph']
    Gamma_Edd = _HYDRO_CACHE['Gamma_Edd']
    
    q = M1 / M2
    if q <= 0 or not np.isfinite(q):
        return 0.0
    
    L1_x = compute_L1_position(q)
    R_L = roche_lobe_radius_eggleton(q) * a
    B, C = roche_coefficients(q, L1_x)
    
    if B <= 0 or C <= 0 or not np.isfinite(B) or not np.isfinite(C):
        return 0.0
    
    # Use full BVP solver if enabled
    if _HYDRO_CACHE.get('use_full_bvp', False):
        try:
            from nozzle_flow import compute_M_dot_cached
            T_ph = _HYDRO_CACHE.get('T_ph', 10000.0)
            mu = _HYDRO_CACHE.get('mu', 0.6)
            return compute_M_dot_cached(
                a, M1, M2, R_donor, rho_ph, T_ph, Gamma_Edd, mu
            )
        except Exception as e:
            # Fallback to simplified formula
            print(f"BVP solver failed, using simplified formula: {e}")
    
    # Simplified formula (Eqs 32/36)
    if R_donor >= R_L:
        return _compute_M_dot_rlof(rho_ph, P_gas_ph, Gamma_Edd, B, C)
    else:
        return _compute_M_dot_underfilling(
            rho_ph, P_gas_ph, Gamma_Edd, B, C, M1, R_donor, R_L
        )


def compute_derivs_hydro(t, y, M_c1, M_c2, hydro_params):
    """
    ODE derivatives with DYNAMIC Ṁ - evolves [a, e, M1, M2].
    
    Conservative mass transfer: mass lost by M1 is gained by M2.
    State vector y = [a, e, M1, M2] where:
        M1 = donor mass (decreasing)
        M2 = accretor mass (increasing for conservative transfer)
    """
    from equations import _combine_scalings, _dadt, _dedt
    
    # Check state vector size
    if len(y) == 2:
        # Old 2-variable mode (a, e only) - use initial masses
        a = y[0]
        e = min(max(y[1], 0.0), 1.0 - 1e-8)
        M1 = M_c1
        M2 = M_c2
        evolve_mass = False
    elif len(y) == 3:
        # 3-variable mode [a, e, M1] - backwards compatibility
        a = y[0]
        e = min(max(y[1], 0.0), 1.0 - 1e-8)
        M1 = max(y[2], 1e-10)
        M2 = _HYDRO_CACHE['M2_initial']  # Constant
        evolve_mass = 'donor_only'
    else:
        # New 4-variable mode [a, e, M1, M2] - conservative transfer
        a = y[0]
        e = min(max(y[1], 0.0), 1.0 - 1e-8)
        M1 = max(y[2], 1e-10)  # Current donor mass
        M2 = max(y[3], 1e-10)  # Current accretor mass
        evolve_mass = 'both'
    
    M_c = M1 + M2
    
    # Compute dynamic mass transfer rate
    M_dot = compute_dynamic_M_dot(a, M1, M2)  # Negative: mass leaving donor
    
    # Get accretion efficiency
    beta = _HYDRO_CACHE.get('beta', 1.0)
    
    # Mass scaling for M1 (donor - losing mass)
    f1 = 1.0  # Using actual masses, no initial scaling
    df1 = M_dot / M1 if M1 > 1e-10 else 0.0  # Negative
    d2f1 = 0.0  # Assume constant rate for now
    d3f1 = 0.0
    
    # Mass scaling for M2 (accretor - gaining mass)
    f2 = 1.0
    # M2 gains mass at rate beta * |M_dot| = -beta * M_dot (since M_dot < 0)
    dM2_dt = -beta * M_dot  # Positive for M_dot < 0
    df2 = dM2_dt / M2 if M2 > 1e-10 else 0.0  # Positive (mass increasing)
    d2f2 = 0.0
    d3f2 = 0.0
    
    f_M, df_M, d2f_M, f_mu, df_mu, d2f_mu, d3f_mu = _combine_scalings(
        M1, M2, M_c, f1, df1, d2f1, d3f1, f2, df2, d2f2, d3f2
    )
    
    mu_c = (M1 * M2) / M_c
    
    dadt_val = _dadt(
        mu_c, M_c, M1, M2,
        f1, f2, df1, df2,
        f_M, f_mu, df_M, df_mu,
        d2f_M, d2f_mu, d3f_mu,
        a, e,
    )
    
    dedt_val = _dedt(
        M1, M2, M_c, mu_c,
        f1, f2, df1, df2,
        f_M, df_M, f_mu, df_mu,
        d2f_M, d2f_mu, d3f_mu,
        a, e,
    )
    
    if evolve_mass == 'both':
        # Conservative mass transfer: dM2/dt = -beta * M_dot
        return np.array([dadt_val, dedt_val, M_dot, dM2_dt], dtype=np.float64)
    elif evolve_mass == 'donor_only':
        return np.array([dadt_val, dedt_val, M_dot], dtype=np.float64)
    else:
        return np.array([dadt_val, dedt_val], dtype=np.float64)


# Legacy functions for compatibility
@njit(cache=True)
def _hydro_mass_scaling(t: float, M_dot: float, M_initial: float) -> tuple:
    """Compute mass scaling from constant M_dot approximation."""
    f = 1.0 + (M_dot * t) / M_initial
    if f < 0.0:
        f = 0.0
    df = M_dot / M_initial
    return f, df, 0.0, 0.0


class HydroMassDecay:
    """Hydrodynamic mass decay (placeholder for compatibility)."""
    def __init__(self, M_dot, M_initial):
        self.M_dot = M_dot
        self.M_initial = M_initial
    def value(self, t):
        return max(1.0 + (self.M_dot * t) / self.M_initial, 0.0)
    def first(self, t):
        return self.M_dot / self.M_initial
    def second(self, t):
        return 0.0
    def third(self, t):
        return 0.0


def make_hydro_mass_function(M_dot, M_initial):
    """Factory for hydrodynamic mass function."""
    from equations import AnalyticMassFunction
    decay = HydroMassDecay(M_dot, M_initial)
    return AnalyticMassFunction(decay.value, decay.first, decay.second, decay.third)
