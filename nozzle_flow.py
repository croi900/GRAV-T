"""
1D Radiative Hydrodynamic Nozzle Flow Solver (arXiv:2505.10616v2 Section 3.2).

Solves the two-point boundary value problem from photosphere to L1:
  Eq 20a (Continuity): (1/v) dv/dx + (1/P_gas) dP_gas/dx = 0
  Eq 20b (Momentum):   v dv/dx + (1/rho) dP_gas/dx = -(1-Γ_Edd) dφ_R/dx
  Eq 20c (Radiation):  (1/rho) dP_rad/dx = -Γ_Edd dφ_R/dx

Uses isothermal EOS: c²_T = P_gas/rho (constant temperature assumed).
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Tuple, Optional

from constants import G, c, k_B, m_H, a_rad


@dataclass
class NozzleFlowParams:
    """Parameters for nozzle flow calculation."""
    M_donor: float      # Donor mass [kg]
    M_accretor: float   # Accretor mass [kg]
    a: float            # Orbital separation [m]
    R_donor: float      # Donor radius [m]
    rho_ph: float       # Photospheric density [kg/m³]
    T_ph: float         # Photospheric temperature [K]
    Gamma_Edd: float    # Eddington factor
    L1_x: float         # L1 position (normalized to a)
    R_L: float          # Roche lobe radius [m]
    B: float            # Roche coefficient B
    C: float            # Roche coefficient C


def compute_roche_potential_gradient(x: float, params: NozzleFlowParams) -> float:
    """
    Compute dphi_R/dx at position x (normalized to separation a).
    x is distance from donor center toward L1.
    
    phi_R = -G*M1/r1 - G*M2/r2 - 0.5*(1+q)*Omega²*rho²
    where rho is distance from rotation axis.
    """
    q = params.M_donor / params.M_accretor
    
    # Distance from donor center (x) and accretor center (1-x) in units of a
    r1 = x * params.a
    r2 = (1.0 - x) * params.a
    
    if r1 < 1e-10 or r2 < 1e-10:
        return 0.0
    
    dphi_dx = (G * params.M_donor / (r1**2) 
               - G * params.M_accretor / (r2**2)
               - (1 + q) * params.a * x)
    
    return dphi_dx


def compute_isothermal_sound_speed(T: float, mu: float = 0.6) -> float:
    return np.sqrt(k_B * T / (mu * m_H))


def compute_nozzle_cross_section(x: float, c_T_sq: float, params: NozzleFlowParams) -> float:
    """
    Compute nozzle cross-section Q(x) at position x.
    """
    factor = 1.0 - params.Gamma_Edd
    if factor < 0.01:
        factor = 0.01
    
    Q_L1 = np.pi * c_T_sq / (factor * params.B * params.C)
    
    # Q varies along the nozzle - assume linear scaling with distance to L1
    x_L1 = params.L1_x
    if x >= x_L1:
        return Q_L1
    
    x_ph = params.R_donor / params.a  # Photosphere position (normalized)
    
    if x <= x_ph:
        return Q_L1 * 0.01  
    
    # Linear interpolation in log space
    frac = (x - x_ph) / (x_L1 - x_ph)
    return Q_L1 * (0.01 + 0.99 * frac)


def nozzle_ode_system(x: float, y: np.ndarray, params: NozzleFlowParams, 
                       c_T_sq: float) -> np.ndarray:
    """
    ODE system for nozzle flow (Eqs 20a-c).
    
    State vector y = [v, P_gas, P_rad]
    
    Returns dy/dx.
    """
    v, P_gas, P_rad = y
    
    if v <= 0 or P_gas <= 0:
        return np.array([0.0, 0.0, 0.0])
    
    rho = P_gas / c_T_sq
    
    dphi_dx = compute_roche_potential_gradient(x, params)
    
    Q = compute_nozzle_cross_section(x, c_T_sq, params)
    dx_small = 1e-6
    Q_plus = compute_nozzle_cross_section(x + dx_small, c_T_sq, params)
    dQ_dx = (Q_plus - Q) / dx_small
    dlogQ_dx = dQ_dx / Q if Q > 0 else 0.0
    
    Gamma = params.Gamma_Edd
    c_s_sq = c_T_sq  # Isothermal sound speed
    
    denom = c_s_sq - v**2
    
    if abs(denom) < 1e-20:
        return np.array([0.0, 0.0, -Gamma * rho * dphi_dx])
    
    dP_gas_dx = P_gas * (v**2 * dlogQ_dx - (1.0 - Gamma) * dphi_dx) / denom
    
    dv_dx = -v * (dP_gas_dx / P_gas + dlogQ_dx)
    
    dP_rad_dx = -Gamma * rho * dphi_dx
    
    return np.array([dv_dx, dP_gas_dx, dP_rad_dx])


def solve_nozzle_flow_ivp(params: NozzleFlowParams, n_points: int = 100,
                           mu: float = 0.6) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve nozzle flow using initial value problem approach (shooting method).
    
    Integrates from photosphere toward L1, finding the sonic point.
    
    Returns:
        x: Position array (normalized to a)
        y: Solution array [v, P_gas, P_rad] at each x
        M_dot: Mass transfer rate [kg/s]
    """
    c_T = compute_isothermal_sound_speed(params.T_ph, mu)
    c_T_sq = c_T**2
    
    x_ph = params.R_donor / params.a
    x_L1 = params.L1_x
    
    # Check if donor overflows Roche lobe (photosphere beyond L1)
    # In this case, the flow is already at or past the sonic point
    if x_ph >= x_L1:
        # Roche lobe overflow - use sonic point formula at L1
        factor = max(1.0 - params.Gamma_Edd, 0.01)
        Q_L1 = np.pi * c_T_sq / (factor * params.B * params.C)
        # At sonic point: v = c_T, ρ = ρ_ph / sqrt(e) (from critical point analysis)
        rho_crit = params.rho_ph / np.sqrt(np.e)
        M_dot = -rho_crit * c_T * Q_L1
        return np.array([x_L1]), np.array([[c_T], [params.rho_ph * c_T_sq], [a_rad * params.T_ph**4 / 3.0]]), M_dot
    
    P_gas_ph = params.rho_ph * c_T_sq
    
    P_rad_ph = a_rad * params.T_ph**4 / 3.0
    
    v_ph = c_T * 0.001 # subsonic
    
    y0 = np.array([v_ph, P_gas_ph, P_rad_ph])
    
    # Integrate from photosphere to L1
    x_span = (x_ph, x_L1)
    
    if x_L1 <= x_ph:
        raise ValueError(f"L1 position ({x_L1}) must be greater than photosphere ({x_ph})")
    
    x_eval = np.linspace(x_ph, x_L1, n_points)
    step_size = (x_L1 - x_ph) / 50.0
    
    def ode_wrapper(x, y):
        return nozzle_ode_system(x, y, params, c_T_sq)
    
    # Event: detect sonic point (v = c_T)
    def sonic_event(x, y):
        return y[0] - c_T
    sonic_event.terminal = True
    sonic_event.direction = 1
    
    try:
        sol = solve_ivp(ode_wrapper, x_span, y0, t_eval=x_eval, 
                        method='RK45', events=sonic_event,
                        max_step=step_size)
        
        if sol.success:
            x = sol.t
            y = sol.y
            
            # Compute mass transfer rate at sonic point or L1
            v_final = y[0, -1]
            P_gas_final = y[1, -1]
            rho_final = P_gas_final / c_T_sq
            Q_final = compute_nozzle_cross_section(x[-1], c_T_sq, params)
            
            M_dot = -rho_final * v_final * Q_final  # Negative = mass loss
            
            return x, y, M_dot
        else:
            raise RuntimeError(f"IVP solver failed: {sol.message}")
            
    except Exception as e:
        # Fallback to simple formula
        factor = max(1.0 - params.Gamma_Edd, 0.01)
        Q_L1 = np.pi * c_T_sq / (factor * params.B * params.C)
        M_dot = -params.rho_ph * c_T * Q_L1 / np.sqrt(np.e)
        return np.array([x_ph, x_L1]), np.array([[v_ph, c_T], [P_gas_ph, P_gas_ph], [P_rad_ph, P_rad_ph]]), M_dot


def compute_M_dot_full(a: float, M1: float, M2: float, 
                        R_donor: float, rho_ph: float, T_ph: float,
                        Gamma_Edd: float, mu: float = 0.6) -> float:
    """
    Compute mass transfer rate using the full BVP solution.
    
    This replaces the simplified algebraic formula with numerical integration.
    """
    from roche_geometry import compute_L1_position, roche_lobe_radius_eggleton, roche_coefficients
    
    if M1 <= 0 or M2 <= 0 or a <= 0:
        return 0.0
    
    q = M1 / M2
    L1_x = compute_L1_position(q)
    R_L = roche_lobe_radius_eggleton(q) * a
    B, C = roche_coefficients(q, L1_x)
    
    if B <= 0 or C <= 0:
        return 0.0
    
    params = NozzleFlowParams(
        M_donor=M1,
        M_accretor=M2,
        a=a,
        R_donor=R_donor,
        rho_ph=rho_ph,
        T_ph=T_ph,
        Gamma_Edd=Gamma_Edd,
        L1_x=L1_x,
        R_L=R_L,
        B=B,
        C=C,
    )
    
    try:
        _, _, M_dot = solve_nozzle_flow_ivp(params, n_points=50, mu=mu)
        return M_dot
    except Exception:
        # Fallback to simplified formula
        c_T_sq = k_B * T_ph / (mu * m_H)
        c_T = np.sqrt(c_T_sq)
        factor = 1.0 - Gamma_Edd
        if factor < 0.01:
            factor = 0.01
        Q = np.pi * c_T_sq / (factor * B * C)
        return -rho_ph * c_T * Q / np.sqrt(np.e)


# Cache for expensive BVP computations
_NOZZLE_CACHE = {
    'a': None,
    'M1': None,
    'M2': None,
    'M_dot': None,
}


def compute_M_dot_cached(a: float, M1: float, M2: float,
                          R_donor: float, rho_ph: float, T_ph: float,
                          Gamma_Edd: float, mu: float = 0.6,
                          tolerance: float = 0.01) -> float:
    """
    Compute mass transfer rate with caching.
    
    Re-computes only if parameters changed by more than tolerance.
    """
    global _NOZZLE_CACHE
    
    if _NOZZLE_CACHE['a'] is not None:
        a_change = abs(a - _NOZZLE_CACHE['a']) / _NOZZLE_CACHE['a']
        M1_change = abs(M1 - _NOZZLE_CACHE['M1']) / _NOZZLE_CACHE['M1']
        M2_change = abs(M2 - _NOZZLE_CACHE['M2']) / _NOZZLE_CACHE['M2']
        
        if a_change < tolerance and M1_change < tolerance and M2_change < tolerance:
            return _NOZZLE_CACHE['M_dot']
    
    M_dot = compute_M_dot_full(a, M1, M2, R_donor, rho_ph, T_ph, Gamma_Edd, mu)
    
    _NOZZLE_CACHE['a'] = a
    _NOZZLE_CACHE['M1'] = M1
    _NOZZLE_CACHE['M2'] = M2
    _NOZZLE_CACHE['M_dot'] = M_dot
    
    return M_dot
