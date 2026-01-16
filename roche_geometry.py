"""Roche geometry calculations for binary mass transfer (from arXiv:2505.10616v2)."""

import numpy as np
from numba import njit
from scipy.optimize import brentq

from constants import G


@njit(cache=True)
def roche_lobe_radius_eggleton(q: float) -> float:
    """
    Eggleton (1983) approximation for Roche lobe radius.
    Returns R_L/a (normalized to orbital separation).
    q = M_donor / M_accretor (mass ratio)
    """
    q_cbrt = q ** (1.0 / 3.0)
    q_2_3 = q ** (2.0 / 3.0)
    return 0.49 * q_2_3 / (0.6 * q_2_3 + np.log(1.0 + q_cbrt))


def compute_L1_position(q: float) -> float:
    """
    Compute L1 position from donor (normalized to separation a).
    Solves the quintic equation for L1 point.
    q = M_donor / M_accretor
    """
    # Guard against invalid mass ratios
    if q <= 0 or q > 1e6 or not np.isfinite(q):
        return 0.5  # Return midpoint as fallback
    
    def L1_equation(x):
        # x is distance from donor center, normalized to a
        return (1.0 / (x ** 2) 
                - q / ((1.0 - x) ** 2) 
                - (1.0 + q) * x 
                + q)
    
    # L1 is between donor and accretor
    x_min = 0.01
    x_max = 0.99
    
    try:
        return brentq(L1_equation, x_min, x_max)
    except ValueError:
        # Fallback if root finding fails
        return 0.5


@njit(cache=True)
def roche_potential(x: float, y: float, z: float, q: float) -> float:
    """
    Roche potential at point (x,y,z) normalized to separation a.
    Origin at donor center, accretor at x=1.
    """
    r1 = np.sqrt(x**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0)**2 + y**2 + z**2)
    
    if r1 < 1e-10 or r2 < 1e-10:
        return -1e30
    
    omega = -1.0 / r1 - q / r2 - 0.5 * (1.0 + q) * (x**2 + y**2)
    return omega


@njit(cache=True)
def roche_coefficients(q: float, L1_x: float) -> tuple:
    """
    Compute B and C coefficients for Roche potential expansion near L1.
    phi_R ≈ phi_1 + B*y^2 + C*z^2 (Eq 29)
    """
    r1 = L1_x
    r2 = 1.0 - L1_x
    
    B = 1.0 / (r1 ** 3) + q / (r2 ** 3) - (1.0 + q)
    
    C = 1.0 / (r1 ** 3) + q / (r2 ** 3)
    
    return B, C


def compute_nozzle_cross_section(c_T_gas_sq: float, Gamma_Edd: float, 
                                  B: float, C: float) -> float:
    """
    Compute cross-section Q at L1 (Eq 31).
    Q = pi * c^2_T,gas / [(1 - Γ_Edd) * B * C]
    """
    if Gamma_Edd >= 1.0:
        return np.inf  # Diverges at Eddington limit
    
    return np.pi * c_T_gas_sq / ((1.0 - Gamma_Edd) * B * C)


def compute_potential_difference(a: float, M_donor: float, R_donor: float, 
                                  R_L: float) -> float:
    """
    Compute potential difference (phi_1 - phi_ph) for underfilling donors.
    Returns value in units of c²_T,gas when normalized.
    """
    # phi_1 - phi_ph ≈ G*M_donor * (1/R_donor - 1/R_L)
    if R_donor >= R_L:
        return 0.0  # Overflow case
    
    return G * M_donor * (1.0 / R_donor - 1.0 / R_L)


@njit(cache=True)
def eddington_factor(L_rad: float, M: float, kappa_R: float) -> float:
    """
    Compute Eddington factor Γ_Edd = L_rad / L_Edd.
    kappa_R is Rosseland mean opacity in m^2/kg.
    """
    c_light = 2.998e8
    L_Edd = 4.0 * np.pi * c_light * G * M / kappa_R
    return L_rad / L_Edd
