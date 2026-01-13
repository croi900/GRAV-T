from __future__ import annotations

import numpy as np
from numba import njit, cfunc
from NumbaQuadpack import dqags, quadpack_sig

# Solar mass in kg (approx)
M_SUN_KG = 1.9891e30

# =============================================================================
# JIT-compiled helper functions for Lander mass decay model
# These are the core performance-critical functions that get called many times
# =============================================================================

@njit(cache=True)
def _L_nu(t: float) -> float:
    """
    Neutrino luminosity L_nu / 10^52 erg/s
    """
    if t < 50:
        return 0.7 * np.exp(-t / 1.5) + 0.3 * (1 - t / 50)**4
    else:
        return 0.7 * np.exp(-t / 1.5)


@njit(cache=True)
def _E_nu(t: float) -> float:
    """
    Neutrino energy E_nu / 10 MeV
    """
    part1 = 0.3 * np.exp(-t / 4)
    if t < 60:
        part2 = 1 - t / 60
        return part1 + part2
    else:
        return part1


@njit(cache=True)
def _M_dot_normalized(t: float) -> float:
    """
    Normalized mass loss rate (dimensionless)
    """
    l_norm = _L_nu(t)
    e_norm = _E_nu(t)
    
    if l_norm < 0:
        l_norm = 0.0
    if e_norm < 0:
        e_norm = 0.0
    
    val = (l_norm)**(5/3) * (e_norm)**(10/3)
    return val


@njit(cache=True)
def _f_dot_val(t: float) -> float:
    """
    df/dt = k * M_dot_normalized / 2.8
    This is the integrand for computing f(t) = 1 + integral(f_dot, 0, t)
    """
    k = -6.8e-5
    return k * _M_dot_normalized(t) / 2.8


@cfunc(quadpack_sig)
def _f_dot_cfunc(t, data):
    """
    C-function wrapper for _f_dot_val to be used with dqags.
    """
    return _f_dot_val(t)


# Get the function address at module level (python time) so it can be used in JIT
FUNC_PTR = _f_dot_cfunc.address


@njit(cache=True)
def _integrate_f_dot(t: float) -> float:
    """
    Compute integral of _f_dot_val from 0 to t using dqags from NumbaQuadpack.
    """
    if t <= 0:
        return 0.0
    
    # Use the pre-computed address
    sol, _, _ = dqags(FUNC_PTR, 0.0, t, epsabs=1e-9, epsrel=1e-9)
    return sol


@njit(cache=True)
def _f_value(t: float) -> float:
    """
    f(t) = 1 + integral(_f_dot_val, 0, t)
    """
    return 1.0 + _integrate_f_dot(t)


@njit(cache=True)
def _f_first(t: float) -> float:
    """First derivative df/dt"""
    return _f_dot_val(t)


@njit(cache=True)
def _f_second(t: float) -> float:
    """Second derivative d²f/dt²"""
    C = -6.8e-5 / 2.8
    
    L = _L_nu(t)
    if L < 1e-9:
        L = 0.0
    
    E = _E_nu(t)
    if E < 1e-9:
        E = 0.0
    
    # L' derivative
    L_prime = 0.7 * (-1/1.5) * np.exp(-t/1.5)
    if t < 50:
        L_prime += 1.2 * (1 - t/50)**3 * (-0.02)
    
    # E' derivative
    E_prime = 0.3 * (-0.25) * np.exp(-t/4)
    if t < 60:
        E_prime -= 1/60
    
    term1 = 0.0
    term2 = 0.0
    
    if L > 0 and E > 0:
        term1 = (5/3) * (L**(2/3)) * L_prime * (E**(10/3))
        term2 = (L**(5/3)) * (10/3) * (E**(7/3)) * E_prime
    
    return C * (term1 + term2)


@njit(cache=True)
def _f_third(t: float) -> float:
    """Third derivative d³f/dt³"""
    C = -6.8e-5 / 2.8
    
    L = _L_nu(t)
    if L < 1e-9:
        L = 0.0
    E = _E_nu(t)
    if E < 1e-9:
        E = 0.0
    
    # L' and L'' derivatives
    L_prime = 0.7 * (-1/1.5) * np.exp(-t/1.5)
    if t < 50:
        L_prime += 1.2 * (1 - t/50)**3 * (-0.02)
    
    L_double = 0.7 * (1/2.25) * np.exp(-t/1.5)
    if t < 50:
        L_double += 3.6 * (1 - t/50)**2 * 0.0004
    
    # E' and E'' derivatives
    E_prime = 0.3 * (-0.25) * np.exp(-t/4)
    if t < 60:
        E_prime -= 1/60
    
    E_double = 0.3 * (1/16) * np.exp(-t/4)
    
    if L <= 1e-9 or E <= 1e-9:
        return 0.0
    
    T1_part1 = (2/3) * (L**(-1/3)) * (L_prime**2) * (E**(10/3))
    T1_part2 = (L**(2/3)) * L_double * (E**(10/3))
    T1_part3 = (L**(2/3)) * L_prime * (10/3) * (E**(7/3)) * E_prime
    
    dT1 = (5/3) * (T1_part1 + T1_part2 + T1_part3)
    
    T2_part1 = (5/3) * (L**(2/3)) * L_prime * (E**(7/3)) * E_prime
    T2_part2 = (L**(5/3)) * (7/3) * (E**(4/3)) * (E_prime**2)
    T2_part3 = (L**(5/3)) * (E**(7/3)) * E_double
    
    dT2 = (10/3) * (T2_part1 + T2_part2 + T2_part3)
    
    return C * (dT1 + dT2)


@njit(cache=True)
def _eval_lander_mass(t: float, mass_coef: float):
    """
    JIT-compiled evaluation of Lander mass function and derivatives.
    mass_coef is ignored (Lander model doesn't use it).
    """
    f = _f_value(t)
    df = _f_first(t)
    d2f = _f_second(t)
    d3f = _f_third(t)
    return f, df, d2f, d3f


# =============================================================================
# Class-based API for backward compatibility
# =============================================================================

class LanderMassDecay:
    """
    Lander mass decay model.
    This class wraps the JIT-compiled functions for use with AnalyticMassFunction.
    """
    def __init__(self, mass_coef: float):
        self.mass_coef = mass_coef  # Lander doesn't use mass_coef, fix to 1

    def value(self, t: float) -> float:
        return _f_value(t)

    def first(self, t: float) -> float:
        return _f_first(t)

    def second(self, t: float) -> float:
        return _f_second(t)

    def third(self, t: float) -> float:
        return _f_third(t)


# =============================================================================
# compute_derivs_lander - uses late import to avoid circular dependency
# This function is NOT @njit because it needs to import from equations.py
# However, the mass evaluation (_eval_lander_mass) IS JIT-compiled, which is
# where most of the computation time was being spent (scipy.quad calls)
# =============================================================================

def compute_derivs_lander(t, y, M_c1, M_c2, mass_coef):
    """
    Derivative computation for Lander mass decay model.
    
    Note: This function is NOT JIT-compiled because it imports from equations.py
    which has a circular dependency. However, _eval_lander_mass IS JIT-compiled,
    which eliminates the scipy.quad bottleneck.
    """
    from equations import _combine_scalings, _dadt, _dedt
    
    a = y[0]
    e = min(max(y[1], 0.0), 1.0 - 1e-8)
    # M_c1 *= mass_coef
    M_c = M_c1 + M_c2

    f1, df1, d2f1, d3f1 = _eval_lander_mass(t, mass_coef)
    f2, df2, d2f2, d3f2 = f1, df1, d2f1, d3f1

    if f1 <= 1e-8 or f2 <= 1e-8:
        return np.zeros(2)

    f_M, df_M, d2f_M, f_mu, df_mu, d2f_mu, d3f_mu = _combine_scalings(
        M_c1, M_c2, M_c, f1, df1, d2f1, d3f1, f2, df2, d2f2, d3f2
    )

    mu_c = (M_c1 * M_c2) / M_c

    dadt_val = _dadt(
        mu_c, M_c, M_c1, M_c2,
        f1, f2, df1, df2,
        f_M, f_mu, df_M, df_mu,
        d2f_M, d2f_mu, d3f_mu,
        a, e,
    )

    dedt_val = _dedt(
        M_c1, M_c2, M_c, mu_c,
        f1, f2, df1, df2,
        f_M, df_M, f_mu, df_mu,
        d2f_M, d2f_mu, d3f_mu,
        a, e,
    )

    return np.array([dadt_val, dedt_val], dtype=np.float64)


# Late import for AnalyticMassFunction to avoid circular dependency
def make_lander_mass_function(mass_coef: float):
    from equations import AnalyticMassFunction
    decay = LanderMassDecay(mass_coef)
    return AnalyticMassFunction(decay.value, decay.first, decay.second, decay.third)
