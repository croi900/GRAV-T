from __future__ import annotations
import numpy as np
from numba import njit, cfunc
from NumbaQuadpack import dqags, quadpack_sig

M_SUN_KG = 1.9891e30


@njit(cache=True)
def _L_nu(t: float) -> float:
    if t < 50:
        return 0.7 * np.exp(-t / 1.5) + 0.3 * (1 - t / 50)**4
    else:
        return 0.7 * np.exp(-t / 1.5)


@njit(cache=True)
def _E_nu(t: float) -> float:
    part1 = 0.3 * np.exp(-t / 4)
    if t < 60:
        part2 = 1 - t / 60
        return part1 + part2
    else:
        return part1


@njit(cache=True)
def _M_dot_normalized(t: float) -> float:
    l_norm = _L_nu(t)
    e_norm = _E_nu(t)
    if l_norm < 0:
        l_norm = 0.0
    if e_norm < 0:
        e_norm = 0.0
    val = l_norm**(5 / 3) * e_norm**(10 / 3)
    return val


@njit(cache=True)
def _f_dot_val(t: float) -> float:
    k = -6.8e-05
    return k * _M_dot_normalized(t) / 2.8


@cfunc(quadpack_sig)
def _f_dot_cfunc(t, data):
    return _f_dot_val(t)


FUNC_PTR = _f_dot_cfunc.address


@njit(cache=True)
def _integrate_f_dot(t: float) -> float:
    if t <= 0:
        return 0.0
    sol, _, _ = dqags(FUNC_PTR, 0.0, t, epsabs=1e-09, epsrel=1e-09)
    return sol


@njit(cache=True)
def _f_value(t: float) -> float:
    return 1.0 + _integrate_f_dot(t)


@njit(cache=True)
def _f_first(t: float) -> float:
    return _f_dot_val(t)


@njit(cache=True)
def _f_second(t: float) -> float:
    C = -6.8e-05 / 2.8
    L = _L_nu(t)
    if L < 1e-09:
        L = 0.0
    E = _E_nu(t)
    if E < 1e-09:
        E = 0.0
    L_prime = 0.7 * (-1 / 1.5) * np.exp(-t / 1.5)
    if t < 50:
        L_prime += 1.2 * (1 - t / 50)**3 * -0.02
    E_prime = 0.3 * -0.25 * np.exp(-t / 4)
    if t < 60:
        E_prime -= 1 / 60
    term1 = 0.0
    term2 = 0.0
    if L > 0 and E > 0:
        term1 = 5 / 3 * L**(2 / 3) * L_prime * E**(10 / 3)
        term2 = L**(5 / 3) * (10 / 3) * E**(7 / 3) * E_prime
    return C * (term1 + term2)


@njit(cache=True)
def _f_third(t: float) -> float:
    C = -6.8e-05 / 2.8
    L = _L_nu(t)
    if L < 1e-09:
        L = 0.0
    E = _E_nu(t)
    if E < 1e-09:
        E = 0.0
    L_prime = 0.7 * (-1 / 1.5) * np.exp(-t / 1.5)
    if t < 50:
        L_prime += 1.2 * (1 - t / 50)**3 * -0.02
    L_double = 0.7 * (1 / 2.25) * np.exp(-t / 1.5)
    if t < 50:
        L_double += 3.6 * (1 - t / 50)**2 * 0.0004
    E_prime = 0.3 * -0.25 * np.exp(-t / 4)
    if t < 60:
        E_prime -= 1 / 60
    E_double = 0.3 * (1 / 16) * np.exp(-t / 4)
    if L <= 1e-09 or E <= 1e-09:
        return 0.0
    T1_part1 = 2 / 3 * L**(-1 / 3) * L_prime**2 * E**(10 / 3)
    T1_part2 = L**(2 / 3) * L_double * E**(10 / 3)
    T1_part3 = L**(2 / 3) * L_prime * (10 / 3) * E**(7 / 3) * E_prime
    dT1 = 5 / 3 * (T1_part1 + T1_part2 + T1_part3)
    T2_part1 = 5 / 3 * L**(2 / 3) * L_prime * E**(7 / 3) * E_prime
    T2_part2 = L**(5 / 3) * (7 / 3) * E**(4 / 3) * E_prime**2
    T2_part3 = L**(5 / 3) * E**(7 / 3) * E_double
    dT2 = 10 / 3 * (T2_part1 + T2_part2 + T2_part3)
    return C * (dT1 + dT2)


@njit(cache=True)
def _eval_lander_mass(t: float, mass_coef: float):
    f = _f_value(t)
    df = _f_first(t)
    d2f = _f_second(t)
    d3f = _f_third(t)
    return (f, df, d2f, d3f)


class LanderMassDecay:

    def __init__(self, mass_coef: float):
        self.mass_coef = mass_coef

    def value(self, t: float) -> float:
        return _f_value(t)

    def first(self, t: float) -> float:
        return _f_first(t)

    def second(self, t: float) -> float:
        return _f_second(t)

    def third(self, t: float) -> float:
        return _f_third(t)


def compute_derivs_lander(t, y, M_c1, M_c2, mass_coef):
    from equations import _combine_scalings, _dadt, _dedt

    a = y[0]
    e = min(max(y[1], 0.0), 1.0 - 1e-08)
    M_c = M_c1 + M_c2
    f1, df1, d2f1, d3f1 = _eval_lander_mass(t, mass_coef)
    f2, df2, d2f2, d3f2 = (f1, df1, d2f1, d3f1)
    if f1 <= 1e-08 or f2 <= 1e-08:
        return np.zeros(2)
    f_M, df_M, d2f_M, f_mu, df_mu, d2f_mu, d3f_mu = _combine_scalings(M_c1, M_c2, M_c, f1, df1,
                                                                      d2f1, d3f1, f2, df2, d2f2,
                                                                      d3f2)
    mu_c = M_c1 * M_c2 / M_c
    dadt_val = _dadt(
        mu_c,
        M_c,
        M_c1,
        M_c2,
        f1,
        f2,
        df1,
        df2,
        f_M,
        f_mu,
        df_M,
        df_mu,
        d2f_M,
        d2f_mu,
        d3f_mu,
        a,
        e,
    )
    dedt_val = _dedt(
        M_c1,
        M_c2,
        M_c,
        mu_c,
        f1,
        f2,
        df1,
        df2,
        f_M,
        df_M,
        f_mu,
        df_mu,
        d2f_M,
        d2f_mu,
        d3f_mu,
        a,
        e,
    )
    return np.array([dadt_val, dedt_val], dtype=np.float64)


def make_lander_mass_function(mass_coef: float):
    from equations import AnalyticMassFunction

    decay = LanderMassDecay(mass_coef)
    return AnalyticMassFunction(
        decay.value,
        decay.first,
        decay.second,
        decay.third
    )
