from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from equations import AnalyticMassFunction, _combine_scalings, _dadt, _dedt

# Solar mass in kg (approx)
M_SUN_KG = 1.9891e30

class LanderMassDecay:
    def __init__(self, decay_rate: float):
        self.nu = 1 #lander doesnt have decay_rate, fix to 1 

    def _L_nu(self, t: float) -> float:
        # L_nu / 10^52
        return 0.7 * np.exp(-t / 1.5) + 0.3 * (1 - t / 50)**4 if t < 50 else 0.7 * np.exp(-t / 1.5)

    def _E_nu(self, t: float) -> float:
        # E_nu / 10 MeV
        part1 = 0.3 * np.exp(-t / 4)
        part2 = 1 - t / 60
        return part1 + part2 if t < 60 else part1

    def _M_dot_normalized(self, t: float) -> float:
        l_norm = self._L_nu(t)
        e_norm = self._E_nu(t)

        if l_norm < 0: l_norm = 0
        if e_norm < 0: e_norm = 0

        val = (l_norm)**(5/3) * (e_norm)**(10/3)
        return val

    def _f_dot_val(self, t: float) -> float:
        # Constants
        k = -6.8e-5
        val = k * self._M_dot_normalized(t) * 1 / 2.8
        return val

    def value(self, t: float) -> float:
        res, err = quad(self._f_dot_val, 0, t, limit=100)
        return 1.0 + res

    def first(self, t: float) -> float:
        return self._f_dot_val(t)

    def second(self, t: float) -> float:
        C = -6.8e-5 * 1 / 2.8
        
        L = self._L_nu(t)
        if L < 1e-9: L = 0
        
        E = self._E_nu(t)
        if E < 1e-9: E = 0
        
        L_prime = 0.7 * (-1/1.5) * np.exp(-t/1.5)
        if t < 50:
            L_prime += 1.2 * (1 - t/50)**3 * (-0.02)
            
        E_prime = 0.3 * (-0.25) * np.exp(-t/4)
        if t < 60:
            E_prime -= 1/60
            
        term1 = (5/3) * (L**(2/3) if L>0 else 0) * L_prime * (E**(10/3) if E>0 else 0)
        term2 = (L**(5/3) if L>0 else 0) * (10/3) * (E**(7/3) if E>0 else 0) * E_prime
        
        return C * (term1 + term2)

    def third(self, t: float) -> float:
        C = -6.8e-5 * 1 / 2.8
        
        L = self._L_nu(t)
        if L < 1e-9: L = 0
        E = self._E_nu(t)
        if E < 1e-9: E = 0
        
        L_prime = 0.7 * (-1/1.5) * np.exp(-t/1.5)
        if t < 50:
            L_prime += 1.2 * (1 - t/50)**3 * (-0.02)
            
        E_prime = 0.3 * (-0.25) * np.exp(-t/4)
        if t < 60:
            E_prime -= 1/60
            
        L_double = 0.7 * (1/2.25) * np.exp(-t/1.5)
        if t < 50:
            L_double += 3.6 * (1 - t/50)**2 * 0.0004
            
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


def make_lander_mass_function(decay_rate: float) -> AnalyticMassFunction:
    decay = LanderMassDecay(decay_rate)
    return AnalyticMassFunction(decay.value, decay.first, decay.second, decay.third)


def _eval_lander_mass(t: float, decay_rate: float):
    # This function is NOT JIT compiled because it uses quad (via the class)
    # We instantiate the class momentarily or reuse logic.
    # For performance, maybe reuse logic without quad if possible? No, f(t) needs integral.
    # So we must use the class.
    decay = LanderMassDecay(decay_rate)
    f = decay.value(t)
    df = decay.first(t)
    d2f = decay.second(t)
    d3f = decay.third(t)
    return f, df, d2f, d3f


def compute_derivs_lander(t, y, M_c1, M_c2, decay_rate):
    # This is a Python function, not JIT compiled.
    a = y[0]
    e = min(max(y[1], 0.0), 1.0 - 1e-8)
    M_c = M_c1 + M_c2

    f1, df1, d2f1, d3f1 = _eval_lander_mass(t, decay_rate)
    f2, df2, d2f2, d3f2 = f1, df1, d2f1, d3f1

    if f1 <= 1e-8 or f2 <= 1e-8:
        return np.zeros(2)

    # _combine_scalings and others are JIT compiled but can be called from python
    f_M, df_M, d2f_M, f_mu, df_mu, d2f_mu, d3f_mu = _combine_scalings(
        M_c1, M_c2, M_c, f1, df1, d2f1, d3f1, f2, df2, d2f2, d3f2
    )

    mu_c = (M_c1 * M_c2) / M_c

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
