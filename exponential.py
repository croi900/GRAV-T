from __future__ import annotations

from typing import Tuple
import numpy as np
from numba import njit

from equations import AnalyticMassFunction

class ExpMassDecay:
    def __init__(self, decay_rate: float):
        self.decay_rate = decay_rate

    def value(self, t: float) -> float:
        return np.exp(-self.decay_rate * t)

    def first(self, t: float) -> float:
        return -self.decay_rate * np.exp(-self.decay_rate * t)

    def second(self, t: float) -> float:
        return self.decay_rate**2 * np.exp(-self.decay_rate * t)

    def third(self, t: float) -> float:
        return -(self.decay_rate**3) * np.exp(-self.decay_rate * t)


def make_exp_mass_function(decay_rate: float) -> AnalyticMassFunction:
    decay = ExpMassDecay(decay_rate)
    return AnalyticMassFunction(decay.value, decay.first, decay.second, decay.third)


@njit(cache=True)
def _eval_exp_mass(t: float, decay_rate: float):
    exp_val = np.exp(-decay_rate * t)
    f = exp_val
    df = -decay_rate * exp_val
    d2f = decay_rate**2 * exp_val
    d3f = -(decay_rate**3) * exp_val
    return f, df, d2f, d3f


from equations import _combine_scalings, _dadt, _dedt

@njit(cache=True)
def compute_derivs_exp(t, y, M_c1, M_c2, decay_rate):
    
    a = y[0]

    e = y[1]
    M_c = M_c1 + M_c2

    f1, df1, d2f1, d3f1 = _eval_exp_mass(t, decay_rate)
    f2, df2, d2f2, d3f2 = f1, df1, d2f1, d3f1

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

    return np.array([dadt_val, dedt_val])
