from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

import config as cfg
from constants import G, c


def isco(m1: float, m2: float) -> float:
    """Calculate ISCO radius for binary: r = 6*G*M/c^2."""
    M_total = m1 + m2
    return 6.0 * G * M_total / (c ** 2)


FloatFunc = Callable[[float], float]


def _ensure_float(fun: FloatFunc, t: float) -> float:
    return float(fun(t))


@dataclass(frozen=True)
class AnalyticMassFunction:
    value: FloatFunc
    first: FloatFunc
    second: FloatFunc
    third: FloatFunc

    def evaluate_all(self, t: float) -> Tuple[float, float, float, float]:
        return (
            _ensure_float(self.value, t),
            _ensure_float(self.first, t),
            _ensure_float(self.second, t),
            _ensure_float(self.third, t),
        )


@dataclass(frozen=True)
class ScalingSnapshot:
    f1: float
    f2: float
    df1: float
    df2: float
    f_M: float
    df_M: float
    d2f_M: float
    f_mu: float
    df_mu: float
    d2f_mu: float
    d3f_mu: float

    def as_tuple(self) -> Tuple[float, ...]:
        return (
            self.f1,
            self.f2,
            self.df1,
            self.df2,
            self.f_M,
            self.df_M,
            self.d2f_M,
            self.f_mu,
            self.df_mu,
            self.d2f_mu,
            self.d3f_mu,
        )


@njit(cache=True)
def _combine_scalings(
    M_c1: float,
    M_c2: float,
    M_c: float,
    f1: float,
    df1: float,
    d2f1: float,
    d3f1: float,
    f2: float,
    df2: float,
    d2f2: float,
    d3f2: float,
):
    alpha1 = M_c1 / M_c
    alpha2 = M_c2 / M_c

    f_M = alpha1 * f1 + alpha2 * f2
    df_M = alpha1 * df1 + alpha2 * df2
    d2f_M = alpha1 * d2f1 + alpha2 * d2f2

    f_mu, df_mu, d2f_mu, d3f_mu = _compute_fmu(
        M_c, M_c1, M_c2, f1, df1, d2f1, d3f1, f2, df2, d2f2, d3f2
    )

    return f_M, df_M, d2f_M, f_mu, df_mu, d2f_mu, d3f_mu


@njit(cache=True)
def _compute_fmu(
    M_c: float,
    M_c1: float,
    M_c2: float,
    f1: float,
    df1: float,
    d2f1: float,
    d3f1: float,
    f2: float,
    df2: float,
    d2f2: float,
    d3f2: float,
):
    N = f1 * f2
    dN = df1 * f2 + f1 * df2
    d2N = d2f1 * f2 + 2.0 * df1 * df2 + f1 * d2f2
    d3N = d3f1 * f2 + 3.0 * d2f1 * df2 + 3.0 * df1 * d2f2 + f1 * d3f2

    D = M_c1 * f1 + M_c2 * f2
    dD = M_c1 * df1 + M_c2 * df2
    d2D = M_c1 * d2f1 + M_c2 * d2f2
    d3D = M_c1 * d3f1 + M_c2 * d3f2

    invD = 1.0 / D
    dinvD = -invD * invD * dD
    d2invD = 2.0 * invD**3 * dD**2 - invD * invD * d2D
    d3invD = -6.0 * invD**4 * dD**3 + 6.0 * invD**3 * dD * d2D - invD * invD * d3D

    R = N * invD
    dR = dN * invD + N * dinvD
    d2R = d2N * invD + 2.0 * dN * dinvD + N * d2invD
    d3R = d3N * invD + 3.0 * d2N * dinvD + 3.0 * dN * d2invD + N * d3invD

    return M_c * R, M_c * dR, M_c * d2R, M_c * d3R


@njit(cache=True)
def _F_factors(f_M, f_mu, df_M, df_mu, d2f_M, d2f_mu, d3f_mu):
    F1 = d3f_mu
    sqrt_f_M = np.sqrt(f_M)
    F2 = (
        sqrt_f_M * 3.0 * d2f_mu
        + (3.0 * df_mu * df_M) / (2.0 * sqrt_f_M)
        + f_mu / (4.0 * f_M ** (1.5)) * (2.0 * d2f_M * f_M - df_M**2)
    )
    F3 = 3.0 * df_mu * f_M + 1.5 * df_M * f_mu
    F4 = f_M**1.5 * f_mu
    return F1, F2, F3, F4


@njit(cache=True)
def _Fp_factors(f_M, f_mu, df_M, df_mu, d3f_mu):
    sqrt_f_M = np.sqrt(f_M)
    Fp1 = d3f_mu
    Fp2 = 2.0 * df_mu * sqrt_f_M + f_mu * df_M / (2.0 * sqrt_f_M)
    Fp3 = f_M * f_mu
    return Fp1, Fp2, Fp3


@njit(cache=True)
def _classical_dEdt(mu_c, M_c, a, e):
    M1_sq_M2_sq_M = mu_c**2 * M_c**3
    sqrt_1_e2 = np.sqrt(1.0 - e**2)
    prefactor = (8.0 * G**4 * M1_sq_M2_sq_M) / (15.0 * c**5 * a**5 * sqrt_1_e2**10)
    eccentricity_factor = 1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4
    return prefactor * eccentricity_factor


@njit(cache=True)
def _classical_dLdt(mu_c, M_c, a, e):
    prefactor = (
        32.0
        * G**3.5
        * np.sqrt(M_c)
        * mu_c**2
        * M_c
        / (5.0 * c**5 * a**3.5 * (1.0 - e**2) ** 2)
    )
    eccentricity_factor = 1.0 + (7.0 / 8.0) * e**2
    return prefactor * eccentricity_factor


@njit(cache=True)
def _variable_mass_dEdt(
    M_c,
    mu_c,
    f_M,
    f_mu,
    df_M,
    df_mu,
    d2f_M,
    d2f_mu,
    d3f_mu,
    a,
    e,
):
    dEdt_classical = _classical_dEdt(mu_c, M_c, a, e)
    F1, F2, F3, F4 = _F_factors(f_M, f_mu, df_M, df_mu, d2f_M, d2f_mu, d3f_mu)

    sqrt_1_e2 = np.sqrt(1.0 - e**2)
    correction_1 = (16.0 * G**2 * M_c**2 * (4.0 - sqrt_1_e2) / (a**2 * sqrt_1_e2)) * (
        3.0 * F3**2 - 2.0 * F2 * F4
    )
    correction_2 = 4.0 * G * M_c * a * (3.0 - e**2) * (3.0 * F2**2 - 2.0 * F1 * F3)
    correction_3 = a**4 * (8.0 + 40.0 * e**2 + 15.0 * e**4) / 6.0 * F1**2

    return -(F4**2) * dEdt_classical + (G * mu_c**2 / (10.0 * c**5)) * (
        correction_1 + correction_2 + correction_3
    )


@njit(cache=True)
def _variable_mass_dLdt(
    M_c,
    mu_c,
    f_M,
    f_mu,
    df_M,
    df_mu,
    d2f_M,
    d2f_mu,
    d3f_mu,
    a,
    e,
):
    dLdt_classical = _classical_dLdt(mu_c, M_c, a, e)
    F1, F2, F3, F4 = _F_factors(f_M, f_mu, df_M, df_mu, d2f_M, d2f_mu, d3f_mu)
    Fp1, Fp2, Fp3 = _Fp_factors(f_M, f_mu, df_M, df_mu, d3f_mu)

    sqrt_factor = np.sqrt(G * M_c * (1.0 - e**2) / a)
    Lz_correction_1 = (
        sqrt_factor * 8.0 * G * M_c * (Fp1 * F4 + 3.0 * Fp3 * F2 - 6.0 * Fp2 * F3)
    )
    Lz_correction_2 = (
        sqrt_factor * a**3 * (2.0 + 3.0 * e**2) * (2.0 * Fp2 * F1 - 3.0 * Fp1 * F2)
    )

    return -Fp3 * F4 * dLdt_classical - (G * mu_c**2 / (5.0 * c**5)) * (
        Lz_correction_1 + Lz_correction_2
    )


@njit(cache=True)
def _dadt(
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
):
    dEdt_classical = _classical_dEdt(mu_c, M_c, a, e)
    F1, F2, F3, F4 = _F_factors(f_M, f_mu, df_M, df_mu, d2f_M, d2f_mu, d3f_mu)
    dadt_classical = (2.0 * a**2 / (G * M_c1 * M_c2)) * (-dEdt_classical)
    sqrt_1_e2 = np.sqrt(1.0 - e**2)

    correction = (
        mu_c
        / (5.0 * c**5 * f1 * f2)
        * (
            (16.0 * G**2 * M_c * (4.0 - sqrt_1_e2) / sqrt_1_e2)
            * (3.0 * F3**2 - 2.0 * F2 * F4)
            + 4.0 * G * a**3 * (3.0 - e**2) * (3.0 * F2**2 - 2.0 * F1 * F3)
            + (a**6 / (6.0 * M_c)) * (8.0 + 40.0 * e**2 + 15.0 * e**4) * F1**2
        )
    )

    return (F4**2 / (f1 * f2)) * dadt_classical - correction - a * (df_M / f_M)


@njit(cache=True)
def _dedt(
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
):
    epsilon_sq = 1e-12
    inv_e = e / (e**2 + epsilon_sq)

    e_sq = e**2
    sqrt_1_e2 = np.sqrt(1.0 - e_sq)

    F1, F2, F3, F4 = _F_factors(f_M, f_mu, df_M, df_mu, d2f_M, d2f_mu, d3f_mu)
    Fp1, Fp2, Fp3 = _Fp_factors(f_M, f_mu, df_M, df_mu, d3f_mu)

    dedt_classical = (
        -304
        * G**3
        * M_c1
        * M_c2
        * M_c
        * e
        / (15.0 * c**5 * a**4 * sqrt_1_e2**5)
        * (1 + 121 / 304 * e_sq)
    )

    correction_1 = (
        mu_c
        * a
        * (1.0 - e_sq)
        / (10.0 * c**5 * f1 * f2)
        * inv_e
        * (
            (16.0 * G**2 * M_c * (4.0 - sqrt_1_e2) / (a**2 * sqrt_1_e2))
            * (3.0 * F3**2 - 2.0 * F2 * F4)
            + 4.0 * G * a * (3.0 - e_sq) * (3.0 * F2**2 - 2.0 * F1 * F3)
            + (a**4 / (6.0 * M_c)) * (8.0 + 40.0 * e_sq + 15.0 * e_sq**2) * F1**2
        )
    )

    correction_2 = (
        G
        * mu_c
        * (1.0 - e_sq)
        / (5.0 * c**5 * a)
        * inv_e
        * np.sqrt(f_M)
        / (f1 * f2)
        * (
            8.0 * G * M_c * (Fp1 * F4 + 3.0 * Fp3 * F2 - 6.0 * Fp2 * F3)
            + a**3 * (2.0 + 3.0 * e_sq) * (2.0 * Fp2 * F1 - 3.0 * Fp1 * F2)
        )
    )

    return (
        (F4**2 / (f1 * f2)) * dedt_classical
        - correction_1
        - correction_2
        - (1.0 - e_sq) / 2.0 * inv_e * (df_M / f_M - 3.0 * df1 / f1 - 3.0 * df2 / f2)
    )


@njit(cache=True)
def _dPdt(
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
):
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
    dPdt_val = (
        2
        * np.pi
        / np.sqrt(G * M_c)
        * np.sqrt(a / f_M)
        * (1.5 * dadt_val - a / 2 * df_M / f_M)
    )
    return dPdt_val


class BinarySystemModelFast:
    def __init__(self, config) -> None:
        from name_maps import decay_map

        if isinstance(config, cfg.Config):
            self.M_c1 = float(config.state.M1)
            self.M_c2 = float(config.state.M2)
            self.M_c = self.M_c1 + self.M_c2
            self.mu_c = (self.M_c1 * self.M_c2) / self.M_c
            self.a0 = float(config.state.a)
            self.e0 = float(config.state.e)

            self.mass_fun_1 = decay_map(config.decay_type, config.state.decay_rate)
            self.mass_fun_2 = decay_map(config.decay_type, config.state.decay_rate)
        elif isinstance(config, cfg.State):
            self.M_c1 = config.M1
            self.M_c2 = config.M2
            self.M_c = self.M_c1 + self.M_c2
            self.mu_c = (self.M_c1 * self.M_c2) / self.M_c
            self.a0 = config.a
            self.e0 = config.e

            self.mass_fun_1 = decay_map(config.decay_type, config.state.decay_rate)
            self.mass_fun_2 = decay_map(config.decay_type, config.state.decay_rate)

    def _snapshot(self, t: float) -> ScalingSnapshot:
        f1, df1, d2f1, d3f1 = self.mass_fun_1.evaluate_all(t)
        f2, df2, d2f2, d3f2 = self.mass_fun_2.evaluate_all(t)
        (
            f_M,
            df_M,
            d2f_M,
            f_mu,
            df_mu,
            d2f_mu,
            d3f_mu,
        ) = _combine_scalings(
            self.M_c1,
            self.M_c2,
            self.M_c,
            f1,
            df1,
            d2f1,
            d3f1,
            f2,
            df2,
            d2f2,
            d3f2,
        )

        return ScalingSnapshot(
            f1,
            f2,
            df1,
            df2,
            f_M,
            df_M,
            d2f_M,
            f_mu,
            df_mu,
            d2f_mu,
            d3f_mu,
        )

    def variable_mass_dEdt(self, t: float, a: float, e: float) -> float:
        snap = self._snapshot(t)
        return _variable_mass_dEdt(
            self.M_c,
            self.mu_c,
            snap.f_M,
            snap.f_mu,
            snap.df_M,
            snap.df_mu,
            snap.d2f_M,
            snap.d2f_mu,
            snap.d3f_mu,
            a,
            e,
        )

    def variable_mass_dLdt(self, t: float, a: float, e: float) -> float:
        snap = self._snapshot(t)
        return _variable_mass_dLdt(
            self.M_c,
            self.mu_c,
            snap.f_M,
            snap.f_mu,
            snap.df_M,
            snap.df_mu,
            snap.d2f_M,
            snap.d2f_mu,
            snap.d3f_mu,
            a,
            e,
        )

    def dadt(self, t: float, a: float, e: float) -> float:
        snap = self._snapshot(t)
        return _dadt(
            self.mu_c,
            self.M_c,
            self.M_c1,
            self.M_c2,
            snap.f1,
            snap.f2,
            snap.df1,
            snap.df2,
            snap.f_M,
            snap.f_mu,
            snap.df_M,
            snap.df_mu,
            snap.d2f_M,
            snap.d2f_mu,
            snap.d3f_mu,
            a,
            e,
        )

    def dedt(self, t: float, a: float, e: float) -> float:
        snap = self._snapshot(t)
        return _dedt(
            self.M_c1,
            self.M_c2,
            self.M_c,
            self.mu_c,
            snap.f1,
            snap.f2,
            snap.df1,
            snap.df2,
            snap.f_M,
            snap.df_M,
            snap.f_mu,
            snap.df_mu,
            snap.d2f_M,
            snap.d2f_mu,
            snap.d3f_mu,
            a,
            e,
        )

    def dPdt(self, t: float, a: float, e: float) -> float:
        snap = self._snapshot(t)
        return _dPdt(
            self.mu_c,
            self.M_c,
            self.M_c1,
            self.M_c2,
            snap.f1,
            snap.f2,
            snap.df1,
            snap.df2,
            snap.f_M,
            snap.f_mu,
            snap.df_M,
            snap.df_mu,
            snap.d2f_M,
            snap.d2f_mu,
            snap.d3f_mu,
            a,
            e,
        )

    def rhs(self, t: float, y: Sequence[float]) -> np.ndarray:
        a, e = float(y[0]), float(y[1])
        return np.array([self.dadt(t, a, e), self.dedt(t, a, e)], dtype=np.float64)

    def solve(
        self,
        t_span: Tuple[float, float],
        y0: Iterable[float] | None = None,
        method: str = "Radau",
        **solve_kwargs,
    ):
        if y0 is None:
            y0 = (self.a0, self.e0)
        return solve_ivp(self.rhs, t_span, y0, method=method, **solve_kwargs)

    def coalescence_time(
        self, a_min=1e5, max_time=1e20, rtol=1e-4, atol=1e-6, method="Radau"
    ):
        progress = tqdm(
            total=max_time if max_time > 0 else 1.0,
            desc="Coalescence search",
            unit="s",
            dynamic_ncols=True,
        )
        last_t = {"value": 0.0}

        def update_progress(current_t: float):
            current_t = float(current_t)
            if current_t <= last_t["value"]:
                return
            increment = min(current_t, max_time) - last_t["value"]
            if increment > 0:
                progress.update(increment)
                last_t["value"] += increment

        def system(t, y):
            a, e = y
            update_progress(t)
            dadt = self.dadt(t, a, e)
            dedt = self.dedt(t, a, e)
            return [dadt, dedt]

        def event_merger(t, y):
            a, e = y
            # Calculate current masses at time t
            f1 = self.mass_fun_1.value(t)
            f2 = self.mass_fun_2.value(t)
            m1_current = self.M_c1 * f1
            m2_current = self.M_c2 * f2
            # Calculate dynamic ISCO based on current masses
            r_isco = isco(m1_current, m2_current)
            return a - r_isco

        event_merger.terminal = True
        event_merger.direction = -1

        def event_mass_depletion(t, y):
            f1 = self.mass_fun_1.value(t)
            return f1 - 1e-6

        event_mass_depletion.terminal = True
        event_mass_depletion.direction = -1

        try:
            solution = solve_ivp(
                system,
                t_span=[0, max_time],
                y0=[self.a0, self.e0],
                method=method,
                rtol=rtol,
                atol=atol,
                events=[event_merger, event_mass_depletion],
            )
        finally:
            progress.update(max(0.0, progress.total - progress.n))
            progress.close()

        if solution.t_events[0].size > 0:
            return solution.t_events[0][0], solution
        elif solution.t_events[1].size > 0:
            print(f"Mass depletion limit reached at t={solution.t_events[1][0]:.4e} s")
            return solution.t_events[1][0], solution
        else:
            print("Semi-major axis did not reach minimum for the allowed timeframe.")
            return solution.t[-1], solution


__all__ = [
    "AnalyticMassFunction",
    "BinarySystemModelFast",
    "isco",
    "make_exp_mass_function",
    "make_linear_mass_function",
    "make_lander_mass_function",
]


from linear import (
    LinearMassDecay,
    make_linear_mass_function,
    _eval_linear_mass,
    compute_derivs_linear,
)
from exponential import (
    ExpMassDecay,
    make_exp_mass_function,
    _eval_exp_mass,
    compute_derivs_exp,
)
from lander import LanderMassDecay, make_lander_mass_function
