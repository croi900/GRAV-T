
import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from constants import G, c


def compute_phi_r_ode(t, a, e, M):

    a_func = interp1d(t, a, kind="linear", fill_value="extrapolate")
    e_func = interp1d(t, e, kind="linear", fill_value="extrapolate")
    M_func = interp1d(t, M, kind="linear", fill_value="extrapolate")

    padding = (t[-1] - t[0]) * 0.01
    pbar = tqdm(total=t[-1] - t[0] + padding, desc="  Phase integration", unit="s", leave=False)
    last_t = np.array([t[0]])

    def dphi_dt_func(t_val, y):
        if t_val > last_t[0]:
            pbar.update(t_val - last_t[0])
            last_t[0] = t_val

        phi = y[0]
        a_val = float(a_func(t_val))
        e_val = float(e_func(t_val))
        M_val = float(M_func(t_val))

        e_val = np.clip(e_val, 1e-10, 0.9999)

        r_val = a_val * (1 - e_val**2) / (1 + e_val * np.cos(phi))
        dphi_dt = np.sqrt(G * M_val * a_val * (1 - e_val**2)) / r_val**2

        return [dphi_dt]

    sol = solve_ivp(
        fun=dphi_dt_func,
        t_span=(t[0], t[-1]),
        y0=[0.0],
        t_eval=t,
        method="RK45",
        max_step=(t[-1] - t[0]) / len(t) * 10,
    )
    
    pbar.close()

    phi_arr = sol.y[0]

    e_safe = np.clip(e, 1e-10, 0.9999)
    r_arr = a * (1 - e_safe**2) / (1 + e_safe * np.cos(phi_arr))

    print(
        f"  Phase evolution complete: φ from {phi_arr[0]:.2f} to {phi_arr[-1]:.2e} rad"
    )
    return phi_arr, r_arr


@njit(cache=True)
def _compute_waveform_point(phi, a, e, G_val, M, mu):

    e_safe = max(min(e, 0.9999), 1e-10)

    p = a * (1 - e_safe**2)
    u = 1 + e_safe * np.cos(phi)
    r = p / u

    L = np.sqrt(G_val * M * a * (1 - e_safe**2))
    phi_dot = L / (r * r)

    phi_ddot = (
        -2 * e_safe * np.sin(phi) * G_val * M * u**3 / (a**3 * (1 - e_safe**2) ** 3)
    )

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    dr_dphi = p * e_safe * sin_phi / (u * u)
    d2r_dphi2 = (
        p * e_safe * (cos_phi * u + 2 * e_safe * sin_phi * sin_phi) / (u * u * u)
    )

    A = r * r
    dA_dphi = 2 * r * dr_dphi
    d2A_dphi2 = 2 * (dr_dphi * dr_dphi + r * d2r_dphi2)

    Bxx = 3 * cos_phi * cos_phi - 1
    dBxx_dphi = -6 * cos_phi * sin_phi
    d2Bxx_dphi2 = -6 * (cos_phi * cos_phi - sin_phi * sin_phi)

    dQxx_dphi = mu * (dA_dphi * Bxx + A * dBxx_dphi)
    d2Qxx_dphi2 = mu * (d2A_dphi2 * Bxx + 2 * dA_dphi * dBxx_dphi + A * d2Bxx_dphi2)

    Byy = 3 * sin_phi * sin_phi - 1
    dByy_dphi = 6 * sin_phi * cos_phi
    d2Byy_dphi2 = 6 * (cos_phi * cos_phi - sin_phi * sin_phi)

    dQyy_dphi = mu * (dA_dphi * Byy + A * dByy_dphi)
    d2Qyy_dphi2 = mu * (d2A_dphi2 * Byy + 2 * dA_dphi * dByy_dphi + A * d2Byy_dphi2)

    Bxy = 3 * sin_phi * cos_phi
    dBxy_dphi = 3 * (cos_phi * cos_phi - sin_phi * sin_phi)
    d2Bxy_dphi2 = -12 * sin_phi * cos_phi

    dQxy_dphi = mu * (dA_dphi * Bxy + A * dBxy_dphi)
    d2Qxy_dphi2 = mu * (d2A_dphi2 * Bxy + 2 * dA_dphi * dBxy_dphi + A * d2Bxy_dphi2)

    phi_dot_sq = phi_dot * phi_dot
    ddQxx = d2Qxx_dphi2 * phi_dot_sq + dQxx_dphi * phi_ddot
    ddQyy = d2Qyy_dphi2 * phi_dot_sq + dQyy_dphi * phi_ddot
    ddQxy = d2Qxy_dphi2 * phi_dot_sq + dQxy_dphi * phi_ddot

    return ddQxx - ddQyy, 2 * ddQxy


@njit(parallel=True, cache=True)
def _compute_waveforms_parallel(phi, a, e, M, mu, n_points):
    """
    Compute waveforms in parallel using numba.
    """
    hp_component = np.zeros(n_points)
    hx_component = np.zeros(n_points)

    for i in prange(n_points):
        hp_comp, hx_comp = _compute_waveform_point(phi[i], a[i], e[i], G, M[i], mu[i])
        hp_component[i] = hp_comp
        hx_component[i] = hx_comp

    return hp_component, hx_component


def get_waveforms(phi, a, e, M, mu, D_obs, clight=2.998e8):
    """
    Compute gravitational wave strains h+ and hx using analytical derivatives.
    """

    phi = np.atleast_1d(phi).astype(np.float64)
    a = np.atleast_1d(a).astype(np.float64)
    e = np.atleast_1d(e).astype(np.float64)
    M = np.atleast_1d(M).astype(np.float64)
    mu = np.atleast_1d(mu).astype(np.float64)

    n_points = len(phi)
    print(f"Computing waveforms for {n_points:,} points...")

    hp_component, hx_component = _compute_waveforms_parallel(phi, a, e, M, mu, n_points)

    factor = G / (clight**4 * D_obs)

    h_plus = factor * hp_component
    h_cross = factor * hx_component

    h_amp = np.sqrt(h_plus**2 + h_cross**2)
    print(
        f"  Amplitude range: {h_amp[0]:.3e} to {h_amp[-1]:.3e} (ratio: {h_amp[-1] / h_amp[0]:.1f}x)"
    )

    return h_plus, h_cross


def get_r(phi, a, e):
    e_safe = np.clip(e, 1e-10, 0.9999)
    return a * (1 - e_safe**2) / (1 + e_safe * np.cos(phi))


def get_phi_dot(phi, a, e, G_val, M):
    r = get_r(phi, a, e)
    e_safe = np.clip(e, 1e-10, 0.9999)
    return np.sqrt(G_val * M * a * (1 - e_safe**2)) / r**2


def get_d2phi_dt2(phi, a, e, G_val, M):
    e_safe = np.clip(e, 1e-10, 0.9999)
    return (
        -2
        * e_safe
        * np.sin(phi)
        * G_val
        * M
        * (1 + e_safe * np.cos(phi)) ** 3
        / (a**3 * (1 - e_safe**2) ** 3)
    )
