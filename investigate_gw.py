#!/usr/bin/env python3


import h5py
import numpy as np
from polarizations import compute_phi_r_ode, get_waveforms, G


def analyze_run(f, run_name):
    print(f"\n{'=' * 60}")
    print(f"=== {run_name.upper()} ===")
    print(f"{'=' * 60}")

    a = np.array(f[f"{run_name}/a"])
    e = np.array(f[f"{run_name}/e"])
    t = np.array(f[f"{run_name}/times"])
    m1 = np.array(f[f"{run_name}/m1"])
    m2 = np.array(f[f"{run_name}/m2"])

    print(f"\n--- Raw Data Statistics ---")
    print(f"  Time span: {t[0]:.4e} to {t[-1]:.4e} s ({len(t)} points)")
    print(f"  Duration: {t[-1] - t[0]:.4e} s = {(t[-1] - t[0]) / 3600:.2f} hours")
    print(f"  Semi-major axis a: {a[0]:.4e} to {a[-1]:.4e} m")
    print(f"  Eccentricity e: {e[0]:.6f} to {e[-1]:.6f}")
    print(
        f"  Mass m1: {m1[0]:.4e} to {m1[-1]:.4e} kg ({m1[0] / 1.989e30:.2f} to {m1[-1] / 1.989e30:.2f} M_sun)"
    )
    print(f"  Mass m2: {m2[0]:.4e} to {m2[-1]:.4e} kg")

    print(f"\n--- Data Quality ---")
    print(f"  a < 0: {np.sum(a < 0)} points")
    print(f"  e < 0: {np.sum(e < 0)} points")
    print(f"  e > 1: {np.sum(e > 1)} points")
    print(f"  NaN in a: {np.sum(np.isnan(a))}, NaN in e: {np.sum(np.isnan(e))}")

    valid = (e > 0) & (e < 1) & (a > 0) & ~np.isnan(a) & ~np.isnan(e)
    print(
        f"  Valid points: {np.sum(valid)} / {len(t)} ({100 * np.sum(valid) / len(t):.1f}%)"
    )

    if np.sum(valid) < 10:
        print("  ERROR: Too few valid points!")
        return

    a_v, e_v, t_v = a[valid], e[valid], t[valid]
    m1_v, m2_v = m1[valid], m2[valid]
    M_v = m1_v + m2_v
    mu_v = m1_v * m2_v / M_v

    omega = np.sqrt(G * M_v / a_v**3)
    f_orb = omega / (2 * np.pi)
    f_GW_approx = 2 * f_orb

    print(f"\n--- Orbital Dynamics ---")
    print(f"  GW frequency (Kepler): {f_GW_approx[0]:.2f} to {f_GW_approx[-1]:.2f} Hz")
    print(
        f"  Orbital period: {2 * np.pi / omega[0]:.4e} to {2 * np.pi / omega[-1]:.4e} s"
    )

    c = 2.998e8
    f_e = (1 + (73 / 24) * e_v**2 + (37 / 96) * e_v**4) / (1 - e_v**2) ** (7 / 2)
    P_gw = (32 / 5) * (G**4 / c**5) * (mu_v**2 * M_v**3) / a_v**5 * f_e

    print(f"\n--- GW Power (Peters & Mathews) ---")
    print(f"  Power: {P_gw[0]:.4e} to {P_gw[-1]:.4e} W")
    print(f"  Power ratio (end/start): {P_gw[-1] / P_gw[0]:.2f}x")

    M_chirp = (mu_v ** (3 / 5)) * (M_v ** (2 / 5))
    D_obs = 10 * 3.086e22
    h_estimated = (
        (4 / D_obs)
        * (G * M_chirp / c**2) ** (5 / 3)
        * (np.pi * f_GW_approx / c) ** (2 / 3)
    )

    print(f"\n--- Strain Estimate (Leading Order) ---")
    print(f"  h estimated: {h_estimated[0]:.4e} to {h_estimated[-1]:.4e}")
    print(f"  Strain ratio (end/start): {h_estimated[-1] / h_estimated[0]:.2f}x")

    step = max(1, len(t_v) // 1000)
    idx = np.arange(0, len(t_v), step)

    print(f"\n--- Actual Polarization Computation (subsample: {len(idx)} points) ---")
    try:
        phi, r = compute_phi_r_ode(t_v[idx], a_v[idx], e_v[idx], M_v[idx])
        print(
            f"  phi: {phi[0]:.4f} to {phi[-1]:.4f} rad ({phi[-1] / (2 * np.pi):.2f} orbits)"
        )
        print(f"  r: {r[0]:.4e} to {r[-1]:.4e} m")

        hx, hp = get_waveforms(phi, a_v[idx], e_v[idx], M_v[idx], mu_v[idx], D_obs)
        h_amp = np.sqrt(hp**2 + hx**2)

        print(f"  h_plus: {hp[0]:.4e} to {hp[-1]:.4e}")
        print(f"  h_cross: {hx[0]:.4e} to {hx[-1]:.4e}")
        print(f"  |h|: {h_amp[0]:.4e} to {h_amp[-1]:.4e}")
        print(f"  Amplitude ratio (end/start): {h_amp[-1] / h_amp[0]:.2f}x")

        mid = len(idx) // 2
        print(f"\n  At middle point:")
        print(
            f"    t={t_v[idx[mid]]:.4e}, a={a_v[idx[mid]]:.4e}, e={e_v[idx[mid]]:.6f}"
        )
        print(f"    phi={phi[mid]:.4f}, h_amp={h_amp[mid]:.4e}")

    except Exception as ex:
        print(f"  ERROR computing waveforms: {ex}")
        import traceback

        traceback.print_exc()

    return


if __name__ == "__main__":
    print("=" * 60)
    print("GRAVITATIONAL WAVE WAVEFORM INVESTIGATION")
    print("=" * 60)

    with h5py.File("exponential_better/exponential_better.h5", "r") as f:
        print(f"Available runs: {list(f.keys())}")

        for run in ["circularization", "merger"]:
            if run in f:
                analyze_run(f, run)
