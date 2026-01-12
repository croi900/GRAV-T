import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from config import State
from equations import BinarySystemModelFast
from plotter import Plotter
from polarizations import compute_phi_r_ode, get_waveforms

G = 6.674e-11


def compute_merger_peak(t_ds, hp, hx, a_ds, e_ds, m1_ds, m2_ds, D_obs):
    """
    Compute the merger peak values for strain and frequency.
    """
    M_total = m1_ds[-1] + m2_ds[-1]
    c = SPEED_OF_LIGHT

    f_ISCO = c**3 / (np.pi * G * M_total * 6**1.5)

    h_amp = np.sqrt(hp[-1] ** 2 + hx[-1] ** 2)
    h_peak = h_amp * 2.0

    dt = t_ds[-1] - t_ds[-2] if len(t_ds) > 1 else 1e-4
    t_peak = t_ds[-1] + dt

    return t_peak, h_peak, 0.0, f_ISCO


def find_chirp_region(t_ds, h_amp, window_seconds=1.0):
    """
    Find the chirp region where amplitude increases most rapidly.

    Returns the time range [t_start, t_end] for the chirp.
    """

    dh_dt = np.gradient(h_amp, t_ds)

    dh_dt_smooth = gaussian_filter1d(dh_dt, sigma=min(100, len(dh_dt) // 100))

    max_rate_idx = np.argmax(dh_dt_smooth)
    t_max_rate = t_ds[max_rate_idx]

    t_chirp_end = t_ds[-1]
    t_chirp_start = max(t_ds[0], t_chirp_end - window_seconds)

    return t_chirp_start, t_chirp_end, max_rate_idx


def find_merger_region(t_ds, h_amp, threshold_fraction=0.01):
    """
    Find the merger region where the GW signal becomes interesting.

    Detects where the strain amplitude starts increasing significantly above its
    baseline. This allows automatic zooming to the interesting part of the waveform
    even for long simulations like "circularization" that contain the merger.

    IMPORTANT: The merger is defined as the PEAK amplitude. We only include data
    up to the merger, not after it (post-merger data is unphysical in this model).

    Args:
        t_ds: Array of time values
        h_amp: Array of strain amplitude values
        threshold_fraction: Fraction of max amplitude to use as detection threshold

    Returns:
        (merger_start_idx, merger_peak_idx): Start and end indices of the interesting region
    """
    if len(h_amp) < 10:
        return 0, len(h_amp) - 1

    merger_peak_idx = np.argmax(h_amp)

    h_amp_inspiral = h_amp[: merger_peak_idx + 1]

    if len(h_amp_inspiral) < 10:
        return 0, merger_peak_idx

    h_max = h_amp[merger_peak_idx]
    h_min = np.min(h_amp_inspiral)
    h_range = h_max - h_min

    if h_range < 1e-100:
        return 0, merger_peak_idx

    h_normalized = (h_amp_inspiral - h_min) / h_range

    threshold = threshold_fraction
    above_threshold = h_normalized > threshold

    if not np.any(above_threshold):
        return 0, merger_peak_idx

    merger_start_idx = np.argmax(above_threshold)

    lead_up_points = max(1, (merger_peak_idx - merger_start_idx) // 10)
    merger_start_idx = max(0, merger_start_idx - lead_up_points)

    return merger_start_idx, merger_peak_idx


def compute_gw_frequency(hp, hx, t_ds, a=None, M=None):
    """
    Compute instantaneous GW frequency.

    If orbital parameters (a, M) are provided, uses the physically correct formula:
        f_GW = 2 × f_orbital = (1/π) × sqrt(G×M/a³)

    Otherwise falls back to phase derivative method (less accurate for coarse sampling).
    """
    if a is not None and M is not None:
        f_orbital = (1 / (2 * np.pi)) * np.sqrt(G * M / a**3)
        f_GW = 2 * f_orbital
        return f_GW

    h_squared = hp**2 + hx**2
    h_squared = np.where(h_squared < 1e-100, 1e-100, h_squared)

    dhp_dt = np.gradient(hp, t_ds)
    dhx_dt = np.gradient(hx, t_ds)

    f_GW = (1 / (2 * np.pi)) * (hp * dhx_dt - hx * dhp_dt) / h_squared

    return np.abs(f_GW)


class PolarizationPlotter(Plotter):
    def __init__(self, config):
        super().__init__(config)

    def plot(self, run_name):
        print(f"\n{'=' * 60}")
        print(f"Polarization analysis for {run_name}")
        print(f"{'=' * 60}")

        with h5py.File(self.h5path, "r") as f:
            a_ds = np.array(f[f"{run_name}/a"])
            e_ds = np.array(f[f"{run_name}/e"])
            t_ds = np.array(f[f"{run_name}/times"])
            m1_ds = np.array(f[f"{run_name}/m1"])
            m2_ds = np.array(f[f"{run_name}/m2"])

        valid_mask = e_ds > 0
        a_ds = a_ds[valid_mask]
        t_ds = t_ds[valid_mask]
        m1_ds = m1_ds[valid_mask]
        m2_ds = m2_ds[valid_mask]
        e_ds = e_ds[valid_mask]

        M_total_init = m1_ds[0] + m2_ds[0]
        r_isco = 6 * G * M_total_init / SPEED_OF_LIGHT**2
        print(f"  ISCO radius: {r_isco / 1e3:.1f} km")

        isco_mask = a_ds > r_isco
        if not np.all(isco_mask):
            isco_idx = np.argmax(~isco_mask)
            print(
                f"  Filtering: orbit reaches ISCO at index {isco_idx} (t={t_ds[isco_idx]:.6e}s)"
            )

            a_ds = a_ds[:isco_idx]
            t_ds = t_ds[:isco_idx]
            m1_ds = m1_ds[:isco_idx]
            m2_ds = m2_ds[:isco_idx]
            e_ds = e_ds[:isco_idx]
        else:
            print(f"  All data is above ISCO (a_min={a_ds[-1] / 1e3:.1f} km)")

        print(f"  Data points (after ISCO filter): {len(t_ds)}")
        print(
            f"  Time range: {t_ds[0]:.6e} to {t_ds[-1]:.6e} s (span: {t_ds[-1] - t_ds[0]:.6e} s)"
        )
        print(f"  Semi-major axis: {a_ds[0] / 1e3:.1f} to {a_ds[-1] / 1e3:.1f} km")
        print(f"  Eccentricity: {e_ds[0]:.6f} to {e_ds[-1]:.6f}")

        phi_ds, r_ds = compute_phi_r_ode(t_ds, a_ds, e_ds, m1_ds + m2_ds)

        M_total = m1_ds + m2_ds
        mu = m1_ds * m2_ds / M_total

        hp, hx = get_waveforms(
            phi_ds,
            a_ds,
            e_ds,
            M_total,
            mu,
            self.config.observer_distance,
        )

        h_amp = np.sqrt(hp**2 + hx**2)

        f_GW = compute_gw_frequency(hp, hx, t_ds, a=a_ds, M=M_total)

        print(f"  Computed f_GW range: {f_GW[0]:.1f} to {f_GW[-1]:.1f} Hz")
        print(f"  Amplitude range: {h_amp[0]:.4e} to {h_amp[-1]:.4e}")

        dhp_dt = np.gradient(hp, t_ds)
        dhx_dt = np.gradient(hx, t_ds)
        F_GW = (SPEED_OF_LIGHT**3 / (16 * np.pi * G)) * (dhp_dt**2 + dhx_dt**2)

        print(f"Saving polarization data to HDF5...")
        with h5py.File(self.h5path, "a") as f:
            grp = f.require_group(f"{run_name}/polarizations")

            def save_dataset(name, data):
                if name in grp:
                    del grp[name]
                grp.create_dataset(name, data=data, compression="gzip")

            save_dataset("h_plus", hp)
            save_dataset("h_cross", hx)
            save_dataset("h_amplitude", h_amp)
            save_dataset("phi", phi_ds)
            save_dataset("r", r_ds)
            save_dataset("f_GW", f_GW)
            save_dataset("F_GW", F_GW)
            save_dataset("times", t_ds)

            grp.attrs["observer_distance_m"] = self.config.observer_distance
            grp.attrs["n_points"] = len(t_ds)
            grp.attrs["t_start"] = t_ds[0]
            grp.attrs["t_end"] = t_ds[-1]
            grp.attrs["h_amp_start"] = float(h_amp[0])
            grp.attrs["h_amp_end"] = float(h_amp[-1])
            grp.attrs["amplitude_ratio"] = (
                float(h_amp[-1] / h_amp[0]) if h_amp[0] > 0 else 0
            )

        print(f"  Saved: h_plus, h_cross, h_amplitude, phi, r, f_GW, F_GW")

        merger_start_idx, merger_peak_idx = find_merger_region(
            t_ds, h_amp, threshold_fraction=0.01
        )

        n_points = len(t_ds)
        needs_focus = n_points > 10000

        print(f"  Total points: {n_points}")
        print(
            f"  Merger peak at t={t_ds[merger_peak_idx]:.4e}s (index {merger_peak_idx})"
        )
        print(f"  Interesting region: indices {merger_start_idx} to {merger_peak_idx}")
        print(f"  Creating focused plots: {needs_focus}")

        t_focus = t_ds[merger_start_idx : merger_peak_idx + 1]
        hp_focus = hp[merger_start_idx : merger_peak_idx + 1]
        hx_focus = hx[merger_start_idx : merger_peak_idx + 1]
        h_amp_focus = h_amp[merger_start_idx : merger_peak_idx + 1]
        f_GW_focus = f_GW[merger_start_idx : merger_peak_idx + 1]
        F_GW_focus = F_GW[merger_start_idx : merger_peak_idx + 1]
        phi_focus = phi_ds[merger_start_idx : merger_peak_idx + 1]
        r_focus = r_ds[merger_start_idx : merger_peak_idx + 1]

        t_before_merger = t_focus - t_focus[-1]

        print(f"Creating standard plots...")

        plt.plot(t_ds, phi_ds)
        self.saveplot(
            f"{run_name}/phi_t", xlabel=r"$t$ [s]", ylabel=r"$\phi$ [rad]"
        )

        plt.plot(t_ds, hp)
        self.saveplot(f"{run_name}/t_hp", xlabel=r"$t$ [s]", ylabel=r"$h_+$")

        plt.plot(t_ds, hx)
        self.saveplot(f"{run_name}/t_hx", xlabel=r"$t$ [s]", ylabel=r"$h_\times$")

        plt.plot(t_ds, r_ds / 1e3)
        self.saveplot(
            f"{run_name}/t_r", xlabel=r"$t$ [s]", ylabel=r"$r$ [km]"
        )

        plt.plot(t_ds, f_GW)
        self.saveplot(f"{run_name}/t_f_GW", xlabel=r"$t$ [s]", ylabel=r"$f_{GW}$ [Hz]")

        plt.plot(t_ds, h_amp)
        self.saveplot(f"{run_name}/t_h_amp", xlabel=r"$t$ [s]", ylabel=r"$|h|$")

        plt.plot(t_ds, F_GW)
        self.saveplot(f"{run_name}/t_F_GW", xlabel=r"$t$ [s]", ylabel=r"$F_{GW}$ [W/m²]")

        plt.semilogy(t_ds, np.abs(F_GW) + 1e-100)
        self.saveplot(
            f"{run_name}/t_F_GW_log", xlabel=r"$t$ [s]", ylabel=r"$F_{GW}$ [W/m²]"
        )

        if len(t_focus) > 10:
            print(f"Creating focused merger region plots ({len(t_focus)} points)...")

            plt.figure(figsize=(12, 4))
            plt.plot(t_before_merger, hp_focus, "b-", linewidth=0.5)
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$h_+$")
            plt.xlim(-1, 0)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.saveplot(f"{run_name}/focus_hp")

            plt.figure(figsize=(12, 4))
            plt.plot(t_before_merger, hx_focus, "r-", linewidth=0.5)
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$h_\times$")
            plt.xlim(-1, 0)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.saveplot(f"{run_name}/focus_hx")

            plt.figure(figsize=(10, 6))
            plt.plot(t_before_merger, h_amp_focus, "b-", linewidth=1)
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$|h|$")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.saveplot(f"{run_name}/focus_h_amp")

            plt.figure(figsize=(10, 6))
            plt.plot(t_before_merger, f_GW_focus, "g-", linewidth=1)
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$f_{GW}$ [Hz]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.saveplot(f"{run_name}/focus_f_GW")

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

            axes[0].plot(t_before_merger, hp_focus, "b-", linewidth=0.5, label="h₊")
            axes[0].plot(
                t_before_merger, hx_focus, "r-", linewidth=0.5, alpha=0.5, label="h×"
            )
            axes[0].fill_between(
                t_before_merger,
                -h_amp_focus,
                h_amp_focus,
                alpha=0.2,
                color="blue",
                label="Envelope",
            )
            axes[0].set_ylabel(r"$h$")
            axes[0].legend(loc="upper left", fontsize=8)
            axes[0].grid(True, alpha=0.3)

            axes[1].semilogy(-t_before_merger + 1e-10, h_amp_focus)
            axes[1].set_ylabel(r"$|h|$")
            axes[1].grid(True, alpha=0.3)
            axes[1].invert_xaxis()

            axes[2].plot(t_before_merger, f_GW_focus, "g-", linewidth=1)
            axes[2].set_ylabel(r"$f_{GW}$ [Hz]")
            axes[2].set_xlabel(r"$t$ [s]")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            self.saveplot(f"{run_name}/focus_combined")

        print(f"Creating chirp analysis plots...")

        t_chirp_start, t_chirp_end, peak_idx = find_chirp_region(
            t_ds, h_amp, window_seconds=1.0
        )
        chirp_mask = (t_ds >= t_chirp_start) & (t_ds <= t_chirp_end)

        if np.sum(chirp_mask) > 10:
            t_chirp = t_ds[chirp_mask]
            hp_chirp = hp[chirp_mask]
            hx_chirp = hx[chirp_mask]
            h_amp_chirp = h_amp[chirp_mask]
            f_GW_chirp = f_GW[chirp_mask]

            t_before_merger = t_chirp - t_chirp[-1]

            plt.figure(figsize=(12, 4))
            plt.plot(t_before_merger, hp_chirp, "b-", linewidth=0.5)
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$h_+$")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.saveplot(f"{run_name}/chirp_hp")

            plt.figure(figsize=(10, 6))
            plt.semilogy(-t_before_merger + 1e-10, h_amp_chirp)
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$|h|$")
            plt.gca().invert_xaxis()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.saveplot(f"{run_name}/chirp_amplitude")

            plt.figure(figsize=(10, 6))
            plt.plot(t_before_merger, f_GW_chirp)
            plt.xlabel(r"$t$ [s]")
            plt.ylabel(r"$f_{GW}$ [Hz]")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self.saveplot(f"{run_name}/chirp_frequency")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            ax1.plot(t_before_merger, hp_chirp, "b-", linewidth=0.5, label="h₊")
            ax1.fill_between(
                t_before_merger,
                -h_amp_chirp,
                h_amp_chirp,
                alpha=0.3,
                color="blue",
                label="Envelope",
            )
            ax1.set_ylabel(r"$h$")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            ax2.plot(t_before_merger, f_GW_chirp, "r-", linewidth=1)
            ax2.set_ylabel(r"$f$ [Hz]")
            ax2.set_xlabel(r"$t$ [s]")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            self.saveplot(f"{run_name}/chirp_combined")

            plt.figure(figsize=(10, 6))

            sort_idx = np.argsort(f_GW_chirp)
            f_sorted = f_GW_chirp[sort_idx]
            h_sorted = h_amp_chirp[sort_idx]
            plt.loglog(f_sorted, h_sorted)
            plt.xlabel(r"$f$ [Hz]")
            plt.ylabel(r"$|h|$")
            plt.grid(True, alpha=0.3, which="both")
            plt.tight_layout()
            self.saveplot(f"{run_name}/strain_vs_frequency")

        for window_sec in [10.0, 1.0, 0.1]:
            t_start = t_ds[-1] - window_sec
            mask = t_ds >= t_start

            if np.sum(mask) > 10:
                t_window = t_ds[mask] - t_ds[-1]
                hp_window = hp[mask]
                h_amp_window = h_amp[mask]
                f_GW_window = f_GW[mask]

                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

                axes[0].plot(t_window, hp_window, "b-", linewidth=0.5)
                axes[0].set_ylabel(r"$h_+$")
                axes[0].grid(True, alpha=0.3)

                axes[1].semilogy(-t_window + 1e-10, h_amp_window)
                axes[1].set_ylabel(r"$|h|$")
                axes[1].grid(True, alpha=0.3)
                axes[1].invert_xaxis()

                axes[2].plot(t_window, f_GW_window, "r-")
                axes[2].set_ylabel(r"$f_{GW}$ [Hz]")
                axes[2].set_xlabel(r"$t$ [s]")
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                self.saveplot(f"{run_name}/final_{window_sec:.1f}s")

        print(f"\n--- Summary for {run_name} ---")
        print(f"  Duration: {t_ds[-1] - t_ds[0]:.4e} s")
        print(
            f"  Amplitude: {h_amp[0]:.4e} → {h_amp[-1]:.4e} (ratio: {h_amp[-1] / h_amp[0]:.2f}x)"
        )
        print(f"  Frequency: {f_GW[0]:.2f} → {f_GW[-1]:.2f} Hz")
        print(f"  Polarization plots saved for {run_name}")
