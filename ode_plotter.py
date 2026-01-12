import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from config import State
from equations import BinarySystemModelFast
from plotter import Plotter


class ODEPlotter(Plotter):
    def __init__(self, config):
        super().__init__(config)

    def plot(self, run_name):
        with h5py.File(self.h5path, "r") as f:
            a_ds = np.array(f[f"{run_name}/a"])
            e_ds = np.array(f[f"{run_name}/e"])
            t_ds = np.array(f[f"{run_name}/times"])
            m1_ds = np.array(f[f"{run_name}/m1"])
            m2_ds = np.array(f[f"{run_name}/m2"])

        valid_mask = e_ds > 1e-6
        t_valid = t_ds[valid_mask]
        a_valid = a_ds[valid_mask]
        e_valid = e_ds[valid_mask]
        m1_valid = m1_ds[valid_mask]
        m2_valid = m2_ds[valid_mask]

        plt.plot(t_ds, a_ds / 1e3)
        self.saveplot(
            f"{run_name}/t_a", xlabel=r"$t$ [s]", ylabel=r"$a$ [km]"
        )

        plt.plot(t_valid, e_valid)
        self.saveplot(f"{run_name}/t_e", xlabel=r"$t$ [s]", ylabel=r"$e$")

        plt.plot(t_ds, m1_ds)
        self.saveplot(f"{run_name}/t_m1", xlabel=r"$t$ [s]", ylabel=r"$M_1$ [kg]")

        plt.plot(t_ds, m2_ds)
        self.saveplot(f"{run_name}/t_m2", xlabel=r"$t$ [s]", ylabel=r"$M_2$ [kg]")

        print(f"Computing E, L, P for {run_name}...")

        system = BinarySystemModelFast(self.config)

        n_points = len(t_valid)
        dEdt = np.zeros(n_points)
        dLdt = np.zeros(n_points)
        dPdt = np.zeros(n_points)

        for i in range(n_points):
            t_i, a_i, e_i = t_valid[i], a_valid[i], e_valid[i]
            dEdt[i] = system.variable_mass_dEdt(t_i, a_i, e_i)
            dLdt[i] = system.variable_mass_dLdt(t_i, a_i, e_i)
            dPdt[i] = system.dPdt(t_i, a_i, e_i)

        if len(t_valid) > 1:
            delta_E = np.zeros(n_points)
            delta_L = np.zeros(n_points)
            delta_P = np.zeros(n_points)

            delta_E[1:] = cumulative_trapezoid(dEdt, t_valid)
            delta_L[1:] = cumulative_trapezoid(dLdt, t_valid)
            delta_P[1:] = cumulative_trapezoid(dPdt, t_valid)

            plt.plot(t_valid, delta_E)
            self.saveplot(f"{run_name}/t_E", xlabel=r"$t$ [s]", ylabel=r"$\Delta E$ [J]")

            plt.plot(t_valid, delta_L)
            self.saveplot(f"{run_name}/t_L", xlabel=r"$t$ [s]", ylabel=r"$\Delta L$ [kg$\cdot$m$^2$/s]")

            plt.plot(t_valid, delta_P)
            self.saveplot(f"{run_name}/t_P", xlabel=r"$t$ [s]", ylabel=r"$\Delta P$ [s]")

            plt.plot(t_valid, -dEdt)
            self.saveplot(f"{run_name}/t_dEdt", xlabel=r"$t$ [s]", ylabel=r"$-dE/dt$ [W]")

            plt.plot(t_valid, -dLdt)
            self.saveplot(
                f"{run_name}/t_dLdt", xlabel=r"$t$ [s]", ylabel=r"$-dL/dt$ [kg$\cdot$m$^2$/s$^2$]"
            )

            plt.plot(t_valid, dPdt)
            self.saveplot(
                f"{run_name}/t_dPdt", xlabel=r"$t$ [s]", ylabel=r"$dP/dt$ [s/s]"
            )

        print(f"ODE plots saved for {run_name}")
