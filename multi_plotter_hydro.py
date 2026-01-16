"""
Multi-plotter for hydrodynamics simulations.
Generates comparison plots for parameter studies (e.g., varying Gamma_Edd).
"""

import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from config import Config, M_SUN
from equations import BinarySystemModelFast

_INKSCAPE_AVAILABLE = shutil.which('inkscape') is not None


def _pdf_to_eps(pdf_path: str, eps_path: str) -> bool:
    """Convert PDF to EPS using inkscape."""
    try:
        subprocess.run(
            ['inkscape', pdf_path, f'--export-filename={eps_path}'],
            check=True, capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class MultiPlotterHydro:
    """
    Plotter for hydrodynamics parameter studies.
    Compares runs with different Gamma_Edd values.
    """

    def __init__(self, config: Config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"
        self.output_dir = f"{config.name}/ode_plots/multi"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.datasets: list[tuple[str, float]] = []

    def add_dataset(self, run_name: str, gamma_val: float):
        """Add a dataset to the plotter."""
        self.datasets.append((run_name, gamma_val))
        print(f"MultiPlotterHydro: Added '{run_name}' with Γ_Edd={gamma_val:.2f}")

    def _load_run_data(self, f: h5py.File, run_name: str) -> dict:
        return {
            "a": np.array(f[f"{run_name}/a"]),
            "e": np.array(f[f"{run_name}/e"]),
            "t": np.array(f[f"{run_name}/times"]),
            "m1": np.array(f[f"{run_name}/m1"]),
            "m2": np.array(f[f"{run_name}/m2"]),
        }

    def _compute_derived_quantities(self, data: dict) -> dict:
        """Compute dE/dt, dL/dt, dP/dt from orbital parameters."""
        system = BinarySystemModelFast(self.config)
        
        t = data["t_clip"]
        a = data["a_clip"]
        e = data["e_clip"]
        n_points = len(t)
        
        dEdt = np.zeros(n_points)
        dLdt = np.zeros(n_points)
        dPdt = np.zeros(n_points)

        for i in range(n_points):
            dEdt[i] = system.variable_mass_dEdt(t[i], a[i], e[i])
            dLdt[i] = system.variable_mass_dLdt(t[i], a[i], e[i])
            dPdt[i] = system.dPdt(t[i], a[i], e[i])

        derived = {"dEdt": dEdt, "dLdt": dLdt, "dPdt": dPdt}

        if n_points > 1:
            derived["delta_E"] = np.zeros(n_points)
            derived["delta_L"] = np.zeros(n_points)
            derived["delta_P"] = np.zeros(n_points)
            derived["delta_E"][1:] = cumulative_trapezoid(dEdt, t)
            derived["delta_L"][1:] = cumulative_trapezoid(dLdt, t)
            derived["delta_P"][1:] = cumulative_trapezoid(dPdt, t)

        return derived

    def saveplot(self, name: str, xlabel: str, ylabel: str):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best", fontsize=10)
        plt.tight_layout()
        base_path = f"{self.output_dir}/{name}"
        plt.savefig(f"{base_path}.png", dpi=150)
        
        if _INKSCAPE_AVAILABLE:
            pdf_path = f"{base_path}.pdf"
            eps_path = f"{base_path}.eps"
            plt.savefig(pdf_path, format='pdf')
            if _pdf_to_eps(pdf_path, eps_path):
                os.remove(pdf_path)
            else:
                plt.savefig(eps_path, format='eps')
                os.remove(pdf_path)
        else:
            plt.savefig(f"{base_path}.eps", format='eps')
        
        plt.close()

    def generate_plots(self):
        if not self.datasets:
            print("No datasets registered.")
            return

        print(f"Generating comparison plots for {len(self.datasets)} datasets...")

        n = len(self.datasets)
        cmap = plt.cm.viridis
        colors = [cmap(i / (n - 1)) if n > 1 else cmap(0.5) for i in range(n)]

        all_data = []
        with h5py.File(self.h5path, "r") as f:
            for run_name, gamma_val in self.datasets:
                data = self._load_run_data(f, run_name)
                data["gamma"] = gamma_val
                data["run_name"] = run_name
                all_data.append(data)

        t_max = min(d["t"][-1] for d in all_data if len(d["t"]) > 0)

        for data in all_data:
            mask = data["t"] <= t_max
            data["t_clip"] = data["t"][mask]
            data["a_clip"] = data["a"][mask]
            data["e_clip"] = data["e"][mask]
            data["m1_clip"] = data["m1"][mask]
            data["m2_clip"] = data["m2"][mask]

        labels = [rf"$\Gamma_{{\rm Edd}} = {d['gamma']:.2f}$" for d in all_data]
        AU = 1.496e11

        # Semi-major axis
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(data["t_clip"][2:], data["a_clip"][2:] / AU,
                    color=colors[i], label=labels[i], alpha=0.8, linewidth=1.5)
        self.saveplot("t_a_comparison", r"$t$ [s]", r"$a$ [AU]")

        # Eccentricity
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(data["t_clip"][2:], data["e_clip"][2:],
                    color=colors[i], label=labels[i], alpha=0.8, linewidth=1.5)
        self.saveplot("t_e_comparison", r"$t$ [s]", r"$e$")
        
        # Donor mass
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(data["t_clip"][2:], data["m1_clip"][2:] / M_SUN,
                    color=colors[i], label=labels[i], alpha=0.8, linewidth=1.5)
        self.saveplot("t_m1_comparison", r"$t$ [s]", r"$M_1$ [$M_\odot$]")

        print("Computing dE/dt, dL/dt, dP/dt...")
        for data in all_data:
            if len(data["t_clip"]) > 1:
                derived = self._compute_derived_quantities(data)
                data.update(derived)

        # dE/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dEdt" in data:
                plt.plot(data["t_clip"][2:], -data["dEdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8, linewidth=1.5)
        self.saveplot("t_dEdt_comparison", r"$t$ [s]", r"$-dE/dt$ [W]")

        # dL/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dLdt" in data:
                plt.plot(data["t_clip"][2:], -data["dLdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8, linewidth=1.5)
        self.saveplot("t_dLdt_comparison", r"$t$ [s]", r"$-dL/dt$")

        # dP/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dPdt" in data:
                plt.plot(data["t_clip"][2:], data["dPdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8, linewidth=1.5)
        self.saveplot("t_dPdt_comparison", r"$t$ [s]", r"$dP/dt$")

        print(f"Plots saved to {self.output_dir}/")
