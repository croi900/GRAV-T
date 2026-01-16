"""
Multi-plotter for mass parameter studies.
Generates comparison plots for different stellar masses.
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


class MultiPlotterMass:
    """
    Plotter for mass parameter studies.
    Compares runs with different M1, M2 values.
    """

    def __init__(self, config: Config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"
        self.output_dir = f"{config.name}/ode_plots/multi"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.datasets: list[tuple[str, float, float]] = []

    def add_dataset(self, run_name: str, m1: float, m2: float):
        """Add dataset with masses in solar masses."""
        self.datasets.append((run_name, m1, m2))
        print(f"MultiPlotterMass: Added '{run_name}' with M1={m1:.2f}, M2={m2:.2f} M_sun")

    def _load_run_data(self, f: h5py.File, run_name: str) -> dict:
        return {
            "a": np.array(f[f"{run_name}/a"]),
            "e": np.array(f[f"{run_name}/e"]),
            "t": np.array(f[f"{run_name}/times"]),
            "m1": np.array(f[f"{run_name}/m1"]),
            "m2": np.array(f[f"{run_name}/m2"]),
        }

    def _compute_derived_quantities(self, data: dict) -> dict:
        """Compute dE/dt, dL/dt, dP/dt with correct masses."""
        original_m1 = self.config.state.M1
        original_m2 = self.config.state.M2
        
        self.config.state.M1 = data["m1_val"] * M_SUN
        self.config.state.M2 = data["m2_val"] * M_SUN
        
        system = BinarySystemModelFast(self.config)
        t = data["t_valid_clip"]
        a = data["a_valid_clip"]
        e = data["e_valid_clip"]
        n_points = len(t)
        
        dEdt = np.zeros(n_points)
        dLdt = np.zeros(n_points)
        dPdt = np.zeros(n_points)

        for i in range(n_points):
            dEdt[i] = system.variable_mass_dEdt(t[i], a[i], e[i])
            dLdt[i] = system.variable_mass_dLdt(t[i], a[i], e[i])
            dPdt[i] = system.dPdt(t[i], a[i], e[i])

        self.config.state.M1 = original_m1
        self.config.state.M2 = original_m2

        derived = {"dEdt": dEdt, "dLdt": dLdt, "dPdt": dPdt}
        if n_points > 1:
            derived["delta_E"] = np.zeros(n_points)
            derived["delta_L"] = np.zeros(n_points)
            derived["delta_P"] = np.zeros(n_points)
            derived["delta_E"][1:] = cumulative_trapezoid(dEdt, t)
            derived["delta_L"][1:] = cumulative_trapezoid(dLdt, t)
            derived["delta_P"][1:] = cumulative_trapezoid(dPdt, t)
        return derived

    def saveplot(self, name: str, xlabel: str, ylabel: str, xlim: tuple = None):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if xlim:
            plt.xlim(xlim)
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
            for run_name, m1, m2 in self.datasets:
                data = self._load_run_data(f, run_name)
                data["m1_val"] = m1
                data["m2_val"] = m2
                data["run_name"] = run_name

                valid_mask = data["e"] > 1e-6
                data["t_valid"] = data["t"][valid_mask]
                data["a_valid"] = data["a"][valid_mask]
                data["e_valid"] = data["e"][valid_mask]
                all_data.append(data)

        t_max = min(d["t"][-1] for d in all_data if len(d["t"]) > 0)
        t_max_valid = min(d["t_valid"][-1] for d in all_data if len(d["t_valid"]) > 0)

        for data in all_data:
            mask = data["t"] <= t_max
            data["t_clip"] = data["t"][mask]
            data["a_clip"] = data["a"][mask]

            mask_valid = data["t_valid"] <= t_max_valid
            data["t_valid_clip"] = data["t_valid"][mask_valid]
            data["a_valid_clip"] = data["a_valid"][mask_valid]
            data["e_valid_clip"] = data["e_valid"][mask_valid]

        labels = [rf"$M_1={d['m1_val']:.1f}M_\odot, M_2={d['m2_val']:.1f}M_\odot$" for d in all_data]
        AU = 1.496e11

        # Semi-major axis
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(data["t_clip"][2:], data["a_clip"][2:] / AU,
                    color=colors[i], label=labels[i], alpha=0.8)
        self.saveplot("t_a_comparison", r"$t$ [s]", r"$a$ [AU]")

        # Eccentricity
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(data["t_valid_clip"][2:], data["e_valid_clip"][2:],
                    color=colors[i], label=labels[i], alpha=0.8)
        self.saveplot("t_e_comparison", r"$t$ [s]", r"$e$", xlim=(0, t_max_valid))

        # Derived quantities
        print("Computing dE/dt, dL/dt, dP/dt...")
        for data in all_data:
            if len(data["t_valid_clip"]) > 1:
                derived = self._compute_derived_quantities(data)
                data.update(derived)

        # dE/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dEdt" in data:
                plt.plot(data["t_valid_clip"][2:], -data["dEdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8)
        self.saveplot("t_dEdt_comparison", r"$t$ [s]", r"$-dE/dt$ [W]", xlim=(0, t_max_valid))

        # dL/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dLdt" in data:
                plt.plot(data["t_valid_clip"][2:], -data["dLdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8)
        self.saveplot("t_dLdt_comparison", r"$t$ [s]", r"$-dL/dt$", xlim=(0, t_max_valid))

        # dP/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dPdt" in data:
                plt.plot(data["t_valid_clip"][2:], data["dPdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8)
        self.saveplot("t_dPdt_comparison", r"$t$ [s]", r"$dP/dt$", xlim=(0, t_max_valid))

        print(f"Plots saved to {self.output_dir}/")
