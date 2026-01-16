"""
Multi-plotter for decay rate parameter studies.
Generates comparison plots for different decay rates.
"""

import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from config import Config
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


class MultiPlotter:
    """
    Plotter for decay rate parameter studies.
    Compares runs with different decay rates.
    """

    def __init__(self, config: Config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"
        self.output_dir = f"{config.name}/ode_plots/multi"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.datasets: list[tuple[str, float, float]] = []

    def add_dataset(self, run_name: str, k_factor: float, actual_decay_rate: float):
        self.datasets.append((run_name, k_factor, actual_decay_rate))
        print(f"MultiPlotter: Added '{run_name}' with decay_rate={actual_decay_rate:.3e}")

    def _load_run_data(self, f: h5py.File, run_name: str) -> dict:
        return {
            "a": np.array(f[f"{run_name}/a"]),
            "e": np.array(f[f"{run_name}/e"]),
            "t": np.array(f[f"{run_name}/times"]),
            "m1": np.array(f[f"{run_name}/m1"]),
            "m2": np.array(f[f"{run_name}/m2"]),
        }

    def _compute_derived_quantities(self, t: np.ndarray, a: np.ndarray, e: np.ndarray) -> dict:
        system = BinarySystemModelFast(self.config)
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
            for run_name, k_factor, actual_decay_rate in self.datasets:
                data = self._load_run_data(f, run_name)
                data["k_factor"] = k_factor
                data["actual_decay_rate"] = actual_decay_rate
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

        # Generate labels
        labels = []
        for d in all_data:
            rate = d["actual_decay_rate"]
            symbol = r"\omega" if self.config.decay_type == "exponential" else "k"
            mantissa = f"{rate:.2e}".split("e")[0]
            exponent = int(f"{rate:.2e}".split("e")[1])
            labels.append(rf"${symbol} = {mantissa} \times 10^{{{exponent}}}$ $s^{{-1}}$")

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
        self.saveplot("t_e_comparison", r"$t$ [s]", r"$e$")

        # Derived quantities
        print("Computing dE/dt, dL/dt, dP/dt...")
        for data in all_data:
            if len(data["t_valid_clip"]) > 1:
                derived = self._compute_derived_quantities(
                    data["t_valid_clip"], data["a_valid_clip"], data["e_valid_clip"]
                )
                data.update(derived)

        # dE/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dEdt" in data:
                plt.plot(data["t_valid_clip"][2:], -data["dEdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8)
        self.saveplot("t_dEdt_comparison", r"$t$ [s]", r"$-dE/dt$ [W]")

        # dL/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dLdt" in data:
                plt.plot(data["t_valid_clip"][2:], -data["dLdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8)
        self.saveplot("t_dLdt_comparison", r"$t$ [s]", r"$-dL/dt$")

        # dP/dt
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            if "dPdt" in data:
                plt.plot(data["t_valid_clip"][2:], data["dPdt"][2:],
                        color=colors[i], label=labels[i], alpha=0.8)
        self.saveplot("t_dPdt_comparison", r"$t$ [s]", r"$dP/dt$")

        print(f"Plots saved to {self.output_dir}/")
