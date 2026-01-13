import os
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from config import Config
from equations import BinarySystemModelFast, AnalyticMassFunction

# Check if inkscape is available for PDF→EPS conversion with transparency
_INKSCAPE_AVAILABLE = shutil.which('inkscape') is not None


def _pdf_to_eps(pdf_path: str, eps_path: str) -> bool:
    """Convert PDF to EPS using inkscape. Returns True on success."""
    try:
        subprocess.run(
            ['inkscape', pdf_path, f'--export-filename={eps_path}'],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class MultiPlotterMass:
    """
    Plotter that accumulates multiple datasets varying by Mass (M1, M2) and creates overlayed comparison plots.
    Call add_dataset() for each run, then generate_plots() at the end.
    """

    def __init__(self, config: Config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"
        self.output_dir = f"{config.name}/ode_plots/multi"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Store run_name, m1, m2
        self.datasets: list[tuple[str, float, float]] = []

    def add_dataset(self, run_name: str, m1: float, m2: float):
        """
        Add a dataset to the plotter.
        Args:
            run_name: Name of the HDF5 group
            m1: Mass 1 in Solar Masses
            m2: Mass 2 in Solar Masses
        """
        self.datasets.append((run_name, m1, m2))
        print(
            f"MultiPlotterMass: Added dataset '{run_name}' with M1={m1:.2f} M_sun, M2={m2:.2f} M_sun"
        )

    def _load_run_data(self, f: h5py.File, run_name: str) -> dict:
        return {
            "a": np.array(f[f"{run_name}/a"]),
            "e": np.array(f[f"{run_name}/e"]),
            "t": np.array(f[f"{run_name}/times"]),
            "m1": np.array(f[f"{run_name}/m1"]),
            "m2": np.array(f[f"{run_name}/m2"]),
        }

    def _compute_derived_quantities(
        self, data: dict, t_valid: np.ndarray, a_valid: np.ndarray, e_valid: np.ndarray
    ) -> dict:
        # Crucial: Update config with the correct masses for this run to compute derived quantities correctly
        # We assume masses are constant parameters in config but varying here?
        # BinarySystemModelFast uses config.state.M1/M2 if passed config
        
        # Save original state
        original_m1 = self.config.state.M1
        original_m2 = self.config.state.M2
        
        # Update state (converting M_sun to kg handled where? 
        # In variable_mass.py: config.state.m1 = k * M_SUN. 
        # Note: Config object has properties M1, M2 that might read from state?
        # Let's check how BinarySystemModelFast initializes.
        # It reads config.state.M1 / M2.
        
        # We need to ensure we set the correct values. 
        # data["m1_val"] and data["m2_val"] stored in generate_plots are in Solar Masses?
        # variable_mass.py passes m, k (which are multipliers of M_SUN).
        # stored in self.datasets as multipliers.
        from config import M_SUN
        
        self.config.state.M1 = data["m1_val"] * M_SUN
        self.config.state.M2 = data["m2_val"] * M_SUN
        
        # Also need to reset mass functions? 
        # BinarySystemModelFast __init__ creates mass_fun_1/2 based on decay_rate.
        # If decay rate is constant across mass variations, that's fine.
        
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

        derived = {
            "dEdt": dEdt,
            "dLdt": dLdt,
            "dPdt": dPdt,
        }

        if n_points > 1:
            delta_E = np.zeros(n_points)
            delta_L = np.zeros(n_points)
            delta_P = np.zeros(n_points)

            delta_E[1:] = cumulative_trapezoid(dEdt, t_valid)
            delta_L[1:] = cumulative_trapezoid(dLdt, t_valid)
            delta_P[1:] = cumulative_trapezoid(dPdt, t_valid)

            derived.update(
                {
                    "delta_E": delta_E,
                    "delta_L": delta_L,
                    "delta_P": delta_P,
                }
            )

        # Restore original state
        self.config.state.M1 = original_m1
        self.config.state.M2 = original_m2

        return derived

    def saveplot(
        self,
        name: str,
        xlabel: str = "Unnamed",
        ylabel: str = "Unnamed",
        xlim: tuple = None,
    ):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if xlim is not None:
            plt.xlim(xlim)
        plt.legend(loc="best", fontsize=8 * 1.25)  # 25% larger legend font
        plt.tight_layout()
        base_path = f"{self.output_dir}/{name}"
        # Save PNG
        plt.savefig(f"{base_path}.png", dpi=150)
        
        # Save EPS
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
            print("No datasets registered. Call add_dataset() first.")
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
                data["m1_valid"] = data["m1"][valid_mask]
                data["m2_valid"] = data["m2"][valid_mask]

                all_data.append(data)

        # Calculate common x-axis limit
        t_max = min(data["t"][-1] for data in all_data if len(data["t"]) > 0)
        t_max_valid = min(
            data["t_valid"][-1] for data in all_data if len(data["t_valid"]) > 0
        )

        for data in all_data:
            clip_mask = data["t"] <= t_max
            data["t_clip"] = data["t"][clip_mask]
            data["a_clip"] = data["a"][clip_mask]
            data["m1_clip"] = data["m1"][clip_mask]
            data["m2_clip"] = data["m2"][clip_mask]

            clip_mask_valid = data["t_valid"] <= t_max_valid
            data["t_valid_clip"] = data["t_valid"][clip_mask_valid]
            data["a_valid_clip"] = data["a_valid"][clip_mask_valid]
            data["e_valid_clip"] = data["e_valid"][clip_mask_valid]
            data["m1_valid_clip"] = data["m1_valid"][clip_mask_valid]
            data["m2_valid_clip"] = data["m2_valid"][clip_mask_valid]

        # Labels based on Mass
        labels = [
            rf"$M_1={d['m1_val']:.1f}M_\odot, M_2={d['m2_val']:.1f}M_\odot$"
            for d in all_data
        ]

        AU = 1.496e11  # meters per AU
        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(
                data["t_clip"][2:], 
                data["a_clip"][2:] / AU,
                color=colors[i],
                label=labels[i],
                alpha=0.8,
            )
        self.saveplot(
            "t_a_comparison",
            xlabel=r"$t$ [s]",
            ylabel=r"$a$ [AU]",
            xlim=(0, t_max_valid),
        )

        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(
                data["t_valid_clip"][2:],
                data["e_valid_clip"][2:],
                color=colors[i],
                label=labels[i],
                alpha=0.8,
            )
        self.saveplot(
            "t_e_comparison",
            xlabel=r"$t$ [s]",
            ylabel=r"$e$",
            xlim=(0, t_max_valid),
        )

        print("Computing E, L, P for all datasets...")
        for data in all_data:
            if len(data["t_valid_clip"]) > 1:
                derived = self._compute_derived_quantities(
                    data,
                    data["t_valid_clip"],
                    data["a_valid_clip"],
                    data["e_valid_clip"],
                )
                data.update(derived)

        has_derived = all(
            "delta_E" in d for d in all_data if len(d["t_valid_clip"]) > 1
        )

        if has_derived:
            plt.figure(figsize=(6, 6))
            for i, data in enumerate(all_data):
                if "dEdt" in data:
                    plt.plot(
                        data["t_valid_clip"][2:],
                        -data["dEdt"][2:],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_dEdt_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$-dE/dt$ [W]",
                xlim=(0, t_max_valid),
            )

            plt.figure(figsize=(6, 6))
            for i, data in enumerate(all_data):
                if "dLdt" in data:
                    plt.plot(
                        data["t_valid_clip"][2:],
                        -data["dLdt"][2:],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_dLdt_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$-dL/dt$ [kg$\cdot$m$^2$/s$^2$]",
                xlim=(0, t_max_valid),
            )

            plt.figure(figsize=(6, 6))
            for i, data in enumerate(all_data):
                if "dPdt" in data:
                    plt.plot(
                        data["t_valid_clip"][2:],
                        data["dPdt"][2:],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_dPdt_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$dP/dt$ [s/s]",
                xlim=(0, t_max_valid),
            )

        print(f"Multi-comparison plots saved to {self.output_dir}/")
