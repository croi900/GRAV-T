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
from equations import BinarySystemModelFast

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


class MultiPlotter:
    """
    Plotter that accumulates multiple datasets and creates overlayed comparison plots.
    Call add_dataset() for each run, then generate_plots() at the end.
    """

    def __init__(self, config: Config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"
        self.output_dir = f"{config.name}/ode_plots/multi"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.datasets: list[tuple[str, float, float]] = []

    def add_dataset(self, run_name: str, k_factor: float, actual_decay_rate: float):
        self.datasets.append((run_name, k_factor, actual_decay_rate))
        print(
            f"MultiPlotter: Added dataset '{run_name}' with decay_rate={actual_decay_rate:.3e}"
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

        return derived

    def saveplot(
        self,
        name: str,
        xlabel: str = "Unnamed",
        ylabel: str = "Unnamed",
    ):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best", fontsize=8 * 1.25)  # 25% larger legend font
        plt.tight_layout()
        base_path = f"{self.output_dir}/{name}"
        # Save PNG
        plt.savefig(f"{base_path}.png", dpi=150)
        
        # Save EPS - use PDF→EPS conversion if inkscape available (preserves transparency)
        if _INKSCAPE_AVAILABLE:
            pdf_path = f"{base_path}.pdf"
            eps_path = f"{base_path}.eps"
            plt.savefig(pdf_path, format='pdf')
            if _pdf_to_eps(pdf_path, eps_path):
                os.remove(pdf_path)  # Clean up intermediate PDF
            else:
                # Fallback to direct EPS if conversion failed
                plt.savefig(eps_path, format='eps')
                os.remove(pdf_path)
        else:
            # No inkscape available, use direct EPS (no transparency support)
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
            for run_name, k_factor, actual_decay_rate in self.datasets:
                data = self._load_run_data(f, run_name)
                data["k_factor"] = k_factor
                data["actual_decay_rate"] = actual_decay_rate
                data["run_name"] = run_name

                valid_mask = data["e"] > 1e-6
                data["t_valid"] = data["t"][valid_mask]
                data["a_valid"] = data["a"][valid_mask]
                data["e_valid"] = data["e"][valid_mask]
                data["m1_valid"] = data["m1"][valid_mask]
                data["m2_valid"] = data["m2"][valid_mask]

                all_data.append(data)

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

        labels = [
            rf"${'\omega' if self.config.decay_type == 'exponential' else 'k'} = {f'{d["actual_decay_rate"]:.2e}'.split('e')[0]} \times 10^{{{int(f'{d["actual_decay_rate"]:.2e}'.split('e')[1])}}}$"
            for d in all_data
        ]

        # Time to impact plot: X = decay rate, Y = time to impact
        plt.figure(figsize=(6, 6))
        decay_rates = [d["actual_decay_rate"] for d in all_data]
        impact_times = [d["t"][-1] for d in all_data]  # Final time = time to impact
        
        for i, data in enumerate(all_data):
            plt.scatter(
                data["actual_decay_rate"],
                data["t"][-1],
                color=colors[i],
                marker='s',
                s=100,
                alpha=0.8,
                zorder=3,
            )
            # Add label next to each point
            plt.annotate(
                labels[i],
                (data["actual_decay_rate"], data["t"][-1]),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=8 * 1.25,
                color=colors[i],
            )
        
        plt.xscale('log')
        self.saveplot(
            "time_to_impact",
            xlabel=r"Decay Rate $\omega$ [s$^{-1}$]" if self.config.decay_type == 'exponential' else r"Decay Rate $k$ [kg/s]",
            ylabel=r"Time to Impact [s]",
        )

        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(
                data["t_valid_clip"],
                data["e_valid_clip"],
                color=colors[i],
                label=labels[i],
                alpha=0.8,
            )
        self.saveplot(
            "t_e_comparison",
            xlabel=r"$t$ [s]",
            ylabel=r"$e$",
        )

        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(
                data["t_clip"],
                data["m1_clip"],
                color=colors[i],
                label=labels[i],
                alpha=0.8,
            )
        self.saveplot(
            "t_m1_comparison",
            xlabel=r"$t$ [s]",
            ylabel=r"$M_1$ [kg]",
        )

        plt.figure(figsize=(6, 6))
        for i, data in enumerate(all_data):
            plt.plot(
                data["t_clip"],
                data["m2_clip"],
                color=colors[i],
                label=labels[i],
                alpha=0.8,
            )
        self.saveplot(
            "t_m2_comparison",
            xlabel=r"$t$ [s]",
            ylabel=r"$M_2$ [kg]",
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
                if "delta_E" in data:
                    plt.plot(
                        data["t_valid_clip"],
                        data["delta_E"],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_E_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$\Delta E$ [J]",
            )

            plt.figure(figsize=(6, 6))
            for i, data in enumerate(all_data):
                if "delta_L" in data:
                    plt.plot(
                        data["t_valid_clip"],
                        data["delta_L"],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_L_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$\Delta L$ [kg$\cdot$m$^2$/s]",
            )

            plt.figure(figsize=(6, 6))
            for i, data in enumerate(all_data):
                if "delta_P" in data:
                    plt.plot(
                        data["t_valid_clip"],
                        data["delta_P"],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_P_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$\Delta P$ [s]",
            )

            plt.figure(figsize=(6, 6))
            for i, data in enumerate(all_data):
                if "dEdt" in data:
                    plt.plot(
                        data["t_valid_clip"],
                        -data["dEdt"],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_dEdt_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$-dE/dt$ [W]",
            )

            plt.figure(figsize=(6, 6))
            for i, data in enumerate(all_data):
                if "dLdt" in data:
                    plt.plot(
                        data["t_valid_clip"],
                        -data["dLdt"],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_dLdt_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$-dL/dt$ [kg$\cdot$m$^2$/s$^2$]",
            )

            plt.figure(figsize=(6, 6))
            for i, data in enumerate(all_data):
                if "dPdt" in data:
                    plt.plot(
                        data["t_valid_clip"],
                        data["dPdt"],
                        color=colors[i],
                        label=labels[i],
                        alpha=0.8,
                    )
            self.saveplot(
                "t_dPdt_comparison",
                xlabel=r"$t$ [s]",
                ylabel=r"$dP/dt$ [s/s]",
            )

        print(f"Multi-comparison plots saved to {self.output_dir}/")
