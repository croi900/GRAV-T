"""
Plots-only mode: Generate plots from existing HDF5 data without running integration.

Usage:
    python plots_only.py --problem <problem.toml>

This script loads the configuration and runs all plotters on existing data in the HDF5 file.
"""

import os
import argparse

import h5py
import tomli as tomllib

from config import Config
from ode_plotter import ODEPlotter
from polarization_plotter import PolarizationPlotter
from orbit_plotter import OrbitPlotter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots from existing HDF5 simulation data"
    )
    parser.add_argument("--problem", type=str, required=True, help="Problem file TOML")
    return parser.parse_args()


def get_available_runs(h5path: str) -> list[str]:
    """Discover available run names in the HDF5 file."""
    runs = []
    with h5py.File(h5path, "r") as f:
        for key in f.keys():
            # Check if this is a run group (has required datasets)
            if isinstance(f[key], h5py.Group) and "a" in f[key] and "e" in f[key]:
                runs.append(key)
    return runs


if __name__ == "__main__":
    args = parse_args()

    if not args.problem or not args.problem.endswith(".toml"):
        print("Problem file must be a TOML file")
        exit(1)

    config = Config(args.problem)
    h5path = f"{config.name}/{config.name}.h5"

    if not os.path.exists(h5path):
        print(f"HDF5 file not found: {h5path}")
        print("Run the main simulation first to generate data.")
        exit(1)

   
    pol_plotter = PolarizationPlotter(config)
    pol_plotter.plot("merger")
