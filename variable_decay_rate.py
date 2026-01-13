import copy
import os
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import h5py
from scipy import constants
from scipy.integrate import RK45, Radau, DOP853, LSODA
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

import domain_gen
from ode_plotter import ODEPlotter
from config import Config
from equations import *
import tomli as tomllib
import argparse

from integration_run import IntegrationRun
from name_maps import domain_type_map
from orbit_plotter import OrbitPlotter
from polarization_plotter import PolarizationPlotter

from multi_plotter import MultiPlotter


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--problem", type=str, help="Problem file TOML")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.problem or not args.problem.endswith(".toml"):
        print("Problem file be a TOML file / Unspecified problem")
        exit(1)

    config = Config(args.problem)
    config.output_points //= 100
    initial_decay_rate = config.state.decay_rate
    if not os.path.exists(config.name):
        os.mkdir(config.name)

    multi_plotter = MultiPlotter(config)
    # for i, k in enumerate(np.array([1, 10, 20, 30])): # FOR EXPONENTIAL
    for i, k in enumerate(np.array([1, 100, 200, 300])): # FOR LINEAR
    # for i, k in enumerate(np.array([1])): # FOR LANDER
        config.state.decay_rate = k * initial_decay_rate

        cotime, _ = BinarySystemModelFast(config).coalescence_time(
            config.cotime_a_min,
            config.cotime_max_time,
            rtol=1e-9,
            atol=1e-12,
            method=config.method,
        )

        print("COALESCENCE TIME: {:5e}".format(cotime))
        print(f"COMPUTING FOR: {k}")
        if config.initial_points_exponent > 0:
            t_eval = domain_gen.exponential_domain(
                0, cotime, 2, config.exponent_offset, config.initial_points_exponent
            )
        else:
            t_eval = domain_gen.uniform_domain(
                0, cotime * 1.01, int(config.output_points)
            )

        circularization = IntegrationRun(
            f"k_{i}_circ_{k}",
            config,
            0,
            cotime - config.merger_seconds,
            decay_type=config.decay_type,
            solver=config.method,
        )
        circularization.run()

        multi_plotter.add_dataset(f"k_{i}_circ_{k}", k, config.state.decay_rate)

    multi_plotter.generate_plots()
