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

from multi_plotter import MultiPlotter

from integration_run import IntegrationRun
from name_maps import domain_type_map
from orbit_plotter import OrbitPlotter
from polarization_plotter import PolarizationPlotter


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

    if not os.path.exists(config.name):
        os.mkdir(config.name)

    cotime, _ = BinarySystemModelFast(config).coalescence_time(
        config.cotime_a_min,
        config.cotime_max_time,
        rtol=1e-9,
        atol=1e-12,
        method=config.method,
    )

    print("COALESCENCE TIME: {:5e}".format(cotime))

    if config.initial_points_exponent > 0:
        t_eval = domain_gen.exponential_domain(
            0, cotime, 2, config.exponent_offset, config.initial_points_exponent
        )
    else:
        t_eval = domain_gen.uniform_domain(
            0, cotime - config.merger_seconds, int(config.output_points // 10)
        )

    circularization = IntegrationRun(
        "circularization",
        config,
        0,
        cotime - config.merger_seconds,
        t_eval=t_eval,
        decay_type=config.decay_type,
        solver=config.method,
    )

    merger_cfg = copy.deepcopy(config)
    merger_cfg.state = circularization.run()

    if config.initial_points_exponent > 0:
        t_eval = domain_gen.exponential_domain(
            0, cotime, 2, config.exponent_offset, config.initial_points_exponent
        )
    else:
        merger_window = merger_cfg.merger_seconds * 2.0
        t_eval = domain_gen.uniform_domain(0, merger_window, int(config.output_points))
        print(
            f"Merger sampling: {config.output_points} points over {merger_window}s (2x margin)"
        )
        print(
            f"  → Resolution: {merger_window / config.output_points * 1e6:.2f} μs per point"
        )

    merger = IntegrationRun(
        "merger",
        merger_cfg,
        0,
        merger_cfg.merger_seconds * 2.0,
        t_eval=t_eval,
        decay_type=config.decay_type,
        solver=config.method,
    )

    merger.run()
    # plotter = MultiPlotter(config)
    # plotter.plot("circularization")

    plotter = PolarizationPlotter(config)
    plotter.plot("merger")

    # plotter = OrbitPlotter(config)

    #
