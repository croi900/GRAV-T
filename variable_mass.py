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

from multi_plotter_mass import MultiPlotterMass
from config import M_SUN

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
    
    if not os.path.exists(config.name):
        os.mkdir(config.name)

    multi_plotter = MultiPlotterMass(config)
  

    for i, k in enumerate(np.array([1.2, 1.4, 1.6, 1.8])):
        for j, m in enumerate(np.array([1.4])) :
            config.state.M1 = k * M_SUN 
            config.state.M2 = m * M_SUN

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
                f"k_{i}_m_{j}_circ_{k}_{m}",
                config,
                0,
                cotime - config.merger_seconds,
                decay_type=config.decay_type,
                solver=config.method,
            )
            circularization.run()

            multi_plotter.add_dataset(f"k_{i}_m_{j}_circ_{k}_{m}", k, m)

    multi_plotter.generate_plots()
