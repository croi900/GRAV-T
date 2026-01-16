"""
Parameter study for decay rate variations.
Compares orbital evolution with different mass decay rates.
"""

import os
import argparse
import numpy as np

import domain_gen
from config import Config
from equations import BinarySystemModelFast
from integration_run import IntegrationRun
from multi_plotter import MultiPlotter


def parse_args():
    parser = argparse.ArgumentParser(description="Decay rate parameter study")
    parser.add_argument("--problem", type=str, required=True, help="Problem file TOML")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.problem.endswith(".toml"):
        print("Problem file must be a TOML file")
        exit(1)

    config = Config(args.problem)
    config.output_points //= 100
    initial_decay_rate = config.state.decay_rate
    
    if not os.path.exists(config.name):
        os.mkdir(config.name)

    multi_plotter = MultiPlotter(config)
    
    # FOR LINEAR: k = [1, 100, 200, 300]
    # FOR EXPONENTIAL: k = [1, 10, 20, 30]
    # FOR LANDER: k = [0.9, 1, 1.1, 1.2]
    k_values = np.array([1, 100, 200, 300])
    
    for i, k in enumerate(k_values):
        config.state.decay_rate = k * initial_decay_rate

        cotime, _ = BinarySystemModelFast(config).coalescence_time(
            config.cotime_a_min,
            config.cotime_max_time,
            rtol=1e-9,
            atol=1e-12,
            method=config.method,
        )

        print(f"COALESCENCE TIME: {cotime:.5e}")
        print(f"COMPUTING FOR: k={k}")
        
        if config.initial_points_exponent > 0:
            t_eval = domain_gen.exponential_domain(
                0, cotime, 2, config.exponent_offset, config.initial_points_exponent
            )
        else:
            t_eval = domain_gen.uniform_domain(0, cotime * 1.01, config.output_points)

        run = IntegrationRun(
            f"k_{i}_circ_{k}",
            config,
            0,
            cotime - config.merger_seconds,
            decay_type=config.decay_type,
            solver=config.method,
        )
        run.run()

        multi_plotter.add_dataset(f"k_{i}_circ_{k}", k, config.state.decay_rate)

    multi_plotter.generate_plots()
