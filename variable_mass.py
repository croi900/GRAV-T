"""
Parameter study for mass variations.
Compares orbital evolution with different stellar masses.
"""

import os
import argparse
import numpy as np

import domain_gen
from config import Config, M_SUN
from equations import BinarySystemModelFast
from integration_run import IntegrationRun
from multi_plotter_mass import MultiPlotterMass


def parse_args():
    parser = argparse.ArgumentParser(description="Mass parameter study")
    parser.add_argument("--problem", type=str, required=True, help="Problem file TOML")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.problem.endswith(".toml"):
        print("Problem file must be a TOML file")
        exit(1)

    config = Config(args.problem)
    config.output_points //= 100
    
    if not os.path.exists(config.name):
        os.mkdir(config.name)

    multi_plotter = MultiPlotterMass(config)
    
    m1_values = np.array([1.2, 1.4, 1.6, 1.8])
    m2_values = np.array([1.4])
    
    for i, m1 in enumerate(m1_values):
        for j, m2 in enumerate(m2_values):
            config.state.M1 = m1 * M_SUN 
            config.state.M2 = m2 * M_SUN

            cotime, _ = BinarySystemModelFast(config).coalescence_time(
                config.cotime_a_min,
                config.cotime_max_time,
                rtol=1e-9,
                atol=1e-12,
                method=config.method,
            )

            print(f"COALESCENCE TIME: {cotime:.5e}")
            print(f"COMPUTING FOR: M1={m1}, M2={m2}")
            
            if config.initial_points_exponent > 0:
                t_eval = domain_gen.exponential_domain(
                    0, cotime, 2, config.exponent_offset, config.initial_points_exponent
                )
            else:
                t_eval = domain_gen.uniform_domain(0, cotime * 1.01, config.output_points)

            run = IntegrationRun(
                f"m1_{i}_m2_{j}_{m1}_{m2}",
                config,
                0,
                cotime - config.merger_seconds,
                decay_type=config.decay_type,
                solver=config.method,
            )
            run.run()

            multi_plotter.add_dataset(f"m1_{i}_m2_{j}_{m1}_{m2}", m1, m2)

    multi_plotter.generate_plots()
