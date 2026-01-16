#!/usr/bin/env python
"""
Hydrodynamics visualization - compare different Eddington factors.
Uses MultiPlotterHydro for consistent styling.

This script runs parameter studies varying Gamma_Edd and generates
comparison plots showing how radiation pressure affects mass transfer
and orbital evolution.
"""

import argparse
import copy
import os
import numpy as np

from config import Config
from constants import M_SUN
from equations import BinarySystemModelFast
from integration_run import IntegrationRun
from hydrodynamics import set_hydro_params, compute_dynamic_M_dot
from multi_plotter_hydro import MultiPlotterHydro


def run_hydro_comparison(config: Config, gamma_values: list):
    """Run simulations with different Eddington factors."""
    
    multi_plotter = MultiPlotterHydro(config)
    
    if not os.path.exists(config.name):
        os.makedirs(config.name, exist_ok=True)
        
    for gamma in gamma_values:
        print(f"\n=== Running Γ_Edd = {gamma:.2f} ===")
        
        cfg = copy.deepcopy(config)
        cfg.hydro.Gamma_Edd = gamma
        
        model = BinarySystemModelFast(cfg)
        cotime, _ = model.coalescence_time(
            cfg.cotime_a_min, cfg.cotime_max_time,
            rtol=1e-6, atol=1e-9, method="Radau"
        )
        print(f"Coalescence time: {cotime:.3e} s")
        
        run_name = f"gamma_{gamma:.2f}".replace(".", "_")
        t_end = cotime - cfg.merger_seconds
        if t_end < 0:
            t_end = cotime * 0.99
        
        run = IntegrationRun(
            run_name, cfg,
            t0=0, t1=t_end,
            decay_type="hydrodynamics",
            solver=cfg.method,
        )
        run.run()
        
        multi_plotter.add_dataset(run_name, gamma)
    
    multi_plotter.generate_plots()


def main():
    parser = argparse.ArgumentParser(description="Hydrodynamics comparison study")
    parser.add_argument("--problem", required=True, help="TOML problem file")
    parser.add_argument("--gammas", type=str, default="0.1,0.3,0.5,0.7",
                       help="Comma-separated Eddington factors to compare")
    args = parser.parse_args()
    
    config = Config(args.problem)
    gamma_values = [float(g) for g in args.gammas.split(",")]
    
    print(f"Comparing Γ_Edd values: {gamma_values}")
    
    set_hydro_params(
        config.hydro.R_donor, config.hydro.rho_ph, config.hydro.P_gas_ph,
        config.hydro.L_rad, config.state.M1, config.hydro.kappa_R, 
        config.hydro.Gamma_Edd, config.state.M2
    )
    M_dot_0 = compute_dynamic_M_dot(config.state.a, config.state.M1, config.state.M2)
    print(f"Initial M_dot: {M_dot_0:.3e} kg/s = {M_dot_0*3.154e7/M_SUN:.3e} M_sun/yr")
    
    run_hydro_comparison(config, gamma_values)


if __name__ == "__main__":
    main()
