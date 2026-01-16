"""Visualization for hydrodynamic mass transfer simulations."""

import numpy as np
import h5py
from matplotlib import pyplot as plt
from pathlib import Path

from plotter import Plotter
from roche_geometry import roche_lobe_radius_eggleton, compute_L1_position, roche_coefficients
from constants import M_SUN


class HydroPlotter(Plotter):
    """Plots for hydrodynamic mass transfer analysis."""
    
    def plot(self, run_name):
        """Generate all hydro-related plots for a run."""
        with h5py.File(self.h5path, "r") as f:
            t = np.array(f[f"{run_name}/times"])
            a = np.array(f[f"{run_name}/a"])
            e = np.array(f[f"{run_name}/e"])
            m1 = np.array(f[f"{run_name}/m1"])
            m2 = np.array(f[f"{run_name}/m2"])
        
        self._plot_mass_evolution(run_name, t, m1, m2)
        self._plot_mass_ratio(run_name, t, m1, m2)
        self._plot_roche_lobe_evolution(run_name, t, a, m1, m2)
        self._plot_separation_vs_mass(run_name, a, m1, m2)
        
        print(f"Hydro plots saved to {self.config.name}/ode_plots/{run_name}/")
    
    def _plot_mass_evolution(self, run_name, t, m1, m2):
        """Plot M1, M2, and M_total vs time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t_scaled = t / 3.154e7  # Convert to years
        
        ax.plot(t_scaled, m1 / M_SUN, 'b-', label=r'$M_1$ (donor)', linewidth=2)
        ax.plot(t_scaled, m2 / M_SUN, 'r-', label=r'$M_2$ (accretor)', linewidth=2)
        ax.plot(t_scaled, (m1 + m2) / M_SUN, 'k--', label=r'$M_{total}$', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(r'Mass ($M_\odot$)')
        ax.legend()
        ax.set_title('Mass Evolution')
        ax.grid(True, alpha=0.3)
        
        self.saveplot(f"{run_name}/hydro_mass_evolution")
    
    def _plot_mass_ratio(self, run_name, t, m1, m2):
        """Plot mass ratio q = M1/M2 vs time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t_scaled = t / 3.154e7
        q = m1 / m2
        
        ax.plot(t_scaled, q, 'g-', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='q=1')
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(r'Mass ratio $q = M_1/M_2$')
        ax.set_title('Mass Ratio Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.saveplot(f"{run_name}/hydro_mass_ratio")
    
    def _plot_roche_lobe_evolution(self, run_name, t, a, m1, m2):
        """Plot Roche lobe radius evolution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t_scaled = t / 3.154e7
        q = m1 / m2
        
        # Vectorized Roche lobe calculation
        R_L = np.array([roche_lobe_radius_eggleton(q_i) * a_i for q_i, a_i in zip(q, a)])
        
        ax.plot(t_scaled, R_L / 1e3, 'purple', linewidth=2, label=r'$R_L$ (Roche lobe)')
        ax.plot(t_scaled, a / 1e3, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Separation $a$')
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Radius (km)')
        ax.set_title('Roche Lobe Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.saveplot(f"{run_name}/hydro_roche_lobe")
    
    def _plot_separation_vs_mass(self, run_name, a, m1, m2):
        """Phase plot: separation vs total mass."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        M_total = (m1 + m2) / M_SUN
        a_km = a / 1e3
        
        # Color by time (index)
        colors = np.linspace(0, 1, len(a))
        scatter = ax.scatter(M_total, a_km, c=colors, cmap='viridis', s=1, alpha=0.5)
        
        ax.set_xlabel(r'Total Mass ($M_\odot$)')
        ax.set_ylabel('Separation (km)')
        ax.set_title('Phase Space: a vs M')
        
        cbar = plt.colorbar(scatter, label='Time (normalized)')
        ax.grid(True, alpha=0.3)
        
        self.saveplot(f"{run_name}/hydro_phase_space")


def plot_roche_potential_contours(q, n_points=200):
    """Generate Roche equipotential contour plot."""
    from roche_geometry import roche_potential
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.linspace(-0.5, 1.5, n_points)
    y = np.linspace(-1.0, 1.0, n_points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = roche_potential(X[i, j], Y[i, j], 0.0, q)
    
    Z = np.clip(Z, -10, 0)
    
    levels = np.linspace(Z.min(), Z.max(), 30)
    contour = ax.contour(X, Y, Z, levels=levels, cmap='RdBu_r')
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    L1_x = compute_L1_position(q)
    ax.plot(L1_x, 0, 'ko', markersize=8, label=f'L1 at x={L1_x:.3f}')
    
    ax.plot(0, 0, 'b*', markersize=15, label='Donor')
    ax.plot(1, 0, 'r*', markersize=15, label='Accretor')
    
    ax.set_xlabel('x (units of a)')
    ax.set_ylabel('y (units of a)')
    ax.set_title(f'Roche Potential (q={q:.2f})')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax
