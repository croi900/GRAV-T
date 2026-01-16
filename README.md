# GRAV-T: GRAVitational wave simulation from Time dependant variable mass binaries

A high-performance Python simulation framework for modeling gravitational wave emission from inspiraling binary systems with variable mass decay.

## Overview

This project numerically integrates the orbital evolution of compact binary systems (e.g., neutron star-neutron star mergers) while accounting for gravitational wave radiation reaction and time-dependent mass loss. The simulator produces gravitational wave polarizations (h+ and hx) and orbital visualizations throughout the inspiral and merger phases.

### Key Features

- **Variable Mass Decay**: Supports exponential and linear mass decay functions with configurable rates
- **High-Precision ODE Integration**: Uses scipy's stiff solvers (Radau, DOP853, LSODA) with adaptive time-stepping
- **Quadrupole Radiation**: Analytical gravitational wave polarization computation using the quadrupole formula
- **Coalescence Time Estimation**: Automatic detection of merger events based on semi-major axis evolution
- **HDF5 Output**: Efficient chunked storage with gzip compression for large datasets
- **Visualization**: Orbital phase plots, waveform animations, and polarization ellipse diagrams

## Installation

Requires Python 3.13+. Install dependencies using `uv`:

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

### Dependencies

Core scientific stack:
- `numpy`, `scipy` - Numerical integration and interpolation
- `numba` - JIT compilation for performance-critical routines
- `h5py` - HDF5 data storage
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress tracking

## Usage

### Basic Simulation

Run a simulation by providing a TOML configuration file:

```bash
python main.py --problem problems/exponential_better.toml
```

### Configuration Format

Problem configurations are defined in TOML files. Example:

```toml
# Binary system parameters (solar masses for M, AU for a)
M1 = 1.4                      # Primary mass (solar masses)
M2 = 1.4                      # Secondary mass (solar masses)
a = 10e-4                     # Semi-major axis (AU)
e = 0.1                       # Eccentricity

# Mass decay parameters
decay_rate = 1e-17            # Decay rate coefficient
decay_type = "exponential"    # "exponential" or "linear"

# Integration settings
output_points = 10000000      # Number of output points
use_cotime = true             # Enable coalescence time estimation
cotime_a_min = 3e4            # Minimum semi-major axis (m) for merger detection
cotime_max_time = 1e20        # Maximum integration time (s)
method = "Radau"              # ODE solver ("Radau", "DOP853", "LSODA")

# Merger focus
merger_focus = true
merger_seconds = 5.0          # Time window around merger (s)

# Observer parameters
observer_distance_mpc = 10    # Distance to observer (Megaparsecs)

# Rendering settings
width = 960
height = 540
fps = 60
stride = 1000
```

### Example Configurations

Pre-configured problem files are available in the `problems/` directory:

| File | Description |
|------|-------------|
| `exponential_better.toml` | Compact NS-NS binary with exponential mass decay |
| `linear_better.toml` | Linear decay model for comparison |
| `verify.toml` | Test configuration with verification parameters |

## Project Structure

```
mom2/
├── main.py                 # Entry point and orchestration
├── config.py               # State and Config dataclasses
├── equations.py            # ODE system definitions and physics
├── integration_run.py      # IntegrationRun class for ODE solving
├── polarizations.py        # Gravitational wave polarization computation
├── plotter.py              # Base plotter class
├── ode_plotter.py          # Orbital parameter plots
├── orbit_plotter.py        # Orbit animation generation
├── polarization_plotter.py # Waveform visualization
├── multi_plotter.py        # Overlay comparison plots
├── domain_gen.py           # Time domain generation utilities
├── name_maps.py            # Decay/solver type mappings
├── h5utils.py              # HDF5 helper functions
├── problems/               # TOML configuration files
└── pyproject.toml          # Project metadata and dependencies
```

## Physics

### Orbital Evolution

The binary orbit evolves according to gravitational wave radiation reaction:

- **Semi-major axis decay**: da/dt driven by energy loss
- **Eccentricity evolution**: de/dt from angular momentum loss
- **Peters-Mathews formalism** with extensions for variable mass

### Mass Functions

Time-dependent mass scaling functions:

- **Exponential**: f(t) = exp(-k*t) with derivatives
- **Linear**: f(t) = 1 - k*t with cutoff

### Gravitational Waves

Polarizations computed using the quadrupole formula:

```
h+ = (G/c^4*D) * (d^2Q_xx/dt^2 - d^2Q_yy/dt^2)
hx = (G/c^4*D) * 2 * d^2Q_xy/dt^2
```

where Q_ij is the reduced quadrupole moment tensor.

## Output

Simulations create a directory named after the configuration file containing:

- `{name}.h5` - HDF5 file with all time series data
  - `times`, `a`, `e`, `m1`, `m2` arrays for each integration phase
- `ode_plots/` - Semi-major axis and eccentricity evolution plots
- `polarization_plots/` - Waveform plots (h+, hx, amplitude)
- `orbit.mp4` - Animated orbital visualization

## Performance

Critical computational kernels are optimized using:

- **Numba JIT**: `@njit` compiled functions for physics calculations
- **Parallel computation**: `@njit(parallel=True)` for waveform generation
- **Chunked I/O**: Streaming to HDF5 during integration

## License

This project is for academic and research purposes.
