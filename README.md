# GRAV-T: Gravitational-wave inspiral with time‑dependent masses

GRAV‑T is a Python tool for simulating compact binary inspirals with **variable mass loss**, computing the orbital evolution, coalescence time and (optionally) gravitational‑wave polarizations, and producing high‑quality **orbit movies**.

The code uses an analytic variable–mass formalism implemented in `equations.py`, high‑accuracy ODE solvers from SciPy, and chunked HDF5 output so that very long integrations remain tractable.

## Overview

### Key features

- **Variable mass decay models** via analytic scaling functions:
  - `decay_type = "exponential"` – exponential mass loss
  - `decay_type = "linear"` – linear mass loss with cutoff
  - `decay_type = "lander"` – alternative model implemented in `lander.py`
- **Coalescence search** using `BinarySystemModelFast.coalescence_time`, with merger detected when the semi‑major axis reaches the ISCO radius of the time‑varying binary.
- **Two‑phase integration pipeline**:
  - long **circularization** run up to `cotime - merger_seconds`
  - short high‑resolution **merger window** around coalescence (via `IntegrationRun`)
- **HDF5 output** with streaming writes for `a(t)`, `e(t)`, and time‑dependent masses.
- **Orbit visualization** with `OrbitPlotter`, generating modern 16:9 MP4 animations that stay visually readable even for extreme inspirals.
- **Optional GW polarization analysis** with `PolarizationPlotter`, generating \(h_+\), \(h_\times\), \(|h|\), and \(f_{\rm GW}\) near merger.

---

## Installation

Requires **Python ≥ 3.11**.

From the top‑level of this repository (the directory that contains `GRAV-T/` and `problems/`):

### With `uv`

```bash
cd GRAV-T
uv sync
```

### With `pip`

```bash
cd GRAV-T
pip install -e .
```

The core dependencies (from `pyproject.toml`) include:

- **SciPy / NumPy** – numerical integration and interpolation
- **Numba** – JIT compilation for performance‑critical routines
- **h5py** – HDF5 data storage
- **matplotlib**, **tqdm** – visualization and progress reporting

---

## Running a simulation

The main entry point is `main.py` inside the `GRAV-T/` directory.

From the repository root:

```bash
cd GRAV-T
python main.py --problem ../problems/exponential_better.toml
```

Current `main.py` behaviour:

- loads the TOML problem with `Config`
- computes the coalescence time with `BinarySystemModelFast.coalescence_time`
- runs a **circularization** integration with `IntegrationRun`
- writes HDF5 output into `<problem_name>/<problem_name>.h5`
- builds an **orbit animation** for the circularization phase with `OrbitPlotter`

The template for a merger‑focused run (`IntegrationRun("merger", ...)`) and polarization analysis (`PolarizationPlotter`) is present in `main.py` but commented out. You can either:

- uncomment those lines, or
- call `IntegrationRun` / `PolarizationPlotter` from your own driver script

to generate merger‑window datasets and GW polarization plots.

---

## Configuration files (`problems/*.toml`)

Simulation problems are described in TOML files such as `problems/exponential_better.toml`. They are read by `Config` and converted into SI units where appropriate.

Typical fields (as used in the current code) are:

- **Binary parameters**
  - `M1`, `M2` – component masses in solar masses
  - `a` – initial semi‑major axis in astronomical units (AU)
  - `e` – initial eccentricity
- **Mass decay**
  - `decay_rate` – decay coefficient (model‑dependent)
  - `decay_type` – `"exponential"`, `"linear"`, or `"lander"`
- **Coalescence search / integration**
  - `output_points` – number of output samples for the ODE solution
  - `use_cotime` – whether to use the coalescence time search
  - `cotime_a_min` – minimum semi‑major axis (in metres) used as a merger cutoff
  - `cotime_max_time` – maximum physical integration time for the search
  - `method` – ODE solver name: `"Radau"`, `"DOP853"`, `"LSODA"`, `"RK45"`, ...
  - `initial_points_exponent`, `exponent_offset` – control exponential vs uniform sampling in time (see `domain_gen.py` and `main.py`)
  - `merger_focus`, `merger_seconds` – control the high‑resolution merger window
  - `observer_distance_mpc` – observer distance in megaparsecs (used in polarization analysis)
- **Rendering / storage**
  - `width`, `height` – resolution of the orbit animation
  - `fps` – frames per second for the video
  - `tail_length` – number of frames of trailing orbit history
  - `star_scale` – visual scaling of the stars in the plot
  - `memory_gb` – approximate memory budget used when planning chunk sizes
  - `stride` – sampling stride for various plotting utilities

Example (adapted from `problems/exponential_better.toml`):

```toml
# Binary (solar masses) and orbit (AU)
M1 = 1.4
M2 = 1.4
a = 10e-4
e = 0.1

# Variable mass model
decay_rate = 2e-17
decay_type = "exponential"

# Coalescence search and integration
output_points = 10000000
use_cotime = true
cotime_a_min = 3e4
cotime_max_time = 1e20
observer_distance_mpc = 1
method = "Radau"
initial_points_exponent = 0
exponent_offset = 12
merger_focus = true
merger_seconds = 5.0

# Visualization / storage
width = 960
height = 540
fps = 60
tail_length = 600
star_scale = 5.0
memory_gb = 10.0
stride = 1
```

---

## Outputs

For a problem file like `problems/exponential_better.toml`, simulations write into a directory named after the problem (here `exponential_better/`) relative to the current working directory:

- `exponential_better/exponential_better.h5` – main HDF5 file containing:
  - `circularization/times`, `a`, `e`, `m1`, `m2`
  - (if enabled) `merger/times`, `a`, `e`, `m1`, `m2`
- `exponential_better/ode_plots/<run_name>/orbit.mp4` – orbit animation created by `OrbitPlotter`
- (if `PolarizationPlotter` is used) polarization data and plots stored under a `polarizations/` group and image files saved via `Plotter.saveplot`

You can inspect the HDF5 contents with tools like `h5ls`, `h5dump`, or Python + `h5py`.

---

## Project structure

At the package root (`GRAV-T/`):

```text
GRAV-T/
├── main.py                # CLI entry point (coalescence search + circularization + orbit movie)
├── config.py              # State / Config dataclasses and TOML loader
├── equations.py           # Variable-mass inspiral formalism and BinarySystemModelFast
├── linear.py              # Linear mass decay model and ODE RHS
├── exponential.py         # Exponential mass decay model and ODE RHS
├── lander.py              # Alternative decay model and ODE RHS
├── integration_run.py     # IntegrationRun: streaming ODE integration + HDF5 writer
├── polarizations.py       # GW polarization building blocks (phase, radius, waveforms)
├── polarization_plotter.py# Post-processing: h+, h×, |h|, f_GW near merger
├── orbit_plotter.py       # OrbitPlotter: high-quality inspiral movie generator
├── plotter.py             # Shared plotting utilities / base class
├── multi_plotter.py       # Comparison plots across runs (e.g. decay models)
├── domain_gen.py          # Time-domain sampling utilities (uniform / exponential)
├── name_maps.py           # Maps decay / system / integrator names to implementations
├── h5utils.py             # HDF5 helper functions and safe dataset creation
├── verbprint.py           # Verbose logging helper
└── pyproject.toml         # Project metadata and dependency list
```

Top‑level (repository root):

- `GRAV-T/` – the package itself (this directory)
- `problems/` – example TOML problem files

---

## Physics model (brief)

The inspiral follows a variable‑mass extension of the standard gravitational‑wave driven binary evolution:

- orbital elements \(a(t)\), \(e(t)\) evolve under radiation reaction
- time‑dependent masses are encoded via analytic scaling functions \(f_1(t)\), \(f_2(t)\)
- combined scalings \(f_M(t)\), \(f_\mu(t)\) and their derivatives feed into corrected energy and angular‑momentum fluxes
- the ODE system is evaluated through Numba‑compiled kernels in `equations.py` / `linear.py` / `exponential.py` / `lander.py`

Polarization computation (when enabled) uses the orbit and mass history to reconstruct the quadrupole‑based GW polarizations, their amplitude, and instantaneous frequency near merger.

---

## License

This project is intended for academic and research use.

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
- **Visualization Suite**: Orbital phase plots, waveform animations, and polarization ellipse diagrams

## Installation

Requires Python 3.11+. Install dependencies using `uv`:

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
