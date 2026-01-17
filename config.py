"""Configuration classes for GRAV-T simulations."""

import copy

from name_maps import *
import tomli
from dataclasses import dataclass, field

from constants import M_SUN, AU, MPC


@dataclass
class HydrodynamicsParams:
    """Donor stellar properties for hydrodynamic mass transfer (arXiv:2505.10616v2)."""
    R_donor: float = 1e7           # Donor radius (m)
    T_eff: float = 40000.0         # Effective temperature (K)
    T_ph: float = 10000.0          # Photosphere temperature (K) for isothermal EOS
    L_rad: float = 1e32            # Radiative luminosity (W)
    rho_ph: float = 1e-6           # Photosphere density (kg/m³)
    P_gas_ph: float = 1e4          # Photosphere gas pressure (Pa)
    kappa_R: float = 0.034         # Rosseland mean opacity (m²/kg)
    Gamma_Edd: float = 0.0         # Fixed Eddington factor (0 = compute from L_rad)
    mu: float = 0.6                # Mean molecular weight (for isothermal EOS)
    use_full_bvp: bool = False     # Use full BVP solver (Eqs 20) vs simplified formula


@dataclass
class State:
    M1: float
    M2: float
    a: float
    e: float
    decay_rate: float

    def __init__(self, M1, M2, a, e, decay, decay_type):
        self.M1 = M1 * M_SUN
        self.M2 = M2 * M_SUN
        self.a = a * AU
        self.e = e
        self.decay_rate = decay
        self.decay_type = decay_type

    @classmethod
    def from_si(cls, M1_kg, M2_kg, a_m, e, decay):
        """Construct a State from SI units (bypasses M_SUN/AU scaling)."""
        obj = cls.__new__(cls)
        obj.M1 = M1_kg
        obj.M2 = M2_kg
        obj.a = a_m
        obj.e = e
        obj.decay_rate = decay
        return obj

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.M1 = self.M1
        new_instance.M2 = self.M2
        new_instance.a = self.a
        new_instance.e = self.e
        new_instance.decay_rate = self.decay_rate
        new_instance.decay_type = self.decay_type
        memo[id(self)] = new_instance
        return new_instance


@dataclass
class Config:
    name: str = field(init=False)
    state: State = field(init=False)
    output_points: int = field(init=False)
    use_cotime: bool = field(init=False)
    cotime_a_min: float = field(init=False)
    cotime_max_time: float = field(init=False)
    method: callable = field(init=False)
    decay_type: str = field(init=False)
    initial_points_exponent: int = field(init=False)
    exponent_offset: int = field(init=False)
    merger_focus: bool = field(init=False)
    merger_seconds: float = field(init=False)
    stride: int = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)
    fps: int = field(init=False)
    tail_length: int = field(init=False)
    star_scale: float = field(init=False)
    memory_gb: float = field(init=False)
    hydro: HydrodynamicsParams = field(init=False)

    def __init__(self, toml_file):
        file = tomli.load(open(toml_file, "rb"))

        self.state = State(
            file["M1"],
            file["M2"],
            file["a"],
            file["e"],
            file["decay_rate"],
            file["decay_type"],
        )
        self.name = toml_file.split("/")[-1].split(".")[0]
        self.output_points = file["output_points"]
        self.use_cotime = file["use_cotime"]
        self.cotime_a_min = file["cotime_a_min"]
        self.cotime_max_time = file["cotime_max_time"]
        self.method = file["method"]
        self.decay_type = file["decay_type"]
        self.initial_points_exponent = file["initial_points_exponent"]
        self.exponent_offset = file["exponent_offset"]
        self.merger_focus = file["merger_focus"]
        self.merger_seconds = file["merger_seconds"]
        self.stride = file["stride"]
        self.width = file["width"]
        self.height = file["height"]
        self.fps = file["fps"]
        self.tail_length = file["tail_length"]
        self.star_scale = file["star_scale"]
        self.memory_gb = file["memory_gb"]
        self.observer_distance = file.get("observer_distance_mpc", 10) * MPC

        # Parse hydrodynamics section if present
        hydro_section = file.get("hydrodynamics", {})
        R_SUN = 6.957e8  # Solar radius in meters
        self.hydro = HydrodynamicsParams(
            R_donor=hydro_section.get("R_donor", 0.01) * R_SUN,
            T_eff=hydro_section.get("T_eff", 40000.0),
            T_ph=hydro_section.get("T_ph", 10000.0),
            L_rad=hydro_section.get("L_rad", 1e32),
            rho_ph=hydro_section.get("rho_ph", 1e-6),
            P_gas_ph=hydro_section.get("P_gas_ph", 1e4),
            kappa_R=hydro_section.get("kappa_R", 0.034),
            Gamma_Edd=hydro_section.get("Gamma_Edd", 0.0),
            mu=hydro_section.get("mu", 0.6),
            use_full_bvp=hydro_section.get("use_full_bvp", False),
        )

    def __deepcopy__(self, memo):
        new_config = object.__new__(Config)
        memo[id(self)] = new_config

        new_config.state = copy.deepcopy(self.state, memo)
        new_config.hydro = copy.deepcopy(self.hydro, memo)

        new_config.name = self.name
        new_config.output_points = self.output_points
        new_config.use_cotime = self.use_cotime
        new_config.cotime_a_min = self.cotime_a_min
        new_config.cotime_max_time = self.cotime_max_time
        new_config.method = self.method
        new_config.decay_type = self.decay_type
        new_config.initial_points_exponent = self.initial_points_exponent
        new_config.exponent_offset = self.exponent_offset
        new_config.merger_focus = self.merger_focus
        new_config.merger_seconds = self.merger_seconds
        new_config.stride = self.stride
        new_config.width = self.width
        new_config.height = self.height
        new_config.fps = self.fps
        new_config.tail_length = self.tail_length
        new_config.star_scale = self.star_scale
        new_config.memory_gb = self.memory_gb
        new_config.observer_distance = self.observer_distance

        return new_config
