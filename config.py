"""
EXAMPLE TOML CONFIG

decay_rate = 1e-20
M1 = 1.4
M2 = 1.4
a = 1
e = 0.8

[integration]
output_points = 10000000
use_cotime = true
cotime_a_min = 1e1
cotime_max_time = 1e30
method = "Radau"
decay_type = "exponential"
initial_points_exponent = 0
exponent_offset = 12
merger_focus = true
merger_seconds = 5.0

[rendering]
width = 1920
height = 1080
fps = 60
tail_length = 200
star_scale = 5.0
memory_gb = 10.0
stride = 1000
"""

import copy

from name_maps import *
import tomli
from dataclasses import dataclass, field
from scipy.constants import astronomical_unit as AU

M_SUN = 1.98847e30


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
        """
        Construct a State when the inputs are already in SI units.
        Avoids the solar-mass / AU scaling done in __init__.
        """
        obj = cls.__new__(cls)
        obj.M1 = M1_kg
        obj.M2 = M2_kg
        obj.a = a_m
        obj.e = e
        obj.decay_rate = decay
        return obj

    def __deepcopy__(self, memo):
        new_instance = self.__class__(
            copy.deepcopy(self.M1, memo),
            copy.deepcopy(self.M2, memo),
            copy.deepcopy(self.a, memo),
            copy.deepcopy(self.e, memo),
            copy.deepcopy(self.decay_rate, memo),
            copy.deepcopy(self.decay_type, memo),
        )
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

        MPC = 3.086e22
        self.observer_distance = file.get("observer_distance_mpc", 10) * MPC

    def __deepcopy__(self, memo):
        new_config = object.__new__(Config)
        memo[id(self)] = new_config

        new_config.state = copy.deepcopy(self.state, memo)

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
