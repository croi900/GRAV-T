from functools import partial

from numbergen import ExponentialDecay
from scipy.integrate import *
from domain_gen import *
from equations import (
    compute_derivs_linear,
    compute_derivs_exp,
    LinearMassDecay,
    ExpMassDecay,
    make_linear_mass_function,
    make_exp_mass_function,
    make_lander_mass_function,
)
from lander import compute_derivs_lander


def integrator_map(name):
    if name == "RK45":
        return RK45
    if name == "Radau":
        return Radau
    if name == "DOP853":
        return DOP853
    if name == "LSODA":
        return LSODA
    raise ValueError(f"Unknown integration method: {name}")


def domain_type_map(name):
    if name == "uniform":
        return uniform_domain
    if name == "exponential":
        return exponential_domain
    raise ValueError(f"Unknown domain type: {name}")


def system_map(name):
    if name == "linear":
        return compute_derivs_linear
    if name == "exponential":
        return compute_derivs_exp
    if name == "lander":
        return compute_derivs_lander
    raise ValueError(f"Unknown system type: {name}")


def decay_map(name, decay_rate):
    if name == "linear":
        return partial(make_linear_mass_function, decay_rate)()
    if name == "exponential":
        return partial(make_exp_mass_function, decay_rate)()
    if name == "lander":
        return partial(make_lander_mass_function, decay_rate)()
    raise ValueError(f"Unknown decay type: {name}")
