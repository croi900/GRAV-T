"""
Centralized physical constants for GRAV-T.

All constants are in SI units unless otherwise noted.
"""

from scipy import constants as scipy_const

G = scipy_const.G                    # Gravitational constant: 6.67430e-11 m^3/(kg·s^2)
c = scipy_const.c                    # Speed of light: 299792458 m/s
k_B = scipy_const.Boltzmann          # Boltzmann constant: 1.38065e-23 J/K
m_H = scipy_const.proton_mass        # Hydrogen/proton mass: 1.6726e-27 kg

M_SUN = 1.98847e30                   # Solar mass in kg
AU = scipy_const.astronomical_unit   # Astronomical unit in meters
MPC = 3.086e22                       # Megaparsec in meters

a_rad = 4 * scipy_const.sigma / c    # Radiation constant: 4sigma/c ≈ 7.566e-16 J/(m^3 K^4)

