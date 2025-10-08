"""
conversion between SI unit and other unit systems.

Including atomic unit (a.u.), natural unit (n.u.) and Planck unit.
"""

from .._data.values import Math
from ..quantity import constant
from ..utils.constclass import ConstClass
from .constants import SI

__all__ = ['au', 'nu', 'Planck']


class au(ConstClass):
    """
    atomic unit (a.u.) used in atomic physics

    setting ħ = e = me = 4πε0 = 1, then c = 1/α ≈ 137

    example: `au.length` gives the value of a.u. of length in SI unit
    """

    # defined base

    charge = SI.e
    action = SI.hbar
    mass = SI.me
    permittivity = 4 * Math.PI * SI.epsilon0

    # derived dynamics

    length = SI.a0  # Bohr radius
    energy = SI.Eh  # Hartree energy
    time = constant(action / energy)
    force = constant(energy / length)
    velocity = constant(length / time)
    momentum = constant(mass * velocity)
    temperature = constant(energy / SI.kB)

    # derived electromagnetic

    current = constant(charge / time, simplify=True)
    charge_density = constant(charge / length**3)
    electric_potential = constant(energy / charge, simplify=True)
    electric_field = constant(electric_potential / length)
    electric_dipole_moment = constant(charge * length)
    electric_quadrupole_moment = constant(charge * length**2)
    electric_polarizability = constant(electric_dipole_moment / electric_field)
    magnetic_flux_density = constant(electric_field / velocity, simplify=True)
    magnetic_dipole_moment = constant(electric_dipole_moment * velocity, 'J/T')
    magnetizability = constant(magnetic_dipole_moment / magnetic_flux_density)


class nu(ConstClass):
    """
    natural unit (n.u.) used in particle and atomic physics

    setting c = ħ = me = ε0 = 1, then e = √4πα

    example: `nu.length` gives the value of n.u. of length in SI unit
    """

    # defined base

    velocity = SI.c
    action = SI.hbar
    mass = SI.me
    permittivity = SI.epsilon0

    # derived dynamics

    energy = constant(mass * velocity**2, simplify=True)
    momentum = constant(mass * velocity)
    length = constant(action / momentum, simplify=True)
    time = constant(length / velocity)
    force = constant(energy / length, simplify=True)

    # derived electromagnetic

    charge = constant((permittivity * action * velocity).root(2), simplify=True)
    current = constant(charge / time, simplify=True)
    charge_density = constant(charge / length**3)
    electric_potential = constant(energy / charge, simplify=True)
    electric_field = constant(electric_potential / length)
    electric_dipole_moment = constant(charge * length)
    electric_quadrupole_moment = constant(charge * length**2)
    electric_polarizability = constant(electric_dipole_moment / electric_field)
    magnetic_flux_density = constant(electric_field / velocity, simplify=True)
    magnetic_dipole_moment = constant(electric_dipole_moment * velocity, 'J/T')
    magnetizability = constant(magnetic_dipole_moment / magnetic_flux_density)


class Planck(ConstClass):
    """
    Planck unit

    setting c = ħ = G = kB = 1

    example: `Planck.length` gives the value of Planck length in SI unit
    """

    # defined base

    velocity = SI.c
    action = SI.hbar

    # derived base

    mass = constant((action * velocity / SI.G).root(2), simplify=True)
    energy = constant(mass * velocity**2, simplify=True)
    momentum = constant(mass * velocity)
    length = constant(action / momentum, simplify=True)
    time = constant(length / velocity)

    temperature = constant(energy / SI.kB)

