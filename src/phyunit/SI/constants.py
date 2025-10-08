from .._data.values import Math, Physic
from ..quantity import Constant, constant
from ..utils.constclass import ConstClass
from .units import si

__all__ = ['SI', 'particle']


class SI(ConstClass):
    '''
    The `SI` class contains common physical constants in SI units,
    like speed of light (_c_), Planck constant (_h_)...

    For example, the standard acceleration of gravity (_g_) is:
    >>> print(SI.g)
    9.80665 m/s²

    These constants can be directly used in formulas or code.

    Ref: http://physics.nist.gov/constants
    '''

    # exact constants defined by SI

    # DeltanuCs = Constant(9_192_631_770, si.Hz)
    # '''hyperfine transition frequency of Cs-133, definition of second'''
    c = Constant(Physic.C, 'm/s')
    '''speed of light in vacuum'''
    h = Constant(6.626_070_15e-34, 'J.s')
    '''Planck constant'''
    e = Constant(Physic.ELC, si.C)
    '''elementary charge'''
    kB = Constant(1.380_649e-23, 'J/K')
    '''Boltzmann constant'''
    NA = Constant(6.022_140_76e23, 'mol-1')
    '''Avogadro constant'''
    # Kcd = Constant(683, 'lm/W')
    # '''luminous efficacy of 540 THz radiation, definition of candela'''

    hbar = constant(h / (2 * Math.PI))
    '''reduced Planck constant, ħ = h / 2π'''
    g = Constant(Physic.GRAVITY, 'm/s2')
    '''standard acceleration of gravity'''
    T0 = Constant(Physic.KELVIN_ZERO, si.K)
    '''standard temperature'''

    # universal

    G = Constant(6.674_30e-11, 'm3/kg.s2')
    '''Newtonian constant of gravitation'''
    kappa = constant(8 * Math.PI * G / c**4, simplify=True)
    '''Einstein gravitational constant'''
    H0 = Constant(70, 'km/s.Mpc')
    '''Hubble constant, approx'''
    Lambda = Constant(1.089e-52, 'm-2')
    '''cosmological constant, approx'''

    # electromagnetic constants

    mu0 = Constant(1.256_637_061_27e-6, 'H/m')
    '''vacuum magnetic permeability, μ₀ ≈ 4π×10⁻⁷ H/m'''
    epsilon0 = constant(1 / (mu0 * c**2), 'F/m')
    '''vacuum electric permittivity'''
    Z0 = constant(mu0 * c, simplify=True)
    '''characteristic impedance of vacuum'''
    ke = constant(1 / (4 * Math.PI * epsilon0), 'N.m2/C2')
    '''Coulomb constant'''
    KJ = constant(2 * e / h, 'Hz/V')
    '''Josephson constant'''
    Phi0 = constant(1 / KJ, simplify=True)
    '''magnetic flux quantum'''
    G0 = constant(2 * e**2 / h, simplify=True)
    '''conductance quantum'''
    RK = constant(h / e**2, simplify=True)
    '''von Klitzing constant'''

    # atomic and nuclear

    me = Constant(9.109_383_7139e-31, si.kg)
    '''electron mass'''
    mp = Constant(1.672_621_925_95e-27, si.kg)
    '''proton mass'''
    mn = Constant(1.674_927_500_56e-27, si.kg)
    '''neutron mass'''

    alpha = constant(e**2 / (2 * epsilon0 * h * c), simplify=True)
    '''fine-structure constant, α ≈ 1/137'''
    a0 = constant(hbar / (alpha * me * c), simplify=True)
    '''Bohr radius'''
    lambdaC = constant(h / (me * c), simplify=True)
    '''Compton wavelength'''
    Rinf = constant(alpha**2 / (2 * lambdaC))
    '''Rydberg constant'''
    Eh = constant(me * (alpha * c)**2, simplify=True)
    '''Hartree energy'''
    re = constant(alpha**2 * a0)
    '''classical electron radius'''
    sigmae = constant(8 * Math.PI / 3 * re**2)
    '''Thomson cross section'''
    muB = constant(e * hbar / (2 * me), 'J/T')
    '''Bohr magneton'''
    muN = constant(e * hbar / (2 * mp), 'J/T')
    '''nuclear magneton'''

    # physico-chemical
    
    Mu = constant(NA * si.u, 'kg/mol')
    '''molar mass constant'''
    R = constant(kB * NA)
    '''molar gas constant'''
    F = constant(NA * e)
    '''Faraday constant'''

    sigma = constant(Math.PI**2 * kB**4 / (60 * hbar**3 * c**2), 'W/m2.K4')
    '''Stefan-Boltzmann constant'''
    c1L = constant(2 * h * c**2, 'W.m2/sr')
    '''first radiation constant for spectral radiance'''
    c1 = constant(c1L * Math.PI * si.sr)
    '''first radiation constant'''
    c2 = constant(h * c / kB)
    '''second radiation constant'''
    b = constant(c2 / Math.WEIN_W)
    '''Wien wavelength displacement law constant'''
    b_ = constant(Math.WEIN_F * c / c2, 'Hz/K')
    '''Wien frequency displacement law constant'''

    Vm = constant(R * T0 / si.bar, 'm3/mol')
    '''molar volume of ideal gas (273.15 K, 100 kPa)'''
    Vmatm = constant(R * T0 / si.atm, 'm3/mol')
    '''molar volume of ideal gas (273.15 K, 101.325 kPa)'''
    n0 = constant(NA / Vm)
    '''Loschmidt constant (273.15 K, 100 kPa)'''
    n0atm = constant(NA / Vmatm)
    '''Loschmidt constant (273.15 K, 101.32 kPa)'''


class particle(ConstClass):
    '''
    Common particle data, including mass and magnetic moment of
    electron (e), muon (μ), tau (τ), proton (p), neutron (n),
    deuteron (D), triton (T), helion (h) and alpha particle (α).

    Separated from `SI` class for symbol and namespace clarity.
    '''

    me = SI.me
    '''electron (e) mass'''
    mmu = Constant(1.883_531_627e-28, si.kg)
    '''muon (μ) mass'''
    mtau = Constant(3.167_54e-27, si.kg)
    '''tau (τ) mass'''
    mp = SI.mp
    '''proton (p = ¹H) mass'''
    mn = SI.mn
    '''neutron (n) mass'''
    mD = Constant(3.343_583_7768e-27, si.kg)
    '''deuteron (D = ²H) mass'''
    mT = Constant(5.007_356_7512e-27, si.kg)
    '''triton (T = ³H) mass'''
    mh = Constant(5.006_412_7862e-27, si.kg)
    '''helion (h = ³He) mass'''
    malpha = Constant(6.644_657_3450e-27, si.kg)
    '''alpha (α = ⁴He) particle mass'''

    mue = Constant(-9.284_764_6917e-24, 'J/T')
    '''electron magnetic moment'''
    ge = constant(-2 * mue / SI.muB)
    '''electron g-factor'''
    mumu = Constant(-4.490_448_30e-26, 'J/T')
    '''muon magnetic moment'''
    gmu = constant(-2 * mumu / (SI.e * SI.hbar / (2 * mmu)))
    '''muon g-factor'''
    mup = Constant(1.410_606_795_45e-26, 'J/T')
    '''proton magnetic moment'''
    gp = constant(2 * mup / SI.muN)
    '''proton g-factor'''
    mun = Constant(-9.662_3653e-27, 'J/T')
    '''neutron magnetic moment'''
    gn = constant(-2 * mun / SI.muN)
    '''neutron g-factor'''
    muD = Constant(4.330_735_087e-27, 'J/T')
    '''deuteron magnetic moment'''
    gD = constant(2 * muD / SI.muN)
    '''deuteron g-factor'''
    muT = Constant(1.504_609_5178e-26, 'J/T')
    '''triton magnetic moment'''
    gT = constant(2 * muT / SI.muN)
    '''triton g-factor'''
    muh = Constant(-1.074_617_551_98e-26, 'J/T')
    '''helion magnetic moment'''
    gh = constant(-2 * muh / SI.muN)
    '''helion g-factor'''
