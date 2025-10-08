import enum
from decimal import Decimal
from fractions import Fraction

from ..dimension import DIMENSIONLESS, Dimension
from ..dimensionconst import DimensionConst
from ..exceptions import UnitDeprecationWarning
from ..utils.iter_tools import firstof
from .values import Math, Physic, Time


class UnitSystem(enum.Flag):
    SI = enum.auto()
    Gaussian = enum.auto()
    ESU = enum.auto()  # Electrostatic unit
    EMU = enum.auto()  # Electromagnetic unit
    Heaviside = enum.auto()  # Heaviside-Lorentz unit
    CGS = SI | Gaussian | ESU | EMU | Heaviside
    Imperial = enum.auto()
    All = SI | CGS | Imperial


class UnitData:
    '''
    data of unit (without prefix)

    Attributes:
        factor (float): e.g., 1e-3 for 'gram'
        name (str | list[str]): e.g., 'meter', ['litre', 'liter']
        dimension (Dimension): e.g., length, time
        system (enum.Flag): e.g., SI, CGS
        noprefix (bool): Whether the unit should never be prefixed
        log (bool): Whether the unit is logarithmic ratio
        recommend (str | None): The recommended unit when the unit is deprecated
        SIbase (str | None): SI base unit equivalent (e.g., 'cd·sr/m²' for 'lux')
    '''

    __slots__ = ('factor', 'name', 'dimension', 'system', 'alter')

    def __init__(self, 
                 factor: float | Fraction, 
                 name: str | list[str],
                 dimension: Dimension = DIMENSIONLESS, *,
                 system: UnitSystem = UnitSystem.All,
                 **alternative_attribute
                ) -> None:
        self.factor = factor
        self.name = [name] if isinstance(name, str) else name
        self.dimension = dimension
        self.system = system
        self.alter = alternative_attribute

    def __hash__(self) -> int: return hash((self.factor, self.name[0]))

    @property
    def noprefix(self) -> bool: return self.alter.get('noprefix', False)
    @property
    def log(self) -> bool: return self.alter.get('log', False)
    @property
    def recommend(self) -> str | None: return self.alter.get('recommend')
    @property
    def SIbase(self) -> str | None: return self.alter.get('SIbase')

    def deprecation_warning(self):
        if self.recommend is None:
            return
        import warnings
        warnings.warn(
            f"'{self.name[0]}' is deprecated, use '{self.recommend}' instead.",
            UnitDeprecationWarning, stacklevel=2
        )


BASE_SI = ('s', 'm', 'kg', 'A', 'K', 'mol', 'cd')

# unit library, classified by dimension, internal use only
__UNIT_LIB: dict[Dimension, dict[str | tuple[str, ...], UnitData]] = {
    DimensionConst.DIMENSIONLESS: {
        '': UnitData(1, ''),
        'rad': UnitData(1, 'radian'),
        'sr': UnitData(1, 'steradian'),
        '°': UnitData(Math.DEGREE, 'degree', noprefix=True),
        ('′', "'"): UnitData(Math.ARCMIN, 'arcminute', noprefix=True),  # chr(0x2032), chr(0x27)
        ('″', '"'): UnitData(Math.ARCSEC, 'arcsecond', noprefix=True),  # chr(0x2033), chr(0x22)
        ('%', '٪'): UnitData(1e-2, 'percent', noprefix=True),  # chr(0x25), chr(0x66A)
        '‰': UnitData(1e-3, 'permille', noprefix=True),
        '‱': UnitData(1e-4, ['permyriad', 'per ten thousand'], noprefix=True),
        ('B', 'b'): UnitData(1, 'bel', log=True),
        'Np': UnitData(1, 'neper', log=True),
    },
    DimensionConst.TIME: {
        's': UnitData(1, 'second'),
        'min': UnitData(Time.MINUTE, 'minute', noprefix=True),
        'h': UnitData(Time.HOUR, 'hour', noprefix=True),
        'd': UnitData(Time.DAY, 'day', noprefix=True),
        'yr': UnitData(Time.SIMPLE_YEAR, 'year'),
        'a': UnitData(Time.JULIAN_YEAR, ['annum', 'Julian year']),
    },
    DimensionConst.LENGTH: {
        'm': UnitData(1, ['metre', 'meter']),
        'fm': UnitData(1e-15, 'fermi', noprefix=True),  # femtometer
        ('Å', 'Å', 'Å'): UnitData(1e-10, ['ångström', 'angstrom']),  # chr(0xC5), chr(0x212B), 'A'+chr(0x30A)
        ('au', 'AU'): UnitData(Physic.AU, 'astronomical unit'),
        'pc': UnitData(Physic.PC, 'parsec'),
        'ly': UnitData(Physic.LIGHT_YEAR, 'light year'),
    },
    DimensionConst.MASS: {
        'g': UnitData(1e-3, 'gram'),
        't': UnitData(1000, ['tonne', 'ton']),
        'u': UnitData(Physic.DALTON, 'unified atomic mass unit'),
        'Da': UnitData(Physic.DALTON, 'dalton'),
    },
    DimensionConst.ELECTRIC_CURRENT: {
        'A': UnitData(1, 'ampere'),
    },
    DimensionConst.THERMODYNAMIC_TEMPERATURE: {
        'K': UnitData(1, 'kelvin'),
        ('°C', '℃'): UnitData(1, 'degree Celsius', noprefix=True),
        ('°F', '℉'): UnitData(Fraction(5, 9), 'degree Fahrenheit', noprefix=True),
        '°R': UnitData(Fraction(9, 5), 'degree Rankine', noprefix=True)
    },
    DimensionConst.AMOUNT_OF_SUBSTANCE: {
        'mol': UnitData(1, 'mole'),
    },
    DimensionConst.LUMINOUS_INTENSITY: {
        'cd': UnitData(1, 'candela'),
        'lm': UnitData(1, 'lumen', SIbase='cd·sr'),  # luminous flux
    },
    # derived
    DimensionConst.FREQUENCY: {
        'Hz': UnitData(1, 'hertz'),
        'cps': UnitData(1, 'cycles per second', recommend='s⁻¹'),
        'Bq': UnitData(1, 'becquerel'),
        'Ci': UnitData(3.7e10, 'curie', recommend='Bq'),
        'Rd': UnitData(1e6, 'rutherford', recommend='Bq'),
    },
    DimensionConst.WAVENUMBER: {
        'Ky': UnitData(100, 'kayser', system=UnitSystem.CGS),
    },
    DimensionConst.AREA: {
        'b': UnitData(1e-28, 'barn'),
        'ha': UnitData(10000, 'hectare', noprefix=True),
    },
    DimensionConst.VOLUME: {
        ('L', 'l'): UnitData(1e-3, ['litre', 'liter']),
    },
    DimensionConst.VELOCITY: {
        'c': UnitData(Physic.C, 'speed of light', noprefix=True),
    },
    DimensionConst.ACCELERATION: {
        'Gal': UnitData(0.01, ['gal', 'galileo'], system=UnitSystem.CGS),
    },
    DimensionConst.FORCE: {
        'N': UnitData(1, 'newton'),
        'gf': UnitData(Physic.GRAVITY / 1000, 'gram fore'),
        'dyn': UnitData(1e-5, 'dyne', system=UnitSystem.CGS),
    },
    DimensionConst.PRESSURE: {
        'Pa': UnitData(1, 'pascal'),
        'bar': UnitData(Physic.SSP, 'bar'),
        'atm': UnitData(Physic.ATM, 'standard atmosphere'),
        'mHg': UnitData(Physic.MMHG * 1000, 'meter of mercury'),
        'Torr': UnitData(Physic.MMHG, ['torr', 'torricelli']),
        'Ba': UnitData(0.1, 'barye', system=UnitSystem.CGS),
    },
    DimensionConst.ENERGY: {
        'J': UnitData(1, 'joule'),
        'Wh': UnitData(Time.HOUR, 'watthour'),
        'eV': UnitData(Physic.EV, 'electronvolt'),
        'cal': UnitData(Physic.CAL, 'calorie'),
        'erg': UnitData(1e-7, 'erg', system=UnitSystem.CGS),
    },
    DimensionConst.POWER: {
        'W': UnitData(1, 'watt'),
        ('var', 'VAR', 'VAr'): UnitData(1, 'volt-ampere reactive'),
        'VA': UnitData(1, 'volt-ampere'),
        'statW': UnitData(1, 'statwatt', system=UnitSystem.ESU),
    },
    DimensionConst.DYNAMIC_VISCOSITY: {
        'P': UnitData(0.1, 'poise', system=UnitSystem.CGS),
    },
    DimensionConst.KINEMATIC_VISCOSITY: {
        'St': UnitData(1e-4, 'stokes', system=UnitSystem.CGS),
    },
    DimensionConst.ELECTRIC_CHARGE: {
        'C': UnitData(1, 'coulomb', system=UnitSystem.SI),
        'Ah': UnitData(Time.HOUR, 'ampere-hour'),
        'statC': UnitData(1, 'statcoulomb', system=UnitSystem.ESU),
        'Fr': UnitData(1, 'franklin', system=UnitSystem.Gaussian),
    },
    DimensionConst.VOLTAGE: {
        'V': UnitData(1, 'volt', system=UnitSystem.SI),
        'statV': UnitData(1, 'statvolt', system=UnitSystem.ESU),
    },
    DimensionConst.CAPACITANCE: {
        'F': UnitData(1, 'farad'),
    },
    DimensionConst.RESISTANCE: {
        'Ω': UnitData(1, 'ohm'),
    },
    DimensionConst.CONDUCTANCE: {
        'S': UnitData(1, 'siemens'),
        '℧': UnitData(1, 'mho', recommend='S'),
    },
    DimensionConst.MAGNETIC_FLUX: {
        'Wb': UnitData(1, 'weber'),
        'Mx': UnitData(1e-8, 'maxwell', system=UnitSystem.Gaussian),
    },
    DimensionConst.MAGNETIC_FLUX_DENSITY: {
        'T': UnitData(1, 'tesla'),
        'G': UnitData(1e-4, 'gauss', system=UnitSystem.Gaussian),
    },
    DimensionConst.MAGNETIC_FIELD_STRENGTH: {
        'Oe': UnitData(1, ['oersted', 'ørsted'], system=UnitSystem.Gaussian),
    },
    DimensionConst.INDUCTANCE: {
        'H': UnitData(1, 'henry'),
    },
    DimensionConst.ILLUMINANCE: {
        'lx': UnitData(1, 'lux', SIbase='cd·sr/m²'),  # illuminance [lm/m²]
        'nt': UnitData(1, 'nit'),  # luminance [cd/m²]
        'sb': UnitData(10000, 'stilb', system=UnitSystem.CGS),  # luminance [cd/cm²]
        'ph': UnitData(10000, 'phot', system=UnitSystem.CGS),  # illuminance [lm/cm²]
    },
    DimensionConst.KERMA: {
        'Gy': UnitData(1, 'gray'),
        'Sv': UnitData(1, 'sievert'),
        'rem': UnitData(0.01, 'roentgen equivalent man', recommend='Sv')
    },
    DimensionConst.EXPOSURE: {
        'R': UnitData(2.58e-4, 'roentgen', recommend='C/kg'),
    },
    DimensionConst.CATALYTIC_ACTIVITY: {
        'kat': UnitData(1, 'katal'),
    },
}

# add dimension property
for dim, unit_dict in __UNIT_LIB.items():
    for basedata in unit_dict.values():
        if basedata.system is UnitSystem.Gaussian:
            basedata.dimension = dim.asGaussian()
        else:
            basedata.dimension = dim

# concatenate all dict in __UNIT_LIB
UNIT: dict[str, UnitData] = {
    unit[0] if isinstance(unit, tuple) else unit: data
    for unit_dict in __UNIT_LIB.values()
    for unit, data in unit_dict.items()
}
'''unit {symbol: data}'''

UNIT_NAME: dict[str, str] = {
    name: unit for unit, data in UNIT.items() for name in data.name
}
'''unit {name: symbol}'''

UNIT_ALIAS = {
    alias: unit[0]
    for unit_dict in __UNIT_LIB.values()
    for unit in unit_dict
    for alias in unit[1:] if isinstance(unit, tuple)
}
'''unit {alias symbol: symbol}'''


# unit standard, every dimension has one SI basic/standard unit
# values() = {s m kg A K mol cd Hz N Pa J W C V F Ω S Wb T H lx Gy kat}
__UNIT_STD_DIM: set[Dimension] = {
    # 7 SI base
    DimensionConst.TIME, DimensionConst.LENGTH, DimensionConst.MASS,
    DimensionConst.ELECTRIC_CURRENT, DimensionConst.THERMODYNAMIC_TEMPERATURE,
    DimensionConst.AMOUNT_OF_SUBSTANCE, DimensionConst.LUMINOUS_INTENSITY,
    # directly derived
    DimensionConst.FREQUENCY,
    # kinematics and dynamics
    DimensionConst.FORCE, DimensionConst.PRESSURE, DimensionConst.ENERGY,
    DimensionConst.POWER,
    # electrodynamics
    DimensionConst.ELECTRIC_CHARGE, DimensionConst.VOLTAGE,
    DimensionConst.CAPACITANCE, DimensionConst.RESISTANCE,
    DimensionConst.CONDUCTANCE, DimensionConst.MAGNETIC_FLUX,
    DimensionConst.MAGNETIC_FLUX_DENSITY, DimensionConst.INDUCTANCE,
    # luminous
    DimensionConst.ILLUMINANCE,
    # nuclear radiation
    DimensionConst.KERMA,
    # chemistry
    DimensionConst.CATALYTIC_ACTIVITY
}
UNIT_STD: dict[Dimension, str] = {
    dim: u[0] if isinstance(u := firstof(unit_dict, default=''), tuple) else u
    for dim, unit_dict in __UNIT_LIB.items()
    if dim in __UNIT_STD_DIM
}
'''standard unit for dimension'''
UNIT_STD[DimensionConst.MASS] = 'kg'


assert sum(map(len, __UNIT_LIB.values())) == len(UNIT), "unit symbol conflict"

