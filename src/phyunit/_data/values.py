# from decimal import Decimal
from fractions import Fraction

from ..utils.constclass import ConstClass


class Math(ConstClass):
    PI = 3.1415926535897932384626433832795028841971693993751
    '''the ratio of the circumference of a circle to its diameter'''
    WEIN_W = 4.9651142317442763036987591313228939440555849867973
    '''solution of: x = 5*(1 - exp(-x)) = 5 + LambertW(-5*exp(-5))'''
    WEIN_F = 2.8214393721220788934031913302944851953458817440731
    '''solution of: x = 3*(1 - exp(-x)) = 3 + LambertW(-3*exp(-3))'''
    DEGREE = PI / 180
    ARCMIN = DEGREE / 60
    ARCSEC = ARCMIN / 60


class Time(ConstClass):
    MINUTE = 60
    HOUR = 60 * MINUTE
    DAY = 24 * HOUR
    SIMPLE_YEAR = 365 * DAY
    '''1 simple year = 365 day [s]'''
    JULIAN_YEAR = SIMPLE_YEAR + DAY // 4
    '''1 Julian year = 365.25 day [s]'''


class Physic(ConstClass):
    # physical constants
    C = 299_792_458
    '''speed of light [m/s]'''
    ELC = 1.602_176_634e-19
    '''elementary charge [C]'''
    AU = 149597870700
    '''astronomical unit [m]'''
    PC = AU / Math.ARCSEC
    '''parsec [m]'''
    LIGHT_YEAR = C * Time.JULIAN_YEAR
    '''1 ly = c * 1 yr [m]'''
    DALTON = 1.660_539_068_92e-27
    '''1 Dalton = mass(12C) / 12'''
    EV = ELC
    '''electron volt'''
    GRAVITY = 9.80665
    '''standard acceleration of gravity [m/s2]'''
    KELVIN_ZERO = 273.15
    '''Kelvin zero point'''
    ATM = 101325
    '''standard atmosphere [Pa]'''
    SSP = 100000
    '''standard-state pressure [Pa]'''
    MMHG = Fraction(ATM, 760)
    '''1 mmHg = 1 atm / 760 [Pa]'''
    KCAL = 4184
    '''kilo-calorie [J]'''
    CAL = Fraction(KCAL, 1000)
    '''calorie [J]'''
    NAUTICAL_MILE = 1852


class Imperial(ConstClass):
    INCH = Fraction(127, 5000)
    TWIP = INCH / 1440
    THOU = INCH / 1000
    BARLEYCORN = INCH / 3
    HAND = 4 * INCH
    FOOT = 12 * INCH
    YARD = 3 * FOOT
    CHAIN = 22 * YARD
    FURLONG = 10 * CHAIN
    MILE = 8 * FURLONG
    LEAGUE = 3 * MILE
