from decimal import Decimal
from fractions import Fraction

ZERO = Fraction(0)
ONE = Fraction(1)
TWO = Fraction(2)
THREE = Fraction(3)
TEN = Fraction(10)


def common_fraction(number: int | float | Decimal | Fraction) -> Fraction:
    '''
    common means the denominator is small, like 1, 42, -2/3...

    >>> Fraction(1.47)
    Fraction(6620291452234629, 4503599627370496)
    >>> common_fraction(1.47)
    Fraction(147, 100)
    '''
    if isinstance(number, Fraction):
        return number
    frac = Fraction(number)
    return frac.limit_denominator() if isinstance(number, float) else frac

