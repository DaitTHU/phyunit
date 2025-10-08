import re
from fractions import Fraction

__all__ = ['superscript', 'small_frac']

DIGIT = '0123456789+-=()'
SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾'
SUBSCRIPT = '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎'
SUP_TRANS = str.maketrans(SUPERSCRIPT, DIGIT)
SUB_TRANS = str.maketrans(SUBSCRIPT, DIGIT)

FRAC_CHAR = {Fraction(k).limit_denominator(10): v for k, v in {
    1/2: '½',
    1/3: '⅓', 2/3: '⅔',
    1/4: '¼', 3/4: '¾',
    1/5: '⅕', 2/5: '⅖', 3/5: '⅗', 4/5: '⅘',
    1/6: '⅙', 5/6: '⅚',
    # 1/7: '⅐',
    1/8: '⅛', 3/8: '⅜', 5/8: '⅝', 7/8: '⅞',
    # 1/9: '⅑', 1/10: '⅒',
}.items()}


def superscript(ratio: int | Fraction, /, *, omit1=True) -> str:
    '''
    turn a number (Fraction/int) into superscript, ¹ will be omitted if omit1=True
    >>> superscript(2)  # ²
    >>> superscript(-1) # ⁻¹
    >>> superscript(Fraction(3, 4))  # ³ᐟ⁴
    '''
    if ratio < 0:
        return '⁻' + superscript(-ratio, omit1=False)
    if ratio.denominator == 1:
        if omit1 and ratio.numerator == 1:
            return ''
        return _sup(ratio.numerator)
    assert isinstance(ratio, Fraction)
    return _sup(ratio.numerator) + 'ᐟ' + _sup(ratio.denominator)


def small_frac(ratio: Fraction) -> str:
    '''
    turn a fraction into a small fraction string.
    >>> small_frac(Fraction(3, 4))  # ¾
    '''
    if ratio < 0:
        return '-' + small_frac(-ratio)
    if ratio in FRAC_CHAR:
        return FRAC_CHAR[ratio]
    if ratio.numerator == 1:
        return '⅟' + _sub(ratio.denominator)
    return _sup(ratio.numerator) + '⁄' + _sub(ratio.denominator)


def _sup(number: int) -> str:
    return ''.join(SUPERSCRIPT[int(digit)] for digit in str(number))


def _sub(number: int) -> str:
    return ''.join(SUBSCRIPT[int(digit)] for digit in str(number))
