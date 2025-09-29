from fractions import Fraction
from typing import Iterable, SupportsIndex

from .utils.number import common_fraction
from .utils.operator import inplace, xnor
from .utils.special_char import superscript

_VARIABLE = ('T', 'L', 'M', 'I', 'Theta', 'N', 'J')
_SYMBOL = ('T', 'L', 'M', 'I', 'Θ', 'N', 'J')
_DICT = {s: i for i, s in enumerate(_SYMBOL)}
_LEN = len(_SYMBOL)


class Dimension:

    __slots__ = ('__vector',)

    def __init__(self, T=0, L=0, M=0, I=0, Theta=0, N=0, J=0):
        self.__vector = tuple(map(common_fraction, (T, L, M, I, Theta, N, J)))

    @classmethod
    def _move(cls, vector: Iterable[Fraction], /):
        '''internal use only, assert len(vector) == _DIM_NUM'''
        obj = super().__new__(cls)
        obj.__vector = tuple(vector)
        return obj

    def astuple(self): return self.__vector

    def __getitem__(self, key: SupportsIndex | str):
        if isinstance(key, str):
            return self.__vector[_DICT[key]]
        return self.__vector[key]

    def __iter__(self): return iter(self.__vector)

    @staticmethod
    def __unpack_vector():
        '''properties of Dimension.'''
        def __getter(i: int):
            return lambda self: self[i]  # closure
        return (property(__getter(i)) for i in range(_LEN))

    T, L, M, I, Theta, N, J = __unpack_vector()
    time, length, mass, electric_current, thermodynamic_temperature, \
        amount_of_substance, luminous_intensity = T, L, M, I, Theta, N, J

    def __repr__(self) -> str:
        para = ', '.join(f'{s}={v}' for s, v in zip(_VARIABLE, self) if v)
        return '{}({})'.format(self.__class__.__name__, para)

    def __str__(self) -> str:
        if self.isdimensionless():
            return '1'
        return ''.join(s + superscript(v) for s, v in zip(_SYMBOL, self) if v)

    def __len__(self) -> int: return _LEN

    def __hash__(self) -> int: return hash(self.__vector)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self.__vector == other.__vector

    def isdimensionless(self):
        '''all dimension is zero.'''
        return self is DIMENSIONLESS or all(v == 0 for v in self)

    def iscomposedof(self, *composition: str) -> bool:
        '''
        check if a dimension is composed of the compostion.

        >>> Dimension(T=1, M=-2).iscomposedof('T', 'M')
        True
        '''
        return all(xnor(v != 0, s in composition) for v, s in zip(self, _VARIABLE))

    def isgeometric(self):
        '''only L != 0.'''
        return self.iscomposedof('L')

    def iskinematic(self):
        '''only T and L != 0.'''
        return self.iscomposedof('T', 'L')

    def isdynamic(self):
        '''only T, L and M != 0'''
        return self.iscomposedof('T', 'L', 'M')

    def asGaussian(self):
        '''Gaussian unit system: 1 statC = 1 g¹ᐟ²·cm³ᐟ²/s'''
        I = self.I
        T = self.T - I
        L = self.L + I * 3 / 2
        M = self.M + I / 2
        return self.__class__(T, L, M, 0, self.Theta, self.N, self.J)

    def inverse(self):
        '''inverse of the Dimension.'''
        if self is DIMENSIONLESS:
            return self
        return self._move(-x for x in self.__vector)

    def __mul__(self, other: object):
        if not isinstance(other, Dimension):
            raise TypeError('unsupported type for *')
        return self._move(x + y for x, y in zip(self.__vector, other.__vector))

    def __truediv__(self, other: object):
        if not isinstance(other, Dimension):
            raise TypeError('unsupported type for /')
        return self._move(x - y for x, y in zip(self.__vector, other.__vector))

    def __pow__(self, n: int | float | Fraction):
        if self is DIMENSIONLESS:
            return self
        if not isinstance(n, (int, Fraction)):
            n = common_fraction(n)
        return self._move(x * n for x in self.__vector)

    __imul__ = inplace(__mul__)
    __itruediv__ = inplace(__truediv__)
    __ipow__ = inplace(__pow__)

    def __rtruediv__(self, one: object):
        '''only used in 1/dimension.'''
        if one != 1:
            raise ValueError(
                'Only 1 or Dimension object can divide Dimension object.')
        return self.inverse()

    def root(self, n: int | float | Fraction):
        '''inverse operation of power.'''
        if self is DIMENSIONLESS:
            return self
        if not isinstance(n, (int, Fraction)):
            n = common_fraction(n)
        return self._move(x / n for x in self.__vector)

    @staticmethod
    def product(dims: Iterable['Dimension']):
        '''return the product of dimension objects.'''
        start = DIMENSIONLESS
        for dim in dims:
            start *= dim
        return start


DIMENSIONLESS = Dimension()

