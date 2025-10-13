from fractions import Fraction
from typing import Iterable, SupportsIndex

from .utils.number import common_fraction
from .utils.operator import inplace
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

    def as_tuple(self): return self.__vector

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
        if self.is_dimensionless():
            return '1'
        return ''.join(s + superscript(v) for s, v in zip(_SYMBOL, self) if v)

    def __len__(self) -> int: return _LEN

    def __hash__(self) -> int: return hash(self.__vector)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self.__vector == other.__vector

    def is_dimensionless(self):
        '''all dimension is zero.'''
        return self is DIMENSIONLESS or all(v == 0 for v in self)

    def components(self) -> set[str]:
        '''
        return the composition of the dimension.

        >>> Dimension(T=1, M=-2).components()
        {'T', 'M'}
        '''
        return {s for v, s in zip(self, _VARIABLE) if v}

    def is_geometric(self):
        '''components only L != 0.'''
        return self.components() == {'L'}

    def is_kinematic(self):
        '''components only T and L != 0.'''
        return self.components() == {'T', 'L'}

    def is_dynamic(self):
        '''components only T, L and M != 0'''
        return self.components() == {'T', 'L', 'M'}

    def __contains__(self, dim: str) -> bool:
        '''
        check if the dimension contains the base dimension.

        Example
        ---
        >>> 'L' in Dimension(L=1, T=-1)
        True
        >>> 'M' in Dimension(L=1, T=-1)
        False
        >>> 'X' in Dimension(L=1, T=-1)
        ValueError: 'X' is not a base dimension symbol.
        '''
        if not isinstance(dim, str):
            raise TypeError(f"{type(dim) = } is not 'str'.")
        if dim not in _DICT:
            raise ValueError(f"'{dim}' is not a base dimension symbol.")
        return self.__vector[_DICT[dim]] != 0

    def as_Gaussian(self):
        '''Gaussian unit system: 1 statC = 1 g¹ᐟ²·cm³ᐟ²/s'''
        T = self.T - self.I
        L = self.L + self.I * 3 / 2
        M = self.M + self.I / 2
        return self.__class__(T, L, M, 0, self.Theta, self.N, self.J)

    @property
    def inv(self):
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
        return self.inv

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
    
    def corresponding_quantity(self):
        """
        return the corresponding physical quantity names of the dimension.
        If not found, return None.

        >>> Dimension(T=1, L=1).corresponding_quantity()
        ('velocity', 'speed')
        >>> Dimenesion(T=-3, L=1).corresponding_quantity()
        None
        """
        from .dimensionconst import _CORR_QUANTITY
        return _CORR_QUANTITY.get(self)


DIMENSIONLESS = Dimension()

