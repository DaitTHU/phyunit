import re
from collections import Counter
from fractions import Fraction
from math import prod as float_product
from typing import Self

from ._data.units import BASE_SI, UNIT, UNIT_STD, UnitData
from .compound import Compound
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import UnitOverwriteWarning
from .singleunit import SingleUnit
from .utils import inplace
from .utils.iter_tools import neg_after
from .utils.number import common_fraction
from .utils.special_char import superscript as sup

_SEP = re.compile(r'[/.·]')  # unit separator pattern
_SEPS = re.compile(r'[/.· ]')  # unit separator pattern with space
_NUM = re.compile(r'[+-]?[0-9]+$')  # number pattern
_EXPO = re.compile(r'\^?[+-]?[0-9]+$')  # exponent pattern


def _resolve_multi(symbol: str, sep: re.Pattern[str]) -> Compound[SingleUnit]:
    '''
    Resolve a unit symbol string into its constituent unit elements as a Compound of SimpleUnit.
    This function parses a unit symbol (e.g., "m/s^2", "kg·m^2/s^2") and decomposes it into its
    base units and their corresponding exponents. It handles unit separators (/, ., ·),
    parses exponents, and correctly negates exponents for units following a division ("/").
    The result is a Compound object mapping SimpleUnit instances to their integer exponents.
    Args:
        symbol (str): The unit symbol string to resolve.
        sep (re.Pattern): The regex pattern used to split the symbol into units.
    Returns:
        Compound[SimpleUnit]: A mapping of SimpleUnit objects to their exponents representing the parsed unit.
    Raises:
        ValueError: If the symbol cannot be parsed into valid units.
    '''
    # split symbol into unit+exponent via separator
    unites = [unite for unite in sep.split(symbol) if unite]
    expos = [1 if m is None else int(m.group()) for m in map(_NUM.search, unites)]
    # find the first '/' and negate all exponents after it
    for i, sep_match in enumerate(sep.finditer(symbol)):
        if '/' in sep_match.group():
            neg_after(expos, i)
            break
    elements: Compound[SingleUnit] = Compound()
    for unite, e in zip(unites, expos):
        if e != 0:
            elements[SingleUnit(_EXPO.sub('', unite))] += e
    return elements


class MultiUnit:
    
    __slots__ = ('_elements', '_dimension', '_factor', '_symbol', '_name')
    
    def __init__(self, symbol: str, /):
        if not isinstance(symbol, str):
            raise TypeError(f"{type(symbol)=} is not 'str''.")
        try:
            element = _resolve_multi(symbol, _SEP)
        except ValueError:
            element = _resolve_multi(symbol, _SEPS)
        self.__derive_properties(element)

    @classmethod
    def _move(cls, elements: Compound[SingleUnit], /):
        obj = super().__new__(cls)
        obj.__derive_properties(elements)
        return obj
    
    def __derive_properties(self, elements: Compound[SingleUnit]):
        '''derive properties from the elements.'''
        self._elements = elements
        self._dimension = Dimension.product(u.dimension**e for u, e in elements.items())
        self._factor = float_product(u.factor**e for u, e in elements.items())
        # symbol and name
        self._symbol = '·'.join(u.symbol + sup(e) for u, e in elements.pos_items())
        self._name = '·'.join(u.name + sup(e) for u, e in elements.pos_items())
        if any(e < 0 for e in elements.values()):
            self._symbol += '/' + '·'.join(u.symbol + sup(-e) for u, e in elements.neg_items())
            self._name += '/' + '·'.join(u.name + sup(-e) for u, e in elements.neg_items())
            
    @classmethod
    def register(cls, symbol: str, name: str | list[str], factor: float, 
                 dimension: Dimension = DIMENSIONLESS, *,
                 alias: None | str | list[str] = None, noprefix=False):
        if _SEP.match(symbol) is not None:
            raise ValueError(
                f"'{symbol}' is not a valid single-unit symbol: "
                "it contains unit separator characters (/, ., ·, space)."
            )
        if symbol in UNIT:
            import warnings
            warnings.warn(
                f"'{symbol}' is already registered as a unit, "
                f"overwriting it with the new definition.",
                UnitOverwriteWarning
            )
        UNIT[symbol] = UnitData(factor, name, dimension, alias=alias, noprefix=noprefix)
        
    @classmethod
    def as_unit(cls, unit: str | Self):
        '''transform a str/Unit object to a Unit object.'''
        if isinstance(unit, cls):
            return unit
        if isinstance(unit, str):
            return cls(unit)
        raise TypeError(f"Unit must be 'str' or '{cls}', not '{type(unit)}'.")
    
    @property
    def dimension(self) -> Dimension: return self._dimension
    @property
    def factor(self) -> float: return self._factor
    @property
    def symbol(self) -> str: return self._symbol
    @property
    def name(self) -> str: return self._name
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.symbol)})'

    def __str__(self) -> str: return self.symbol

    def __hash__(self) -> int: return hash((self.dimension, self.factor))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiUnit):
            return NotImplemented
        return self.dimension == other.dimension and self.factor == other.factor

    def sameas(self, other: Self) -> bool:
        return self._elements == other._elements
    
    def isdimensionless(self) -> bool: return self.dimension.isdimensionless()

    def deprefix(self):
        '''return a new unit that remove all the prefix.'''
        if all(unit.hasnoprefix() for unit in self._elements):
            return self
        elements = self._elements.copy()
        for unit in filter(lambda u: u.hasprefix(), self._elements):
            elements[unit.deprefix()] += elements.pop(unit)
        return self._move(elements)
    
    def toSIbase(self):
        '''return a combination of base SI unit with the same dimension.'''
        elems = {SingleUnit(s): e for s, e in zip(BASE_SI, self.dimension) if e}
        return self._move(Compound._move(elems))  # type: ignore
        
    def simplify(self):
        '''
        try if the complex unit can be simplified as a single unit
        (i.e. `u`, `u⁻¹`, `u²`, `u⁻²`). 
        
        `u` is one of the chosen standard SI units for different dimensions,
        like mass for kg, length for m, time for s, etc.
        Here is the full list of them:
        - Base: m[L], kg[M], s[T], A[I], K[H], mol[N], cd[J];
        - Mechanic: Hz[T⁻¹], N[T⁻²LM], Pa[T⁻²L⁻¹M], J[T⁻²L²M], W[T⁻³L²M];
        - Electromagnetic: C[TI], V[T⁻³L²MI⁻¹], F[T⁴L⁻²M⁻¹I²], Ω[T⁻³L²MI⁻²], 
            S[T³L⁻²M⁻¹I²], Wb[T⁻²L²MI⁻¹], T[T⁻²MI⁻¹], H[T⁻²L²MI⁻²];
        - Other: lx[L⁻²J], Gy[T⁻²L²], kat[T⁻¹N]
        '''
        # single unit itself
        if len(self._elements) < 2:
            return self
        # single unit with simple exponent
        _SIMPLE_EXPONENT = tuple(map(common_fraction, (1, -1, 2, -2)))
        for e in _SIMPLE_EXPONENT:
            symbol = UNIT_STD.get(self.dimension.root(e))
            if symbol is None:
                continue
            return self._move(Compound._move({SingleUnit(symbol): e}))  # type: ignore
        # reduce units with same dimension
        dim_counter = Counter(u.dimension for u in self._elements)
        if all(count < 2 for count in dim_counter.values()):
            return self  # fail to simplify
        elements = self._elements.copy()
        for dim, count in dim_counter.items():
            if count < 2:
                continue
            symbol = UNIT_STD.get(dim)
            if symbol is None:
                continue
            for unit in filter(lambda u: u.dimension == dim, self._elements):
                e = elements.pop(unit)
                elements[SingleUnit(symbol)] += e
        return self._move(elements)

    def inverse(self): return self._move(-self._elements)

    def __mul__(self, other):
        if not isinstance(other, MultiUnit):
            return NotImplemented
        return self._move(self._elements + other._elements)

    def __truediv__(self, other):
        if not isinstance(other, MultiUnit):
            return NotImplemented
        return self._move(self._elements - other._elements)

    def __pow__(self, n: int | float | Fraction):
        return self._move(self._elements * n)

    __imul__ = inplace(__mul__)
    __itruediv__ = inplace(__truediv__)
    __ipow__ = inplace(__pow__)

    def root(self, n: int | float | Fraction):
        return self._move(self._elements / n)




