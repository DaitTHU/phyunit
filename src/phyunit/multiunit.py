import re
from collections import Counter
from fractions import Fraction
from math import prod as float_product

from ._data.units import BASE_SI, UNIT_STD
from .compound import Compound
from .dimension import Dimension
from .singleunit import SingleUnit
from .utils.iter_tools import neg_after
from .utils.operator import inplace
from .utils.special_char import SUP_TRANS
from .utils.special_char import superscript as sup

_SEP = re.compile(r'[/.·]')  # unit separator pattern
_SEPS = re.compile(r'[/.· ]')  # unit separator pattern with space
_NUM = re.compile(r'[+-]?[0-9]+$')  # number pattern
_EXPO = re.compile(r'\^?[+-]?[0-9]+$')  # exponent pattern


def _resolve_multi(symbol: str, sep: re.Pattern[str]) -> Compound[SingleUnit]:
    '''
    Resolve a unit symbol string into its constituent unit elements as a Compound of SingleUnit.
    This function parses a unit symbol (e.g., "m/s^2", "kg·m^2/s^2") and decomposes it into its
    base units and their corresponding exponents. It handles unit separators (/, ., ·),
    parses exponents, and correctly negates exponents for units following a division ("/").
    The result is a Compound object mapping SingleUnit instances to their integer exponents.
    Args:
        symbol (str): The unit symbol string to resolve.
        sep (re.Pattern): The regex pattern used to split the symbol into units.
    Returns:
        Compound[SingleUnit]: A mapping of SingleUnit objects to their exponents representing the parsed unit.
    Raises:
        ValueError: If the symbol cannot be parsed into valid units.
    '''
    symbol = symbol.translate(SUP_TRANS)  # translate superscript to digit
    # split symbol into unit+exponent via separator
    unites = [unite for unite in sep.split(symbol)]
    expos = [1 if m is None else int(m.group()) for m in map(_NUM.search, unites)]
    # find the first '/' and negate all exponents after it
    for i, sep_match in enumerate(sep.finditer(symbol)):
        if '/' in sep_match.group():
            neg_after(expos, i)
            break
    elements: Compound[SingleUnit] = Compound()
    for unite, e in zip(unites, expos):
        if e != 0 and unite:
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
    def as_unit(cls, unit):
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

    def sameas(self, other) -> bool:
        if not isinstance(other, MultiUnit):
            return NotImplemented
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
        Simplify the complex unit to a simple unit with the same dimension.

        The form will be the one of _u_, _u⁻¹_, _u²_, _u⁻²_,
        where _u_ stands for the standard SI unit,
        like mass for _kg_, length for _m_, time for _s_, etc.

        Here list the standard SI units for different dimensions:
        - Base: 
          _m_ [L], _kg_ [M], _s_ [T], _A_ [I], _K_ [H], _mol_ [N], _cd_ [J];
        - Mechanic: 
          _Hz_ [T⁻¹], _N_ [T⁻²LM], _Pa_ [T⁻²L⁻¹M], _J_ [T⁻²L²M], _W_ [T⁻³L²M];
        - Electromagnetic: 
          _C_ [TI], _V_ [T⁻³L²MI⁻¹], _F_ [T⁴L⁻²M⁻¹I²], _Ω_ [T⁻³L²MI⁻²],
          _S_ [T³L⁻²M⁻¹I²], _Wb_ [T⁻²L²MI⁻¹], _T_ [T⁻²MI⁻¹], _H_ [T⁻²L²MI⁻²];
        - Other:
          _lx_ [L⁻²J], _Gy_ [T⁻²L²], _kat_ [T⁻¹N]
        '''
        # single unit itself
        if len(self._elements) < 2:
            return self
        if self.isdimensionless():
            return self._move(Compound._move({}))  # type: ignore
        # single unit with simple exponent
        _SIMPLE_EXPONENT = tuple(map(Fraction, (1, -1, 2, -2)))
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

    def __pow__(self, n: int | Fraction):
        return self._move(self._elements * n)

    __imul__ = inplace(__mul__)
    __itruediv__ = inplace(__truediv__)
    __ipow__ = inplace(__pow__)

    def root(self, n: int | Fraction):
        return self._move(self._elements / n)




