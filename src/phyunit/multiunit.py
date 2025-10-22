import re
from collections import Counter
from fractions import Fraction
from math import prod as float_product

from ._data.units import BASE_SI, UNIT_STD
from .compound import Compound
from .dimension import Dimension
from .singleunit import SingleUnit, UnitSymbolError
from .utils.iter_tools import neg_after, firstof
from .utils.operator import inplace
from .utils.special_char import SUP_TRANS
from .utils.special_char import superscript as sup

_UNIT_STD = {d: SingleUnit(s) for d, s in UNIT_STD.items()}
_UNIT_SI = tuple(u for u in _UNIT_STD.values() if u.symbol in BASE_SI)
ONE, TWO = Fraction(1), Fraction(2)
_SIMPLE_EXPONENT = (ONE, -ONE, TWO, -TWO)

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

    def __init__(self, symbol: str = '', /):
        """
        TODO: cache for simple units.
        """
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
    
    @classmethod
    def _move_dict(cls, elements: dict[SingleUnit, Fraction], /):
        return cls._move(Compound._move(elements))  # type: ignore

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
    def from_dimension(cls, dimension: Dimension, /):
        """
        Create a combination of base SI units with the same dimension.

        Example: for dimension of force, the unit is 'm·kg/s²' (newton in SI base units).
        """
        if not isinstance(dimension, Dimension):
            raise TypeError(f"{type(dimension) = } is not 'Dimension'.")
        return cls._move_dict({u: e for u, e in zip(_UNIT_SI, dimension) if e})

    @classmethod
    def ensure(cls, unit):
        """
        ensure the output is a Unit instance, as the input can be str or Unit.
        Args:
            unit (str | Unit): the unit to ensure.
        Returns:
            Unit: the ensured Unit instance.
        Raises:
            TypeError: if the input is neither str nor Unit.
        """
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
        symbol = None if self.symbol == '' else repr(self.symbol)
        return f'{self.__class__.__name__}({symbol})'

    def __str__(self) -> str: return self.symbol

    def __hash__(self) -> int: return hash((self.dimension, self.factor))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiUnit):
            return NotImplemented
        return self.dimension == other.dimension and self.factor == other.factor

    def is_dimensionless(self) -> bool: return self.dimension.is_dimensionless()

    def __len__(self) -> int:
        """
        number of single units.
        >>> len(Unit('N'))
        1
        >>> len(Unit('kg·m/s²'))
        3
        """
        return len(self._elements)

    def __bool__(self) -> bool: return len(self._elements) > 0

    def is_single(self) -> bool:
        """
        if this unit is a single unit.
        >>> Unit('m').is_single()
        True
        >>> Unit('m²').is_single()
        False
        >>> Unit('kg·m/s²').is_single()
        False
        """
        return len(self._elements) == 0 or \
            (len(self._elements) == 1 and firstof(self._elements.values()) == 1)

    def has_prefix(self) -> bool:
        return any(unit.has_prefix() for unit in self._elements)
    
    def components(self):
        """return an iterable of unit symbols for the unit."""
        return (u.symbol for u in self._elements)

    def items(self):
        """return an iterable of (unit symbol, exponent) pairs for the unit."""
        return ((u.symbol, e) for u, e in self._elements.items())

    def __contains__(self, unit) -> bool:
        """
        if a single unit is contained in this unit.

        Example
        ---
        >>> 'm' in Unit('m/s')   # str is allowed
        True
        >>> 'meter' in Unit('m/s')  # alias or name is also allowed
        True
        >>> Unit('s') in Unit('m/s')
        True
        >>> Unit('kg') in Unit('m/s')
        False
        >>> Unit('m²') in Unit('m/s')
        False
        >>> Unit('N') in Unit('kg·m/s²')
        False
        >>> Unit('m/s') in Unit('m/s')  # ValueError: 'm/s' is not a single unit.
        """
        if isinstance(unit, MultiUnit) and unit.is_single():
            return all(u in self._elements for u in unit._elements)
        if isinstance(unit, str):
            try:
                return SingleUnit(unit) in self._elements
            except UnitSymbolError:
                unit = MultiUnit(unit)
        if not isinstance(unit, (str, MultiUnit)):
            raise TypeError(f"{type(unit) = } is not 'str' or 'Unit'.")
        raise ValueError(f"'{unit}' is not a single unit.")

    def deprefix(self):
        '''return a new unit that remove all the prefix.'''
        if not self.has_prefix():
            return self
        elements = self._elements.copy()
        for unit in self._elements:
            if unit.has_prefix():
                elements[unit.deprefix()] += elements.pop(unit)
        return self._move(elements)

    def SI_base_form(self):
        '''return the unit in SI base units with the same dimension.'''
        return self.from_dimension(self.dimension)

    def simplify(self):
        """
        Simplify the complex unit to a simple unit with the same dimension.

        The form will be the one of _u_, _u⁻¹_, _u²_, _u⁻²_,
        where _u_ stands for the standard SI unit,
        like mass for _kg_, length for _m_, time for _s_, etc.

        Here list the standard SI units for different dimensions:
        s, m, kg, A, K, mol, cd,
        Hz, N, Pa, J, W, C, V, F, Ω, S, Wb, T, H, lx, Gy, kat.
        """
        # single unit itself
        if len(self._elements) < 2:
            return self
        if self.is_dimensionless():
            return self._move_dict({})
        # single unit with simple exponent
        for e in _SIMPLE_EXPONENT:
            unit_std = _UNIT_STD.get(self.dimension.root(e))
            if unit_std is None:
                continue
            return self._move_dict({unit_std: e})
        # reduce units with same dimension
        dim_counter = Counter(u.dimension for u in self._elements)
        if all(count < 2 for count in dim_counter.values()):
            return self  # fail to simplify
        elements = self._elements.copy()
        for dim, count in dim_counter.items():
            if count < 2:
                continue
            unit_std = _UNIT_STD.get(dim)
            if unit_std is None:
                continue
            for unit in self._elements:
                if unit.dimension == dim and unit != unit_std:
                    elements[unit_std] += elements.pop(unit)
        return self._move(elements)

    @property
    def inv(self): return self._move(-self._elements)

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




