from ._data.prefixes import PREFIX, PREFIX_ALIAS, PREFIX_NAME
from ._data.units import UNIT, UNIT_ALIAS, UNIT_NAME
from .exceptions import UnitSymbolError

_PREFIX_MAXLEN = max(map(len, PREFIX))
_PREFIX_NAME_MINLEN = min(map(len, filter(None, PREFIX_NAME)))
_PREFIX_NAME_MAXLEN = max(map(len, PREFIX_NAME))


def _resolve_single(symbol: str) -> tuple[str, str]:
    '''resolve unit symbol to (prefix, unit) str tuple.'''
    # unit symbol without prefix
    if symbol in UNIT:
        return '', symbol
    # unit alias symbol without prefix
    if symbol in UNIT_ALIAS:
        return '', UNIT_ALIAS[symbol]
    # unit (alias) symbol with prefix symbol
    for plen in range(1, _PREFIX_MAXLEN):
        prefix, unit = symbol[:plen], symbol[plen:]
        if prefix in PREFIX_ALIAS:
            prefix = PREFIX_ALIAS[prefix]
        if unit in UNIT_ALIAS:
            unit = UNIT_ALIAS[unit]
        if prefix in PREFIX and unit in UNIT:
            if UNIT[unit].noprefix:
                continue
            return prefix, unit
    # unit name without prefix
    if symbol in UNIT_NAME:
        return '', UNIT_NAME[symbol]
    # unit name with prefix name
    for plen in range(_PREFIX_NAME_MINLEN, _PREFIX_NAME_MAXLEN):
        prefix, unit = symbol[:plen], symbol[plen:]
        if prefix in PREFIX_NAME and unit in UNIT_NAME:
            if UNIT[UNIT_NAME[unit]].noprefix:
                continue
            return PREFIX_NAME[prefix], UNIT_NAME[unit]
    raise UnitSymbolError(f"'{symbol}' is not a valid unit symbol.")


class SingleUnit:

    __slots__ = ('_unit', '_prefix')

    def __init__(self, symbol: str):
        if not isinstance(symbol, str):
            raise TypeError(f"{type(symbol)=} is not 'str''.")
        self._prefix, self._unit = _resolve_single(symbol)
        UNIT[self._unit].deprecation_warning()
          
    @classmethod
    def _move(cls, prefix: str, unit: str):
        obj = super().__new__(cls)
        obj._prefix, obj._unit = prefix, unit
        return obj

    @property
    def prefix(self) -> str: return self._prefix
    @property
    def unit(self) -> str: return self._unit
    @property
    def symbol(self) -> str: return self.prefix + self.unit
    @property
    def prefix_name(self) -> str: return PREFIX[self.prefix].name[0]
    @property
    def unit_name(self) -> str: return UNIT[self.unit].name[0]
    @property
    def name(self) -> str: return self.prefix_name + self.unit_name
    @property
    def prefix_factor(self): return PREFIX[self.prefix].factor
    @property
    def unit_factor(self): return UNIT[self.unit].factor
    @property
    def factor(self): return self.prefix_factor * self.unit_factor
    @property
    def dimension(self): return UNIT[self.unit].dimension

    def deprefix(self):
        return self if self.hasnoprefix() else SingleUnit._move('', self._unit)

    def hasnoprefix(self) -> bool: return self.prefix == ''

    def hasprefix(self) -> bool: return self.prefix != ''

    def isprefix(self) -> bool: return self.unit == '' and self.prefix != ''
    
    def __repr__(self) -> str: return f'{self.__class__.__name__}({self.symbol})'

    def __str__(self) -> str: return self.symbol

    def __hash__(self) -> int: return hash((self.prefix, self.unit))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SingleUnit):
            return NotImplemented
        return self.prefix == other.prefix and self.unit == other.unit
    
