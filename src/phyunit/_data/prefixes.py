# from decimal import Decimal

class PrefixData:
    '''
    data of prefix

    Attributes:
        factor (str): e.g., 1000 for kilo-
        name (str | list[str]): e.g., 'kilo', 'mega'
        alias (None | str | list[str]): alternative symbols
    '''

    __slots__ = ('factor', 'name', 'alias')

    def __init__(self, factor: float, name: str | list[str], *,
                 alias: None | str | list[str] = None) -> None:
        self.factor = factor
        self.name = [name] if isinstance(name, str) else name
        self.alias = [alias] if isinstance(alias, str) else alias

    def __hash__(self) -> int: return hash((self.factor, self.name[0]))


PREFIX: dict[str, PrefixData] = {
    # whole unit
    'Q': PrefixData(1e30, 'quetta'),
    'R': PrefixData(1e27, 'ronna'),
    'Y': PrefixData(1e24, 'yotta'),
    'Z': PrefixData(1e21, 'zetta'),
    'E': PrefixData(1e18, 'exa'),
    'P': PrefixData(1e15, 'peta'),
    'T': PrefixData(1e12, 'tera'),
    'G': PrefixData(1e9, 'giga'),
    'M': PrefixData(1e6, 'mega'),
    'k': PrefixData(1e3, 'kilo', alias='K'),
    'h': PrefixData(1e2, 'hecto'),
    'da': PrefixData(1e1, 'deka'),
    '': PrefixData(1, ''),
    # sub unit
    'd': PrefixData(1e-1, 'deci'),
    'c': PrefixData(1e-2, 'centi'),
    'm': PrefixData(1e-3, 'milli'),
    'µ': PrefixData(1e-6, 'micro', alias=['u', 'μ']),  # chr(0xB5): chr(0x3BC)
    'n': PrefixData(1e-9, 'nano'),
    'p': PrefixData(1e-12, 'pico'),
    'f': PrefixData(1e-15, 'femto'),
    'a': PrefixData(1e-18, 'atto'),
    'z': PrefixData(1e-21, 'zepto'),
    'y': PrefixData(1e-24, 'yocto'),
    'r': PrefixData(1e-27, 'ronto'),
    'q': PrefixData(1e-30, 'quecto'),
}
'''prefix {symbol: data}'''

PREFIX_NAME: dict[str, str] = {
    name: prefix for prefix, data in PREFIX.items() for name in data.name
}
'''prefix {name: symbol}'''

PREFIX_ALIAS: dict[str, str] = {
    alias: prefix for prefix, data in PREFIX.items() if data.alias is not None
    for alias in data.alias
}
'''prefix {alias symbol: symbol}'''
