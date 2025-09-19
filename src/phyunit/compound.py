from fractions import Fraction
from itertools import chain
from typing import Generic, Iterator, TypeVar

from .utils.number import ZERO, common_fraction
from .utils.operator import inplace

__all__ = ['Compound']

K = TypeVar('K')


class Compound(Generic[K]):
    __slots__ = ('_elements',)

    def __init__(self, elements: dict[K, int | Fraction] = {}, /) -> None:
        if not isinstance(elements, dict):
            raise TypeError(f"{type(elements) = } is not 'dict'.")
        self._elements = {k: common_fraction(v) for k, v in elements.items() if v}
        
    @classmethod
    def _move(cls, elements: dict[K, Fraction], /):
        '''internal __new__, directly move elements.'''
        assert all(isinstance(v, Fraction) and v != 0 for v in elements.values())
        obj = super().__new__(cls)
        obj._elements = elements
        return obj

    def __contains__(self, key: K) -> bool: return key in self._elements

    def __getitem__(self, key: K) -> Fraction:
        return self._elements.get(key, ZERO)

    def __setitem__(self, key: K, value: int | Fraction) -> None:
        if value == 0:
            self._elements.pop(key, ZERO)
        else:
            self._elements[key] = common_fraction(value)

    def __delitem__(self, key: K) -> None: del self._elements[key]

    def __iter__(self) -> Iterator[K]: return iter(self._elements)

    def __repr__(self) -> str:
        kv = ', '.join(f'{repr(k)}: {v}' for k, v in self._elements.items())
        return f'{self.__class__.__name__}(' + '{' + kv + '})'

    def __str__(self) -> str:
        kv = ', '.join(f'{k}: {v}' for k, v in self._elements.items())
        return '{' + kv + '}'

    def __len__(self) -> int: return len(self._elements)

    def copy(self): return self._move(self._elements.copy())

    def keys(self): return self._elements.keys()

    def values(self): return self._elements.values()

    def items(self): return self._elements.items()

    def pos_items(self):
        '''filter items whose value > 0.'''
        return filter(lambda item: item[1] > 0, self._elements.items())

    def neg_items(self):
        '''filter items whose value < 0.'''
        return filter(lambda item: item[1] < 0, self._elements.items())

    def pop(self, key: K, default: Fraction = ZERO) -> Fraction: 
        return self._elements.pop(key, default)

    def clear(self): self._elements.clear()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Compound):
            return NotImplemented
        return self._elements == other._elements

    def __pos__(self):
        return self._move({k: +v for k, v in self.items()})

    def __neg__(self):
        return self._move({k: -v for k, v in self.items()})

    def __add__(self, other):
        if not isinstance(other, Compound):
            return NotImplemented
        return self.__class__({k: self[k] + other[k] for k in chain(self, other)})

    def __sub__(self, other):
        if not isinstance(other, Compound):
            return NotImplemented
        return self.__class__({k: self[k] - other[k] for k in chain(self, other)})

    def __mul__(self, other: int | Fraction):
        if other == 0:
            return self._move({})
        other = common_fraction(other)
        return self._move({k: v * other for k, v in self.items()})

    def __truediv__(self, other: int | Fraction):
        other = common_fraction(other)
        return self._move({k: v / other for k, v in self.items()})

    __iadd__ = inplace(__add__)
    __isub__ = inplace(__sub__)
    __imul__ = inplace(__mul__)
    __itruediv__ = inplace(__truediv__)
    __rmul__ = __mul__
