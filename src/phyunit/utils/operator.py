import operator
from typing import Callable, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')


def xnor(a: bool, b: bool):
    '''(a and b) or (not a and not b)'''
    return not operator.xor(a, b)


def inplace(op: Callable[[X, Y], X]) -> Callable[[X, Y], X]:
    '''
    The easiest way to generate __iop__ using __op__.
    In this way:
    >>> b = a
    >>> b += c  # a no change
    '''

    def iop(self: X, other: Y) -> X:
        self = op(self, other)
        return self
    return iop
