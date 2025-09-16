from . import constclass
from . import number
from . import special_char


from typing import Callable, TypeVar


X = TypeVar('X')
Y = TypeVar('Y')


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

