from typing import Iterable, TypeVar

T = TypeVar('T')


def unzip(zip_iter: Iterable):
    '''
    >>> a = list(zip([1, 2, 3], ['a', 'b', 'c']))
    [(1, 'a'), (2, 'b'), (3, 'c')]
    >>> list(unzip(a))
    [(1, 2, 3), ('a', 'b', 'c')]
    '''
    return zip(*zip_iter)


def firstof(iterable: Iterable[T], /, default: T) -> T:
    '''return the first item of an iterable.'''
    for item in iterable:
        return item
    return default


def neg_after(ls: list, idx: int) -> None:
    '''negate items in-place after an index (exclusive) in the list.'''
    for i in range(idx + 1, len(ls)):
        ls[i] = -ls[i]


