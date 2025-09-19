'''
ConstClass is a class storing static constants.
It's similar to Enum, but members of Enum need an extra value attribute `value`
to access its value, and I think it's unnecessary, so I use ConstClass.
'''

from typing import Any, NoReturn

__all__ = ['ConstMeta', 'ConstClass']


class ConstMeta(type):
    def __new__(metacls, cls, bases, attrs, /, **kwds: Any):
        return super().__new__(metacls, cls, bases, attrs, **kwds)

    def __init__(metacls, cls, bases, attrs, /, **kwds: Any) -> None:
        return super().__init__(cls, bases, attrs, **kwds)

    def __call__(metacls, *args: Any, **kwds: Any) -> NoReturn:
        '''
        No, this class restores constants, you can NOT instantiate objects.
        please access constants via class attributes.
        '''
        raise TypeError(metacls.__call__.__doc__)

    def __setattr__(cls, name, value) -> NoReturn:
        '''
        No, this class restores constants, you can NOT modify them.
        '''
        raise AttributeError(cls.__class__.__setattr__.__doc__)

    def __delattr__(cls, name) -> NoReturn:
        '''
        No, this class restores constants, you can NOT delete them.
        '''
        raise AttributeError(cls.__class__.__delattr__.__doc__)


class ConstClass(metaclass=ConstMeta):
    pass 
