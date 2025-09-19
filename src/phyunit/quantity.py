import operator
from typing import Generic, TypeVar

try:
    import numpy as np  # type: ignore
except ImportError:
    from .utils import numpy_phony as np

from .dimension import Dimension
from .multiunit import MultiUnit
from .utils.numpy_ufunc import ufunc_dict
from .utils.operator import inplace
from .utils.valuetype import ValueType

T = TypeVar('T', bound=ValueType)


class Unit(MultiUnit):
    def __rmul__(self, other):
        '''other is not `Unit` or `Quantity`, treated as value * unit.'''
        return Quantity(other, self)

    def __rtruediv__(self, other):
        '''other is not `Unit` or `Quantity`, treated as value / unit.'''
        return Quantity(other, self.inverse())

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.multiply:
            return Quantity(inputs[0], self)
        if ufunc == np.true_divide:
            return Quantity(inputs[0], self.inverse())
        return NotImplemented


UNITLESS = Unit('')
DIMENSIONLESS = UNITLESS.dimension

DEGREE = Unit('degree')
RADIAN = Unit('rad')


class Constant(Generic[T]):
    
    __slots__ = ('_value', '_unit')
    
    def __init__(self, value: T, /, unit: str | Unit = UNITLESS):
        self._value = value
        self._unit = Unit.as_unit(unit)
        
    @property
    def value(self) -> T: return self._value
    @property
    def unit(self) -> Unit: return self._unit
    @property
    def dimension(self) -> Dimension: return self._unit.dimension
    @property
    def _std_value(self) -> T: return self._value * self._unit.factor
    @property
    def _base_cls(self): return Constant
    
    def __repr__(self) -> str:
        if self.unit == UNITLESS:
            return f'{self.__class__.__name__}({self.value})'
        return f"{self.__class__.__name__}({self.value}, '{self.unit}')"
    
    def __str__(self):
        if self.unit == UNITLESS:
            return str(self.value)
        return f'{self.value} {self.unit}'
    
    def __format__(self, format_spec: str):
        if self.unit == UNITLESS:
            return format(self.value, format_spec)
        return f'{self.value:{format_spec}} {self.unit}'
    
    def __getitem__(self, i): 
        if not hasattr(self._value, '__getitem__'):
            msg = f"'{self._value.__class__.__name__}' object is not subscriptable."
            raise TypeError(msg)
        return Quantity(self._value[i], self._unit)  # type: ignore
        
    
    def isdimensionless(self) -> bool: return self.unit.isdimensionless()

    def copy(self): return self.__class__(self._value, self._unit)
    
    def to(self, new_unit: str | Unit, *, strict: bool = True):
        '''unit conversion.
        
        If strict is True, the unit must have the same dimension.
        '''
        unit = Unit.as_unit(new_unit)
        if strict and unit.dimension != self.unit.dimension:
            raise ValueError(f"Cannot convert '{self.unit}' to '{unit}':"
                             f" dimension {self.dimension} != {unit.dimension}.")
        factor = self.unit.factor / unit.factor
        return self.__class__(self._value * factor, unit)
    
    def deprefix_unit(self):
        '''remove all the prefix of the unit.'''
        return self.to(self.unit.deprefix())

    def toSIbase(self):
        '''convert to SI base unit.'''
        return self.to(self.unit.toSIbase())

    def simplify_unit(self):
        '''simplify the unit.'''
        return self.to(self.unit.simplify())
    
    @staticmethod
    def __comparison(op):
        def __op(self: 'Constant', other):
            if not isinstance(other, self._base_cls):
                if not self.isdimensionless():
                    raise ValueError(f'{self.dimension} cannot be compared with dimensionless value.')
                return op(self._std_value, other)
            if self.dimension != other.dimension:
                raise ValueError(f'dimension {self.dimension} != {other.dimension}')
            return op(self._std_value, other._std_value)
        return __op
    
    __eq__ = __comparison(operator.eq)
    __ne__ = __comparison(operator.ne)
    __gt__ = __comparison(operator.gt)
    __lt__ = __comparison(operator.lt)
    __ge__ = __comparison(operator.ge)
    __le__ = __comparison(operator.le)
    
    @staticmethod
    def __unary(op):
        def __op(self: 'Constant'):
            return Constant(op(self.value), self.unit)
        return __op
    
    __neg__ = __unary(operator.neg)
    __pos__ = __unary(operator.pos)
    
    
    @staticmethod
    def __addsub(op, iop):
        '''construct operator: a + b, a - b.

        if a is dimensionless, b can be non-quantity. 
        Otherwise, a and b should meet dimension consistency.
        '''

        def __op(self: 'Constant', other: 'Constant'):
            if self.isdimensionless() and not isinstance(other, self._base_cls):
                return Quantity(op(self._std_value, other))
            if self.dimension != other.dimension:
                raise ValueError(f'dimension {self.dimension} != {other.dimension}')
            other_var = other.value * (other.unit.factor / self.unit.factor)
            return Quantity(op(self.value, other_var), self.unit)

        def __iop(self: 'Constant', other: 'Constant'):
            if self.isdimensionless() and not isinstance(other, self._base_cls):
                self._value *= self.unit.factor
                self._value = iop(self._value, other)
                self._unit = UNITLESS
                return self
            if self.dimension != other.dimension:
                raise ValueError(f'dimension {self.dimension} != {other.dimension}')
            other_var = other.value * (other.unit.factor / self.unit.factor)
            self._value = iop(self._value, other_var)
            return self

        def __rop(self: 'Constant', other):
            '''type(other) is not Constant.'''
            if not self.isdimensionless():
                raise ValueError(
                    f'{self.dimension} cannot ± dimensionless value.')
            return Quantity(op(other, self._std_value))

        return __op, __iop, __rop

    __add__, __Iadd__, __radd__ = __addsub(operator.add, operator.iadd)
    __sub__, __Isub__, __rsub__ = __addsub(operator.sub, operator.isub)
    __iadd__ = inplace(__add__)
    __isub__ = inplace(__sub__)

    @staticmethod
    def __muldiv(op, iop):
        '''operator: a * b, a / b, a @ b

        when a or b is not a `Constant` object, which will be treated as a
        dimensionless Constant.
        '''
        # unitop: unit [op] unit, [op] = * or /
        unitop = operator.mul if op is operator.matmul else op

        def __op(self: 'Constant', other: 'Constant'):
            if isinstance(other, Unit):
                return Quantity(self.value, unitop(self.unit, other))
            if not isinstance(other, self._base_cls):
                return Quantity(op(self.value, other), self.unit)
            return Quantity(op(self.value, other.value), unitop(self.unit, other.unit))

        def __iop(self: 'Constant', other: 'Constant'):
            if isinstance(other, Unit):
                self._unit = unitop(self.unit, other)
                return self
            if not isinstance(other, self._base_cls):
                self._value = iop(self._value, other)
                return self
            self._value = iop(self._value, other.value)
            self._unit = unitop(self._unit, other.unit)
            return self

        # rop: other is not a `Constant` object.
        if op is operator.truediv:
            def __rop(self: 'Constant', other):
                return Quantity(op(other, self._value), self.unit.inverse())
        else:
            def __rop(self: 'Constant', other):
                return Quantity(op(other, self._value), self.unit)

        return __op, __iop, __rop

    __mul__, __Imul__, __rmul__ = __muldiv(operator.mul, operator.imul)
    __matmul__, __Imatmul__, __rmatmul__ = __muldiv(
        operator.matmul, operator.imatmul)
    __truediv__, __Itruediv__, __rtruediv__ = __muldiv(
        operator.truediv, operator.itruediv)
    __imul__ = inplace(__mul__)
    __imatmul__ = inplace(__matmul__)
    __itruediv__ = inplace(__truediv__)

    def __pow__(self, other):
        return Quantity(self.value ** other, self.unit ** other)

    __ipow__ = inplace(__pow__)

    def __rpow__(self, other):
        if not self.isdimensionless():
            raise ValueError('Quantity must be dimensionless to be exponent.')
        return other ** self.value

    def root(self, n):
        '''n-th root of the quantity.'''
        return Quantity(self.value**(1/n), self.unit.root(n))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ufunc_name = ufunc.__name__
        units = [i.unit if isinstance(i, self._base_cls) else UNITLESS for i in inputs]
        unit = None  # None means output is just value, not Quantity
        if ufunc_name in ufunc_dict['dimless']:
            if not all(u.isdimensionless() for u in units):
                dims = ', '.join(f"'{u.dimension}'" for u in units if not u.isdimensionless())
                msg = f"Quantities in '{ufunc_name}' must be dimensionless, got {dims}."
                raise ValueError(msg)
            unit = None
        elif ufunc_name in ufunc_dict['bool']:
            unit = None
        elif ufunc_name in ufunc_dict['comparison'] or ufunc_name in ufunc_dict['dimsame']:
            if len(set(u.dimension for u in units)) > 1:
                dims = ', '.join(f"'{u.dimension}'" for u in units)
                msg = f"Quantities in '{ufunc_name}' must have the same dimension, got {dims}."
                raise ValueError(msg)
            if ufunc_name in ufunc_dict['dimsame'] and ufunc_name != 'arctan2':
                unit = units[0]
        elif ufunc_name in ufunc_dict['preserve']:
            unit = units[0]
        elif ufunc_name in ufunc_dict['angle']:
            if ufunc_name == 'deg2rad' and units[0] != DEGREE:
                msg = "deg2rad(quantity) requires unit to be degree."
                raise ValueError(msg)
            if ufunc_name == 'rad2deg' and units[0] != RADIAN:
                msg = "deg2rad(quantity) requires unit to be radian."
                raise ValueError(msg)
            if ufunc_name in {'deg2rad', 'radians'}:
                unit = RADIAN
            elif ufunc_name in {'rad2deg', 'degrees'}:
                unit = DEGREE
            else:
                msg = f"ufunc not covered, got '{ufunc_name}'"
                raise ValueError(msg)
        elif ufunc_name in ufunc_dict['product']:
            unit = units[0] * units[1]
        elif ufunc_name in ufunc_dict['nonlinear']:
            if ufunc_name == 'square':
                unit = units[0]**2
            elif ufunc_name == 'true_divide':
                unit = units[0] / units[1]
            elif ufunc_name == 'reciprocal':
                unit = units[0].inverse()
            elif ufunc_name == 'sqrt':
                unit = units[0].root(2)
            elif ufunc_name == 'cbrt':
                unit = units[0].root(3)
            else:
                msg = f"ufunc not covered, got '{ufunc_name}'"
                raise ValueError(msg)
        elif ufunc_name in ufunc_dict['other']:
            if ufunc_name == 'copysign':
                unit = units[0]
            elif ufunc_name == 'heaviside' and not units[1].isdimensionless():
                msg = f"Heaviside(x) must be dimensionless, got '{units[1].dimension}'"
                raise ValueError(msg)
            elif ufunc_name in {'power', 'float_power'}:
                if not units[1].isdimensionless():
                    msg = f"Exponent must be dimensionless, got '{units[1].dimension}'"
                    raise ValueError(msg)
                unit = units[0]**inputs[1]
            elif ufunc_name == 'frexp':
                mantissa, exponent = ufunc(inputs[0].value, **kwargs)
                return Quantity(mantissa, units[0]), exponent
            elif ufunc_name == 'ldexp':
                unit = units[0]
            elif ufunc_name in {'sign', 'signbit'}:
                unit = None
            else:
                msg = f"ufunc not covered, got '{ufunc_name}'"
                raise ValueError(msg)
        else:
            msg = f"ufunc not covered, got '{ufunc_name}'"
            raise ValueError(msg)
        func = getattr(ufunc, method)
        inputs = [i._std_value if isinstance(i, self._base_cls) else i for i in inputs]
        value = func(*inputs, **kwargs)
        return value if unit is None else Quantity(value / unit.factor, unit)
    
    def __array__(self, dtype=None):
        if dtype is None:
            dtype = np.float64
        return np.asarray(self.value, dtype=dtype)
    
    # __array_interface__ = {'dtype'}


class Quantity(Constant[T]):

    __slots__ = ()

    @property
    def value(self) -> T: return self._value
    @value.setter
    def value(self, value: T): self._value = value
    @property
    def unit(self) -> Unit: return self._unit
    @unit.setter
    def unit(self, unit: str | Unit): self._unit = Unit.as_unit(unit)
    @property
    def dimension(self) -> Dimension: return self._unit.dimension
    @property
    def _std_value(self) -> T: return self._value * self._unit.factor

    def isdimensionless(self) -> bool: return self.unit.isdimensionless()

    def copy(self):
        return self.__class__(self._value, self._unit)

    def __to(self, new_unit: Unit, inplace: bool):
        '''internal use only'''
        factor = self.unit.factor / new_unit.factor
        if inplace:
            self._value *= factor
            self._unit = new_unit
            return self
        return self.__class__(self._value * factor, new_unit)

    def to(self, new_unit: str | Unit, *, inplace=False, strict=True):
        '''
        unit conversion.
        If inplace is True, the original object will be modified.
        If strict is True, the unit must have the same dimension.
        '''
        unit = Unit.as_unit(new_unit)
        if strict and unit.dimension != self.unit.dimension:
            raise ValueError(f"Cannot convert {self.unit} to {unit}:"
                             f" dimension mismatch {self.dimension} != {unit.dimension}.")
        return self.__to(unit, inplace)

    def ito(self, new_unit: str | Unit, *, strict=True):
        '''inplace unit conversion.'''
        return self.to(new_unit, inplace=True, strict=strict)

    def deprefix_unit(self, *, inplace=False):
        return self.to(self.unit.deprefix(), inplace=inplace)
    
    def toSIbase(self, *, inplace=False):
        return self.to(self.unit.toSIbase(), inplace=inplace)
    
    def simplify_unit(self, *, inplace=False):
        return self.to(self.unit.simplify(), inplace=inplace)
    
    def __iadd__(self, other): return self.__Iadd__(other)
    def __isub__(self, other): return self.__Isub__(other)
    def __imul__(self, other): return self.__Imul__(other)
    def __imatmul__(self, other): return self.__Imatmul__(other)
    def __itruediv__(self, other): return self.__Itruediv__(other)

    def __ipow__(self, other):
        self._value **= other
        self._unit **= other
        return self


def constant(quantity: Quantity[T], unit: None | str | Unit = None, *, simplify=False):
    '''to make a Quantity object to a Constant.'''
    if unit is not None:
        quantity = quantity.to(unit)
    elif simplify:
        quantity = quantity.simplify_unit()
    return Constant(quantity.value, quantity.unit)
