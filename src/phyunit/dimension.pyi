from fractions import Fraction
from typing import Iterable, Iterator, Literal, Self, SupportsIndex, overload

__all__ = ['Dimension', 'DIMENSIONLESS']


class Dimension:
    """
    implementation of physical dimension defined by SI.

    | symbol  |         dimension         |
    |:-------:|:-------------------------:|
    |   `T`   |           time            |
    |   `L`   |          length           |
    |   `M`   |           mass            |
    |   `I`   |     electric current      |
    | `Theta` | thermodynamic temperature |
    |   `N`   |    amount of substance    |
    |   `J`   |    luminous intensity     |

    `Theta` is the ASCII substitute for Greek letter `Θ`.

    `Dimension` is like a 7-len `namedtuple` of `Fraction`.

    Construct
    ---
    see `__init__` docstring. Very straightforward.
    >>> time_dim = Dimension(T=1)
    >>> vilocity_dim = Dimension(-1, 1)
    >>> force_dim = Dimension(*[-2, 1, 1])

    Get base quantity
    ---
    You can get base quantity property like mass by index, initial
    or fullname.
    >>> force_dim[0]        # -2 (Time property)
    >>> force_dim.M         # 1
    >>> force_dim.length    # 1

    Operation
    ---
    very straight-forward:
    >>> force_dim.inv                           # T⁻²LM
    >>> power_dim = force_dim * vilocity_dim    # T⁻³L²M
    >>> vilocity_dim**2                         # T⁻²L²
    """

    def __init__(self, T=0, L=0, M=0, I=0, Theta=0, N=0, J=0) -> None:
        '''construct a `Dimension` object using 7 int/Fraction arguments,
        default 0.
        >>> time_dim = Dimension(T=1)
        >>> vilocity_dim = Dimension(-1, 1)

        You can also use * and ** operator to construct a `Dimension` object
        from a `Iterable` or `dict`:
        >>> force_dim = Dimension(*[-2, 1, 1])
        >>> power_dim = Dimension(**{'T': -3, 'L': 2, 'M': 1})
        '''

    def as_tuple(self) -> tuple[Fraction]: ...
    @overload
    def __getitem__(self, key: SupportsIndex) -> Fraction: ...
    @overload
    def __getitem__(self, key: Literal['T', 'L', 'M', 'I', 'Theta', 'N', 'J']) -> Fraction: ...
    def __iter__(self) -> Iterator[Fraction]: ...
    @property
    def T(self) -> Fraction: '''time'''
    @property
    def L(self) -> Fraction: '''length'''
    @property
    def M(self) -> Fraction: '''mass'''
    @property
    def I(self) -> Fraction: '''electric current'''
    @property
    def Theta(self) -> Fraction: '''thermodynamic temperature'''
    @property
    def N(self) -> Fraction: '''amount of substance'''
    @property
    def J(self) -> Fraction: '''luminous intensity'''
    @property
    def time(self) -> Fraction: ...
    @property
    def length(self) -> Fraction: ...
    @property
    def mass(self) -> Fraction: ...
    @property
    def electric_current(self) -> Fraction: ...
    @property
    def thermodynamic_temperature(self) -> Fraction: ...
    @property
    def amount_of_substance(self) -> Fraction: ...
    @property
    def luminous_intensity(self) -> Fraction: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def is_dimensionless(self) -> bool: ...
    def components(self) -> set[Literal['T', 'L', 'M', 'I', 'Theta', 'N', 'J']]: ...
    def is_geometric(self) -> bool: ...
    def is_kinematic(self) -> bool: ...
    def is_dynamic(self) -> bool: ...
    def __contains__(self, dim: Literal['T', 'L', 'M', 'I', 'Theta', 'N', 'J']) -> bool: ...
    def as_Gaussian(self) -> Self: ...
    @property
    def inv(self) -> Self: ...
    def __mul__(self, other: Dimension) -> Self: ...
    def __truediv__(self, other: Dimension) -> Self: ...
    def __imul__(self, other: Dimension) -> Self: ...
    def __itruediv__(self, other: Dimension) -> Self: ...
    def __rtruediv__(self, one: Literal[1]) -> Self: ...
    def __pow__(self, n: int | Fraction) -> Self: ...
    def __ipow__(self, n: int | Fraction) -> Self: ...
    def root(self, n: int | Fraction) -> Self: ...
    @staticmethod
    def product(dim_iter: Iterable[Dimension]) -> Dimension: ...
    def corresponding_quantity(self) -> tuple[str] | None: ...


DIMENSIONLESS: Dimension

