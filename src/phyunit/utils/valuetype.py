from abc import abstractmethod
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar('T', covariant=True)


@runtime_checkable
class ValueType(Protocol[T]):
    @abstractmethod
    def __mul__(self, other: float) -> T: ...
    @abstractmethod
    def __pow__(self, other: float) -> T: ...
    

