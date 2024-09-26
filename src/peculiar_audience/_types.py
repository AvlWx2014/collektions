from typing import Any, Hashable, Protocol, TypeVar

T = TypeVar("T")
"""Typically used with iterables as the Type for each item in the iterable."""
R = TypeVar("R")
"""Typically used as a Return or Result type usually from a transformation of some kind."""

K = TypeVar("K")
"""Typically used with mappings as the Key type."""
V = TypeVar("V")
"""Typically used with mappings as the Value type."""

H = TypeVar("H", bound=Hashable)
"""Typically used as a type for items that must be Hashable."""


class Comparable(Protocol):
    def __eq__(self, other: Any) -> bool:
        ...

    def __ne__(self, other: Any) -> bool:
        ...

    def __lt__(self, other: Any) -> bool:
        ...

    def __gt__(self, other: Any) -> bool:
        ...

    def __le__(self, other: Any) -> bool:
        ...

    def __ge__(self, other: Any) -> bool:
        ...


C = TypeVar("C", bound=Comparable)
