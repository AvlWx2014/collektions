from typing import Hashable, TypeVar

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
