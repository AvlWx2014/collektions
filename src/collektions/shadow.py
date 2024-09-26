"""Override some built-in functions in order to provide a consistent interface.

In other words, these functions hide the built-in functions of the same name so
that calling code can have a consistent interface across higher-order function
use in their code.

For example, rather than calling code having

    head = first_or_none(iterable, predicate)
    tail = last_or_none(iterable, predicate)
    flat = flat_map(iterable1, iterable2, ..., mapping)
    sum_ = sum_by(iterable, selector)

mixed in with built-ins of a different interface like

    transformed = map(mapping, iterable)
    at_least_one = any(iterable_of_bools)
    all_of_them = all(iterable_of_bools)

shadows like ``map``, ``any``, and ``all`` can be imported explicitly from this
module to instead have consistent interfaces across their higher-order collections
functions:

    head = first_or_none(iterable, predicate)
    tail = last_or_none(iterable, predicate)
    flat = flat_map(iterable1, iterable2, ..., mapping)
    sum_ = sum_by(iterable, selector)
    transformed = map(iterable, mapping)
    at_least_one = any(iterable, predicate)
    all_of_them = all(iterable, predicate)
"""

from collections.abc import Iterable
from typing import Callable

from ._types import R, T

__pyall = all
__pyany = any
__pyfilter = filter
__pymap = map


__all__ = ["all", "any", "filter", "map"]


def all(iterable: Iterable[T], predicate: Callable[[T], bool]) -> bool:
    """Overrides built-in :py:func:`all` to provide an interface consistent with this library."""
    return __pyall(predicate(item) for item in iterable)


def any(iterable: Iterable[T], predicate: Callable[[T], bool]) -> bool:
    """Overrides built-in :py:func:`any` to provide an interface consistent with this library."""
    return __pyany(predicate(item) for item in iterable)


def filter(iterable: Iterable[T], predicate: Callable[[T], bool]) -> Iterable[T]:
    """Overrides built-in :py:func:`filter` to provide an interface consistent with this library."""
    return __pyfilter(predicate, iterable)


def map(iterable: Iterable[T], mapping: Callable[[T], R]) -> Iterable[R]:
    """Overrides built-in :py:func:`map` to provide an interface consistent with this library."""
    return __pymap(mapping, iterable)
