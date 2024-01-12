"""Functional-esque tools for Python inspired by the Kotlin Collections API."""

from __future__ import annotations

__all__ = [
    "distinct",
    "distinct_by",
    "first",
    "first_or_none",
    "flat_map",
    "flatten",
    "fold",
    "map_not_none",
    "none",
    "sum_by",
]

from typing import Callable, Collection, Hashable, Iterable, Optional, overload

from ._defaults import default_predicate
from ._types import R, T


def distinct(iterable: Iterable[T]) -> Collection[T]:
    """Return a collection of distinct items from ``iterable``."""
    return distinct_by(iterable, hash)


def distinct_by(
    iterable: Iterable[T], selector: Callable[[T], Hashable]
) -> Collection[T]:
    """Return a collection of distinct items from ``iterable`` using ``selector`` as the key.

    If two items in ``iterable`` map to the same value from ``selector``, the first one
    in the iteration order of ``iterable`` wins.

    ``selector`` must return a hashable value.
    """
    unique = {}
    for item in iterable:
        key = selector(item)
        if key not in unique:
            unique[key] = item
    return list(unique.values())


def first(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> T:
    """Return the first item of ``collection`` matching ``predicate`` or raise if no item matches.

    Raises:
        ValueError: If no item matches ``predicate``.
    """
    for item in iterable:
        if predicate(item):
            return item
    raise ValueError("No item found matching predicate.")


def first_or_none(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> Optional[T]:
    """
    Return the first item of ``collection`` matching ``predicate`` or `None` if no item matches.
    """
    try:
        result = first(iterable, predicate)
    except ValueError:
        result = None

    return result


def flat_map(*iterables: Iterable[T], mapping: Callable[[T], R]) -> Iterable[R]:
    """Flatten ``iterables`` in to a single list comprising transformed items from each iterable.

    Items from each iterable in ``iterables`` are transformed according to ``mapping``.
    """
    # use built-in `map` to avoid allocating a list for each `map` operation
    return (item for iterable in iterables for item in map(mapping, iterable))


def flatten(*iterables: Iterable[T]) -> Iterable[T]:
    """Flatten ``iterables`` in to a single list comprising all items from all iterables."""
    return (item for iterable in iterables for item in iterable)


def fold(
    iterable: Iterable[T], initial_value: R, accumulator: Callable[[R, T], R]
) -> R:
    """Accumulates value starting from ``initial_value``.

    Accumulation starts from ``initial_value`` and applies ``accumulator`` from left
    to right across ``iterable`` passing the current accumulated value with each item.
    """
    acc = initial_value
    for item in iterable:
        acc = accumulator(acc, item)
    return acc


def map_not_none(
    iterable: Iterable[T | None], mapping: Callable[[T | None], R | None]
) -> Iterable[R]:
    """Map items in ``iterable`` according to ``mapping``, filtering out any ``None`` values."""
    result = []
    for item in iterable:
        mapped: R | None = mapping(item)
        if mapped is not None:
            result.append(mapped)
    return result


def none(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> bool:
    """Returns ``True`` if no item in iterable matches ``predicate`` and ``False`` otherwise."""
    for item in iterable:
        if predicate(item):
            return False
    return True


@overload
def sum_by(iterable: Iterable[T], selector: Callable[[T], int]) -> int:
    ...


@overload
def sum_by(iterable: Iterable[T], selector: Callable[[T], float]) -> float:
    ...


# Type hints intentionally left out of the signature since @overload is being used
def sum_by(iterable, selector):
    """Accumulate a sum of each item in ``iterable``.

    ``selector`` maps each item in ``iterable`` to a numeric value to add.

    If ``selector`` returns an :py:obj:`int` for each value, then the return value
    will be an :py:obj:`int`. Otherwise, if ``selector`` returns a :py:obj:`float` then
    the return value will be a :py:obj:`float`.
    """
    sum_ = 0
    for item in iterable:
        sum_ += selector(item)
    return sum_
