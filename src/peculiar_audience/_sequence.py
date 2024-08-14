from __future__ import annotations

__all__ = ["drop_last", "drop_last_while", "last", "last_or_none"]

from typing import Callable, Sequence

from ._defaults import default_predicate
from ._types import T
from .preconditions import require


def drop_last(sequence: Sequence[T], number: int) -> Sequence[T]:
    """Drop the last ``number`` items from ``iterable``."""
    require(number >= 0, message="Number of elements to drop must be non-negative.")
    return sequence[:-number]


def drop_last_while(
    sequence: Sequence[T], predicate: Callable[[T], bool] = default_predicate
) -> Sequence[T]:
    """Drop items from ``sequence`` matching ``predicate`` from right to left.

    In other words, all items at the end of ``sequence`` matching ``predicate``
    are dropped.
    """
    result = []
    iterator = reversed(sequence)
    for item in iterator:
        if not predicate(item):
            result.append(item)
            break
    result.extend(iterator)
    # items were iterated over right -> left, but inserted in to
    # result left -> right so reverse the result set to retain the
    # original ordering
    return result[::-1]


def last(
    sequence: Sequence[T], predicate: Callable[[T], bool] = default_predicate
) -> T:
    """Return the last item of ``sequence`` matching ``predicate`` or raise if no item matches.

    Raises:
        ValueError: If no item matches ``predicate``.
    """
    for item in sequence[::-1]:
        if predicate(item):
            return item
    raise ValueError("No item found matching predicate.")


def last_or_none(
    sequence: Sequence[T], predicate: Callable[[T], bool] = default_predicate
) -> T | None:
    """Return the last item of ``sequence`` matching ``predicate`` or `None` if no item matches.

    Returns:
        The last item from ``sequence`` that matches ``predicate`` or `None` if no item matches.
    """
    try:
        result = last(sequence, predicate)
    except ValueError:
        result = None

    return result
