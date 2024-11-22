from __future__ import annotations

__all__ = [
    "drop_last",
    "drop_last_while",
    "find_last",
    "fold_right",
    "fold_right_indexed",
    "last",
    "last_or_none",
    "take_last",
    "take_last_while",
]

from collections.abc import Sequence
from typing import Callable

from ._defaults import default_predicate
from ._types import R, T
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


def find_last(
    sequence: Sequence[T], predicate: Callable[[T], bool] = default_predicate
) -> T | None:
    return last_or_none(sequence, predicate)


def fold_right(
    sequence: Sequence[T], initial_value: R, accumulator: Callable[[T, R], R]
) -> R:
    """Accumulates value starting from ``initial_value``.

    Accumulation starts from ``initial_value`` and applies ``accumulator`` from right
    to left across ``iterable`` passing the current accumulated value with each item.
    """
    acc = initial_value
    for item in reversed(sequence):
        acc = accumulator(item, acc)
    return acc


def fold_right_indexed(
    sequence: Sequence[T], initial_value: R, accumulator: Callable[[int, T, R], R]
) -> R:
    """Accumulates value starting from ``initial_value``.

    Accumulation starts from ``initial_value`` and applies ``accumulator`` from right
    to left across ``iterable`` passing the current accumulated value, the current index,
    and the curren item at that index.
    """
    n = len(sequence)
    acc = initial_value
    for idx, item in enumerate(reversed(sequence)):
        acc = accumulator(n - idx - 1, item, acc)
    return acc


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


def take_last(sequence: Sequence[T], n: int) -> Sequence[T]:
    """Return the last ``n`` items from ``sequence``.

    In almost all cases, the input sequence type is preserved in the return value.

    In other words, if a string is the input sequence, a string will be the output. If a
    range is the input sequence, then a range will be the output, etc.
    """
    require(n >= 0, "n cannot be negative")
    start = max(len(sequence) - n, 0)
    return sequence[start:]


def take_last_while(sequence: Sequence[T], predicate: Callable[[T], bool]) -> list[T]:
    """Return the right-most items in ``sequence`` satisfying ``predicate``.

    In other words, all items in ``sequence`` working backwards from the end satisfying
    ``predicate`` are returned. As soon as the first item not matching predicate is found
    iteration stops and no more items to the left of the first non-matching item are evaluated.
    """
    result: list[T] = []
    for item in reversed(sequence):
        if not predicate(item):
            break
        result.insert(0, item)
    return result
