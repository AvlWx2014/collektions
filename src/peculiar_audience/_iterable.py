"""Functional-esque tools for Python inspired by the Kotlin Collections API."""

from __future__ import annotations

__all__ = [
    "associate",
    "associate_by",
    "associate_by_to",
    "associate_to",
    "associate_with",
    "associate_with_to",
    "average",
    "chunked",
    "distinct",
    "distinct_by",
    "drop",
    "drop_while",
    "first",
    "first_or_none",
    "flat_map",
    "flatten",
    "fold",
    "fold_indexed",
    "map_not_none",
    "none",
    "sum_by",
    "windowed",
]

from collections.abc import MutableSequence
from numbers import Real
from typing import (
    Any,
    Callable,
    Collection,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    overload,
)

from ._defaults import default_predicate, default_predicate_with_index
from ._types import H, K, R, T, V
from .preconditions import require


def associate(
    iterable: Iterable[T], transform: Callable[[T], Tuple[K, V]]
) -> Mapping[K, V]:
    """Transform ``iterable`` in to a mapping of key, value pairs as returned by ``transform``."""
    return associate_to(iterable, transform, {})


def associate_to(
    iterable: Iterable[T],
    transform: Callable[[T], Tuple[K, V]],
    destination: MutableMapping[K, V],
) -> Mapping[K, V]:
    """Update ``destination`` with new entries from ``iterable`` transformed by ``transform``."""
    for item in iterable:
        key, value = transform(item)
        destination[key] = value
    return destination


def associate_by(
    iterable: Iterable[T], key_transform: Callable[[T], K]
) -> Mapping[K, T]:
    """Map items in ``iterable`` by keys prescribed by ``key_transform``.

    Put another way, turn ``iterable`` in to a mapping of key, value pairs where the keys
    are prescribed by ``key_transform`` and the values are the original items in ``iterable``.
    """
    return associate_by_to(iterable, key_transform, {})


def associate_by_to(
    iterable: Iterable[T],
    key_transform: Callable[[T], K],
    destination: MutableMapping[K, T],
) -> Mapping[K, T]:
    """Update ``destination`` with new entries from ``iterable``.

    The keys in the new entries are prescribed by ``key_transform``, and the values are the
    original items in ``iterable``.
    """
    for item in iterable:
        key = key_transform(item)
        destination[key] = item
    return destination


def associate_with(
    iterable: Iterable[H], value_transform: Callable[[H], V]
) -> Mapping[H, V]:
    """Map items in ``iterable`` to values prescribed by ``value_transform``.

    Put another way, turn ``iterable`` in to a mapping of key, value pairs where the keys
    are the original items in ``iterable`` and the values are prescribed by ``value_transform``.

    The items in ``iterable`` must be hashable.
    """
    return associate_with_to(iterable, value_transform, {})


def associate_with_to(
    iterable: Iterable[H],
    value_transform: Callable[[H], V],
    destination: MutableMapping[H, V],
) -> Mapping[H, V]:
    """Update ``destination`` with new entries from ``iterable``.

    The keys in the new entries are the original items in ``iterable`` and the values
    are prescribed by ``value_transform``.
    """
    for item in iterable:
        value = value_transform(item)
        destination[item] = value
    return destination


def average(iterable: Iterable[Real]) -> float:
    """Return the average (mean) of the values in ``iterable``.

    If ``iterable`` has no items, then this function returns ``float("NaN")``,
    which can be checked by ``math.isnan(average(...))``.

    All items in ``iterable`` must be real numbers.
    """

    sum_ = 0
    count = 0
    for number in iterable:
        sum_ += number
        count += 1
    return sum_ / count if count else float("NaN")


def chunked(iterable: Iterable[T], size: int = 1) -> Iterable[Collection[T]]:
    return windowed(iterable, size=size, step=size, allow_partial=True)


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


def drop(iterable: Iterable[T], number: int) -> Collection[T]:
    """Drop the first ``number`` items from ``iterable``."""
    require(number >= 0, message="Number of elements to drop must be non-negative.")
    if isinstance(iterable, Sequence):
        # fast path for Sequences - just slice it
        # note: this includes ranges
        return iterable[number:]

    iterator = iter(iterable)
    while number:
        try:
            next(iterator)
            number -= 1
        except StopIteration:
            # if we exhaust the iterator before reaching ``number`` items, then
            # we break and return an empty list
            break
    return list(iterator)


def drop_while(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> Collection[T]:
    """Drop the first items from ``iterable`` that matching ``predicate``."""
    result = []
    iterator = iter(iterable)
    for item in iterator:
        if not predicate(item):
            result.append(item)
            break
    result.extend(iterator)
    return result


def filter_indexed(
    iterable: Iterable[T],
    predicate: Callable[[int, T], bool] = default_predicate_with_index,
) -> Collection[T]:
    return [item for i, item in enumerate(iterable) if predicate(i, item)]


def filter_isinstance(iterable: Iterable[Any], type_: type[R]) -> Collection[R]:
    return [item for item in iterable if isinstance(item, type_)]


def filter_not(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> Collection[T]:
    return [item for item in iterable if not predicate(item)]


def filter_not_none(iterable: Iterable[T | None]) -> Collection[T]:
    return [item for item in iterable if item is not None]


def find(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> T | None:
    return first_or_none(iterable, predicate)


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


def flat_map(
    iterable: Iterable[T], transform: Callable[[T], Iterable[R]]
) -> Collection[R]:
    """
    Return the collection of items yielded from calling ``transform`` on each item of ``iterable``.
    """
    result = []
    for item in iterable:
        result.extend(transform(item))
    return result


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


def fold_indexed(
    iterable: Iterable[T], initial_value: R, accumulator: Callable[[int, R, T], R]
) -> R:
    """Accumulates value starting from ``initial_value``.

    Accumulation starts from ``initial_value`` and applies ``accumulator`` from left
    to right across ``iterable`` passing the current accumulated value, the current index,
    and the curren item at that index.
    """
    acc = initial_value
    for idx, item in enumerate(iterable):
        acc = accumulator(idx, acc, item)
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


def windowed(
    iterable: Iterable[T], size: int = 1, step: int = 1, allow_partial: bool = False
) -> Iterable[Collection[T]]:
    # TODO: add windowed_iterator function that does not require casting the whole
    #  iterable to a list first
    sequence = list(iterable) if not isinstance(iterable, Sequence) else iterable
    return _windowed_iterator_sliced(sequence, size, step, allow_partial)


def _windowed_iterator_sliced(
    sequence: Sequence[T], size: int, step: int, allow_partial: bool
) -> Iterable[Collection[T]]:
    left = 0
    while left < len(sequence):
        right = left + size
        window = sequence[left:right]
        if len(window) < size and not allow_partial:
            break
        yield window
        left = min(left + step, len(sequence))
