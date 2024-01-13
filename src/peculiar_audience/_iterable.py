"""Functional-esque tools for Python inspired by the Kotlin Collections API."""

from __future__ import annotations

__all__ = [
    "associate",
    "associate_by",
    "associate_by_to",
    "associate_to",
    "associate_with",
    "associate_with_to",
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

from typing import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    overload,
)

from ._defaults import default_predicate
from ._types import H, K, R, T, V


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
