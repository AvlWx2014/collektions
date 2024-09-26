from __future__ import annotations

__all__ = [
    "filter_keys",
    "filter_values",
    "map_keys",
    "map_keys_to",
    "map_values",
    "map_values_to",
]

from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence
from typing import Callable

from ._types import K, R, V


def filter_keys(
    mapping: Mapping[K, V], predicate: Callable[[K, V], bool]
) -> Iterable[K]:
    """Filter ``mapping``'s key set based on ``predicate``.

    The predicate function takes both the key and value from each key/value pair
    so that filtering can be done on either.

    Returns:
        A collection of the key from each key/value pair for which predicate returns ``True``.
    """
    result: MutableSequence[K] = []
    for key, value in mapping.items():
        if predicate(key, value):
            result.append(key)
    return result


def filter_values(
    mapping: Mapping[K, V], predicate: Callable[[K, V], bool]
) -> Iterable[V]:
    """Filter ``mapping``'s value set based on ``predicate``.

    The predicate function takes both the key and value from each key/value pair
    so that filtering can be done on either.

    Returns:
        A collection of the value from each key/value pair for which predicate returns ``True``.
    """
    result: MutableSequence[V] = []
    for key, value in mapping.items():
        if predicate(key, value):
            result.append(value)
    return result


def map_keys(mapping: Mapping[K, V], transform: Callable[[K, V], R]) -> Mapping[R, V]:
    """Return a new mapping with a key set formed by applying ``transform`` to the original key set.

    The value set in the new mapping is the same as the value set in the original mapping.
    """
    return map_keys_to(mapping, {}, transform)


def map_keys_to(
    mapping: Mapping[K, V],
    destination: MutableMapping[R, V],
    transform: Callable[[K, V], R],
) -> MutableMapping[R, V]:
    """Update ``destination`` with ``mapping``.

    Entries from the original ``mapping`` have their keys transformed by applying ``transform``,
    and their values unchanged.
    """
    for key, value in mapping.items():
        new_key = transform(key, value)
        destination[new_key] = value
    return destination


def map_values(mapping: Mapping[K, V], transform: Callable[[K, V], R]) -> Mapping[K, R]:
    """
    Return a new mapping w/ a value set formed by applying ``transform`` to the original value set.

    The key set in the new mapping is the same as the key set in the original mapping.
    """
    return map_values_to(mapping, {}, transform)


def map_values_to(
    mapping: Mapping[K, V],
    destination: MutableMapping[K, R],
    transform: Callable[[K, V], R],
) -> MutableMapping[K, R]:
    """Update ``destination`` with ``mapping``.

    Entries from the original ``mapping`` have their keys unchanged, and their values transformed
    by applying ``transform``.
    """
    for key, value in mapping.items():
        destination[key] = transform(key, value)
    return destination
