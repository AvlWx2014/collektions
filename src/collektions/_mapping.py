from __future__ import annotations

__all__ = [
    "filter_keys",
    "filter_values",
    "map_keys",
    "map_keys_to",
    "map_values",
    "map_values_to",
]

from collections.abc import Mapping, MutableMapping
from typing import Callable

from ._types import K, R, V


def filter_keys(
    mapping: Mapping[K, V], predicate: Callable[[K], bool]
) -> Mapping[K, V]:
    """
    Return a new mapping of all key/value pairs from ``mapping`` where `key` satisfies ``predicate``.
    """
    return {k: v for k, v in mapping.items() if predicate(k)}


def filter_values(
    mapping: Mapping[K, V], predicate: Callable[[V], bool]
) -> Mapping[K, V]:
    """
    Return a new mapping of all key/value pairs from ``mapping`` where `value` satisfies ``predicate``.
    """
    return {k: v for k, v in mapping.items() if predicate(v)}


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
