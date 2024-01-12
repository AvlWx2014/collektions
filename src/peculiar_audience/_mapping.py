from __future__ import annotations

__all__ = ["filter_keys", "filter_values"]

from typing import Callable, Iterable, Mapping, MutableSequence

from ._types import K, V


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
