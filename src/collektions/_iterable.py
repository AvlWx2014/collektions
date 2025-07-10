"""Functional-esque tools for Python inspired by the Kotlin Collections API."""

from __future__ import annotations

__all__ = [
    "associate",
    "associate_to",
    "associate_by",
    "associate_by_to",
    "associate_with",
    "associate_with_to",
    "average",
    "chunked",
    "distinct",
    "distinct_by",
    "drop",
    "drop_while",
    "filter_indexed",
    "filter_isinstance",
    "filter_not",
    "filter_not_none",
    "first",
    "first_not_none_of",
    "first_not_none_of_or_none",
    "first_or_none",
    "find",
    "flat_map",
    "flatten",
    "fold",
    "fold_indexed",
    "group_by",
    "group_by",
    "group_by",
    "group_by_to",
    "group_by_to",
    "group_by_to",
    "is_empty",
    "is_not_empty",
    "map_indexed",
    "map_not_none",
    "map_indexed_not_none",
    "max_by",
    "max_of",
    "min_by",
    "min_of",
    "none",
    "on_each",
    "on_each_indexed",
    "partition",
    "reduce",
    "reduce_indexed",
    "reduce_indexed_or_none",
    "reduce_or_none",
    "running_fold",
    "running_fold_indexed",
    "scan",
    "scan_indexed",
    "single",
    "single",
    "single",
    "single_or_none",
    "single_or_none",
    "single_or_none",
    "sum_of",
    "sum_of",
    "sum_of",
    "take",
    "take_while",
    "unzip",
    "windowed",
]


from collections.abc import (
    Collection,
    Generator,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import suppress
from numbers import Real
from typing import (
    Any,
    Callable,
    overload,
)

from ._defaults import default_predicate, default_predicate_with_index, identity
from ._types import C, K, R, T, V
from .preconditions import require


def associate(
    iterable: Iterable[T], transform: Callable[[T], tuple[K, V]]
) -> Mapping[K, V]:
    """Transform ``iterable`` in to a mapping of key, value pairs as returned by ``transform``."""
    return associate_to(iterable, transform, {})


def associate_to(
    iterable: Iterable[T],
    transform: Callable[[T], tuple[K, V]],
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
    iterable: Iterable[T], value_transform: Callable[[T], V]
) -> Mapping[T, V]:
    """Map items in ``iterable`` to values prescribed by ``value_transform``.

    Put another way, turn ``iterable`` in to a mapping of key, value pairs where the keys
    are the original items in ``iterable`` and the values are prescribed by ``value_transform``.

    The items in ``iterable`` must be hashable.
    """
    return associate_with_to(iterable, value_transform, {})


def associate_with_to(
    iterable: Iterable[T],
    value_transform: Callable[[T], V],
    destination: MutableMapping[T, V],
) -> Mapping[T, V]:
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


def chunked(iterable: Iterable[T], size: int = 1) -> Iterable[Sequence[T]]:
    return windowed(iterable, size=size, step=size, allow_partial=True)


def distinct(iterable: Iterable[T]) -> list[T]:
    """Return a collection of distinct items from ``iterable``."""
    return distinct_by(iterable, hash)


def distinct_by(iterable: Iterable[T], selector: Callable[[T], Hashable]) -> list[T]:
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


def drop(iterable: Iterable[T], number: int) -> Sequence[T]:
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
) -> list[T]:
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
) -> list[T]:
    """Filter ``iterable`` to items satisfying ``predicate``.

    ``predicate`` is called with both the item and its index in the original iterable.
    """
    return [item for i, item in enumerate(iterable) if predicate(i, item)]


def filter_isinstance(iterable: Iterable[Any], type_: type[R]) -> list[R]:
    """Filter ``iterable`` for instances of ``type_``."""
    return [item for item in iterable if isinstance(item, type_)]


def filter_not(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> list[T]:
    """Filter ``iterable`` to items that don't satisfy ``predicate``."""
    return [item for item in iterable if not predicate(item)]


def filter_not_none(iterable: Iterable[T | None]) -> list[T]:
    """Filter ``iterable`` to items that are not `None`."""
    return [item for item in iterable if item is not None]


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


def first_not_none_of(iterable: Iterable[T], transform: Callable[[T], R | None]) -> R:
    """Returns the first item of `iterable` that is not `None` after mapping with `transform`.

    Raises:
        ValueError: If `transform` maps all items in `iterable` to `None`.
    """
    if (result := first_not_none_of_or_none(iterable, transform)) is None:
        raise ValueError("All elements mapped to None by the given transform.")
    return result


def first_not_none_of_or_none(
    iterable: Iterable[T], transform: Callable[[T], R | None]
) -> R | None:
    """Returns the first item of `iterable` that is not `None` after mapping with `transform`.

    If `transform` maps all items in `iterable` to `None`, then `None` is returned.
    """
    for item in iterable:
        if (result := transform(item)) is not None:
            return result
    return None


def first_or_none(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> T | None:
    """
    Return the first item of ``collection`` matching ``predicate`` or `None` if no item matches.
    """
    try:
        result = first(iterable, predicate)
    except ValueError:
        result = None

    return result


find = first_or_none
"""An alias for first_or_none for situations where ``find`` makes more sense contextually."""


def flat_map(iterable: Iterable[T], transform: Callable[[T], Iterable[R]]) -> list[R]:
    """
    Return the collection of items yielded from calling ``transform`` on each item of ``iterable``.
    """
    result: list[R] = []
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


@overload
def group_by(
    iterable: Iterable[T],
    key_selector: Callable[[T], K],
) -> Mapping[K, list[T]]: ...


@overload
def group_by(
    iterable: Iterable[T],
    key_selector: Callable[[T], K],
    value_transform: Callable[[T], V],
) -> Mapping[K, list[V]]: ...


# Ignore: mypy assignment
# Reason: The identity function is type (T) -> T, which is a valid function type
#   for use where (T) -> V is expected it simply means that V == T.
def group_by(
    iterable: Iterable[T],
    key_selector: Callable[[T], K],
    value_transform: Callable[[T], V] = identity,  # type: ignore[assignment]
) -> Mapping[K, list[V]]:
    """Groups elements of the original ``iterable`` by the key returned by ``key_selector``.

    If ``value_transform`` is provided, then each element as transformed by ``value_transform``
    is grouped by the key returned by ``key_selector`` as applied to the original element.
    """
    return group_by_to(iterable, {}, key_selector, value_transform)


@overload
def group_by_to(
    iterable: Iterable[T],
    destination: MutableMapping[K, list[V]],
    key_selector: Callable[[T], K],
) -> Mapping[K, list[V]]: ...


@overload
def group_by_to(
    iterable: Iterable[T],
    destination: MutableMapping[K, list[V]],
    key_selector: Callable[[T], K],
    value_transform: Callable[[T], V],
) -> Mapping[K, list[V]]: ...


# Ignore: mypy assignment
# Reason: The identity function is type (T) -> T, which is a valid function type
#   for use where (T) -> V is expected it simply means that V == T.
def group_by_to(
    iterable: Iterable[T],
    destination: MutableMapping[K, list[V]],
    key_selector: Callable[[T], K],
    value_transform: Callable[[T], V] = identity,  # type: ignore[assignment]
) -> Mapping[K, list[V]]:
    """Groups elements of the original ``iterable`` by the key returned by ``key_selector``.

    The groupings are added to ``destination``, which is modified in-place.

    If ``value_transform`` is provided, then each element as transformed by ``value_transform``
    is grouped by the key returned by ``key_selector`` as applied to the original element.
    """
    for item in iterable:
        group = destination.setdefault(key_selector(item), [])
        group.append(value_transform(item))
    return destination


def is_empty(iterable: Iterable[T]) -> bool:
    """Return ``True`` if ``iterable`` is empty, ``False`` otherwise."""
    # fast-path: if we know iterable is a Collection then just return its
    # truthyness value
    if isinstance(iterable, Collection):
        return not bool(iterable)

    return first_or_none(iterable) is None


def is_not_empty(iterable: Iterable[T]) -> bool:
    """Return ``True`` if ``iterable`` is not empty, ``False`` otherwise."""
    return not is_empty(iterable)


def map_indexed(iterable: Iterable[T], mapping: Callable[[int, T], R]) -> list[R]:
    """Transform elements of ``iterable`` by applying ``mapping`` to each element and its index."""
    return [mapping(idx, item) for idx, item in enumerate(iterable)]


def map_not_none(iterable: Iterable[T], mapping: Callable[[T], R | None]) -> list[R]:
    """Transform items in ``iterable`` by applying ``mapping`` to each element.

    If ``mapping`` returns `None` for an element, then that element is filtered out of the
    result.
    """
    result = []
    for item in iterable:
        mapped: R | None = mapping(item)
        if mapped is not None:
            result.append(mapped)
    return result


def map_indexed_not_none(
    iterable: Iterable[T], mapping: Callable[[int, T], R | None]
) -> list[R]:
    """Transform items in ``iterable`` by applying ``mapping`` to each element and its index.

    If ``mapping`` returns `None` for an element, then that element is filtered out of the
    result.
    """
    result = []
    for idx, item in enumerate(iterable):
        mapped: R | None = mapping(idx, item)
        if mapped is not None:
            result.append(mapped)
    return result


def max_by(iterable: Iterable[T], selector: Callable[[T], C]) -> T:
    """Return the first element yielding the largest value of the given ``selector``.

    ``selector`` must return something that is comparable.

    Raises:
        StopIteration: if ``iterable`` is empty.
    """
    iterator = iter(iterable)
    max_ = next(iterator)
    max_value = selector(max_)
    for item in iterator:
        value = selector(item)
        if value > max_value:
            max_ = item
            max_value = value
    return max_


def max_of(iterable: Iterable[T], transform: Callable[[T], C]) -> C:
    """Return the maximum value resulting from applying ``transform`` to each item in ``iterable``.

    ``transform`` must return something that is comparable.

    Raises:
        StopIteration: if ``iterable`` is empty.
    """
    iterator = iter(iterable)
    max_ = transform(next(iterator))
    for item in iterator:
        value = transform(item)
        if value > max_:
            max_ = value
    return max_


def min_by(iterable: Iterable[T], selector: Callable[[T], C]) -> T:
    """Return the first element yielding the smallest value of the given ``selector``.

    ``selector`` must return something that is comparable.

    Raises:
        StopIteration: if ``iterable`` is empty.
    """
    iterator = iter(iterable)
    min_ = next(iterator)
    min_value = selector(min_)
    for item in iterator:
        value = selector(item)
        if value < min_value:
            min_ = item
            min_value = value
    return min_


def min_of(iterable: Iterable[T], transform: Callable[[T], C]) -> C:
    """Return the minimum value resulting from applying ``transform`` to each item in ``iterable``.

    ``transform`` must return something that is comparable.

    Raises:
        StopIteration: if ``iterable`` is empty.
    """
    iterator = iter(iterable)
    min_ = transform(next(iterator))
    for item in iterator:
        value = transform(item)
        if value < min_:
            min_ = value
    return min_


def none(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> bool:
    """Returns ``True`` if no item in iterable matches ``predicate`` and ``False`` otherwise."""
    return all(not predicate(item) for item in iterable)


def on_each(iterable: Iterable[T], action: Callable[[T], None]) -> Iterable[T]:
    """Invoke ``action`` on each element of ``iterable``.

    Returns:
        The original iterable. If the input iterable is a generator, the generator will have
        been exhausted when this function returns. If the input is a generator expression then
        it will also have been exhausted. In both cases it is probably not useful to retain a
        reference to the value returned by this function.
    """
    for item in iterable:
        action(item)
    return iterable


def on_each_indexed(
    iterable: Iterable[T], action: Callable[[int, T], None]
) -> Iterable[T]:
    """Invoke ``action`` on each element of ``iterable`` and its index.

    Returns:
        The original iterable. If the input iterable is a generator, the generator will have
        been exhausted when this function returns. If the input is a generator expression then
        it will also have been exhausted. In both cases it is probably not useful to retain a
        reference to the value returned by this function.
    """
    for idx, item in enumerate(iterable):
        action(idx, item)
    return iterable


def partition(
    iterable: Iterable[T], predicate: Callable[[T], bool]
) -> tuple[list[T], list[T]]:
    """Partition ``iterable`` in to two disjoint collections using ``predicate``.

    Returns:
        A `left` and `right` collection. Items in the `left` collection yielded a value of `True`
            for the given function, while items in the `right` collection yielded a value of
            `False`.
    """
    left: list[T] = []
    right: list[T] = []
    for item in iterable:
        dest = left if predicate(item) else right
        dest.append(item)
    return left, right


def reduce(iterable: Iterable[T], accumulator: Callable[[T, T], T]) -> T:
    """Accumulates value from left to right starting with the first element.

    ``accumulator`` takes the current accumulated value and the next value in ``iterable``.

    Raises:
        StopIteration: if ``iterable`` is empty.
    """
    iterator = iter(iterable)
    acc = next(iterator)
    for item in iterator:
        acc = accumulator(acc, item)
    return acc


def reduce_indexed(iterable: Iterable[T], accumulator: Callable[[int, T, T], T]) -> T:
    """Accumulates value from left to right starting with the first element.

    ``accumulator`` takes the accumulated value so far, as well as the next element in ``iterable``
    and its index.

    Raises:
        StopIteration: if ``iterable`` is empty.
    """
    iterator = iter(iterable)
    acc = next(iterator)
    for idx, item in enumerate(iterator, start=1):
        acc = accumulator(idx, acc, item)
    return acc


def reduce_indexed_or_none(
    iterable: Iterable[T], accumulator: Callable[[int, T, T], T]
) -> T | None:
    """Accumulates value from left to right starting with the first element.

    ``accumulator`` takes the accumulated value so far, as well as the next element in ``iterable``
    and its index.

    This function returns `None` if ``iterable`` is empty rather than raising.
    """
    try:
        return reduce_indexed(iterable, accumulator)
    except StopIteration:
        return None


def reduce_or_none(iterable: Iterable[T], accumulator: Callable[[T, T], T]) -> T | None:
    """Accumulates value from left to right starting with the first element.

    ``accumulator`` takes the accumulated value so far, as well as the next element in ``iterable``.

    This function returns `None` if ``iterable`` is empty rather than raising.
    """
    try:
        return reduce(iterable, accumulator)
    except StopIteration:
        return None


def running_fold(
    iterable: Iterable[T], initial: R, operation: Callable[[R, T], R]
) -> list[R]:
    """Return a running list of accumulated values applying ``operation`` from left to right.

    The resulting list starts with ``initial`` and ``operation`` is called with the previously
    accumulated value and the next value in ``iterable``.

    For the first item in ``iterable``, ``operation`` is called with ``initial`` and the first
    value.

    Warning:
        The accumulated value should not be mutated otherwise it can affect the preceding values.
    """
    result = [initial]
    acc = initial
    for item in iterable:
        acc = operation(acc, item)
        result.append(acc)
    return result


def running_fold_indexed(
    iterable: Iterable[T], initial: R, operation: Callable[[int, R, T], R]
) -> list[R]:
    """Return a running list of accumulated values applying ``operation`` from left to right.

    The resulting list starts with ``initial`` and ``operation`` is called with the previously
    accumulated value and the next value in ``iterable`` along with its index.

    For the first item in ``iterable``, ``operation`` is called with ``initial`` and the first
    value.

    Warning:
        The accumulated value should not be mutated otherwise it can affect the preceding values.
    """
    result = [initial]
    acc = initial
    for idx, item in enumerate(iterable):
        acc = operation(idx, acc, item)
        result.append(acc)
    return result


scan = running_fold
scan_indexed = running_fold_indexed


@overload
def single(iterable: Iterable[T]) -> T: ...


@overload
def single(iterable: Iterable[T], predicate: Callable[[T], bool]) -> T: ...


def single(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> T:
    """Return the single item in ``iterable``.

    If ``predicate`` is provided, then this returns the single item matching ``predicate``.

    Raises:
        StopIteration: if ``iterable`` is empty.
        ValueError: if ``iterable`` contains more than one item.
    """
    candidate: T | None = None
    for item in iterable:
        if predicate(item):
            if candidate is not None:
                raise ValueError("More than one value found")
            candidate = item
    if candidate is None:
        raise ValueError("No values found")
    return candidate


@overload
def single_or_none(iterable: Iterable[T]) -> T | None: ...


@overload
def single_or_none(
    iterable: Iterable[T], predicate: Callable[[T], bool]
) -> T | None: ...


def single_or_none(
    iterable: Iterable[T], predicate: Callable[[T], bool] = default_predicate
) -> T | None:
    """Return the single item in ``iterable``.

    If ``predicate`` is provided, then this returns the single item matching ``predicate``.

    This function returns `None` if ``iterable`` is empty or has more than one matching
    value instead of raising.
    """
    with suppress(ValueError):
        return single(iterable, predicate)
    return None


@overload
def sum_of(iterable: Iterable[T], selector: Callable[[T], int]) -> int: ...


@overload
def sum_of(iterable: Iterable[T], selector: Callable[[T], float]) -> float: ...


# Type hints intentionally left out of the signature since @overload is being used
def sum_of(iterable, selector):
    """Accumulate a sum of each item in ``iterable`` given the value of ``selector`` for that item.

    If ``selector`` returns an :py:obj:`int` for each value, then the return value
    will be an :py:obj:`int`. Otherwise, if ``selector`` returns a :py:obj:`float` then
    the return value will be a :py:obj:`float`.
    """
    sum_ = 0
    for item in iterable:
        sum_ += selector(item)
    return sum_


def take(iterable: Iterable[T], n: int) -> Generator[T, None, None]:
    """Yield the first ``n`` items of ``iterable``."""
    require(n >= 0, "n cannot be negative")
    for idx, item in enumerate(iterable):
        if idx >= n:
            break
        yield item


def take_while(
    iterable: Iterable[T], predicate: Callable[[T], bool]
) -> Generator[T, None, None]:
    """Consume items from ``iterable`` while they continue matching ``predicate``."""
    iterator = iter(iterable)
    for item in iterator:
        if not predicate(item):
            break
        yield item


def unzip(iterable: Iterable[tuple[T, R]]) -> tuple[list[T], list[R]]:
    """Decompose a zip'd ``iterable`` in to its constituent left and right-hand iterables.

    This function performs the inverse operation that :py:func:`zip` does.
    """
    left = []
    right = []
    for first_, second in iterable:
        left.append(first_)
        right.append(second)
    return left, right


def windowed(
    iterable: Iterable[T], size: int = 1, step: int = 1, allow_partial: bool = False
) -> Iterable[Sequence[T]]:
    """Return a sliding window view of ``iterable`` with window size ``size``.

    If the length of ``iterable`` is not evenly divisible by ``size`` and ``allow_partial``
    is `True`, then the remaining len(iterable) - len(iterable) // size items will be returned
    in a partial window. If ``allow_partial`` is `False`, then only full-size windows
    will be returned.

    The window moves ``step`` steps each time.
    """
    # TODO: add windowed_iterator function that does not require casting the whole
    #  iterable to a list first
    sequence = list(iterable) if not isinstance(iterable, Sequence) else iterable
    return _windowed_iterator_sliced(sequence, size, step, allow_partial)


def _windowed_iterator_sliced(
    sequence: Sequence[T], size: int, step: int, allow_partial: bool
) -> Iterable[Sequence[T]]:
    left = 0
    while left < len(sequence):
        right = left + size
        window = sequence[left:right]
        if len(window) < size and not allow_partial:
            break
        yield window
        left = min(left + step, len(sequence))
