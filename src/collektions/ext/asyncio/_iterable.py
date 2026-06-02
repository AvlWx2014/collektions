"""Async functional-esque tools for Python inspired by the Kotlin Collections API."""

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
    "filter_indexed",
    "filter_isinstance",
    "filter_not",
    "filter_not_none",
    "find",
    "first",
    "first_not_none_of",
    "first_not_none_of_or_none",
    "first_or_none",
    "flat_map",
    "flatten",
    "fold",
    "fold_indexed",
    "group_by",
    "group_by_to",
    "is_empty",
    "is_not_empty",
    "map_indexed",
    "map_indexed_not_none",
    "map_not_none",
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
    "single_or_none",
    "sum_of",
    "take",
    "take_while",
    "unzip",
    "windowed",
]

import inspect
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import suppress
from typing import (
    Any,
    overload,
)

from collektions._defaults import (
    default_predicate,
    default_predicate_with_index,
    identity,
)
from collektions._types import C, K, R, T, V
from collektions.preconditions import require


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value  # type: ignore[misc]
    return value  # type: ignore[return-value]


async def associate(
    iterable: AsyncIterable[T],
    transform: Callable[[T], tuple[K, V] | Awaitable[tuple[K, V]]],
) -> Mapping[K, V]:
    return await associate_to(iterable, transform, {})


async def associate_to(
    iterable: AsyncIterable[T],
    transform: Callable[[T], tuple[K, V] | Awaitable[tuple[K, V]]],
    destination: MutableMapping[K, V],
) -> Mapping[K, V]:
    async for item in iterable:
        key, value = await _maybe_await(transform(item))
        destination[key] = value
    return destination


async def associate_by(
    iterable: AsyncIterable[T], key_transform: Callable[[T], K | Awaitable[K]]
) -> Mapping[K, T]:
    return await associate_by_to(iterable, key_transform, {})


async def associate_by_to(
    iterable: AsyncIterable[T],
    key_transform: Callable[[T], K | Awaitable[K]],
    destination: MutableMapping[K, T],
) -> Mapping[K, T]:
    async for item in iterable:
        key = await _maybe_await(key_transform(item))
        destination[key] = item
    return destination


async def associate_with(
    iterable: AsyncIterable[T], value_transform: Callable[[T], V | Awaitable[V]]
) -> Mapping[T, V]:
    return await associate_with_to(iterable, value_transform, {})


async def associate_with_to(
    iterable: AsyncIterable[T],
    value_transform: Callable[[T], V | Awaitable[V]],
    destination: MutableMapping[T, V],
) -> Mapping[T, V]:
    async for item in iterable:
        value = await _maybe_await(value_transform(item))
        destination[item] = value
    return destination


async def average(iterable: AsyncIterable[float | int]) -> float:
    sum_ = 0
    count: int = 0
    async for number in iterable:
        sum_ += number  # type: ignore[assignment]
        count += 1
    return sum_ / count if count else float("NaN")


async def chunked(
    iterable: AsyncIterable[T], size: int = 1
) -> AsyncIterator[Sequence[T]]:
    async for window in windowed(iterable, size=size, step=size, allow_partial=True):
        yield window


async def distinct(iterable: AsyncIterable[T]) -> list[T]:
    return await distinct_by(iterable, hash)


async def distinct_by(
    iterable: AsyncIterable[T], selector: Callable[[T], Hashable | Awaitable[Hashable]]
) -> list[T]:
    unique: dict[Hashable, T] = {}
    async for item in iterable:
        key = await _maybe_await(selector(item))
        if key not in unique:
            unique[key] = item
    return list(unique.values())


async def drop(iterable: AsyncIterable[T], number: int) -> list[T]:
    require(number >= 0, message="Number of elements to drop must be non-negative.")
    iterator = aiter(iterable)
    count = number
    while count:
        try:
            await anext(iterator)
            count -= 1
        except StopAsyncIteration:
            break
    return [item async for item in iterator]


async def drop_while(
    iterable: AsyncIterable[T],
    predicate: Callable[[T], bool | Awaitable[bool]] = default_predicate,
) -> list[T]:
    result: list[T] = []
    iterator = aiter(iterable)
    async for item in iterator:
        if not await _maybe_await(predicate(item)):
            result.append(item)
            break
    async for item in iterator:
        result.append(item)
    return result


async def filter_indexed(
    iterable: AsyncIterable[T],
    predicate: Callable[
        [int, T], bool | Awaitable[bool]
    ] = default_predicate_with_index,
) -> list[T]:
    result: list[T] = []
    i = 0
    async for item in iterable:
        if await _maybe_await(predicate(i, item)):
            result.append(item)
        i += 1
    return result


async def filter_isinstance(iterable: AsyncIterable[Any], type_: type[R]) -> list[R]:
    return [item async for item in iterable if isinstance(item, type_)]


async def filter_not(
    iterable: AsyncIterable[T],
    predicate: Callable[[T], bool | Awaitable[bool]] = default_predicate,
) -> list[T]:
    result: list[T] = []
    async for item in iterable:
        if not await _maybe_await(predicate(item)):
            result.append(item)
    return result


async def filter_not_none(iterable: AsyncIterable[T | None]) -> list[T]:
    return [item async for item in iterable if item is not None]


async def first(
    iterable: AsyncIterable[T],
    predicate: Callable[[T], bool | Awaitable[bool]] = default_predicate,
) -> T:
    async for item in iterable:
        if await _maybe_await(predicate(item)):
            return item
    raise ValueError("No item found matching predicate.")


async def first_not_none_of(
    iterable: AsyncIterable[T], transform: Callable[[T], R | None | Awaitable[R | None]]
) -> R:
    if (result := await first_not_none_of_or_none(iterable, transform)) is None:
        raise ValueError("All elements mapped to None by the given transform.")
    return result


async def first_not_none_of_or_none(
    iterable: AsyncIterable[T], transform: Callable[[T], R | None | Awaitable[R | None]]
) -> R | None:
    async for item in iterable:
        if (result := await _maybe_await(transform(item))) is not None:
            return result
    return None


async def first_or_none(
    iterable: AsyncIterable[T],
    predicate: Callable[[T], bool | Awaitable[bool]] = default_predicate,
) -> T | None:
    try:
        return await first(iterable, predicate)
    except ValueError:
        return None


find = first_or_none


async def flat_map(
    iterable: AsyncIterable[T],
    transform: Callable[
        [T], Iterable[R] | AsyncIterable[R] | Awaitable[Iterable[R] | AsyncIterable[R]]
    ],
) -> list[R]:
    result: list[R] = []
    async for item in iterable:
        inner = await _maybe_await(transform(item))
        if isinstance(inner, AsyncIterable):
            async for sub in inner:
                result.append(sub)
        else:
            result.extend(inner)  # type: ignore[arg-type]
    return result


async def flatten(*iterables: AsyncIterable[T]) -> AsyncIterator[T]:
    for iterable in iterables:
        async for item in iterable:
            yield item


async def fold(
    iterable: AsyncIterable[T],
    initial_value: R,
    accumulator: Callable[[R, T], R | Awaitable[R]],
) -> R:
    acc = initial_value
    async for item in iterable:
        acc = await _maybe_await(accumulator(acc, item))
    return acc


async def fold_indexed(
    iterable: AsyncIterable[T],
    initial_value: R,
    accumulator: Callable[[int, R, T], R | Awaitable[R]],
) -> R:
    acc = initial_value
    idx = 0
    async for item in iterable:
        acc = await _maybe_await(accumulator(idx, acc, item))
        idx += 1
    return acc


@overload
async def group_by(
    iterable: AsyncIterable[T],
    key_selector: Callable[[T], K | Awaitable[K]],
) -> Mapping[K, list[T]]: ...


@overload
async def group_by(
    iterable: AsyncIterable[T],
    key_selector: Callable[[T], K | Awaitable[K]],
    value_transform: Callable[[T], V | Awaitable[V]],
) -> Mapping[K, list[V]]: ...


# Ignore: mypy assignment
# Reason: The identity function is type (T) -> T, which is a valid function type
#   for use where (T) -> V is expected it simply means that V == T.
async def group_by(
    iterable: AsyncIterable[T],
    key_selector: Callable[[T], K | Awaitable[K]],
    value_transform: Callable[[T], V | Awaitable[V]] = identity,  # type: ignore[assignment]
) -> Mapping[K, list[V]]:
    return await group_by_to(iterable, {}, key_selector, value_transform)


@overload
async def group_by_to(
    iterable: AsyncIterable[T],
    destination: MutableMapping[K, list[V]],
    key_selector: Callable[[T], K | Awaitable[K]],
) -> Mapping[K, list[V]]: ...


@overload
async def group_by_to(
    iterable: AsyncIterable[T],
    destination: MutableMapping[K, list[V]],
    key_selector: Callable[[T], K | Awaitable[K]],
    value_transform: Callable[[T], V | Awaitable[V]],
) -> Mapping[K, list[V]]: ...


# Ignore: mypy assignment
# Reason: The identity function is type (T) -> T, which is a valid function type
#   for use where (T) -> V is expected it simply means that V == T.
async def group_by_to(
    iterable: AsyncIterable[T],
    destination: MutableMapping[K, list[V]],
    key_selector: Callable[[T], K | Awaitable[K]],
    value_transform: Callable[[T], V | Awaitable[V]] = identity,  # type: ignore[assignment]
) -> Mapping[K, list[V]]:
    async for item in iterable:
        key = await _maybe_await(key_selector(item))
        group = destination.setdefault(key, [])
        group.append(await _maybe_await(value_transform(item)))
    return destination


async def is_empty(iterable: AsyncIterable[T]) -> bool:
    return await first_or_none(iterable) is None


async def is_not_empty(iterable: AsyncIterable[T]) -> bool:
    return not await is_empty(iterable)


async def map_indexed(
    iterable: AsyncIterable[T], mapping: Callable[[int, T], R | Awaitable[R]]
) -> list[R]:
    result: list[R] = []
    idx = 0
    async for item in iterable:
        result.append(await _maybe_await(mapping(idx, item)))
        idx += 1
    return result


async def map_not_none(
    iterable: AsyncIterable[T], mapping: Callable[[T], R | None | Awaitable[R | None]]
) -> list[R]:
    result: list[R] = []
    async for item in iterable:
        mapped = await _maybe_await(mapping(item))
        if mapped is not None:
            result.append(mapped)
    return result


async def map_indexed_not_none(
    iterable: AsyncIterable[T],
    mapping: Callable[[int, T], R | None | Awaitable[R | None]],
) -> list[R]:
    result: list[R] = []
    idx = 0
    async for item in iterable:
        mapped = await _maybe_await(mapping(idx, item))
        if mapped is not None:
            result.append(mapped)
        idx += 1
    return result


async def max_by(
    iterable: AsyncIterable[T], selector: Callable[[T], C | Awaitable[C]]
) -> T:
    iterator = aiter(iterable)
    max_ = await anext(iterator)
    max_value = await _maybe_await(selector(max_))
    async for item in iterator:
        value = await _maybe_await(selector(item))
        if value > max_value:
            max_ = item
            max_value = value
    return max_


async def max_of(
    iterable: AsyncIterable[T], transform: Callable[[T], C | Awaitable[C]]
) -> C:
    iterator = aiter(iterable)
    max_ = await _maybe_await(transform(await anext(iterator)))
    async for item in iterator:
        value = await _maybe_await(transform(item))
        if value > max_:
            max_ = value
    return max_


async def min_by(
    iterable: AsyncIterable[T], selector: Callable[[T], C | Awaitable[C]]
) -> T:
    iterator = aiter(iterable)
    min_ = await anext(iterator)
    min_value = await _maybe_await(selector(min_))
    async for item in iterator:
        value = await _maybe_await(selector(item))
        if value < min_value:
            min_ = item
            min_value = value
    return min_


async def min_of(
    iterable: AsyncIterable[T], transform: Callable[[T], C | Awaitable[C]]
) -> C:
    iterator = aiter(iterable)
    min_ = await _maybe_await(transform(await anext(iterator)))
    async for item in iterator:
        value = await _maybe_await(transform(item))
        if value < min_:
            min_ = value
    return min_


async def none(
    iterable: AsyncIterable[T],
    predicate: Callable[[T], bool | Awaitable[bool]] = default_predicate,
) -> bool:
    async for item in iterable:
        if await _maybe_await(predicate(item)):
            return False
    return True


async def on_each(
    iterable: AsyncIterable[T], action: Callable[[T], None | Awaitable[None]]
) -> AsyncIterable[T]:
    async for item in iterable:
        await _maybe_await(action(item))
    return iterable


async def on_each_indexed(
    iterable: AsyncIterable[T], action: Callable[[int, T], None | Awaitable[None]]
) -> AsyncIterable[T]:
    idx = 0
    async for item in iterable:
        await _maybe_await(action(idx, item))
        idx += 1
    return iterable


async def partition(
    iterable: AsyncIterable[T], predicate: Callable[[T], bool | Awaitable[bool]]
) -> tuple[list[T], list[T]]:
    left: list[T] = []
    right: list[T] = []
    async for item in iterable:
        dest = left if await _maybe_await(predicate(item)) else right
        dest.append(item)
    return left, right


async def reduce(
    iterable: AsyncIterable[T], accumulator: Callable[[T, T], T | Awaitable[T]]
) -> T:
    iterator = aiter(iterable)
    acc = await anext(iterator)
    async for item in iterator:
        acc = await _maybe_await(accumulator(acc, item))
    return acc


async def reduce_indexed(
    iterable: AsyncIterable[T], accumulator: Callable[[int, T, T], T | Awaitable[T]]
) -> T:
    iterator = aiter(iterable)
    acc = await anext(iterator)
    idx = 1
    async for item in iterator:
        acc = await _maybe_await(accumulator(idx, acc, item))
        idx += 1
    return acc


async def reduce_indexed_or_none(
    iterable: AsyncIterable[T], accumulator: Callable[[int, T, T], T | Awaitable[T]]
) -> T | None:
    try:
        return await reduce_indexed(iterable, accumulator)
    except StopAsyncIteration:
        return None


async def reduce_or_none(
    iterable: AsyncIterable[T], accumulator: Callable[[T, T], T | Awaitable[T]]
) -> T | None:
    try:
        return await reduce(iterable, accumulator)
    except StopAsyncIteration:
        return None


async def running_fold(
    iterable: AsyncIterable[T],
    initial: R,
    operation: Callable[[R, T], R | Awaitable[R]],
) -> list[R]:
    result = [initial]
    acc = initial
    async for item in iterable:
        acc = await _maybe_await(operation(acc, item))
        result.append(acc)
    return result


async def running_fold_indexed(
    iterable: AsyncIterable[T],
    initial: R,
    operation: Callable[[int, R, T], R | Awaitable[R]],
) -> list[R]:
    result = [initial]
    acc = initial
    idx = 0
    async for item in iterable:
        acc = await _maybe_await(operation(idx, acc, item))
        result.append(acc)
        idx += 1
    return result


scan = running_fold
scan_indexed = running_fold_indexed


@overload
async def single(iterable: AsyncIterable[T]) -> T: ...


@overload
async def single(
    iterable: AsyncIterable[T], predicate: Callable[[T], bool | Awaitable[bool]]
) -> T: ...


async def single(
    iterable: AsyncIterable[T],
    predicate: Callable[[T], bool | Awaitable[bool]] = default_predicate,
) -> T:
    candidate: T | None = None
    async for item in iterable:
        if await _maybe_await(predicate(item)):
            if candidate is not None:
                raise ValueError("More than one value found")
            candidate = item
    if candidate is None:
        raise ValueError("No values found")
    return candidate


@overload
async def single_or_none(iterable: AsyncIterable[T]) -> T | None: ...


@overload
async def single_or_none(
    iterable: AsyncIterable[T], predicate: Callable[[T], bool | Awaitable[bool]]
) -> T | None: ...


async def single_or_none(
    iterable: AsyncIterable[T],
    predicate: Callable[[T], bool | Awaitable[bool]] = default_predicate,
) -> T | None:
    with suppress(ValueError):
        return await single(iterable, predicate)
    return None


@overload
async def sum_of(
    iterable: AsyncIterable[T], selector: Callable[[T], int | Awaitable[int]]
) -> int: ...


@overload
async def sum_of(
    iterable: AsyncIterable[T], selector: Callable[[T], float | Awaitable[float]]
) -> float: ...


async def sum_of(iterable, selector):  # type: ignore[no-untyped-def]
    sum_ = 0
    async for item in iterable:
        sum_ += await _maybe_await(selector(item))
    return sum_


async def take(iterable: AsyncIterable[T], n: int) -> AsyncIterator[T]:
    require(n >= 0, "n cannot be negative")
    idx = 0
    async for item in iterable:
        if idx >= n:
            break
        yield item
        idx += 1


async def take_while(
    iterable: AsyncIterable[T], predicate: Callable[[T], bool | Awaitable[bool]]
) -> AsyncIterator[T]:
    async for item in iterable:
        if not await _maybe_await(predicate(item)):
            break
        yield item


async def unzip(iterable: AsyncIterable[tuple[T, R]]) -> tuple[list[T], list[R]]:
    left: list[T] = []
    right: list[R] = []
    async for first_, second in iterable:
        left.append(first_)
        right.append(second)
    return left, right


async def windowed(
    iterable: AsyncIterable[T],
    size: int = 1,
    step: int = 1,
    allow_partial: bool = False,
) -> AsyncIterator[Sequence[T]]:
    sequence: list[T] = []
    async for item in iterable:
        sequence.append(item)

    left = 0
    while left < len(sequence):
        right = left + size
        window = sequence[left:right]
        if len(window) < size and not allow_partial:
            break
        yield window
        left = min(left + step, len(sequence))
