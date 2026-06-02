from __future__ import annotations

from collections import namedtuple
from collections.abc import AsyncIterator
from math import isnan
from string import ascii_lowercase
from typing import NamedTuple, TypeVar

import pytest
from hamcrest import (
    assert_that,
    contains_exactly,
    contains_inanyorder,
    empty,
    equal_to,
    has_entries,
    instance_of,
    is_,
    not_,
    same_instance,
)

from collektions.ext.asyncio import (
    associate,
    associate_by,
    associate_with,
    average,
    chunked,
    distinct,
    distinct_by,
    drop,
    drop_while,
    filter_indexed,
    filter_isinstance,
    filter_not,
    filter_not_none,
    first,
    first_not_none_of,
    first_not_none_of_or_none,
    first_or_none,
    flat_map,
    flatten,
    fold,
    fold_indexed,
    group_by,
    is_empty,
    is_not_empty,
    map_indexed,
    map_indexed_not_none,
    map_not_none,
    max_by,
    max_of,
    min_by,
    min_of,
    none,
    on_each,
    on_each_indexed,
    partition,
    reduce,
    reduce_indexed,
    reduce_indexed_or_none,
    reduce_or_none,
    running_fold,
    running_fold_indexed,
    single,
    single_or_none,
    sum_of,
    take,
    take_while,
    unzip,
    windowed,
)

T = TypeVar("T")


class RandomObject(NamedTuple):
    property1: str
    property2: int


async def aiter_of(*items: T) -> AsyncIterator[T]:
    for item in items:
        yield item


async def async_collect(ait: AsyncIterator[T]) -> list[T]:
    return [item async for item in ait]


# --- Association ---


@pytest.mark.asyncio
async def test_associate():
    expected = {i: letter.upper() for i, letter in enumerate(ascii_lowercase)}

    def _transform(letter: str) -> tuple[int, str]:
        return ascii_lowercase.index(letter), letter.upper()

    actual = await associate(aiter_of(*ascii_lowercase), _transform)
    assert_that(actual, has_entries(expected))


@pytest.mark.asyncio
async def test_associate_async_transform():
    async def _transform(letter: str) -> tuple[int, str]:
        return ascii_lowercase.index(letter), letter.upper()

    actual = await associate(aiter_of(*ascii_lowercase), _transform)
    expected = {i: letter.upper() for i, letter in enumerate(ascii_lowercase)}
    assert_that(actual, has_entries(expected))


@pytest.mark.asyncio
async def test_associate_by():
    expected = dict(enumerate(ascii_lowercase))
    actual = await associate_by(
        aiter_of(*ascii_lowercase), lambda letter: ascii_lowercase.index(letter)
    )
    assert_that(actual, has_entries(expected))


@pytest.mark.asyncio
async def test_associate_with():
    expected = {letter: letter.upper() for letter in ascii_lowercase}
    actual = await associate_with(aiter_of(*ascii_lowercase), str.upper)
    assert_that(actual, has_entries(expected))


# --- Aggregation ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values,expected",
    [
        (list(range(10)), 4.5),
        ([1, 3, 5, 7, 9, 2, 4, 6, 8, 10], 5.5),
        ([1] * 10, 1),
        ([4, 0, 69, 6, 54, 45, 99, 9, 25, 26], 33.7),
    ],
)
async def test_average_non_empty(values: list[int], expected: float):
    actual = await average(aiter_of(*values))
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_average_returns_nan_on_empty():
    actual = await average(aiter_of())
    assert_that(isnan(actual))


@pytest.mark.asyncio
async def test_sum_of():
    T_ = namedtuple("T_", ("value",))
    ts = [T_(value=i) for i in range(10)]
    actual = await sum_of(aiter_of(*ts), lambda o: o.value)
    assert_that(actual, instance_of(int))
    assert_that(actual, equal_to(45))


@pytest.mark.asyncio
async def test_sum_of_async_selector():
    T_ = namedtuple("T_", ("value",))
    ts = [T_(value=float(i)) for i in range(10)]

    async def _selector(o):
        return o.value

    actual = await sum_of(aiter_of(*ts), _selector)
    assert_that(actual, equal_to(45.0))


@pytest.mark.asyncio
async def test_max_by():
    inputs = [
        RandomObject(letter, index)
        for index, letter in enumerate(ascii_lowercase, start=1)
    ]
    expected = RandomObject("z", 26)
    actual = await max_by(aiter_of(*inputs), lambda it: it.property2)
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_max_by_empty():
    with pytest.raises(StopAsyncIteration):
        await max_by(aiter_of(), lambda it: it)


@pytest.mark.asyncio
async def test_max_of():
    inputs = [
        RandomObject(letter, index)
        for index, letter in enumerate(ascii_lowercase, start=1)
    ]
    actual = await max_of(aiter_of(*inputs), lambda it: it.property2)
    assert_that(actual, equal_to(26))


@pytest.mark.asyncio
async def test_max_of_empty():
    with pytest.raises(StopAsyncIteration):
        await max_of(aiter_of(), lambda it: it)


@pytest.mark.asyncio
async def test_min_by():
    inputs = [
        RandomObject(letter, index)
        for index, letter in enumerate(ascii_lowercase, start=1)
    ]
    expected = RandomObject("a", 1)
    actual = await min_by(aiter_of(*inputs), lambda it: it.property1)
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_min_by_empty():
    with pytest.raises(StopAsyncIteration):
        await min_by(aiter_of(), lambda it: it)


@pytest.mark.asyncio
async def test_min_of():
    inputs = [
        RandomObject(letter, index)
        for index, letter in enumerate(ascii_lowercase, start=1)
    ]
    actual = await min_of(aiter_of(*inputs), lambda it: it.property1)
    assert_that(actual, equal_to("a"))


@pytest.mark.asyncio
async def test_min_of_empty():
    with pytest.raises(StopAsyncIteration):
        await min_of(aiter_of(), lambda it: it)


# --- Filtering ---


@pytest.mark.asyncio
async def test_filter_not():
    actual = await filter_not(aiter_of(*range(10)), lambda x: x % 2 == 0)
    assert_that(actual, contains_exactly(1, 3, 5, 7, 9))


@pytest.mark.asyncio
async def test_filter_not_async_predicate():
    async def _pred(x):
        return x % 2 == 0

    actual = await filter_not(aiter_of(*range(10)), _pred)
    assert_that(actual, contains_exactly(1, 3, 5, 7, 9))


@pytest.mark.asyncio
async def test_filter_not_none():
    actual = await filter_not_none(aiter_of(1, None, 2, None, 3))
    assert_that(actual, contains_exactly(1, 2, 3))


@pytest.mark.asyncio
async def test_filter_indexed():
    actual = await filter_indexed(aiter_of(*range(10)), lambda i, x: i % 2 == 0)
    assert_that(actual, contains_exactly(0, 2, 4, 6, 8))


@pytest.mark.asyncio
async def test_filter_isinstance():
    actual = await filter_isinstance(aiter_of(1, "a", 2, "b", 3), int)
    assert_that(actual, contains_exactly(1, 2, 3))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "predicate,expected",
    [
        (lambda i: i % 10 == 0, True),
        (lambda i: i % 2 == 0, False),
        (lambda i: i % 3 == 0, False),
        (lambda i: i.bit_length() > 4, True),
    ],
)
async def test_none(predicate, expected):
    actual = await none(aiter_of(*range(1, 10)), predicate)
    assert_that(actual, is_(expected))


@pytest.mark.asyncio
async def test_partition():
    evens, odds = await partition(aiter_of(*range(10)), lambda it: it % 2 == 0)
    assert_that(evens, equal_to([0, 2, 4, 6, 8]))
    assert_that(odds, equal_to([1, 3, 5, 7, 9]))


@pytest.mark.asyncio
async def test_partition_async_predicate():
    async def _pred(it):
        return it % 2 == 0

    evens, odds = await partition(aiter_of(*range(10)), _pred)
    assert_that(evens, equal_to([0, 2, 4, 6, 8]))
    assert_that(odds, equal_to([1, 3, 5, 7, 9]))


# --- Mapping ---


@pytest.mark.asyncio
async def test_map_indexed():
    actual = await map_indexed(aiter_of(*range(10)), lambda idx, i: str(i + idx))
    expected = ["0", "2", "4", "6", "8", "10", "12", "14", "16", "18"]
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_map_not_none():
    actual = await map_not_none(
        aiter_of(*range(10)), lambda i: str(i) if i % 2 == 0 else None
    )
    assert_that(actual, contains_exactly("0", "2", "4", "6", "8"))


@pytest.mark.asyncio
async def test_map_indexed_not_none():
    actual = await map_indexed_not_none(
        aiter_of(*range(10)), lambda idx, i: str(i + idx) if i % 2 == 0 else None
    )
    assert_that(actual, equal_to(["0", "4", "8", "12", "16"]))


@pytest.mark.asyncio
async def test_flat_map():
    expected = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]
    actual = await flat_map(aiter_of(*range(6)), lambda i: range(i, 6))
    assert_that(actual, contains_exactly(*expected))


@pytest.mark.asyncio
async def test_flat_map_async_iterable_result():
    async def _transform(i):
        for x in range(i, 6):
            yield x

    expected = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]
    actual = await flat_map(aiter_of(*range(6)), _transform)
    assert_that(actual, contains_exactly(*expected))


@pytest.mark.asyncio
async def test_flatten():
    even = aiter_of(0, 2, 4, 6, 8)
    odd = aiter_of(1, 3, 5, 7, 9)
    actual = await async_collect(flatten(even, odd))
    assert_that(actual, contains_inanyorder(*range(10)))


@pytest.mark.asyncio
async def test_distinct():
    actual = await distinct(aiter_of(1, 1, 2, 3, 3, 4, 5))
    assert_that(actual, contains_inanyorder(1, 2, 3, 4, 5))


@pytest.mark.asyncio
async def test_distinct_by():
    inputs = [
        RandomObject("Test 1", 1),
        RandomObject("Test 1", 2),
        RandomObject("Test 2", 1),
        RandomObject("Test 2", 2),
        RandomObject("Test 3", 1),
    ]
    actual = await distinct_by(aiter_of(*inputs), lambda obj: obj.property1)
    expected = [
        RandomObject("Test 1", 1),
        RandomObject("Test 2", 1),
        RandomObject("Test 3", 1),
    ]
    assert_that(actual, contains_inanyorder(*expected))


# --- Finding ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "predicate,expected",
    [
        (lambda i: i % 2 == 0, 2),
        (lambda i: i % 3 == 0, 3),
    ],
)
async def test_first(predicate, expected):
    actual = await first(aiter_of(*range(1, 10)), predicate)
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_first_raises():
    with pytest.raises(ValueError):
        await first(aiter_of(*range(1, 10)), lambda i: i == 10)


@pytest.mark.asyncio
async def test_first_async_predicate():
    async def _pred(i):
        return i % 2 == 0

    actual = await first(aiter_of(*range(1, 10)), _pred)
    assert_that(actual, equal_to(2))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "predicate,expected",
    [
        (lambda i: i % 2 == 0, 2),
        (lambda i: i % 3 == 0, 3),
        (lambda i: i == 10, None),
    ],
)
async def test_first_or_none(predicate, expected):
    actual = await first_or_none(aiter_of(*range(1, 10)), predicate)
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_first_not_none_of():
    actual = await first_not_none_of(
        aiter_of(2, 4, 6, 8, 1, 3), lambda i: None if not i % 2 else i
    )
    assert_that(actual, equal_to(1))


@pytest.mark.asyncio
async def test_first_not_none_of_raises():
    with pytest.raises(ValueError):
        await first_not_none_of(aiter_of(*range(1, 10)), lambda _: None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items,transform,expected",
    [
        ([2, 4, 6, 8, 1, 3], lambda i: None if not i % 2 else i, 1),
        ([1, 3, 5, 7, 9, 0, 2], lambda i: i if not i % 2 else None, 0),
        (list(range(1, 10)), lambda _: None, None),
    ],
)
async def test_first_not_none_of_or_none(items, transform, expected):
    actual = await first_not_none_of_or_none(aiter_of(*items), transform)
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_single():
    actual = await single(aiter_of(10))
    assert_that(actual, equal_to(10))


@pytest.mark.asyncio
async def test_single_empty():
    with pytest.raises(ValueError):
        await single(aiter_of())


@pytest.mark.asyncio
async def test_single_with_predicate():
    actual = await single(aiter_of(*range(1, 10)), lambda it: it % 5 == 0)
    assert_that(actual, equal_to(5))


@pytest.mark.asyncio
async def test_single_with_predicate_raises():
    with pytest.raises(ValueError):
        await single(aiter_of(*range(10)), lambda it: it % 5 == 0)


@pytest.mark.asyncio
async def test_single_or_none_empty():
    actual = await single_or_none(aiter_of())
    assert_that(actual, is_(None))


@pytest.mark.asyncio
async def test_single_or_none_multiple_candidates():
    actual = await single_or_none(aiter_of(*range(10)), lambda it: it % 5 == 0)
    assert_that(actual, is_(None))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items,expected_empty",
    [
        (list(range(10)), False),
        ([], True),
    ],
)
async def test_is_empty_family(items, expected_empty):
    assert_that(await is_empty(aiter_of(*items)), is_(expected_empty))
    assert_that(await is_not_empty(aiter_of(*items)), is_(not_(expected_empty)))


# --- Reduction ---


@pytest.mark.asyncio
async def test_reduce():
    actual = await reduce(aiter_of(*range(10)), lambda acc, value: acc + value)
    assert_that(actual, equal_to(45))


@pytest.mark.asyncio
async def test_reduce_empty():
    with pytest.raises(StopAsyncIteration):
        await reduce(aiter_of(), lambda acc, it: it)


@pytest.mark.asyncio
async def test_reduce_async_accumulator():
    async def _acc(acc, value):
        return acc + value

    actual = await reduce(aiter_of(*range(10)), _acc)
    assert_that(actual, equal_to(45))


@pytest.mark.asyncio
async def test_reduce_indexed():
    actual = await reduce_indexed(
        aiter_of(*range(10)), lambda idx, acc, value: acc + value - idx + 1
    )
    assert_that(actual, equal_to(9))


@pytest.mark.asyncio
async def test_reduce_indexed_empty():
    with pytest.raises(StopAsyncIteration):
        await reduce_indexed(aiter_of(), lambda idx, acc, it: it)


@pytest.mark.asyncio
async def test_reduce_indexed_or_none():
    actual = await reduce_indexed_or_none(aiter_of(), lambda idx, acc, it: it)
    assert_that(actual, is_(None))


@pytest.mark.asyncio
async def test_reduce_or_none():
    actual = await reduce_or_none(aiter_of(), lambda acc, it: it)
    assert_that(actual, is_(None))


@pytest.mark.asyncio
async def test_fold():
    actual = await fold(aiter_of(*range(10)), 0, lambda x, y: x + y)
    assert_that(actual, equal_to(45))


@pytest.mark.asyncio
async def test_fold_async_accumulator():
    async def _acc(x, y):
        return x + y

    actual = await fold(aiter_of(*range(10)), 0, _acc)
    assert_that(actual, equal_to(45))


@pytest.mark.asyncio
async def test_fold_indexed():
    actual = await fold_indexed(aiter_of(*range(10)), 0, lambda x, i, y: x + y - i + 1)
    assert_that(actual, equal_to(10))


@pytest.mark.asyncio
async def test_running_fold():
    actual = await running_fold(aiter_of(*range(10)), 10, lambda acc, it: acc + it)
    expected = [10, 10, 11, 13, 16, 20, 25, 31, 38, 46, 55]
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_running_fold_indexed():
    actual = await running_fold_indexed(
        aiter_of(*range(10)), 10, lambda idx, acc, it: acc + it + idx
    )
    expected = [10, 10, 12, 16, 22, 30, 40, 52, 66, 82, 100]
    assert_that(actual, equal_to(expected))


# --- Grouping ---


@pytest.mark.asyncio
async def test_group_by():
    actual = await group_by(
        aiter_of(*range(10)), lambda k: "even" if k % 2 == 0 else "odd"
    )
    expected = {"odd": [1, 3, 5, 7, 9], "even": [0, 2, 4, 6, 8]}
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_group_by_with_value_transform():
    actual = await group_by(
        aiter_of(*range(10)),
        lambda k: "even" if k % 2 == 0 else "odd",
        lambda v: v + 1,
    )
    expected = {"odd": [2, 4, 6, 8, 10], "even": [1, 3, 5, 7, 9]}
    assert_that(actual, equal_to(expected))


# --- Slicing ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items,n,expected",
    [
        (list(range(10)), 5, [5, 6, 7, 8, 9]),
        (list(range(100)), 99, [99]),
        (list(range(10)), 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
async def test_drop(items, n, expected):
    actual = await drop(aiter_of(*items), n)
    assert_that(actual, contains_exactly(*expected))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items,n",
    [
        ([], 1),
        (list(range(10)), 10),
        (list(range(5)), 10),
    ],
)
async def test_drop_all(items, n):
    actual = await drop(aiter_of(*items), n)
    assert_that(actual, empty())


@pytest.mark.asyncio
async def test_drop_while():
    input_ = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    actual = await drop_while(aiter_of(*input_), lambda x: x < 5)
    assert_that(actual, contains_exactly(6, 8, 1, 3, 5, 7, 9))


@pytest.mark.asyncio
async def test_drop_while_all():
    actual = await drop_while(aiter_of(*range(10)))
    assert_that(actual, empty())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items,n,expected",
    [
        (list(range(10)), 5, [0, 1, 2, 3, 4]),
        ([], 5, []),
        (list(range(10)), 13, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
async def test_take(items, n, expected):
    actual = await async_collect(take(aiter_of(*items), n))
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_take_negative():
    with pytest.raises(ValueError):
        await async_collect(take(aiter_of(*range(10)), -5))


@pytest.mark.asyncio
async def test_take_while():
    actual = await async_collect(take_while(aiter_of(*range(10)), lambda it: it < 5))
    assert_that(actual, equal_to(list(range(5))))


@pytest.mark.asyncio
async def test_take_while_async_predicate():
    async def _pred(it):
        return it < 5

    actual = await async_collect(take_while(aiter_of(*range(10)), _pred))
    assert_that(actual, equal_to(list(range(5))))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items,size,step,partial,expected",
    [
        (
            list(range(10)),
            5,
            5,
            False,
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        ),
        (
            list(range(10)),
            3,
            3,
            True,
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
        ),
        (
            list(range(10)),
            10,
            1,
            False,
            [list(range(10))],
        ),
    ],
)
async def test_windowed(items, size, step, partial, expected):
    actual = await async_collect(
        windowed(aiter_of(*items), size, step, allow_partial=partial)
    )
    assert_that(actual, equal_to(expected))


@pytest.mark.asyncio
async def test_chunked():
    actual = await async_collect(chunked(aiter_of(*range(10)), size=3))
    assert_that(actual, equal_to([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]))


# --- Side Effects ---


@pytest.mark.asyncio
async def test_on_each():
    log = []
    inputs = aiter_of(*range(10))
    return_value = await on_each(inputs, log.append)
    assert_that(log, equal_to(list(range(10))))
    assert_that(return_value, same_instance(inputs))


@pytest.mark.asyncio
async def test_on_each_async_action():
    log = []

    async def _action(item):
        log.append(item)

    await on_each(aiter_of(*range(10)), _action)
    assert_that(log, equal_to(list(range(10))))


@pytest.mark.asyncio
async def test_on_each_indexed():
    log = []
    inputs = aiter_of(*range(10))
    return_value = await on_each_indexed(inputs, lambda idx, it: log.append(it + idx))
    assert_that(log, equal_to([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]))
    assert_that(return_value, same_instance(inputs))


# --- Other ---


@pytest.mark.asyncio
async def test_unzip():
    zipped = list(zip(range(1, 27), ascii_lowercase, strict=False))
    actual_left, actual_right = await unzip(aiter_of(*zipped))
    assert_that(actual_left, equal_to(list(range(1, 27))))
    assert_that(actual_right, equal_to(list(ascii_lowercase)))
