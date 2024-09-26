import string
from collections import namedtuple
from collections.abc import Collection, Sized
from math import isnan
from numbers import Real
from string import ascii_lowercase
from typing import Callable, Hashable, Iterable, NamedTuple, Optional, Tuple, TypeVar

import pytest
from hamcrest import (
    assert_that,
    calling,
    contains_exactly,
    contains_inanyorder,
    empty,
    equal_to,
    has_entries,
    instance_of,
    is_,
    not_,
    raises,
    same_instance,
)

from peculiar_audience import (
    associate,
    associate_by,
    associate_with,
    average,
    chunked,
    distinct,
    distinct_by,
    first,
    first_or_none,
    flat_map,
    flatten,
    fold,
    map_not_none,
    none,
    sum_by,
    windowed,
)
from peculiar_audience._iterable import (
    drop,
    drop_while,
    fold_indexed,
    group_by,
    is_empty,
    is_not_empty,
    map_indexed,
    map_indexed_not_none,
    max_by,
    max_of,
    min_by,
    min_of,
    on_each,
    on_each_indexed,
    partition,
)

T = TypeVar("T")


class RandomObject(NamedTuple):
    property1: str
    property2: int


def test_associate():
    expected = dict((i, letter.upper()) for i, letter in enumerate(ascii_lowercase))

    def _transform(letter: str) -> Tuple[int, str]:
        return ascii_lowercase.index(letter), letter.upper()

    actual = associate(ascii_lowercase, _transform)
    assert_that(actual, has_entries(expected))


def test_associate_by():
    expected = dict((i, letter) for i, letter in enumerate(ascii_lowercase))
    actual = associate_by(ascii_lowercase, lambda letter: ascii_lowercase.index(letter))
    assert_that(actual, has_entries(expected))


def test_associate_with():
    expected = dict((letter, letter.upper()) for letter in ascii_lowercase)
    actual = associate_with(ascii_lowercase, str.upper)
    assert_that(actual, has_entries(expected))


@pytest.mark.parametrize(
    "values,expected",
    [
        (
            range(10),
            4.5,
        ),
        (
            [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            5.5,
        ),
        (
            [
                1,
            ]
            * 10,
            1,
        ),
        (
            [4, 0, 69, 6, 54, 45, 99, 9, 25, 26],
            33.7,
        ),
    ],
)
def test_average_non_empty(values: Iterable[Real], expected: float):
    actual = average(values)
    assert_that(actual, equal_to(expected))


def test_average_returns_nan_on_empty():
    actual = average([])
    assert_that(isnan(actual))


@pytest.mark.parametrize(
    "iterable,chunk_size,expected",
    [
        (
            range(10),
            1,
            [
                range(0, 1),
                range(1, 2),
                range(2, 3),
                range(3, 4),
                range(4, 5),
                range(5, 6),
                range(6, 7),
                range(7, 8),
                range(8, 9),
                range(9, 10),
            ],
        ),
        (range(10), 5, [range(5), range(5, 10)]),
        (range(0), 1, []),
        (range(10), 3, [range(3), range(3, 6), range(6, 9), range(9, 10)]),
    ],
)
def test_chunked(
    iterable: Iterable[int], chunk_size: int, expected: Iterable[Iterable[int]]
):
    actual = list(chunked(iterable, size=chunk_size))
    assert_that(actual, equal_to(expected))


@pytest.mark.parametrize(
    "iterable,predicate,expected",
    [
        ([], None, []),
        (
            [
                1,
                1,
                1,
                2,
                3,
                3,
                4,
                5,
                6,
                7,
                7,
                8,
                9,
                9,
                9,
                9,
                9,
            ],
            None,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ),
        (
            "aaaAAAAbbbCBBBcccccDDDeDDeEE",
            lambda letter: letter.upper(),
            ["a", "b", "C", "D", "e"],
        ),
        (
            [
                RandomObject("Test 1", 1),
                RandomObject("Test 1", 2),
                RandomObject("Test 2", 1),
                RandomObject("Test 2", 2),
                RandomObject("Test 3", 1),
                RandomObject("Test 3", 2),
                RandomObject("Test 3", 3),
            ],
            lambda obj: obj.property1,
            [
                RandomObject("Test 1", 1),
                RandomObject("Test 2", 1),
                RandomObject("Test 3", 1),
            ],
        ),
        (
            [
                RandomObject("Test 1", 1),
                RandomObject("Test 1", 2),
                RandomObject("Test 2", 1),
                RandomObject("Test 2", 2),
                RandomObject("Test 3", 1),
                RandomObject("Test 3", 2),
                RandomObject("Test 3", 3),
            ],
            lambda obj: obj.property2,
            [
                RandomObject("Test 1", 1),
                RandomObject("Test 1", 2),
                RandomObject("Test 3", 3),
            ],
        ),
    ],
)
def test_distinct_family(
    iterable: Iterable[T],
    predicate: Optional[Callable[[T], Hashable]],
    expected: Iterable[T],
):
    if predicate is None:
        actual = distinct(iterable)
    else:
        actual = distinct_by(iterable, predicate)

    assert_that(actual, contains_inanyorder(*expected))


@pytest.mark.parametrize(
    "iterable,n,expected",
    [
        (range(10), 5, [5, 6, 7, 8, 9]),
        (range(100), 99, [99]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5, [5, 6, 7, 8, 9]),
        (string.ascii_lowercase, 13, "nopqrstuvwxyz"),
        ((i for i in range(10)), 5, [5, 6, 7, 8, 9]),
    ],
)
def test_drop(iterable: Iterable[T], n, expected: Collection[T]):
    actual = drop(iterable, n)
    assert_that(actual, contains_exactly(*expected))


@pytest.mark.parametrize(
    "iterable,n",
    [
        (range(0), 1),
        (range(10), 10),
        (range(5), 10),
        ((i for i in range(5)), 10),
    ],
)
def test_drop_all(iterable, n):
    actual: Sized[int] = list(drop(iterable, n))
    assert_that(actual, empty())


def test_drop_while():
    input_ = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    expected = [6, 8, 1, 3, 5, 7, 9]
    actual = drop_while(input_, lambda x: x < 5)
    assert_that(actual, contains_exactly(*expected))


def test_drop_while_all():
    actual: Sized[int] = list(drop_while(range(10)))
    assert_that(actual, empty())


@pytest.mark.parametrize(
    "iterable,predicate,expected",
    [
        (range(1, 10), lambda i: i % 2 == 0, 2),
        (range(1, 10), lambda i: i % 3 == 0, 3),
        (range(1, 10), lambda i: i == 10, ValueError()),
    ],
)
def test_first(iterable, predicate, expected):
    if isinstance(expected, Exception):
        assert_that(
            calling(first).with_args(iterable, predicate), raises(type(expected))
        )
    else:
        actual = first(iterable, predicate)
        assert_that(actual, equal_to(expected))


@pytest.mark.parametrize(
    "iterable,predicate,expected",
    [
        (range(1, 10), lambda i: i % 2 == 0, 2),
        (range(1, 10), lambda i: i % 3 == 0, 3),
        (range(1, 10), lambda i: i == 10, None),
    ],
)
def test_first_or_none(iterable, predicate, expected):
    actual = first_or_none(iterable, predicate)
    assert_that(actual, equal_to(expected))


def test_flat_map():
    expected = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]
    actual = flat_map(range(6), lambda i: range(i, 6))
    assert_that(actual, contains_exactly(*expected))


def test_flatten():
    expected = range(10)
    even = range(0, 10, 2)
    odd = range(1, 10, 2)
    actual = flatten(even, odd)
    assert_that(actual, contains_inanyorder(*expected))


def test_fold():
    inputs = range(10)
    expected = sum(inputs)
    actual = fold(inputs, 0, lambda x, y: x + y)
    assert_that(actual, equal_to(expected))


def test_fold_indexed():
    n = 10
    inputs = range(n)
    expected = n
    # quite possibly the hardest way of adding 1 together n times
    actual = fold_indexed(inputs, 0, lambda x, i, y: x + y - i + 1)
    assert_that(actual, equal_to(expected))


def test_group_by_family():
    inputs = range(10)
    expected = {"odd": [1, 3, 5, 7, 9], "even": [0, 2, 4, 6, 8]}
    actual = group_by(inputs, lambda k: "even" if k % 2 == 0 else "odd")
    assert_that(actual, equal_to(expected))


def test_group_by_family_non_identity():
    inputs = range(10)
    expected = {"odd": [2, 4, 6, 8, 10], "even": [1, 3, 5, 7, 9]}
    actual = group_by(
        inputs, lambda k: "even" if k % 2 == 0 else "odd", lambda v: v + 1
    )
    assert_that(actual, equal_to(expected))


@pytest.mark.parametrize(
    "inputs, empty",
    [
        (range(10), False),
        ((i for i in range(10)), False),
        ((i for i in range(0)), True),
        ([], True),
        ("", True),
    ],
)
def test_is_empty_family(inputs: Iterable[T], empty: bool):
    assert_that(is_empty(inputs), is_(empty))
    assert_that(is_not_empty(inputs), is_(not_(empty)))


def test_map_not_none():
    expected = ["0", "2", "4", "6", "8"]
    actual = map_not_none(range(10), lambda i: str(i) if i % 2 == 0 else None)
    assert_that(actual, contains_exactly(*expected))


def test_map_indexed():
    inputs = range(10)
    expected = [
        "0",
        "2",
        "4",
        "6",
        "8",
        "10",
        "12",
        "14",
        "16",
        "18",
    ]
    actual = map_indexed(inputs, lambda idx, i: str(i + idx))
    assert_that(actual, equal_to(expected))


def test_map_indexed_not_none():
    inputs = range(10)
    expected = [
        "0",
        "4",
        "8",
        "12",
        "16",
    ]
    actual = map_indexed_not_none(
        inputs, lambda idx, i: str(i + idx) if i % 2 == 0 else None
    )
    assert_that(actual, equal_to(expected))


def test_max_by():
    inputs = [
        RandomObject(letter, index)
        for index, letter in enumerate(ascii_lowercase, start=1)
    ]
    expected = RandomObject("z", 26)
    actual = max_by(inputs, lambda it: it.property2)
    assert_that(actual, equal_to(expected))


def test_max_by_empty():
    with pytest.raises(StopIteration):
        max_by((), lambda it: it)


def test_max_of():
    inputs = [
        RandomObject(letter, index)
        for index, letter in enumerate(ascii_lowercase, start=1)
    ]
    expected = 26
    actual = max_of(inputs, lambda it: it.property2)
    assert_that(actual, equal_to(expected))


def test_max_of_empty():
    with pytest.raises(StopIteration):
        max_of((), lambda it: it)


def test_min_by():
    inputs = [
        RandomObject(letter, index)
        for index, letter in enumerate(ascii_lowercase, start=1)
    ]
    expected = RandomObject("a", 1)
    actual = min_by(inputs, lambda it: it.property1)
    assert_that(actual, equal_to(expected))


def test_min_by_empty():
    with pytest.raises(StopIteration):
        min_by((), lambda it: it)


def test_min_of():
    inputs = [
        RandomObject(letter, index)
        for index, letter in enumerate(ascii_lowercase, start=1)
    ]
    expected = "a"
    actual = min_of(inputs, lambda it: it.property1)
    assert_that(actual, equal_to(expected))


def test_min_of_empty():
    with pytest.raises(StopIteration):
        min_of((), lambda it: it)


@pytest.mark.parametrize(
    "iterable,predicate,expected",
    [
        (range(1, 10), lambda i: i % 10 == 0, True),
        (range(1, 10), lambda i: i % 2 == 0, False),
        (range(1, 10), lambda i: i % 3 == 0, False),
        (range(1, 10), lambda i: i.bit_length() > 4, True),
    ],
)
def test_none(iterable, predicate, expected):
    actual = none(iterable, predicate)
    assert_that(actual, is_(expected))


def test_on_each():
    log = []
    action = log.append
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    inputs = range(10)
    return_value = on_each(inputs, action)
    assert_that(log, equal_to(expected))
    assert_that(return_value, same_instance(inputs))


def test_partition():
    inputs = range(10)
    expected_left, expected_right = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9]
    evens, odds = partition(inputs, lambda it: it % 2 == 0)
    assert_that(evens, equal_to(expected_left))
    assert_that(odds, equal_to(expected_right))


def test_on_each_indexed():
    log = []
    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    inputs = range(10)
    return_value = on_each_indexed(inputs, lambda idx, it: log.append(it + idx))
    assert_that(log, equal_to(expected))
    assert_that(return_value, same_instance(inputs))


@pytest.mark.parametrize(
    "iterable,type_,expected", [(range(10), int, 45), (range(10), float, 45.0)]
)
def test_sum_by(iterable, type_, expected):
    # create some temporary objects in order to exercise ``selector``
    T = namedtuple("T", ("value",))
    ts = [T(value=type_(i)) for i in iterable]
    actual = sum_by(ts, lambda o: o.value)
    assert_that(actual, instance_of(type_))
    assert_that(actual, equal_to(expected))


@pytest.mark.parametrize(
    "iterable,size,step,partial,expected",
    [
        (
            range(10),
            10,
            1,
            True,
            [
                range(0, 10),
                range(1, 10),
                range(2, 10),
                range(3, 10),
                range(4, 10),
                range(5, 10),
                range(6, 10),
                range(7, 10),
                range(8, 10),
                range(9, 10),
            ],
        ),
        (range(10), 10, 1, False, [range(0, 10)]),
    ],
)
def test_windowed(
    iterable: Iterable[int],
    size: int,
    step: int,
    partial: bool,
    expected: Iterable[Iterable[int]],
):
    actual = list(windowed(iterable, size, step, allow_partial=partial))
    assert_that(actual, equal_to(expected))
