from collections import namedtuple
from math import isnan
from numbers import Real
from string import ascii_lowercase
from typing import Callable, Hashable, Iterable, NamedTuple, Optional, Tuple

import pytest
from hamcrest import (
    assert_that,
    calling,
    contains_exactly,
    contains_inanyorder,
    equal_to,
    has_entries,
    instance_of,
    is_,
    raises,
)

from peculiar_audience import (
    associate,
    associate_by,
    associate_with,
    average,
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
)
from peculiar_audience._types import T


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
    expected = ["0", "2", "4", "6", "8", "1", "3", "5", "7", "9"]
    even = range(0, 10, 2)
    odd = range(1, 10, 2)
    actual = flat_map(even, odd, mapping=str)
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


def test_map_not_none():
    expected = ["0", "2", "4", "6", "8"]
    actual = map_not_none(range(10), lambda i: str(i) if i % 2 == 0 else None)
    assert_that(actual, contains_exactly(*expected))


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
