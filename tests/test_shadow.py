import pytest as pytest
from hamcrest import assert_that, is_

from collektions.shadow import (
    __pyall,
    __pyany,
    __pyfilter,
    __pymap,
    all,
    any,
    filter,
    map,
)


@pytest.mark.parametrize(
    "iterable,predicate,expected",
    [
        (range(1, 10), lambda i: i % 10 == 0, False),
        (range(1, 10), lambda i: i % 2 == 0, False),
        (range(1, 10), lambda i: i % 3 == 0, False),
        (range(1, 10), lambda i: i.bit_length() <= 4, True),
        (range(1, 10), lambda i: i < 10, True),
    ],
)
def test_all(iterable, predicate, expected):
    compat = __pyall(__pymap(predicate, iterable))
    actual = all(iterable, predicate)
    assert_that(actual, is_(expected))
    assert_that(actual, is_(compat))


@pytest.mark.parametrize(
    "iterable,predicate,expected",
    [
        (range(1, 10), lambda i: i % 10 == 0, False),
        (range(1, 10), lambda i: i % 2 == 0, True),
        (range(1, 10), lambda i: i % 3 == 0, True),
        (range(1, 10), lambda i: i.bit_length() <= 4, True),
    ],
)
def test_any(iterable, predicate, expected):
    compat = __pyany(__pymap(predicate, iterable))
    actual = any(iterable, predicate)
    assert_that(actual, is_(expected))
    assert_that(actual, is_(compat))


def test_filter():
    iterable = range(11)
    expected = [0, 2, 4, 6, 8, 10]

    def _is_even(num: int) -> bool:
        return not num % 2

    compat = __pyfilter(_is_even, iterable)
    actual = filter(iterable, _is_even)
    for e, a, c in zip(expected, actual, compat):
        assert_that(
            e == c == a, reason=f"Expected: {e} Actual: {a} Python Built-in: {c}"
        )


def test_map():
    iterable = "abcdefghijklmnopqrstuvwxyz"
    expected = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    compat = __pymap(str.upper, iterable)
    actual = list(map(iterable, str.upper))
    for e, a, c in zip(expected, actual, compat):
        assert_that(
            e == c == a, reason=f"Expected: {e} Actual: {a} Python Built-in: {c}"
        )
