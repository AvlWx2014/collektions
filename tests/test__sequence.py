from collections.abc import Sized

import pytest
from hamcrest import assert_that, calling, contains_exactly, empty, equal_to, raises

from peculiar_audience import drop_last, last, last_or_none
from peculiar_audience._sequence import drop_last_while


def test_drop_last():
    expected = [0, 1, 2, 3, 4]
    actual = drop_last(range(10), 5)
    assert_that(actual, contains_exactly(*expected))


def test_drop_last_while():
    input_ = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    expected = [0, 2, 4, 6, 8, 1, 3]
    actual = drop_last_while(input_, lambda x: x >= 5)
    assert_that(actual, contains_exactly(*expected))


def test_drop_last_while_all():
    actual: Sized[int] = list(drop_last_while(range(10)))
    assert_that(actual, empty())


@pytest.mark.parametrize(
    "iterable,predicate,expected",
    [
        (range(1, 10), lambda i: i % 2 == 0, 8),
        (range(1, 10), lambda i: i % 3 == 0, 9),
        (range(1, 10), lambda i: i == 10, ValueError()),
    ],
)
def test_last(iterable, predicate, expected):
    if isinstance(expected, Exception):
        assert_that(
            calling(last).with_args(iterable, predicate), raises(type(expected))
        )
    else:
        actual = last(iterable, predicate)
        assert_that(actual, equal_to(expected))


@pytest.mark.parametrize(
    "iterable,predicate,expected",
    [
        (range(1, 10), lambda i: i % 3 == 0, 9),
        (range(2, 10, 2), lambda i: i % 3 == 0, 6),
        (range(1, 10), lambda i: i % 13 == 0, None),
    ],
)
def test_last_or_none(iterable, predicate, expected):
    actual = last_or_none(iterable, predicate)
    assert_that(actual, equal_to(expected))
