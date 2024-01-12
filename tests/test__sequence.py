import pytest
from hamcrest import assert_that, calling, equal_to, raises

from peculiar_audience import last, last_or_none


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
