from hamcrest import assert_that, equal_to

from peculiar_audience import filter_keys, filter_values


def test_filter_keys():
    mapping = dict((i, str(i)) for i in range(10))
    expected = [0, 2, 4, 6, 8]
    actual = filter_keys(mapping, lambda k, _: k % 2 == 0)
    assert_that(actual, equal_to(expected))


def test_filter_values():
    mapping = dict((i, str(i)) for i in range(10))
    expected = ["0", "2", "4", "6", "8"]
    actual = filter_values(mapping, lambda _, v: int(v) % 2 == 0)
    assert_that(actual, equal_to(expected))
