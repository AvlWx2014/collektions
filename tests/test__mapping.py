import calendar

from hamcrest import assert_that, equal_to

from collektions import filter_keys, filter_values
from collektions._mapping import map_keys, map_values


def test_filter_keys():
    mapping = {i: str(i) for i in range(10)}
    expected = {0: "0", 2: "2", 4: "4", 6: "6", 8: "8"}
    actual = filter_keys(mapping, lambda k: k % 2 == 0)
    assert_that(actual, equal_to(expected))


def test_filter_values():
    mapping = {i: str(i) for i in range(10)}
    expected = {0: "0", 2: "2", 4: "4", 6: "6", 8: "8"}
    actual = filter_values(mapping, lambda v: int(v) % 2 == 0)
    assert_that(actual, equal_to(expected))


def test_map_keys_family():
    input_ = {month: i for i, month in enumerate(calendar.month_name) if month}
    expected = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    actual = map_keys(input_, lambda k, _: k[:3].upper())
    assert_that(actual, equal_to(expected))


def test_map_values_family():
    input_ = {i: month for i, month in enumerate(calendar.month_name) if month}
    expected = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    actual = map_values(input_, lambda _, v: v[:3])
    assert_that(actual, equal_to(expected))
