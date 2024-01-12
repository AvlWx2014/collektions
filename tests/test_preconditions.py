from hamcrest import assert_that, calling, has_string, not_, raises

from peculiar_audience.preconditions import (
    check,
    check_not_none,
    require,
    require_not_none,
)


def test_check():
    string = "Some non-empty string"
    assert_that(
        calling(check).with_args(bool(string), "String was empty!"),
        not_(raises(RuntimeError, matching=has_string("String was empty!"))),
    )


def test_check_raises():
    string = ""
    assert_that(
        calling(check).with_args(bool(string), "String was empty!"),
        raises(RuntimeError, matching=has_string("String was empty!")),
    )


def test_check_not_none():
    string = "Some non-None string"
    assert_that(
        calling(check_not_none).with_args(string),
        not_(
            raises(RuntimeError, matching=has_string("Check failed: value was None."))
        ),
    )


def test_check_not_none_raises():
    string = None
    assert_that(
        calling(check_not_none).with_args(string),
        raises(RuntimeError, matching=has_string("Check failed: value was None.")),
    )


def test_require():
    arg = 1
    assert_that(
        calling(require).with_args(arg >= 0, "Argument must be >= 0"),
        not_(raises(ValueError, matching=has_string("Argument must be >= 0"))),
    )


def test_require_raises():
    arg = -1
    assert_that(
        calling(require).with_args(arg >= 0, "Argument must be >= 0"),
        raises(ValueError, matching=has_string("Argument must be >= 0")),
    )


def test_require_not_none():
    required = object()
    assert_that(
        calling(require_not_none).with_args(required, "Argument must not be None"),
        not_(raises(ValueError, matching=has_string("Argument must not be None"))),
    )


def test_require_not_none_raises():
    optional = None
    assert_that(
        calling(require_not_none).with_args(optional, "Argument must not be None"),
        raises(ValueError, matching=has_string("Argument must not be None")),
    )
