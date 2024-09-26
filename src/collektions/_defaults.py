from ._types import T


def default_predicate(_: T) -> bool:
    return True


def default_predicate_with_index(_: int, __: T) -> bool:
    return True


def identity(value: T) -> T:
    return value
