"""Useful precondition functions modeled after Preconditions.kt.

Implementation-wise, these functions do similar things (minute differences in
which type of Exception is raised where), but they are meant to indicate different
things to a reader.

For example:
  + Use ``require{,_not_none}`` to validate function arguments before
    proceeding with the rest of the function's implementation.
  + Use ``check{,_not_none}`` to test assumptions about an object's state.

"""

from __future__ import annotations

__all__ = ["check", "check_not_none", "require", "require_not_none"]


from ._types import T


def check(
    condition: bool,
    message: str = "Check failed.",
    exc_type: type[Exception] = RuntimeError,
) -> None:
    """Check that ``condition`` is ``True``, and raise an exception of type ``exc_type`` if not.

    By default, ``exc_type`` is a generic :py:obj:`RuntimeError`.
    """
    if not condition:
        e = exc_type(message)
        raise e


def check_not_none(
    value: T | None,
    message: str = "Check failed: value was None.",
) -> None:
    """Check that ``value`` is not ``None`` and raise a ``RuntimeError`` if it is."""
    if value is None:
        raise RuntimeError(message)


def require(
    condition: bool,
    message: str = "Requirement not met.",
    exc_type: type[Exception] = ValueError,
) -> None:
    """Check that the requirement represented by ``condition`` is met.

    Raises an exception of type ``exc_type`` if the requirement is not met.

    By default, ``exc_type`` is :py:obj:`ValueError`.
    """
    if not condition:
        e = exc_type(message)
        raise e


def require_not_none(
    value: T | None, message: str = "Requirement not met: value was None."
) -> None:
    """Check that ``value`` is not ``None`` and raise a ``ValueError`` if it is."""
    if value is None:
        raise ValueError(message)
