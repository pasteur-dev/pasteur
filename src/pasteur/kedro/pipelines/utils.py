from itertools import chain
from typing import TypeVar

A = TypeVar("A")


def gen_closure(fun: callable, *args, _fn: str = None, **kwargs):
    """Creates a closure for function `fun`, by passing the positional arguments
    provided in this function to `fun` before the ones given to the function and
    by passing the sum of named arguments given to both functions.

    The closure retains the original function name. If desired, it can
    be renamed using the `_fn` parameter. If fn contains `%s`, it will be
    replaced with the function name"""

    def closure(*args2, **kwargs2):
        return fun(*args, *args2, **kwargs, **kwargs2)

    if _fn:
        closure.__name__ = _fn.replace("%s", fun.__name__)
    else:
        closure.__name__ = fun.__name__

    return closure


def list_unique(*args: list[A]) -> list[A]:
    return list(dict.fromkeys(chain(*args)))
