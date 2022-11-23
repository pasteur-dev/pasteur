from functools import partial
from itertools import chain
from typing import Any, TypeVar

A = TypeVar("A")


def get_relative_fn(fn: str):
    """Returns the directory of a file relative to the script calling this function."""
    import inspect
    import os

    script_fn = inspect.currentframe().f_back.f_globals["__file__"]  # type: ignore
    dirname = os.path.dirname(script_fn)
    return os.path.join(dirname, fn)


def list_unique(*args: list[A]) -> list[A]:
    return list(dict.fromkeys(chain(*args)))


class gen_closure(partial):
    """Creates a closure for function `fun`, by passing the positional arguments
    provided in this function to `fun` before the ones given to the function and
    by passing the sum of named arguments given to both functions.

    The closure retains the original function name. If desired, it can
    be renamed using the `_fn` parameter. If fn contains `%s`, it will be
    replaced with the function name.

    The `_eat` parameter can be used to consume keyword arguments passed to the
    child function. Ex. if `_eat=["bob"]`, passing `bob=safd` will have no effect
    on the bound function.

    The `_return` parameter can override what the function returns."""

    def __new__(
        cls,
        func,
        /,
        *args,
        _fn: str | None = None,
        _eat: list[str] | None = None,
        _return: Any | None = None,
        **keywords,
    ):
        self = super().__new__(cls, func, *args, **keywords)

        if _fn:
            self.__name__ = _fn.replace("%s", func.__name__)  # type: ignore
        else:
            self.__name__ = func.__name__  # type: ignore

        self._eat = _eat  # type: ignore
        self._return = _return  # type: ignore
        return self

    def __call__(self, /, *args, **keywords):
        kw = keywords.copy()
        if self._eat:  # type: ignore
            for e in self._eat:  # type: ignore
                kw.pop(e, None)
        keywords = {**self.keywords, **kw}
        val = self.func(*self.args, *args, **keywords)
        if self._return is not None:  # type: ignore
            return self._return  # type: ignore
        return val
