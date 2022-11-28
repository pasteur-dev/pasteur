from __future__ import annotations

from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, TypeGuard

if TYPE_CHECKING:
    import pandas as pd

A = TypeVar("A")

""" A DataFrame that might be loaded lazily. """
LazyChunk = Union[Callable[..., "pd.DataFrame"], "pd.DataFrame"]
""" A dictionary of DataFrame partitions or a DataFrame that might be loaded lazily. 
If dictionary and includes partition `_all`, that partition will load the whole dataframe."""
LazyFrame = Union[dict[str, LazyChunk], LazyChunk]


def get_data(lazy: LazyFrame, columns: list[str] | None = None) -> "pd.DataFrame":
    """Converts and concats a LazyFrame into one DataFrame, regardless of structure."""
    if callable(lazy):
        return lazy(columns=columns)
    elif isinstance(lazy, dict):
        import pandas as pd

        if "_all" in lazy:
            return get_data(lazy["_all"], columns=columns)

        return pd.concat([get_data(l, columns=columns) for l in lazy.values()])
    else:
        return lazy


def is_lazy(chunk: LazyChunk) -> TypeGuard[Callable[..., "pd.DataFrame"]]:
    return callable(chunk)


def is_not_lazy(chunk: LazyChunk) -> TypeGuard["pd.DataFrame"]:
    return not callable(chunk)


def is_partitioned(lazy: LazyFrame) -> TypeGuard[dict[str, LazyChunk]]:
    """Checks whether a LazyFrame is partitioned (ie a dict)."""
    return isinstance(lazy, dict)


def is_not_partitioned(lazy: LazyFrame) -> TypeGuard[LazyChunk]:
    """Checks whether a LazyFrame is partitioned (ie a dict)."""
    return not is_partitioned(lazy)


def are_partitioned(
    lazies: dict[str, LazyFrame]
) -> TypeGuard[dict[str, dict[str, LazyChunk]]]:
    """Checks whether a set of LazyFrames are partitioned.

    In addition, runs a check on whether all of them are partitioned and if they
    are that they have the same keys."""
    partitioned = [is_partitioned(lazy) for lazy in lazies.values()]
    assert all(partitioned) or not any(partitioned)

    if not partitioned[0]:
        return False

    names = list(lazies.keys())
    ref_keys = set(lazies[names[0]].keys())

    for name in names[1:]:
        assert (
            set(lazies[name].keys()) == ref_keys
        ), f"Tables '{name}' and '{names[0]}' have different partitions."

    return True


def are_not_partitioned(
    lazies: dict[str, LazyFrame]
) -> TypeGuard[dict[str, LazyChunk]]:
    return not are_partitioned(lazies)


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

def _run_chunked(fun: Callable, **tables: LazyFrame):
    """Ingests the keys of a dataset in a partitioned manner."""
    partitioned_frames = {}
    non_partitioned_frames = {}

    for name, table in tables.items():
        if is_partitioned(table):
            partitioned_frames[name] = table
        else:
            non_partitioned_frames[name] = table

    if not partitioned_frames:
        return gen_closure(fun, **non_partitioned_frames)

    assert are_partitioned(partitioned_frames)
    keys = next(iter(partitioned_frames.values())).keys()

    return {
        pid: gen_closure(
            fun,
            **non_partitioned_frames,
            **{name: frame[pid] for name, frame in partitioned_frames.items()},
        )
        for pid in keys
    }

def to_chunked(fun: Callable):
    return gen_closure(_run_chunked, fun, _fn=fun.__name__)
