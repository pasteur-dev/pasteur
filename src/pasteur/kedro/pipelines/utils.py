from functools import partial
from itertools import chain
from typing import Callable, TypeVar

from ...utils.parser import get_params_for_pipe

A = TypeVar("A")


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
    on the bound function."""

    def __new__(
        cls,
        func,
        /,
        *args,
        _fn: str | None = None,
        _eat: list[str] | None = None,
        **keywords,
    ):
        self = super().__new__(cls, func, *args, **keywords)

        if _fn:
            self.__name__ = _fn.replace("%s", func.__name__)  # type: ignore
        else:
            self.__name__ = func.__name__  # type: ignore

        self.eat = _eat  # type: ignore
        return self

    def __call__(self, /, *args, **keywords):
        kw = keywords.copy()
        if self.eat:  # type: ignore
            for e in self.eat:  # type: ignore
                kw.pop(e, None)
        keywords = {**self.keywords, **kw}
        return self.func(*self.args, *args, **keywords)


def _params_closure(
    fun: Callable, view: str, arguments: list[str], params: dict, **kwargs
):
    meta = get_params_for_pipe(view, params)
    meta_kwargs = {n: meta[n] for n in arguments}

    ext_kwargs = {**meta_kwargs, **kwargs}
    return fun(**ext_kwargs)


def get_params_closure(fun: Callable, view: str, *arguments: str):
    return gen_closure(_params_closure, fun, view, arguments, _fn=fun.__name__)


def _lazy_execute(anchor: str, module: str, fun: str, *args, **kwargs):
    from importlib import import_module

    module = import_module(module, anchor)  # type: ignore

    return getattr(module, fun)(*args, **kwargs)


def lazy_load(anchor, module: str, funs: list[str] | str):
    if isinstance(funs, str):
        return gen_closure(_lazy_execute, anchor, module, funs, _fn=funs)
    return (gen_closure(_lazy_execute, anchor, module, fun, _fn=fun) for fun in funs)
