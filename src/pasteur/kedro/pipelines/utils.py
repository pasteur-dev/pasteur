from typing import Callable

from ...utils import gen_closure, list_unique
from ...utils.parser import get_params_for_pipe

def _params_closure(
    fun: Callable, view: str, arguments: list[str], params: dict, **kwargs
):
    meta = get_params_for_pipe(view, params)
    meta_kwargs = {}
    for name in arguments:
        if name not in meta:
            raise AttributeError(f"Parameter '{name}' not provided for view '{view}'.")
        meta_kwargs[name] = meta[name]

    ext_kwargs = {**meta_kwargs, **kwargs}
    return fun(**ext_kwargs)


def get_params_closure(fun: Callable, view: str, *arguments: str):
    return gen_closure(_params_closure, fun, view, arguments)


def _lazy_execute(anchor: str, module: str, fun: str, *args, **kwargs):
    from importlib import import_module

    module = import_module(module, anchor)  # type: ignore

    return getattr(module, fun)(*args, **kwargs)


def lazy_load(anchor, module: str, funs: list[str] | str):
    if isinstance(funs, str):
        return gen_closure(_lazy_execute, anchor, module, funs, _fn=funs)
    return (gen_closure(_lazy_execute, anchor, module, fun, _fn=fun) for fun in funs)
