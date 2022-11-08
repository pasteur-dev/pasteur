from typing import TypeVar, Any


def _try_convert_to_numeric(value: str):
    """Taken from kedro.framework.cli.utils"""
    try:
        value = float(value)
    except ValueError:
        return value
    return int(value) if value.is_integer() else value


def _try_convert_primitive(value: any):
    """Converts string value to integer/float/bool/None."""
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None
    return _try_convert_to_numeric(value)


def _try_convert_eval(value: str, locals: dict[str, object]):
    return eval(value, {}, locals)


def _update_value_nested_dict(
    nested_dict: dict[str, any], value: any, walking_path: list[str]
) -> dict:
    """Taken from kedro.framework.cli.utils"""
    key = walking_path.pop(0)
    if not walking_path:
        nested_dict[key] = value
        return nested_dict
    nested_dict[key] = _update_value_nested_dict(
        nested_dict.get(key, {}), value, walking_path
    )
    return nested_dict


def str_params_to_dict(params: list[str], locals: dict[str, any] = {}):
    """Converts a list of format ["a.b.c=5", "c=b"] to {a: {b: {c:5}}, c: 'b'}.

    Note the number conversion."""

    param_dict = {}
    for item in params:
        item = item.split("=", 1)
        if len(item) != 2:
            assert False
        key = item[0].strip()
        if not key:
            assert False
        value = item[1].strip()
        param_dict = _update_value_nested_dict(
            param_dict, _try_convert_eval(value, locals), key.split(".")
        )
    return param_dict


def eval_params(params: list[str], locals: dict[str, any] = {}):
    return {
        name: _try_convert_eval(value, locals)
        for name, value in map(lambda x: x.split("=", 1), params)
    }


def merge_params(params: dict[str, Any]):
    param_dict = {}

    for key, val in params.items():
        param_dict = _update_value_nested_dict(param_dict, val, key.split("."))

    return param_dict


def flat_params_to_dict(params: dict[str, any]):
    """Converts a list of format {a.b.c: 5, c: b} to {a: {b: {c:5}}, c: 'b'}.

    Note the number conversion."""

    param_dict = {}
    for key, value in params.items():
        if not key:
            raise
        param_dict = _update_value_nested_dict(
            param_dict, _try_convert_to_numeric(value), key.split(".")
        )
    return param_dict


def dict_to_flat_params(params: dict[str, any]) -> dict[str, str]:
    out = {}
    for param, val in params.items():
        if not isinstance(val, dict):
            out[param] = val
        else:
            exp = dict_to_flat_params(val)
            for nest_param, nest_val in exp.items():
                out[f"{param}.{nest_param}"] = nest_val
    return out


CLS = TypeVar("CLS")


def _find_subclasses(cls: type[CLS]) -> dict[str, type[CLS]]:
    """Returns all the subclasses of a given class."""

    sub_cls = {}

    for c in cls.__subclasses__():
        sub_cls[c.name] = c
        sub_cls.update(_find_subclasses(c))

    sub_cls.pop(None, None)
    return sub_cls


def merge_two_dicts(a: dict, b: dict):
    """Recursively merges dictionaries a, b by prioritizing b."""

    ak = set(a.keys())
    bk = set(b.keys())
    out = {}

    for k in ak - bk:
        out[k] = a[k]
    for k in bk - ak:
        out[k] = b[k]

    for k in ak.intersection(bk):
        if isinstance(a[k], dict) and isinstance(b[k], dict):
            out[k] = merge_two_dicts(a[k], b[k])
        else:
            out[k] = b[k]

    return out


def merge_dicts(*ds: dict):
    out = {}
    for d in ds:
        out = merge_two_dicts(out, d)

    return out


def get_params_for_pipe(name: str, params: dict):
    """Returns the parameters for the provided pipeline by merging
    the nodes `default`, `<view>` and the top level one in one dictionary.

    This allows the user to set default values for all views in the `default`
    namespace, view specific overriding params in the `<view>` namespace and
    override any of them using the `--params` argument without having to use
    the parameter namespace"""
    view = name.split(".")[0]
    return merge_dicts(params.get("default", {}), params.get(view, {}), params)
