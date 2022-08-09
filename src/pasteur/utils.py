def find_subclasses(cls):
    """Returns all the subclasses of a given class."""

    sub_cls = {}

    for c in cls.__subclasses__():
        sub_cls[c.name] = c
        sub_cls.update(find_subclasses(c))

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


def get_params_closure(fun: callable, view: str, *arguments: str):
    def closure(params: dict, **kwargs):
        meta = get_params_for_pipe(view, params)
        meta_kwargs = {n: meta[n] for n in arguments}

        ext_kwargs = {**meta_kwargs, **kwargs}
        return fun(**ext_kwargs)

    return closure
