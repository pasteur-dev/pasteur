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


def merge_dicts(*ds):
    out = {}
    for d in ds:
        out = merge_two_dicts(out, d)

    return out
