def find_subclasses(cls):
    """Returns all the subclasses of a given class."""

    sub_cls = {}

    for c in cls.__subclasses__():
        sub_cls[c.name] = c
        sub_cls.update(find_subclasses(c))

    return sub_cls
