def get_relative_fn(fn: str):
    """Returns the directory of a file relative to the script calling this function."""
    import inspect
    import os

    script_fn = inspect.currentframe().f_back.f_globals["__file__"]
    dirname = os.path.dirname(script_fn)
    return os.path.join(dirname, fn)