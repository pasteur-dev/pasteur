from .base import *
from .misc import *
from .table import TableTransformer, Attribute

DEFAULT_TRANSFORMERS = {
    "num": {
        "numerical": "normdist",
        "ordinal": ("idx", "normalize"),
        "categorical": ("idx", "normalize"),
        "time": ("time", "normalize"),
        "date": ("date", "normalize"),
        "datetime": ("datetime", "normalize"),
        "fixed": "fix",
    },
    "bin": {
        "numerical": ("discrete", "gray"),
        "ordinal": ("idx", "gray"),
        "categorical": "onehot",
        "time": ("time", "gray"),
        "date": ("date", "gray"),
        "datetime": ("datetime", "gray"),
        "fixed": "fix",
    },
    "bhr": {  # Binary hierarchical
        "numerical": ("discrete", "bin"),
        "ordinal": ("idx", "bin"),
        "categorical": "idx",
        "time": ("time", "bin"),
        "date": ("date", "bin"),
        "datetime": ("datetime", "bin"),
        "fixed": "fix",
    },
    "idx": {
        "numerical": "discrete",
        "ordinal": "idx",
        "categorical": "idx",
        "time": "time",
        "date": "date",
        "datetime": "datetime",
        "fixed": "fix",
    },
}

__all__ = ["TableTransformer", "Attribute", "DEFAULT_TRANSFORMERS"]
