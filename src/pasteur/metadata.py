import logging
from types import SimpleNamespace

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_METRICS = {
    "x_log": False,
    "y_log": False,
    "x_min": None,
    "x_max": None,
    "y_min": None,
    "y_max": None,
}

DEFAULT_TRANSFORMERS = {
    "num": {
        "numerical": "normdist",
        "ordinal": ("idx", "normalize"),
        "categorical": ("idx", "normalize"),
        "time": ("time", "normalize"),
        "date": ("date", "normalize"),
        "datetime": ("datetime", "normalize"),
    },
    "bin": {
        "numerical": ("discrete", "gray"),
        "ordinal": ("idx", "gray"),
        "categorical": "onehot",
        "time": ("time", "gray"),
        "date": ("date", "gray"),
        "datetime": ("datetime", "gray"),
    },
    "bhr": {  # Binary hierarchical
        "numerical": ("discrete", "bin"),
        "ordinal": ("idx", "bin"),
        "categorical": "idx",
        "time": ("time", "bin"),
        "date": ("date", "bin"),
        "datetime": ("datetime", "bin"),
    },
    "idx": {
        "numerical": "discrete",
        "ordinal": "idx",
        "categorical": "idx",
        "time": "time",
        "date": "date",
        "datetime": "datetime",
    },
}

DEFAULT_COLUMN_META = {"bins": 20, "metrics": DEFAULT_METRICS}


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


class ColumnMeta:

    DEFAULT_COLUMN_META = DEFAULT_COLUMN_META

    def __init__(self, **kwargs):
        type_val: str = kwargs["type"]

        # Check for type extended syntax
        # <type><?>|<main-param>:<ref>
        # main-param is passed to the first transformer as a positional value
        is_nullable = "?" in type_val
        type_ref = type_val.replace("?", "").split(":")
        type_param = type_ref[0].split("|")

        type = type_param[0]
        main_param = type_param[1] if len(type_param) > 1 else None

        # Ref can be set both by the ref keyword or by extended syntax
        ref = type_ref[1] if len(type_ref) > 1 else None
        ref = kwargs.get("ref", ref)

        # Basic type and dtype data
        self.type = type
        self.dtype = kwargs.get("dtype", None)

        # Add reference column, used for dates and IDs
        # Format: <table>.<col>
        if ref is not None:
            d = ref.split(".")
            if len(d) == 2:
                table = d[0]
                col = d[1]
            # For ids, if . is omitted, the format is assumed:
            # <table>
            elif self.type == "id":
                table = d[0]
                col = None
            # For other types of columns (such as dates) the format is:
            # <col> (the column might be in the same table).
            else:
                table = None
                col = d[0]
            self.ref = SimpleNamespace(table=table, col=col)
        else:
            self.ref = None

        # Create metrics structure by merging the default dict with the
        # user overrides. Bins can be set universally for both transformers and
        # metrics or just metrics
        metrics = merge_dicts(
            self.DEFAULT_COLUMN_META["metrics"], kwargs.get("metrics", {})
        )
        if "bins" in self.DEFAULT_COLUMN_META:
            metrics.update({"bins": self.DEFAULT_COLUMN_META["bins"]})
        if "bins" in kwargs:
            metrics.update({"bins": kwargs["bins"]})
        self.metrics = SimpleNamespace(**metrics)

        # Add transformer chains from data, with fallback to table chains
        # specific to type.
        self.chains = {}
        table_td = kwargs["td"]
        own_td = kwargs.get("transformers", {})
        for key in set(table_td).union(own_td):
            table_chain = table_td.get(key, {}).get(type, [])
            own_chain = own_td.get(key, [])

            self.chains[key] = own_chain or table_chain

            if not self.chains[key] and type != "id":
                logger.warning("Column chain empty")

        # Add untyped version of args to use with transformers
        self.args = kwargs.copy()
        if main_param is not None:
            self.args.update({"main_param": main_param})
        if is_nullable:
            self.args.update({"nullable": True})

    def is_categorical(self) -> bool:
        return self.type == "categorical"

    def is_cat(self) -> bool:
        return self.is_categorical()

    def is_id(self) -> bool:
        return self.type == "id"


class TableMeta:
    COLUMN_CLS = ColumnMeta

    def __init__(
        self,
        meta: dict,
        data: pd.DataFrame | None = None,
        transformers: dict | None = None,
    ):
        self.primary_key = meta["primary_key"]

        self.targets = meta.get("targets", meta.get("target", []))
        if isinstance(self.targets, str):
            self.targets = [self.targets]
        self.sensitive = meta.get("sensitive", [])

        # Update transformer chain dict from table entries
        self.td = merge_dicts(transformers or {}, meta.get("transformers", {}))

        self._columns = {}

        # Run a key check to ensure metadata and table have the same keys
        if data is not None:
            table_keys = set(data.keys())
            meta_keys = set(meta["fields"].keys())

            diff_keys = meta_keys.difference(table_keys, {data.index.name})
            assert not diff_keys, "Columns missing from table: " + str(diff_keys)

        fields = meta["fields"]
        for name, field in fields.items():
            dtype = None
            if data is not None:
                if data.index.name == name:
                    dtype = data.index.dtype
                else:
                    dtype = str(data.dtypes[name])

            if isinstance(field, str):
                args = {"type": field, "td": self.td}
            else:
                args = field.copy()
                if "dtype" not in args:
                    args["dtype"] = dtype
                args["td"] = self.td

            self._columns[name] = self.COLUMN_CLS(**args)

    @property
    def columns(self) -> dict[str, ColumnMeta]:
        return self._columns

    @property
    def cols(self) -> dict[str, ColumnMeta]:
        return self.columns

    def __getitem__(self, col) -> ColumnMeta:
        return self._columns[col]


class DatasetMeta:
    TABLE_CLS = TableMeta

    def __init__(
        self,
        meta: dict,
        data: dict[str, pd.DataFrame] | None = None,
        params: dict = None,
    ):
        # `default` nanespace contains vals that can be overriden by the view namespace
        # all can be overriden by the global namespace (ex by `-p a.b.c:3`) for
        # hyperparameter tuning
        if params is not None:
            meta = merge_dicts(params.get("default", {}), meta, params)

        transformers = merge_dicts(
            DEFAULT_TRANSFORMERS,
            meta.get("transformers", {}),
        )

        self._tables = {
            name: self.TABLE_CLS(
                tmeta, data.get(name, None) if data is not None else None, transformers
            )
            for name, tmeta in meta["tables"].items()
        }

        self.algs = meta.get("algs", {})
        self.seed = meta.get("random_state", None)
        self.random_state = self.seed

    def get_table(self, name):
        return self._tables[name]

    @property
    def tables(self):
        return list(self._tables.keys())

    def __getitem__(self, name) -> TableMeta | ColumnMeta:
        if isinstance(name, tuple):
            return self._tables[name[0]][name[1]]
        return self._tables[name]


class Metadata(DatasetMeta):
    pass
