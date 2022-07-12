from typing import Dict, Optional, Union
from types import SimpleNamespace
from collections import namedtuple
import pandas as pd


class ColumnMeta:
    DEFAULT_METRICS = {
        "bin": 20,
        "x_log": False,
        "y_log": False,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
    }

    def __init__(self, type, dtype, metrics, *args, **kwargs):
        self.type = type
        self.dtype = dtype
        self.metrics = SimpleNamespace(**{**self.DEFAULT_METRICS, **metrics})

    def is_categorical(self) -> bool:
        return self.type == "categorical"

    def is_cat(self) -> bool:
        return self.is_categorical()

    def is_id(self) -> bool:
        return self.type == "id"


class TableMeta:
    def __init__(self, meta: Dict, data: Optional[pd.DataFrame] = None):
        self.primary_key = meta["primary_key"]
        self._columns = {}

        # Run a key check to ensure metadata and table have the same keys
        if data is not None:
            table_keys = set(data.keys())
            meta_keys = set(meta["fields"].keys())

            diff_keys = table_keys.difference(meta_keys)
            assert not diff_keys, "Columns missing from table/metadata: " + str(
                diff_keys
            )

        fields = meta["fields"]
        for name, field in fields.items():
            dtype = str(field.dtypes["name"]) if data is not None else None

            if isinstance(field, str):
                type = field
                metrics = {}
            else:
                type = field["type"]
                dtype = field.get("dtype", dtype)
                metrics = field.get("metrics", {})

            self._columns[name] = ColumnMeta(type, dtype, metrics)

    @property
    def columns(self):
        return self._columns

    @property
    def cols(self):
        return self.columns

    def __getitem__(self, col) -> ColumnMeta:
        return self._columns[col]


class DatasetMeta:
    def __init__(self, meta: Dict, data: Optional[Dict[str, pd.DataFrame]] = None):
        self._tables = {
            name: TableMeta(tmeta, data.get(name, None) if data is not None else None)
            for name, tmeta in meta["tables"].items()
        }

    def get_table(self, name):
        return self._tables[name]

    @property
    def tables(self):
        return list(self._tables.keys())

    def __getitem__(self, name) -> Union[TableMeta, ColumnMeta]:
        if isinstance(name, tuple):
            return self._tables[name[0]][name[1]]
        return self._tables[name]


class Metadata(DatasetMeta):
    pass
