""" This module contains a base class `Metadata` which is used to wrap, type, 
and check all View parameters provided to kedro.

@TODO: refactor this file. """

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast, overload

if TYPE_CHECKING:
    import pandas as pd

import logging

logger = logging.getLogger(__name__)


class MetricsMeta(NamedTuple):
    x_log: bool = False
    y_log: bool = False
    bins: int = 20
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None


class ColumnRef(NamedTuple):
    table: str | None = None
    col: str | None = None


class ColumnMeta:
    ref: ColumnRef | list[ColumnRef] | None

    def __init__(self, **kwargs):
        type_val = cast(str, kwargs["type"])

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
        ref = kwargs.get("ref", kwargs.get("refs", ref))
            
        # Basic type and dtype data
        self.type = type
        self.dtype = kwargs.get("dtype", None)

        # Add reference column, used for dates and IDs
        # Format: <table>.<col>
        # Can contain multiple columns, in the form of an array, or a string
        if ref is not None:
            if isinstance(ref, list):
                ref_arr = ref
            elif isinstance(ref, str):
                ref_arr = [r.strip() for r in ref.split(',')]
            else:
                ref_arr = [ref]
            
            refs: list[ColumnRef] = []
            for ref in ref_arr:
                if isinstance(ref, str):
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
                elif isinstance(ref, dict):
                    table = ref.get("table", None)
                    col = ref["col"]
                else:
                    assert False, f"Unsupported ref format: {ref}"
                
                refs.append(ColumnRef(table, col))

            self.ref = refs if len(refs) > 1 else refs[0]
        else:
            self.ref = None

        metrics = kwargs.get("metrics", {})
        if "bins" not in metrics and "bins" in kwargs:
            metrics["bins"] = kwargs["bins"]
        self.metrics = MetricsMeta(**metrics)

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
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()
    
    def __str__(self) -> str:
        return self.__dict__.__str__()


class TableModelMetrics(NamedTuple):
    expand_table: bool = True
    targets: list[str] = []
    sensitive: list[str] = []


class TableMetrics(NamedTuple):
    model: TableModelMetrics = TableModelMetrics()


class TableMeta:
    COLUMN_CLS = ColumnMeta

    def __init__(self, meta: dict):
        self.primary_key = meta.get("primary_key", None)

        if "metrics" in meta:
            metrics_dict = meta["metrics"]
            if "model" in metrics_dict:
                model_dict = metrics_dict["model"]
                model = TableModelMetrics(
                    expand_table=model_dict.get("expand_table", True),
                    targets=model_dict.get("targets", []),
                    sensitive=model_dict.get("sensitive", []),
                )
            else:
                model = TableModelMetrics()

            self.metrics = TableMetrics(model=model)
        else:
            self.metrics = TableMetrics()

        self._columns: dict[str | tuple[str, ...], ColumnMeta] = {}

        fields = meta["fields"]
        for name_str, field in fields.items():
            names = tuple(n.strip() for n in name_str.split(','))
            if len(names) == 1:
                names = names[0]

            if isinstance(field, str):
                args = {"type": field}
            else:
                args = field.copy()

            self._columns[names] = self.COLUMN_CLS(**args)

    @property
    def columns(self) -> dict[str | tuple[str], ColumnMeta]:
        return self._columns

    @property
    def cols(self) -> dict[str | tuple[str], ColumnMeta]:
        return self.columns

    def __getitem__(self, col) -> ColumnMeta:
        return self._columns[col]

    def check(self, data: pd.DataFrame):
        """Run a key check to ensure metadata and table have the same keys"""
        table_keys = set(data.keys())
        meta_keys = set(self._columns.keys())

        diff_keys = meta_keys.difference(table_keys, {data.index.name})
        assert not diff_keys, "Columns missing from table: " + str(diff_keys)
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()
    
    def __str__(self) -> str:
        return self.__dict__.__str__()


class ViewMeta:
    TABLE_CLS = TableMeta

    def __init__(
        self,
        meta: dict,
    ):
        self._tables: dict[str, ViewMeta.TABLE_CLS] = {
            name: self.TABLE_CLS(tmeta) for name, tmeta in meta["tables"].items()
        }

        self.alg_override = meta.get("alg", {})
        self.algs = meta.get("algs", {})
        self.seed: int | None = meta.get("random_state", None)
        self.random_state = self.seed

    def get_table(self, name):
        return self._tables[name]

    @property
    def tables(self):
        return list(self._tables.keys())

    @overload
    def __getitem__(self, name: str) -> TableMeta:
        ...
    
    @overload
    def __getitem__(self, name: tuple[str, str]) -> ColumnMeta:
        ...
    
    def __getitem__(self, name):
        if isinstance(name, tuple):
            return self._tables[name[0]][name[1]]
        return self._tables[name]

    def check(self, data: dict[str, pd.DataFrame]):
        data_tables = set(data.keys())
        meta_tables = set(self._tables.keys())
        assert (
            data_tables == meta_tables
        ), f"Metadata/data have different tables: {data_tables.symmetric_difference(meta_tables)}"

        for name, meta in self._tables.items():
            meta.check(data[name])
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()
    
    def __str__(self) -> str:
        return self.__dict__.__str__()


class Metadata(ViewMeta):
    pass
