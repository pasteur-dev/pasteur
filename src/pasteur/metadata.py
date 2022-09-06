import logging
from .utils import merge_dicts
import pandas as pd
from typing import NamedTuple

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
            self.ref = ColumnRef(table, col)
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


class TableModelMetrics(NamedTuple):
    expand_table: bool = True
    targets: list[str] = []
    sensitive: list[str] = []


class TableMetrics(NamedTuple):
    model: TableModelMetrics = TableModelMetrics()


class TableMeta:
    COLUMN_CLS = ColumnMeta

    def __init__(
        self,
        meta: dict,
        data: pd.DataFrame | None = None,
    ):
        self.primary_key = meta["primary_key"]

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

        self._columns = {}

        # Run a key check to ensure metadata and table have the same keys
        if data is not None:
            table_keys = set(data.keys())
            meta_keys = set(meta["fields"].keys())

            diff_keys = meta_keys.difference(table_keys, {data.index.name})
            assert not diff_keys, "Columns missing from table: " + str(diff_keys)

        fields = meta["fields"]
        for name, field in fields.items():
            if isinstance(field, str):
                args = {"type": field}
            else:
                args = field.copy()

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
    ):
        self._tables = {
            name: self.TABLE_CLS(
                tmeta, data.get(name, None) if data is not None else None
            )
            for name, tmeta in meta["tables"].items()
        }

        self.alg_override = meta.get("alg", {})
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

    @staticmethod
    def from_kedro_params(
        params: dict, view: str, data: dict[str, pd.DataFrame] | None = None
    ):
        from .utils import get_params_for_pipe

        meta_dict = get_params_for_pipe(view, params)
        return DatasetMeta(meta_dict, data)


class Metadata(DatasetMeta):
    pass
