import logging
from collections import defaultdict
from typing import Generic, TypeVar, cast

import pandas as pd

from .attribute import Attributes
from .metadata import ColumnMeta, Metadata
from .module import ModuleClass, ModuleFactory
from .table import TransformHolder
from .utils import LazyFrame

logger = logging.getLogger(__name__)

# Column metric -> receives one column, can't be encoded.
# RefColumn metric -> one col + ref, can't be encoded

# Table metric -> receives one table and its parents
# Dataset metric -> receives the whole dataset

# One node is allocated per dataset metric
# One node is allocated per table metric
# kedro only knows about tables. For column metrics, one node is allocated per table
# and runs the metrics for all its columns.

# For column metrics, the name of the metric corresponds to column type, for the rest
# it corresponts to their name.


class ColumnMetricFactory(ModuleFactory["ColumnMetric"]):
    ...


class RefColumnMetricFactory(ModuleFactory["RefColumnMetric"]):
    ...


class TableMetricFactory(ModuleFactory["TableMetric"]):
    def __init__(
        self, cls: type["TableMetric"], *args, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(cls, *args, name=name, **kwargs)
        self.encodings = cls.encodings


class DatasetMetricFactory(ModuleFactory["DatasetMetric"]):
    def __init__(
        self, cls: type["DatasetMetric"], *args, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(cls, *args, name=name, **kwargs)
        self.encodings = cls.encodings


A = TypeVar("A")


class Metric(ModuleClass, Generic[A]):
    """Encapsulates a special way to visualize results."""

    WRK_SPLIT = 1
    REF_SPLIT = 2
    SYN_SPLIT = 3

    type: str

    def fit(self, *args, **kwargs):
        """Fit is used to capture information about the table or column the metric
        will process. It should be used to store information such as column value names,
        which is common among different executions of the view."""
        raise NotImplementedError()

    def process(self, split: int, *args, **kwargs) -> A:
        """Process is called with each set of data from the view (reference, work, synthetic).
        It should capture data relevant to each metric but in a synopsis or compressed form,
        that can be used to compute the metric for different algorithm/split combinations."""
        raise NotImplementedError()

    def visualise(
        self,
        data: dict[str, A],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):
        """Visualise is called for dicts of runs that run within the same view.

        It is expected to create detailed visualizations (such as tables, figures)
        which utilize the structure of the view (columns etc.).

        `comparison` is set to False when the method is run when executing a run and to true
        when run to compare multiple runs. It can be used to provide different summaries

        If required by the visualization, `wrk_set` and `ref_set` provide the names
        of the synthesis source data (wrk) and reference data (ref) which can be used
        as a reference."""
        ...

    def summarize(
        self,
        data: dict[str, A],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):
        """Summarize is called for dicts of runs that are not necessarily from the same view.

        It is expected to create detailed summary metrics for the run which are
        dataset structure independent (such as avg KL, etc).

        `comparison` is set to False when the method is run when executing a run and to true
        when run to compare multiple runs. It can be used to provide different summaries"""
        ...

    def unique_name(self) -> str:
        """Provides a unique name for the metric which will be used for the system.
        (currently saving artifacts)."""
        return f"{self.type}_{self.name}"


class ColumnMetric(Metric[dict[str, A]], Generic[A]):
    type = "col"
    _factory = ColumnMetricFactory

    def fit(self, table: str, col: str, meta: ColumnMeta, data: pd.Series):
        raise NotImplementedError()

    def process(self, split: int, data: pd.Series) -> A:
        raise NotImplementedError()


class RefColumnMetric(ColumnMetric[A], Generic[A]):
    def fit(
        self,
        table: str,
        col: str,
        meta: ColumnMeta,
        data: pd.Series,
        ref: pd.Series | None = None,
    ):
        raise NotImplementedError()

    def process(self, split: int, data: pd.Series, ref: pd.Series | None = None) -> A:
        raise NotImplementedError()


class ColumnMetricHolder(Metric[dict[str, list]]):
    name = "holder"
    type = "col"

    def __init__(self, modules: dict[str, list[ColumnMetricFactory]]) -> None:
        self.table = ""
        self.modules = modules
        self.metrics: dict[str, list[ColumnMetric]] = {}

    def fit(
        self,
        table: str,
        meta: Metadata,
        tables: dict[str, LazyFrame],
        ids: LazyFrame | None = None,
    ):
        self.meta = meta
        self.table = table

        for name, col in meta[table].cols.items():
            if col.is_id():
                continue

            self.metrics[name] = []
            for fs in self.modules.get(col.type, []):
                m = fs.build()
                if col.ref:
                    rtable, rcol = col.ref.table, col.ref.col
                    assert rcol and isinstance(m, RefColumnMetric)

                    if rtable:
                        assert ids and rtable in tables
                        ref_col = ids.join(tables[rtable][rcol], on=rtable)[rcol]
                    else:
                        ref_col = tables[table][rcol]
                    m.fit(table, name, col, tables[table][name], ref_col)
                else:
                    m.fit(table, name, col, tables[table][name])

                self.metrics[name].append(m)

            if not self.metrics[name]:
                logger.warning(
                    f"Type {col.type} of {table}.{name} does not support visualisation (Single-Column metrics)."
                )

    def process(
        self,
        split: int,
        tables: dict[str, LazyFrame],
        ids: LazyFrame | None = None,
    ) -> dict[str, list]:
        out = {}
        for name, metrics in self.metrics.items():
            out[name] = []
            ref = self.meta[self.table][name].ref
            for m in metrics:
                if ref:
                    rtable, rcol = ref.table, ref.col
                    assert rcol and isinstance(m, RefColumnMetric)

                    if rtable:
                        assert ids and rtable in tables
                        ref_col = ids.join(tables[rtable][rcol], on=rtable)[rcol]
                    else:
                        ref_col = tables[self.table][rcol]
                    a = m.process(split, tables[self.table][name], ref_col)
                else:
                    a = m.process(split, tables[self.table][name])

                out[name].append(a)

        return out

    def visualise(
        self,
        data: dict[str, dict[str, list]],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):
        for name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                metric.visualise(
                    data={n: d[name][i] for n, d in data.items()},
                    comparison=comparison,
                    wrk_set=wrk_set,
                    ref_set=ref_set,
                )

    def summarize(
        self,
        data: dict[str, dict[str, list]],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):
        for name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                metric.visualise(
                    data={n: d[name][i] for n, d in data.items()},
                    comparison=comparison,
                    wrk_set=wrk_set,
                    ref_set=ref_set,
                )

    def unique_name(self) -> str:
        return f"{self.type}_{self.name}_{self.table}"


class TableMetric(Metric[A], Generic[A]):
    _factory = TableMetricFactory
    type = "tbl"
    table: str
    encodings: list[str] = ["raw"]

    def fit(
        self,
        table: str,
        meta: Metadata,
        attrs: dict[str, dict[str, Attributes]],
        tables: dict[str, dict[str, LazyFrame]],
        ids: LazyFrame | None = None,
    ):
        raise NotImplementedError()

    def process(
        self,
        split: int,
        tables: dict[str, dict[str, LazyFrame]],
        ids: LazyFrame | None = None,
    ) -> A:
        raise NotImplementedError()

    def unique_name(self) -> str:
        return f"{self.type}_{self.name}_{self.table}"


class DatasetMetric(Metric[A], Generic[A]):
    _factory = DatasetMetricFactory
    type = "dst"
    table: str
    encodings: list[str] = ["raw"]

    def fit(
        self,
        meta: Metadata,
        attrs: dict[str, dict[str, Attributes]],
        tables: dict[str, dict[str, LazyFrame]],
        ids: dict[str, LazyFrame],
    ):
        raise NotImplementedError()

    def process(
        self,
        tables: dict[str, dict[str, LazyFrame]],
        ids: dict[str, LazyFrame],
    ) -> A:
        raise NotImplementedError()


def _separate_tables(data: dict[str, A]) -> dict[str, dict[str, A]]:
    """Receives a dict of tables with names prefixed with a split such as `tbl.`.
    Splits on `.` and returns a dictionary of splits containing a dictionary of tables."""

    splits = defaultdict(dict)

    for in_name, d in data.items():
        split, name = in_name.split(".")
        splits[split][name] = d

    return splits


def _separate_tables_2lvl(data: dict[str, A]) -> dict[str, dict[str, dict[str, A]]]:
    splits: dict[str, dict[str, dict[str, A]]] = defaultdict(lambda: defaultdict(dict))

    for in_name, d in data.items():
        split, enc, name = in_name.split(".")
        splits[split][enc][name] = d

    return splits


def fit_column_holder(
    modules: dict[str, list[ColumnMetricFactory]],
    name: str,
    meta: Metadata,
    ids: LazyFrame,
    **tables: LazyFrame,
):
    holder = ColumnMetricHolder(modules)
    holder.fit(table=name, meta=meta, tables=_separate_tables(tables)["raw"], ids=ids)
    return holder


def process_column_holder(
    split: int,
    holder: ColumnMetricHolder,
    ids: LazyFrame,
    **tables: LazyFrame,
):
    splits = _separate_tables(tables)
    tables = dict(splits["raw"])

    return holder.process(split=split, tables=tables, ids=ids)


def fit_table_metric(
    fs: TableMetricFactory,
    name: str,
    meta: Metadata,
    ids: LazyFrame,
    **tables: LazyFrame | TransformHolder,
):
    enc = fs.encodings
    splits = _separate_tables(tables).copy()
    trn = splits.pop("trn")
    attrs = {
        e: {
            n: h.get_attributes() if e == "bst" else h[e].get_attributes()
            for n, h in cast(dict[str, TransformHolder], trn).items()
        }
        for e in enc
    }
    data = cast(dict[str, dict[str, LazyFrame]], splits)

    module = fs.build()
    module.fit(table=name, meta=meta, attrs=attrs, tables=data, ids=ids)
    return module


def process_table_metric(
    split: int,
    metric: TableMetric,
    ids: LazyFrame,
    **tables: LazyFrame,
):
    return metric.process(split=split, tables=_separate_tables(tables), ids=ids)


def fit_dataset_metric(
    fs: DatasetMetricFactory,
    meta: Metadata,
    **tables: LazyFrame | TransformHolder,
):
    enc = fs.encodings
    assert enc and len(enc) > 1
    splits = _separate_tables(tables)
    trn = splits.pop("trn")
    attrs = {
        e: {
            n: h.get_attributes() if e == "bst" else h[e].get_attributes()
            for n, h in cast(dict[str, TransformHolder], trn).items()
        }
        for e in enc
    }

    ids = cast(dict[str, LazyFrame], splits.pop("ids"))
    data = cast(dict[str, dict[str, LazyFrame]], splits)

    module = fs.build()
    module.fit(meta=meta, attrs=attrs, tables=data, ids=ids)
    return module


def process_dataset_metric(
    metric: DatasetMetric,
    **tables: LazyFrame,
):
    splits = _separate_tables(tables).copy()
    ids = splits.pop("ids")
    return metric.process(tables=splits, ids=ids)


def viz_metric(
    metric: Metric[A],
    comparison: bool = False,
    wrk_set: str = "wrk",
    ref_set: str = "ref",
    **splits: A,
):
    metric.visualise(
        data=splits, comparison=comparison, wrk_set=wrk_set, ref_set=ref_set
    )


def sum_metric(
    metric: Metric[A],
    comparison: bool = False,
    wrk_set: str = "wrk",
    ref_set: str = "ref",
    **splits: A,
):
    metric.summarize(
        data=splits, comparison=comparison, wrk_set=wrk_set, ref_set=ref_set
    )


def log_metric(
    metric: Metric[A],
    comparison: bool = False,
    wrk_set: str = "wrk",
    ref_set: str = "ref",
    **splits: A,
):
    from .utils.mlflow import mlflow_log_artifacts

    mlflow_log_artifacts("metrics", metric.unique_name(), metric=metric, **splits)
