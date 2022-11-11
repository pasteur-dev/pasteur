from typing import Generic, TypeVar

import pandas as pd

from .attribute import Attribute, Attributes
from .metadata import ColumnMeta, TableMeta, Metadata
from .module import ModuleClass, ModuleFactory

import logging

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
    ...


class DatasetMetricFactory(ModuleFactory["DatasetMetric"]):
    ...


A = TypeVar("A")


class Metric(ModuleClass, Generic[A]):
    """Encapsulates a special way to visualize results."""

    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    def process(self, *args, **kwargs) -> A:
        raise NotImplementedError()

    def visualise(self, data: dict[str, A]):
        raise NotImplementedError()

    def summarize(self, data: dict[str, A]):
        raise NotImplementedError()


class ColumnMetric(Metric[A], Generic[A]):
    _factory = ColumnMetricFactory

    def fit(self, meta: ColumnMeta, data: pd.Series):
        raise NotImplementedError()

    def process(self, data: pd.Series) -> A:
        raise NotImplementedError()


class RefColumnMetric(ColumnMetric[A], Generic[A]):
    def fit(self, meta: ColumnMeta, data: pd.Series, ref: pd.Series | None = None):
        raise NotImplementedError()

    def process(self, data: pd.Series, ref: pd.Series | None = None) -> A:
        raise NotImplementedError()


class ColumnMetricHolder:
    def __init__(self, modules: dict[str, list[ColumnMetricFactory]]) -> None:
        self.modules = modules
        self.metrics: dict[str, list[ColumnMetric]] = {}

    def fit(
        self,
        table: str,
        meta: TableMeta,
        data: pd.DataFrame,
        ids: pd.DataFrame | None = None,
        parents: dict[str, pd.DataFrame] | None = None,
    ):
        self.meta = meta

        for name, col in meta.cols.items():
            if col.is_id():
                continue

            self.metrics[name] = []
            for fs in self.modules.get(col.type, []):
                m = fs.build()
                if col.ref:
                    rtable, rcol = col.ref.table, col.ref.col
                    assert rcol and isinstance(m, RefColumnMetric)

                    if rtable:
                        assert ids and parents and rtable in parents
                        ref_col = ids.join(parents[rtable][rcol], on=rtable)[rcol]
                    else:
                        ref_col = data[rcol]
                    m.fit(col, data[name], ref_col)
                else:
                    m.fit(col, data[name])

                self.metrics[name].append(m)

            if not self.metrics[name]:
                logger.warning(
                    f"Type {col.type} of {table}.{name} does not support visualisation (Single-Column metrics)."
                )

    def process(
        self,
        data: pd.DataFrame,
        ids: pd.DataFrame | None = None,
        parents: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, list]:
        out = {}
        for name, metrics in self.metrics.items():
            out[name] = []
            ref = self.meta[name].ref
            for m in metrics:
                if ref:
                    rtable, rcol = ref.table, ref.col
                    assert rcol and isinstance(m, RefColumnMetric)

                    if rtable:
                        assert ids and parents and rtable in parents
                        ref_col = ids.join(parents[rtable][rcol], on=rtable)[rcol]
                    else:
                        ref_col = data[rcol]
                    a = m.process(data[name], ref_col)
                else:
                    a = m.process(data[name])

                out[name].append(a)

        return out

    def visualise(self, data: dict[str, dict[str, list]]):
        for name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                metric.visualise({n: d[name][i] for n, d in data.items()})

    def summarize(self, data: dict[str, dict[str, list]]):
        for name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                metric.visualise({n: d[name][i] for n, d in data.items()})


class TableMetric(Metric[A], Generic[A]):
    _factory = TableMetricFactory
    """None = raw, bst = base layer transform, list = return dict of str to type."""
    encodings: list[str] | str | None

    def fit(
        self,
        attrs: dict[str, Attributes] | Attributes | TableMeta,
        data: dict[str, pd.DataFrame] | pd.DataFrame,
        ids: pd.DataFrame | None = None,
        parents: dict[str, dict[str, pd.DataFrame]]
        | dict[str, pd.DataFrame]
        | pd.DataFrame
        | None = None,
    ):
        raise NotImplementedError()

    def process(
        self,
        data: pd.Series,
        ids: pd.DataFrame | None = None,
        parents: dict[str, dict[str, pd.DataFrame]]
        | dict[str, pd.DataFrame]
        | pd.DataFrame
        | None = None,
    ) -> A:
        raise NotImplementedError()


class DatasetMetric(Metric[A], Generic[A]):
    _factory = DatasetMetricFactory
    encodings: list[str] | str | None

    def fit(
        self,
        attrs: dict[str, Attributes] | Attributes | Metadata,
        ids: dict[str, pd.DataFrame] | pd.DataFrame,
        tables: dict[str, dict[str, pd.DataFrame]]
        | dict[str, pd.DataFrame]
        | pd.DataFrame,
    ):
        raise NotImplementedError()

    def process(
        self,
        data: pd.Series,
        ids: pd.DataFrame | None = None,
        **parents: dict[str, pd.DataFrame],
    ) -> A:
        raise NotImplementedError()



def _separate_ids_tables(data: dict[str, pd.DataFrame]):
    """Separates tables and ids from incoming kedro data. ID Dataframes should
    start with `ids.` and table dataframes with `tbl.`"""
    ids = {}
    tables = {}

    for in_name, d in data.items():
        split, name = in_name.split(".")
        match split:
            case "ids":
                ids[name] = d
            case "tbl":
                tables[name] = d

    return tables, ids


def create_fitted_hist_holder(
    name: str, meta: Metadata, ids: pd.DataFrame, **tables: pd.DataFrame
):
    holder = HistHolder(name, meta)
    holder.fit(tables, ids)
    return holder


def project_hists_for_view(
    holder: HistHolder, ids: pd.DataFrame, **tables: pd.DataFrame
):
    return holder.process(tables, ids)
