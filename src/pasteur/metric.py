from typing import Generic, TypeVar, cast

import pandas as pd
from collections import defaultdict

from .attribute import Attribute, Attributes
from .metadata import ColumnMeta, TableMeta, Metadata
from .module import ModuleClass, ModuleFactory
from .table import TransformHolder

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

""" encoding name to table attribute dict """
PertypeAttributes = dict[str, Attributes]
""" table name to table attribute dict """
DatasetAttributes = dict[str, Attributes]
""" encoding name to dataset attribute dict"""
PertypeDatasetAttributes = dict[str, DatasetAttributes]


class ColumnMetricFactory(ModuleFactory["ColumnMetric"]):
    ...


class RefColumnMetricFactory(ModuleFactory["RefColumnMetric"]):
    ...


class TableMetricFactory(ModuleFactory["TableMetric"]):
    def __init__(
        self, cls: type["TableMetric"], *args, name: str | None = None, **_
    ) -> None:
        super().__init__(cls, *args, name=name, **_)
        self.encodings = cls.encodings


class DatasetMetricFactory(ModuleFactory["DatasetMetric"]):
    def __init__(
        self, cls: type["DatasetMetric"], *args, name: str | None = None, **_
    ) -> None:
        super().__init__(cls, *args, name=name, **_)
        self.encodings = cls.encodings


A = TypeVar("A")


class Metric(ModuleClass, Generic[A]):
    """Encapsulates a special way to visualize results."""

    def fit(self, *args, **kwargs):
        """Fit is used to capture information about the table or column the metric
        will process. It should be used to store information such as column value names,
        which is common among different executions of the view."""
        raise NotImplementedError()

    def process(self, *args, **kwargs) -> A:
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
        raise NotImplementedError()

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
        raise NotImplementedError()


class ColumnMetric(Metric[A], Generic[A]):
    _factory = ColumnMetricFactory

    def fit(self, table: str, col: str, meta: ColumnMeta, data: pd.Series):
        raise NotImplementedError()

    def process(self, data: pd.Series) -> A:
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

    def process(self, data: pd.Series, ref: pd.Series | None = None) -> A:
        raise NotImplementedError()


class TableMetric(Metric[A], Generic[A]):
    _factory = TableMetricFactory
    """None = raw, bst = base layer transform, list = return dict of str to type."""
    encodings: list[str] | str | None

    def fit(
        self,
        table: str,
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


class ColumnMetricHolder(TableMetric[dict[str, list]]):
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
                    m.fit(table, name, col, data[name], ref_col)
                else:
                    m.fit(table, name, col, data[name])

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
                    {n: d[name][i] for n, d in data.items()},
                    comparison,
                    wrk_set,
                    ref_set,
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
                    {n: d[name][i] for n, d in data.items()},
                    comparison,
                    wrk_set,
                    ref_set,
                )


class DatasetMetric(Metric[A], Generic[A]):
    _factory = DatasetMetricFactory
    encodings: list[str] | str | None

    def fit(
        self,
        attrs: dict[str, dict[str, Attributes]] | dict[str, Attributes] | Metadata,
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


def _separate_tables(data: dict[str, A]) -> dict[str, dict[str, A]]:
    """Receives a dict of tables with names prefixed with a split such as `tbl.`.
    Splits on `.` and returns a dictionary of splits containing a dictionary of table."""

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
    **tables: pd.DataFrame,
):
    splits = _separate_tables(tables)

    ids = splits["ids"][name]
    parents = dict(splits["tbl"])
    table = parents.pop(name)

    holder = ColumnMetricHolder(modules)
    holder.fit(name, meta[name], table, ids, parents)
    return holder


def fit_table_metric_no_encodings(
    fs: TableMetricFactory,
    name: str,
    meta: Metadata,
    data: pd.DataFrame,
    ids: pd.DataFrame,
    **tables: pd.DataFrame,
):
    assert not fs.encodings
    module = fs.build()
    module.fit(name, meta[name], data, ids, tables)
    return module


def fit_table_metric_1_encoding(
    fs: TableMetricFactory,
    name: str,
    holder: TransformHolder,
    data: pd.DataFrame,
    ids: pd.DataFrame,
    **tables: pd.DataFrame,
):
    assert fs.encodings and len(fs.encodings) == 1
    enc = fs.encodings[0]
    attrs = holder.get_attributes() if enc == "bst" else holder[enc].get_attributes()

    module = fs.build()
    module.fit(name, attrs, data, ids, tables)
    return module


def fit_table_metric_n_encodings(
    fs: TableMetricFactory,
    name: str,
    holder: TransformHolder,
    **tables: pd.DataFrame,
):
    enc = fs.encodings
    assert enc and len(enc) > 1
    attrs = {n: holder[n].get_attributes() for n in enc}
    splits = _separate_tables(tables)

    module = fs.build()
    module.fit(
        name, attrs, {t: splits[t][name] for t in enc}, splits["ids"][name], splits
    )
    return module


def fit_dataset_metric_no_encodings(
    fs: DatasetMetricFactory,
    meta: Metadata,
    **tables: pd.DataFrame,
):
    assert not fs.encodings
    module = fs.build()
    splits = _separate_tables(tables)
    module.fit(meta, splits["ids"], splits["raw"])
    return module


def fit_dataset_metric_1_encoding(
    fs: DatasetMetricFactory,
    **tables: pd.DataFrame | TransformHolder,
):
    assert fs.encodings and len(fs.encodings) == 1
    enc = fs.encodings[0]
    splits = _separate_tables(tables)
    attrs = {
        n: h.get_attributes() if enc == "bst" else h[enc].get_attributes()
        for n, h in cast(dict[str, TransformHolder], splits["trn"]).items()
    }

    ids = cast(dict[str, pd.DataFrame], splits["ids"])
    data = cast(dict[str, pd.DataFrame], splits["raw"])

    module = fs.build()
    module.fit(attrs, ids, data)
    return module


def fit_dataset_metric_n_encodings(
    fs: DatasetMetricFactory,
    name: str,
    **tables: pd.DataFrame | TransformHolder,
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

    ids = cast(dict[str, pd.DataFrame], splits.pop("ids"))
    data = cast(dict[str, dict[str, pd.DataFrame]], splits)

    module = fs.build()
    module.fit(attrs, ids, data)
    return module


def viz_metric(
    comparison: bool = False,
    wrk_set: str = "wrk",
    ref_set: str = "ref",
    metric: Metric[A] | None = None,
    **splits: A,
):
    assert metric
    metric.visualise(splits, comparison, wrk_set, ref_set)


def sum_metric(
    comparison: bool = False,
    wrk_set: str = "wrk",
    ref_set: str = "ref",
    metric: Metric[A] | None = None,
    **splits: A,
):
    assert metric
    metric.summarize(splits, comparison, wrk_set, ref_set)
