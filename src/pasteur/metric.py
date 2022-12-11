import logging
from collections import defaultdict
from typing import Generic, TypeVar, cast, TypedDict, NamedTuple, Any

import pandas as pd

from .attribute import Attributes
from .metadata import ColumnMeta, Metadata
from .module import ModuleClass, ModuleFactory
from .table import TransformHolder
from .utils import LazyChunk, LazyFrame
from .utils.progress import process, process_in_parallel

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

_DATA = TypeVar("_DATA")
_INGEST = TypeVar("_INGEST")
_SUMMARY = TypeVar("_SUMMARY")


class Metric(ModuleClass, Generic[_DATA, _INGEST, _SUMMARY]):
    """Encapsulates a special way to visualize results."""

    type: str

    def fit(self, *args, **kwargs):
        """Fit is used to capture information about the table or column the metric
        will process. It should be used to store information such as column value names,
        which is common among different executions of the view."""
        raise NotImplementedError()

    def preprocess(self, wrk: _DATA, ref: _DATA) -> _INGEST | None:
        """Preprocess is called to cache the summaries for the wrk and ref sets
        during ingest. Implementation is optional."""
        ...

    def process(self, wrk: _DATA, ref: _DATA, syn: _DATA, pre: _INGEST) -> _SUMMARY:
        """Process is called with each set of data from the view (reference, work, synthetic).
        It should capture data relevant to each metric but in a synopsis or compressed form,
        that can be used to compute the metric for different algorithm/split combinations.

        If `preprocess()` is implemented, `pre` will contain the results of the function."""
        raise NotImplementedError()

    def visualise(self, data: dict[str, _SUMMARY]):
        """Visualise is called for dicts of runs that run within the same view.

        It is expected to create detailed visualizations (such as tables, figures)
        which utilize the structure of the view (columns etc.).

        `comparison` is set to False when the method is run when executing a run and to true
        when run to compare multiple runs. It can be used to provide different summaries

        If required by the visualization, `wrk_set` and `ref_set` provide the names
        of the synthesis source data (wrk) and reference data (ref) which can be used
        as a reference."""
        ...

    def summarize(self, data: dict[str, _SUMMARY]):
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


class Summaries(Generic[A]):
    wrk: A
    ref: A
    syn: A | None = None

    def __init__(self, wrk: A, ref: A, syn: A | None = None) -> None:
        self.wrk = wrk
        self.ref = ref
        self.syn = syn

    def replace(self, **kwargs):
        params = {"wrk": self.wrk, "ref": self.ref, "syn": self.syn}
        params.update(kwargs)
        return type(self)(**params)


class ColumnMetric(Metric[Any, Any, Summaries[A]], Generic[A]):
    type = "col"
    _factory = ColumnMetricFactory

    def fit(self, table: str, col: str, meta: ColumnMeta, data: pd.Series):
        raise NotImplementedError()

    def process(self, data: pd.Series) -> A:
        raise NotImplementedError()

    def combine(self, summaries: list[A]) -> A:
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


class ColumnData(TypedDict):
    ids: LazyFrame
    tables: dict[str, LazyFrame]


ColumnSummary = dict[str, list[Any]]


class ColumnMetricHolder(
    Metric[ColumnData, Summaries[ColumnSummary], Summaries[ColumnSummary]]
):
    name = "holder"
    type = "col"

    def __init__(self, modules: dict[str, list[ColumnMetricFactory]]) -> None:
        self.table = ""
        self.modules = modules
        self.metrics: dict[str, list[ColumnMetric]] = {}

    def _fit_chunk(
        self, table: str, meta: Metadata, tables: dict[str, LazyChunk], ids: LazyChunk
    ):
        cached = {table: tables[table]()}
        cached_ids = ids()

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
                        if rtable not in cached:
                            cached[rtable] = tables[rtable]()
                        ref_col = cached_ids.join(cached[rtable][rcol], on=rtable)[rcol]
                    else:
                        ref_col = cached[table][rcol]
                    m.fit(table, name, col, cached[table][name], ref_col)
                else:
                    m.fit(table, name, col, cached[table][name])

                self.metrics[name].append(m)

    def fit(self, table: str, meta: Metadata, data: ColumnData):
        self.meta = meta
        self.table = table

        for name, col in meta[table].cols.items():
            if col.is_id():
                continue

            self.metrics[name] = []
            for fs in self.modules.get(col.type, []):
                self.metrics[name].append(fs.build())

            if not self.metrics[name]:
                logger.warning(
                    f"Type {col.type} of {table}.{name} does not support visualisation (Single-Column metrics)."
                )

        # FIXME: does not fit on all data
        ids = data["ids"]
        tables = data["tables"].copy()
        tables["ids"] = ids
        part = next(iter(LazyFrame.zip_values(**tables))) # FIXME: incorrect type
        self._fit_chunk(table, meta, part, part["ids"]) #type: ignore

    def _process_chunk(
        self,
        **tables: LazyChunk,
    ) -> dict[str, list]:
        cached = {self.table: tables[self.table]()}
        cached_ids = tables["ids"]() if "ids" in tables else None
        out = {}
        for name, metrics in self.metrics.items():
            out[name] = []
            ref = self.meta[self.table][name].ref
            for m in metrics:
                if ref:
                    rtable, rcol = ref.table, ref.col
                    assert rcol and isinstance(m, RefColumnMetric)

                    if rtable:
                        assert cached_ids and rtable in tables
                        ref_col = cached_ids.join(cached[rtable][rcol], on=rtable)[rcol]
                    else:
                        ref_col = cached[self.table][rcol]
                    a = m.process(cached[self.table][name], ref_col)
                else:
                    a = m.process(cached[self.table][name])

                out[name].append(a)

        return out

    def preprocess(self, wrk: ColumnData, ref: ColumnData) -> Summaries:
        chunks_wrk = list(LazyFrame.zip_values(**wrk["tables"], ids=wrk["ids"]))
        chunks_ref = list(LazyFrame.zip_values(**ref["tables"], ids=ref["ids"]))
        summaries = process_in_parallel(
            self._process_chunk,
            per_call_args=[*chunks_wrk, *chunks_ref],
            desc=f"Ingesting metric {self.unique_name()}",
        )
        summaries_wrk = summaries[: len(chunks_wrk)]
        summaries_ref = summaries[len(chunks_ref) :]

        wrk_sum = {}
        ref_sum = {}
        for name, metrics in self.metrics.items():
            wrk_sum[name] = []
            ref_sum[name] = []
            for i, metric in enumerate(metrics):
                wrk_sum[name].append(metric.combine(
                    [chunk[name][i] for chunk in summaries_wrk]
                ))
                ref_sum[name].append(metric.combine(
                    [chunk[name][i] for chunk in summaries_ref]
                ))

        return Summaries(wrk_sum, ref_sum)

    def process(
        self, wrk: ColumnData, ref: ColumnData, syn: ColumnData, pre: Summaries
    ) -> Summaries:
        chunks_syn = list(LazyFrame.zip_values(**syn["tables"], ids=syn["ids"]))
        summaries = process_in_parallel(
            self._process_chunk,
            per_call_args=chunks_syn,
            desc=f"Processing metric {self.unique_name()}",
        )

        syn_sum = {}
        for name, metrics in self.metrics.items():
            syn_sum[name] = []
            for i, metric in enumerate(metrics):
                syn_sum[name].append(metric.combine(
                    [chunk[name][i] for chunk in summaries]
                ))

        return pre.replace(syn=syn_sum)

    def visualise(self, data: dict[str, Summaries]):
        for name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                metric.visualise(
                    data={
                        sname: Summaries(
                            wrk=s.wrk[name][i], ref=s.ref[name][i], syn=s.syn[name][i]  # type: ignore
                        )
                        for sname, s in data.items()
                    },
                )

    def summarize(self, data: dict[str, Summaries]):
        for name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                metric.summarize(
                    data={
                        sname: Summaries(
                            wrk=s.wrk[name][i], ref=s.ref[name][i], syn=s.syn[name][i]  # type: ignore
                        )
                        for sname, s in data.items()
                    },
                )

    def unique_name(self) -> str:
        return f"{self.type}_{self.name}_{self.table}"


class TableData(TypedDict):
    ids: LazyFrame
    tables: dict[str, dict[str, LazyFrame]]


class TableMetric(Metric[TableData, _INGEST, _SUMMARY], Generic[_INGEST, _SUMMARY]):
    _factory = TableMetricFactory
    type = "tbl"
    table: str
    encodings: list[str] = ["raw"]

    def fit(
        self,
        table: str,
        meta: Metadata,
        attrs: dict[str, dict[str, Attributes]],
        data: TableData,
    ):
        raise NotImplementedError()

    def unique_name(self) -> str:
        return f"{self.type}_{self.name}_{self.table}"


class DatasetData(TypedDict):
    tables: dict[str, dict[str, LazyFrame]]
    ids: dict[str, LazyFrame]


class DatasetMetric(Metric[DatasetData, _INGEST, _SUMMARY], Generic[_INGEST, _SUMMARY]):
    _factory = DatasetMetricFactory
    type = "dst"
    table: str
    encodings: list[str] = ["raw"]

    def fit(
        self, meta: Metadata, attrs: dict[str, dict[str, Attributes]], data: DatasetData
    ):
        raise NotImplementedError()


def fit_column_holder(
    modules: dict[str, list[ColumnMetricFactory]],
    name: str,
    meta: Metadata,
    data: ColumnData,
):
    holder = ColumnMetricHolder(modules)
    holder.fit(table=name, meta=meta, data=data)
    return holder


def fit_table_metric(
    fs: TableMetricFactory,
    name: str,
    meta: Metadata,
    trns: dict[str, TransformHolder],
    data: TableData,
):
    enc = fs.encodings
    attrs = {
        e: {
            n: h.get_attributes() if e == "bst" else h[e].get_attributes()
            for n, h in trns.items()
        }
        for e in enc
    }

    module = fs.build()
    module.fit(table=name, meta=meta, attrs=attrs, data=data)
    return module


def fit_dataset_metric(
    fs: DatasetMetricFactory,
    meta: Metadata,
    trns: dict[str, TransformHolder],
    data: DatasetData,
):
    enc = fs.encodings
    attrs = {
        e: {
            n: h.get_attributes() if e == "bst" else h[e].get_attributes()
            for n, h in trns.items()
        }
        for e in enc
    }

    module = fs.build()
    module.fit(meta=meta, attrs=attrs, data=data)
    return module


def log_metric(metric: Metric[Any, Any, _SUMMARY], summary: _SUMMARY):
    from .utils.mlflow import mlflow_log_artifacts

    mlflow_log_artifacts(
        "metrics", metric.unique_name(), metric=metric, summary=summary
    )
