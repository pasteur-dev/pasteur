""" This module provides the definitions for Metric Modules.
Metric modules can fit to a column, a table, or a whole View.
In each case, modules are instanciated as required (for columns one is instantiated
per column type, for tables one per table and View metrics are instantiated once)."""

import logging
from collections import defaultdict
from typing import Any, Generic, Literal, TypedDict, TypeVar, cast

import pandas as pd

from pasteur.utils import LazyDataset

from .attribute import Attributes
from .metadata import ColumnMeta, ColumnRef, Metadata
from .encode import Encoder
from .module import Module, ModuleClass, ModuleFactory, get_module_dict_multiple
from .table import ReferenceManager, _calc_joined_refs, _calc_unjoined_refs
from .utils import LazyChunk, LazyDataset, LazyFrame, LazyPartition, lazy_load_tables
from .utils.progress import piter, process_in_parallel, reduce

logger = logging.getLogger(__name__)


class ColumnMetricFactory(ModuleFactory["AbstractColumnMetric"]):
    ...


class MetricFactory(ModuleFactory["Metric"]):
    def __init__(
        self, cls: type["Metric"], *args, name: str | None = None, **kwargs
    ) -> None:
        super().__init__(cls, *args, name=name, **kwargs)
        self.encodings = cls.encodings


A = TypeVar("A")

_DATA = TypeVar("_DATA")
_INGEST = TypeVar("_INGEST")
_SUMMARY = TypeVar("_SUMMARY")


class Metric(ModuleClass, Generic[_INGEST, _SUMMARY]):
    """Encapsulates a special way to visualize results.

    The metric is provided with the metrics requested in `encodings`.
    If one encoding is requested and `encodings` is a string, `meta` and
    `data` will contain the metadata and data of that encoding.

    If `encodings` is a list, `meta` and `data` will be dictionaries containing
    the metadata and data for each encoding."""

    _factory = MetricFactory
    encodings: str | list[str] = "raw"
    type: str

    def fit(
        self,
        meta: Any | dict[str, Any],
        data: dict[str, LazyDataset] | dict[str, dict[str, LazyDataset]],
    ):
        """Fit is used to capture information about the table or column the metric
        will process. It should be used to store information such as column value names,
        which is common among different executions of the view."""
        raise NotImplementedError()

    def preprocess(
        self,
        wrk: dict[str, LazyDataset] | dict[str, dict[str, LazyDataset]],
        ref: dict[str, LazyDataset] | dict[str, dict[str, LazyDataset]],
    ) -> _INGEST | None:
        """Preprocess is called to cache the summaries for the wrk and ref sets
        during ingest. Implementation is optional."""
        ...

    def process(
        self,
        wrk: dict[str, LazyDataset] | dict[str, dict[str, LazyDataset]],
        ref: dict[str, LazyDataset] | dict[str, dict[str, LazyDataset]],
        syn: dict[str, LazyDataset] | dict[str, dict[str, LazyDataset]],
        pre: _INGEST,
    ) -> _SUMMARY:
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


class RefColumnData(TypedDict):
    data: pd.Series | pd.DataFrame
    ref: pd.Series | pd.DataFrame | None


class SeqColumnData(TypedDict):
    data: pd.Series | pd.DataFrame
    ref: dict[str, pd.DataFrame] | None
    ids: pd.DataFrame | None


class AbstractColumnMetric(ModuleClass, Generic[_DATA, _INGEST, _SUMMARY]):
    type: Literal["col", "ref", "seq"] = "col"
    _factory = ColumnMetricFactory

    def fit(self, table: str, col: str | tuple[str], meta: ColumnMeta, data: _DATA):
        """Fit is used to capture information about the table or column the metric
        will process. It should be used to store information such as column value names,
        which is common among different executions of the view."""
        raise NotImplementedError()

    def reduce(self, other: "AbstractColumnMetric"):
        ...

    def preprocess(self, wrk: _DATA, ref: _DATA) -> _INGEST | None:
        """Preprocess is called to cache the summaries for the wrk and ref sets
        during ingest. Implementation is optional."""
        ...

    def process(self, wrk: _DATA, ref: _DATA, syn: _DATA, pre: _INGEST) -> _SUMMARY:
        raise NotImplementedError()

    def combine(self, summaries: list[_SUMMARY]) -> _SUMMARY:
        raise NotImplementedError()

    def visualise(self, data: dict[str, _SUMMARY]):
        ...

    def summarize(self, data: dict[str, _SUMMARY]):
        ...


class ColumnMetric(
    AbstractColumnMetric[pd.Series | pd.DataFrame, _INGEST, _SUMMARY],
    Generic[_INGEST, _SUMMARY],
):
    pass


class RefColumnMetric(
    AbstractColumnMetric[RefColumnData, _INGEST, _SUMMARY],
    Generic[_INGEST, _SUMMARY],
):
    pass


class SeqColumnMetric(
    AbstractColumnMetric[SeqColumnData, _INGEST, _SUMMARY],
    Generic[_INGEST, _SUMMARY],
):
    pass


B = TypeVar("B", bound="Any")


def _reduce_inner_2d(
    a: dict[str | tuple[str], list[B]],
    b: dict[str | tuple[str], list[B]],
):
    for key in a.keys():
        for i in range(len(a[key])):
            a[key][i].reduce(b[key][i])
    return a


def _fit_column_metrics(
    name: str,
    meta: Metadata,
    ref: ReferenceManager,
    tables: dict[str, LazyChunk],
    metrics: dict[str, list[ColumnMetricFactory]],
):
    get_table = lazy_load_tables(tables)

    if ref.table_has_reference():
        ids = ref.find_foreign_ids(name, get_table)
    else:
        ids = None

    out: dict[str | tuple[str], list[AbstractColumnMetric]] = defaultdict(list)
    for col_name, col in meta[name].cols.items():
        if col.is_id() or col.type not in metrics:
            continue

        for factory in metrics[col.type]:
            # Create metric
            if "main_param" in col.args:
                m = factory.build(col.args["main_param"], **col.args)
            else:
                m = factory.build(**col.args)

            if isinstance(m, ColumnMetric):
                m.fit(name, col_name, col, get_table(name)[col_name])
            elif isinstance(m, RefColumnMetric):
                ref_col = _calc_joined_refs(name, get_table, ids, col.ref)
                m.fit(
                    name,
                    col_name,
                    col,
                    RefColumnData(data=get_table(name)[col_name], ref=ref_col),
                )
            elif isinstance(m, SeqColumnMetric):
                ref_col = _calc_unjoined_refs(name, get_table, col.ref)
                m.fit(
                    name,
                    col_name,
                    col,
                    SeqColumnData(data=get_table(name)[col_name], ref=ref_col, ids=ids),
                )
            else:
                assert False, f"Unknown column metric type: {type(m)}"

            out[col_name].append(m)

    return out


def _preprocess_metrics(
    name: str,
    meta: Metadata,
    ref: ReferenceManager,
    tables_wrk: dict[str, LazyChunk],
    tables_ref: dict[str, LazyChunk],
    metrics: dict[str | tuple[str], list[AbstractColumnMetric]],
):
    get_table_wrk = lazy_load_tables(tables_wrk)
    get_table_ref = lazy_load_tables(tables_ref)

    if ref.table_has_reference():
        ids_wrk = ref.find_foreign_ids(name, get_table_wrk)
        ids_ref = ref.find_foreign_ids(name, get_table_ref)
    else:
        ids_wrk = None
        ids_ref = None

    out = defaultdict(list)
    for col_name, ms in metrics.items():
        for m in ms:
            col = meta[name][col_name]
            if isinstance(m, ColumnMetric):
                prec = m.preprocess(
                    get_table_wrk(name)[col_name],
                    get_table_ref(name)[col_name],
                )
            elif isinstance(m, RefColumnMetric):
                prec = m.preprocess(
                    RefColumnData(
                        data=get_table_wrk(name)[col_name],
                        ref=_calc_joined_refs(name, get_table_wrk, ids_ref, col.ref),
                    ),
                    RefColumnData(
                        data=get_table_ref(name)[col_name],
                        ref=_calc_joined_refs(name, get_table_ref, ids_ref, col.ref),
                    ),
                )
            elif isinstance(m, SeqColumnMetric):
                prec = m.preprocess(
                    SeqColumnData(
                        data=get_table_wrk(name)[col_name],
                        ref=_calc_unjoined_refs(name, get_table_wrk, col.ref),
                        ids=ids_wrk,
                    ),
                    SeqColumnData(
                        data=get_table_ref(name)[col_name],
                        ref=_calc_unjoined_refs(name, get_table_ref, col.ref),
                        ids=ids_ref,
                    ),
                )
            else:
                assert False, f"Unknown column metric type: {type(m)}"

            out[col_name].append(prec)

    return out


def _process_metrics(
    name: str,
    meta: Metadata,
    ref: ReferenceManager,
    tables_wrk: dict[str, LazyChunk],
    tables_ref: dict[str, LazyChunk],
    tables_syn: dict[str, LazyChunk],
    metrics: dict[str | tuple[str], list[AbstractColumnMetric]],
    preprocess: dict[str | tuple[str], list[Any]],
):
    get_table_wrk = lazy_load_tables(tables_wrk)
    get_table_ref = lazy_load_tables(tables_ref)
    get_table_syn = lazy_load_tables(tables_syn)

    if ref.table_has_reference():
        ids_wrk = ref.find_foreign_ids(name, get_table_wrk)
        ids_ref = ref.find_foreign_ids(name, get_table_ref)
        ids_syn = ref.find_foreign_ids(name, get_table_syn)
    else:
        ids_wrk = None
        ids_ref = None
        ids_syn = None

    out = defaultdict(list)
    for col_name, ms in metrics.items():
        for m, prec in zip(ms, preprocess[col_name]):
            col = meta[name][col_name]
            if isinstance(m, ColumnMetric):
                proc = m.process(
                    get_table_wrk(name)[col_name],
                    get_table_ref(name)[col_name],
                    get_table_syn(name)[col_name],
                    prec,
                )
            elif isinstance(m, RefColumnMetric):
                proc = m.process(
                    RefColumnData(
                        data=get_table_wrk(name)[col_name],
                        ref=_calc_joined_refs(name, get_table_wrk, ids_wrk, col.ref),
                    ),
                    RefColumnData(
                        data=get_table_ref(name)[col_name],
                        ref=_calc_joined_refs(name, get_table_ref, ids_ref, col.ref),
                    ),
                    RefColumnData(
                        data=get_table_syn(name)[col_name],
                        ref=_calc_joined_refs(name, get_table_syn, ids_syn, col.ref),
                    ),
                    prec,
                )
            elif isinstance(m, SeqColumnMetric):
                proc = m.process(
                    SeqColumnData(
                        data=get_table_wrk(name)[col_name],
                        ref=_calc_unjoined_refs(name, get_table_wrk, col.ref),
                        ids=ids_wrk,
                    ),
                    SeqColumnData(
                        data=get_table_ref(name)[col_name],
                        ref=_calc_unjoined_refs(name, get_table_ref, col.ref),
                        ids=ids_ref,
                    ),
                    SeqColumnData(
                        data=get_table_syn(name)[col_name],
                        ref=_calc_unjoined_refs(name, get_table_syn, col.ref),
                        ids=ids_syn,
                    ),
                    prec,
                )
            else:
                assert False, f"Unknown column metric type: {type(m)}"

            out[col_name].append(proc)

    return out


class ColumnMetricHolder(
    Metric[
        dict[str, dict[str, dict[str | tuple[str], list[Any]]]],
        dict[str, dict[str | tuple[str], list[Any]]],
    ]
):
    name = "holder"
    type = "col"
    encodings = "raw"

    def __init__(self, modules: list[Module]):
        self.table = ""
        self.metric_cls = get_module_dict_multiple(ColumnMetricFactory, modules)
        self.metrics: dict[str, dict[str | tuple[str], list[AbstractColumnMetric]]] = {}

    def fit(
        self,
        meta: Metadata,
        data: dict[str, LazyFrame],
    ):
        per_call = []
        per_call_meta = []

        # Create fitting tasks
        for name in meta.tables:
            ref_mgr = ReferenceManager(meta, name)

            for _, tables in LazyFrame.zip(data):
                per_call.append(
                    {
                        "name": name,
                        "meta": meta,
                        "ref": ref_mgr,
                        "tables": tables,
                        "metrics": self.metric_cls,
                    }
                )
                per_call_meta.append(name)

        # Process them
        out = process_in_parallel(
            _fit_column_metrics, per_call, desc="Fitting column metrics"
        )

        metrics = defaultdict(list)
        for chunk_metrics, table in zip(out, per_call_meta):
            metrics[table].append(chunk_metrics)

        self.metrics = {}
        for name in piter(meta.tables, desc="Reducing table modules for each table."):
            self.metrics[name] = reduce(_reduce_inner_2d, metrics[name])

        self.meta = meta
        self.fitted = True

    def preprocess(
        self,
        wrk: dict[str, LazyDataset],
        ref: dict[str, LazyDataset],
    ) -> dict[str, dict[str, dict[str | tuple[str], list[Any]]]]:

        per_call = []
        per_call_meta = []

        # Create preprocess tasks
        for name in self.meta.tables:
            ref_mgr = ReferenceManager(self.meta, name)

            for pid, (tables_wrk, tables_ref) in LazyDataset.zip([wrk, ref]).items():
                per_call.append(
                    {
                        "name": name,
                        "meta": self.meta,
                        "ref": ref_mgr,
                        "tables_wrk": tables_wrk,
                        "tables_ref": tables_ref,
                        "metrics": self.metrics,
                    }
                )
                per_call_meta.append((name, pid))

        out = process_in_parallel(
            _preprocess_metrics, per_call, desc="Preprocessing column metric synopsis"
        )

        # Fix by partition
        pre_dict = defaultdict(dict)
        for (name, pid), pre in zip(per_call_meta, out):
            pre_dict[name][pid] = pre
        return pre_dict

    def process(
        self,
        wrk: dict[str, LazyDataset],
        ref: dict[str, LazyDataset],
        syn: dict[str, LazyDataset],
        pre: dict[str, dict[str, dict[str | tuple[str], list[Any]]]],
    ) -> dict[str, dict[str | tuple[str], list[Any]]]:

        per_call = []
        per_call_meta = []

        # Create preprocess tasks
        for name in self.meta.tables:
            ref_mgr = ReferenceManager(self.meta, name)

            for pid, (tables_wrk, tables_ref, tables_syn) in LazyDataset.zip(
                [wrk, ref, syn]
            ).items():
                per_call.append(
                    {
                        "name": name,
                        "meta": self.meta,
                        "ref": ref_mgr,
                        "tables_wrk": tables_wrk,
                        "tables_ref": tables_ref,
                        "tables_syn": tables_syn,
                        "metrics": self.metrics,
                        "preprocess": pre[name][pid],
                    }
                )
                per_call_meta.append((name, pid))

        out = process_in_parallel(
            _process_metrics, per_call, desc="Processing column metric synopsis"
        )

        # Fix by partition
        proc_dict = defaultdict(list)
        for (name, pid), proc in zip(per_call_meta, out):
            proc_dict[name].append(proc)

        procs: dict[str, dict[str | tuple[str], list[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for name, table_metrics in self.metrics.items():
            for cols, col_metrics in table_metrics.items():
                for i, metric in enumerate(col_metrics):
                    d = [
                        proc_dict[name][j][cols][i] for j in range(len(proc_dict[name]))
                    ]
                    o = metric.combine(d)
                    procs[name][cols].append(o)
        return procs

    def unique_name(self) -> str:
        return f"{self.type}_{self.name}_{self.table}"


def fit_column_holder(
    modules: list[Module],
    metadata: Metadata,
    data: dict[str, LazyFrame],
):
    holder = ColumnMetricHolder(modules)
    holder.fit(meta=metadata, data=data)
    return holder


def fit_metric(
    fs: MetricFactory,
    metadata: Metadata,
    encoder: Encoder | dict[str, Encoder],
    data: dict[str, LazyDataset] | dict[str, dict[str, LazyDataset]],
):
    module = fs.build()
    if isinstance(fs.encodings, list):
        assert isinstance(encoder, dict)
        meta = {name: enc.get_metadata() for name, enc in encoder.items()}
        if "raw" in fs.encodings:
            meta['raw'] = metadata
    else:
        if fs.encodings == 'raw':
            meta = metadata
        else:
            assert isinstance(encoder, Encoder)
            meta = encoder.get_metadata()
    module.fit(meta=meta, data=data)
    return module


def log_metric(metric: Metric[Any, _SUMMARY], summary: _SUMMARY):
    from .utils.mlflow import mlflow_log_artifacts

    mlflow_log_artifacts(
        "metrics", metric.unique_name(), metric=metric, summary=summary
    )


__all__ = [
    "ColumnMetricFactory",
    "MetricFactory",
    "Metric",
    "Summaries",
    "ColumnMetric",
    "RefColumnMetric",
    "SeqColumnMetric",
    "Metric",
]
