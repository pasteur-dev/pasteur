""" This module provides the definitions for Metric Modules.
Metric modules can fit to a column, a table, or a whole View.
In each case, modules are instanciated as required (for columns one is instantiated
per column type, for tables one per table and View metrics are instantiated once)."""

import logging
from collections import defaultdict
from typing import Any, Callable, Generic, Literal, TypedDict, TypeVar, cast

import pandas as pd
from pasteur.attribute import SeqValue
from pasteur.metadata import ColumnMeta

from pasteur.utils import LazyDataset

from .attribute import SeqValue
from .encode import Encoder
from .metadata import ColumnMeta, Metadata
from .module import Module, ModuleClass, ModuleFactory, get_module_dict_multiple
from .table import (
    ReferenceManager,
    TableTransformer,
    _calc_joined_refs,
    _calc_unjoined_refs,
    _backref_cols,
)
from .transform import SeqTransformer
from .utils import LazyChunk, LazyDataset, LazyFrame, lazy_load_tables
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
        return self.name


class Summaries(Generic[A]):
    wrk: A
    ref: A
    syn: A

    def __init__(self, wrk: A, ref: A, syn: A | None = None) -> None:
        self.wrk = wrk
        self.ref = ref
        self.syn = cast(A, syn) # Skip lint check for syn being None

    def replace(self, **kwargs):
        params = {"wrk": self.wrk, "ref": self.ref, "syn": self.syn}
        params.update(kwargs)
        return type(self)(**params)


class RefColumnData(TypedDict):
    data: pd.Series | pd.DataFrame
    ref: pd.Series | pd.DataFrame


class SeqColumnData(TypedDict):
    data: pd.Series | pd.DataFrame
    ref: dict[str, pd.DataFrame]
    ids: pd.DataFrame
    seq: pd.Series


class AbstractColumnMetric(ModuleClass, Generic[_DATA, _INGEST, _SUMMARY]):
    type: Literal["col", "ref", "seq"] = "col"
    _factory = ColumnMetricFactory

    def fit(self, table: str, col: str | tuple[str, ...], data: _DATA):
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
    def fit(
        self,
        table: str,
        col: str | tuple[str, ...],
        seq_val: SeqValue | None,
        data: SeqColumnData,
    ):
        raise NotImplementedError()


B = TypeVar("B", bound="Any")


def _reduce_inner_2d(
    a: dict[str | tuple[str, ...], list[B]],
    b: dict[str | tuple[str, ...], list[B]],
):
    for key in a.keys():
        for i in range(len(a[key])):
            a[key][i].reduce(b[key][i])
    return a


def _get_sequence(
    name: str,
    meta: Metadata,
    trn: SeqTransformer,
    ids: pd.DataFrame,
    table: pd.DataFrame,
    get_parent: Callable[[str], pd.DataFrame],
) -> pd.Series | None:
    seq_name = meta[name].sequencer
    assert seq_name
    col = meta[name].cols[seq_name]
    ref_cols = _calc_unjoined_refs(name, get_parent, col.ref, table)
    res = trn.transform(table[seq_name], ref_cols, ids)
    assert len(res) == 3
    _, _, seq = res

    return seq


def _fit_column_metrics(
    name: str,
    meta: Metadata,
    ref: ReferenceManager,
    trn: SeqTransformer | None,
    tables: dict[str, LazyChunk],
    metrics: dict[str, list[ColumnMetricFactory]],
):
    get_table = lazy_load_tables(tables)
    table = get_table(name)
    seq_val = None
    seq = None

    if ref.table_has_reference():
        ids = ref.find_foreign_ids(name, get_table)

        if len(table.index.symmetric_difference(ids.index)):
            old_len = len(table)
            table = table.reindex(ids.index)
            logger.warn(
                f"There are missing ids for rows in {name}, dropping {old_len-len(table)}/{old_len} rows with missing ids."
            )

        if trn is not None:
            seq_val = trn.get_seq_value()
            seq = _get_sequence(name, meta, trn, ids, table, get_table)
    else:
        ids = None

    out: dict[str | tuple[str, ...], list[AbstractColumnMetric]] = defaultdict(list)
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
                m.fit(name, col_name, table[col_name])
            elif isinstance(m, RefColumnMetric):
                cref = col.ref
                ref_col = _calc_joined_refs(name, get_table, ids, cref, table) if cref else None
                m.fit(
                    name,
                    col_name,
                    RefColumnData(data=table[col_name], ref=ref_col), # type: ignore 
                )
            elif isinstance(m, SeqColumnMetric):
                ref_col = _calc_unjoined_refs(name, get_table, col.ref, table)
                assert ids is not None and seq is not None
                m.fit(
                    name,
                    col_name,
                    seq_val,
                    SeqColumnData(data=table[col_name], ref=ref_col, ids=ids, seq=seq),
                )
            else:
                assert False, f"Unknown column metric type: {type(m)}"

            out[col_name].append(m)

    return out


def _preprocess_metrics(
    name: str,
    meta: Metadata,
    ref: ReferenceManager,
    trn: SeqTransformer | None,
    tables_wrk: dict[str, LazyChunk],
    tables_ref: dict[str, LazyChunk],
    metrics: dict[str | tuple[str, ...], list[AbstractColumnMetric]],
):
    get_table_wrk = lazy_load_tables(tables_wrk)
    get_table_ref = lazy_load_tables(tables_ref)
    table_wrk = get_table_wrk(name)
    table_ref = get_table_ref(name)
    seq_wrk = None
    seq_ref = None

    if ref.table_has_reference():
        ids_wrk = ref.find_foreign_ids(name, get_table_wrk)
        ids_ref = ref.find_foreign_ids(name, get_table_ref)

        if len(table_wrk.index.symmetric_difference(ids_wrk.index)):
            old_len = len(table_wrk)
            table_wrk = table_wrk.reindex(ids_wrk.index)
            logger.warn(
                f"There are missing ids for rows in {name}, dropping {old_len-len(table_wrk)}/{old_len} rows with missing ids."
            )
        if len(table_ref.index.symmetric_difference(ids_ref.index)):
            old_len = len(table_ref)
            table_ref = table_ref.reindex(ids_ref.index)
            logger.warn(
                f"There are missing ids for rows in {name}, dropping {old_len-len(table_ref)}/{old_len} rows with missing ids."
            )

        if trn is not None:
            seq_wrk = _get_sequence(name, meta, trn, ids_wrk, table_wrk, get_table_wrk)
            seq_ref = _get_sequence(name, meta, trn, ids_ref, table_ref, get_table_ref)
    else:
        ids_wrk = None
        ids_ref = None

    out = defaultdict(list)
    for col_name, ms in metrics.items():
        for m in ms:
            col = meta[name][col_name]
            cref = col.ref
            if isinstance(m, ColumnMetric):
                prec = m.preprocess(
                    table_wrk[col_name],
                    table_ref[col_name],
                )
            elif isinstance(m, RefColumnMetric):
                prec = m.preprocess(
                    RefColumnData(
                        data=table_wrk[col_name],
                        ref=_calc_joined_refs(
                            name, get_table_wrk, ids_ref, cref, table_wrk
                        ) if cref else None, # type: ignore
                    ),
                    RefColumnData(
                        data=table_ref[col_name],
                        ref=_calc_joined_refs(
                            name, get_table_ref, ids_ref, cref, table_ref
                        ) if cref else None, # type: ignore
                    ),
                )
            elif isinstance(m, SeqColumnMetric):
                assert (
                    ids_wrk is not None
                    and seq_wrk is not None
                    and ids_ref is not None
                    and seq_ref is not None
                )
                prec = m.preprocess(
                    SeqColumnData(
                        data=table_wrk[col_name],
                        ref=_calc_unjoined_refs(
                            name, get_table_wrk, col.ref, table_wrk
                        ),
                        ids=ids_wrk,
                        seq=seq_wrk,
                    ),
                    SeqColumnData(
                        data=table_ref[col_name],
                        ref=_calc_unjoined_refs(
                            name, get_table_ref, col.ref, table_ref
                        ),
                        ids=ids_ref,
                        seq=seq_ref,
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
    trn: SeqTransformer | None,
    tables_wrk: dict[str, LazyChunk],
    tables_ref: dict[str, LazyChunk],
    tables_syn: dict[str, LazyChunk],
    metrics: dict[str | tuple[str, ...], list[AbstractColumnMetric]],
    preprocess: dict[str | tuple[str, ...], list[Any]],
):
    get_table_wrk = lazy_load_tables(tables_wrk)
    get_table_ref = lazy_load_tables(tables_ref)
    get_table_syn = lazy_load_tables(tables_syn)
    table_wrk = get_table_wrk(name)
    table_ref = get_table_ref(name)
    table_syn = get_table_syn(name)
    seq_wrk = None
    seq_ref = None
    seq_syn = None

    if ref.table_has_reference():
        ids_wrk = ref.find_foreign_ids(name, get_table_wrk)
        ids_ref = ref.find_foreign_ids(name, get_table_ref)
        ids_syn = ref.find_foreign_ids(name, get_table_syn)

        if len(table_wrk.index.symmetric_difference(ids_wrk.index)):
            old_len = len(table_wrk)
            table_wrk = table_wrk.reindex(ids_wrk.index)
            logger.warn(
                f"There are missing ids for rows in {name}, dropping {old_len-len(table_wrk)}/{old_len} rows with missing ids."
            )
        if len(table_ref.index.symmetric_difference(ids_ref.index)):
            old_len = len(table_ref)
            table_ref = table_ref.reindex(ids_ref.index)
            logger.warn(
                f"There are missing ids for rows in {name}, dropping {old_len-len(table_ref)}/{old_len} rows with missing ids."
            )
        if len(table_syn.index.symmetric_difference(ids_syn.index)):
            old_len = len(table_syn)
            table_syn = table_syn.reindex(ids_syn.index)
            logger.warn(
                f"There are missing ids for rows in {name}, dropping {old_len-len(table_syn)}/{old_len} rows with missing ids."
            )

        if trn is not None:
            seq_wrk = _get_sequence(name, meta, trn, ids_wrk, table_wrk, get_table_wrk)
            seq_ref = _get_sequence(name, meta, trn, ids_ref, table_ref, get_table_ref)
            seq_syn = _get_sequence(name, meta, trn, ids_syn, table_syn, get_table_syn)
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
                        data=table_wrk[col_name],
                        ref=_calc_joined_refs(
                            name, get_table_wrk, ids_wrk, col.ref, table_wrk
                        ) if col.ref else None, # type: ignore
                    ),
                    RefColumnData(
                        data=table_ref[col_name],
                        ref=_calc_joined_refs(
                            name, get_table_ref, ids_ref, col.ref, table_ref
                        ) if col.ref else None, # type: ignore
                    ),
                    RefColumnData(
                        data=table_syn[col_name],
                        ref=_calc_joined_refs(
                            name, get_table_syn, ids_syn, col.ref, table_syn
                        ) if col.ref else None, # type: ignore
                    ),
                    prec,
                )
            elif isinstance(m, SeqColumnMetric):
                assert (
                    ids_wrk is not None
                    and seq_wrk is not None
                    and ids_ref is not None
                    and seq_ref is not None
                    and ids_syn is not None
                    and seq_syn is not None
                )
                proc = m.process(
                    SeqColumnData(
                        data=table_wrk[col_name],
                        ref=_calc_unjoined_refs(
                            name, get_table_wrk, col.ref, table_wrk
                        ),
                        ids=ids_wrk,
                        seq=seq_wrk,
                    ),
                    SeqColumnData(
                        data=table_ref[col_name],
                        ref=_calc_unjoined_refs(
                            name, get_table_ref, col.ref, table_ref
                        ),
                        ids=ids_ref,
                        seq=seq_ref,
                    ),
                    SeqColumnData(
                        data=table_syn[col_name],
                        ref=_calc_unjoined_refs(
                            name, get_table_syn, col.ref, table_syn
                        ),
                        ids=ids_syn,
                        seq=seq_syn,
                    ),
                    prec,
                )
            else:
                assert False, f"Unknown column metric type: {type(m)}"

            out[col_name].append(proc)

    return out


def name_add_prefix(col: str | tuple[str, ...], suffix: str):
    if isinstance(col, str):
        return (col, suffix)
    return (*col, suffix)


def name_style_fn(col: str | tuple[str, ...], ext: str | None = None):
    if not ext:
        ext = ""
    if isinstance(col, str):
        return col + ext
    return "_".join(col) + ext


def name_style_title(col: str | tuple[str, ...], title: str | None = None):
    if isinstance(col, str):
        if title:
            return f"{col.capitalize()} {title}"
        else:
            return col.capitalize()
    else:
        if title:
            return f"{', '.join([c.capitalize() for c in col])} {title}"
        else:
            return ", ".join([c.capitalize() for c in col])


class ColumnMetricHolder(
    Metric[
        dict[str, list[dict[str | tuple[str, ...], list[Any]]]],
        dict[str, dict[str | tuple[str, ...], list[Any]]],
    ]
):
    name = "cols"
    encodings = "raw"
    metrics: dict[str, dict[str | tuple[str, ...], list[AbstractColumnMetric]]]

    def __init__(self, modules: list[Module]):
        self.table = ""
        self.metric_cls = get_module_dict_multiple(
            ColumnMetricFactory,
            [*modules, SeqMetricWrapper.get_factory(modules=modules)],
        )
        self.metrics = {}

    def fit(
        self,
        meta: Metadata,
        trns: dict[str, TableTransformer],
        data: dict[str, LazyFrame],
    ):
        per_call = []
        per_call_meta = []
        self.seqs = {
            k: v.get_sequencer()
            for k, v in trns.items()
            if v.get_sequencer() is not None
        }

        # Create fitting tasks
        for name in meta.tables:
            ref_mgr = ReferenceManager(meta, name)

            for tables in LazyFrame.zip_values(data):
                per_call.append(
                    {
                        "name": name,
                        "meta": meta,
                        "ref": ref_mgr,
                        "trn": self.seqs.get(name, None),
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
        for name in piter(
            meta.tables, desc="Reducing table modules for each table.", leave=False
        ):
            self.metrics[name] = reduce(_reduce_inner_2d, metrics[name])

        self.meta = meta
        self.fitted = True

    def preprocess(
        self,
        wrk: dict[str, LazyDataset],
        ref: dict[str, LazyDataset],
    ) -> dict[str, list[dict[str | tuple[str, ...], list[Any]]]]:

        per_call = []
        per_call_meta = []

        # Create preprocess tasks
        for name in self.meta.tables:
            ref_mgr = ReferenceManager(self.meta, name)

            for tables_wrk, tables_ref in LazyDataset.zip_values([wrk, ref]):
                per_call.append(
                    {
                        "name": name,
                        "meta": self.meta,
                        "ref": ref_mgr,
                        "trn": self.seqs.get(name, None),
                        "tables_wrk": tables_wrk,
                        "tables_ref": tables_ref,
                        "metrics": self.metrics[name],
                    }
                )
                per_call_meta.append(name)

        out = process_in_parallel(
            _preprocess_metrics, per_call, desc="Preprocessing column metric synopsis"
        )

        # Fix by partition
        pre_dict = defaultdict(list)
        for name, pre in zip(per_call_meta, out):
            pre_dict[name].append(pre)
        return pre_dict

    def process(
        self,
        wrk: dict[str, LazyDataset],
        ref: dict[str, LazyDataset],
        syn: dict[str, LazyDataset],
        pre: dict[str, list[dict[str | tuple[str, ...], list[Any]]]],
    ) -> dict[str, dict[str | tuple[str, ...], list[Any]]]:

        per_call = []
        per_call_meta = []

        # Create preprocess tasks
        for name in self.meta.tables:
            ref_mgr = ReferenceManager(self.meta, name)

            for i, (tables_wrk, tables_ref, tables_syn) in enumerate(
                LazyDataset.zip_values([wrk, ref, syn])
            ):
                per_call.append(
                    {
                        "name": name,
                        "meta": self.meta,
                        "ref": ref_mgr,
                        "trn": self.seqs.get(name, None),
                        "tables_wrk": tables_wrk,
                        "tables_ref": tables_ref,
                        "tables_syn": tables_syn,
                        "metrics": self.metrics[name],
                        "preprocess": pre[name][i],
                    }
                )
                per_call_meta.append(name)

        out = process_in_parallel(
            _process_metrics, per_call, desc="Processing column metric synopsis"
        )

        # Fix by partition
        proc_dict = defaultdict(list)
        for name, proc in zip(per_call_meta, out):
            proc_dict[name].append(proc)

        procs: dict[str, dict[str | tuple[str, ...], list[Any]]] = defaultdict(
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
        return dict(procs)

    def visualise(
        self, data: dict[str, dict[str, dict[str | tuple[str, ...], list[Any]]]]
    ):
        for table, table_metrics in self.metrics.items():
            for col_name, col_metrics in table_metrics.items():
                for i, metric in enumerate(col_metrics):
                    metric.visualise(
                        {n: d[table][col_name][i] for n, d in data.items()}
                    )

    def summarize(
        self, data: dict[str, dict[str, dict[str | tuple[str, ...], list[Any]]]]
    ):
        for table, table_metrics in self.metrics.items():
            for col_name, col_metrics in table_metrics.items():
                for i, metric in enumerate(col_metrics):
                    metric.summarize(
                        {n: d[table][col_name][i] for n, d in data.items()}
                    )


def fit_column_holder(
    modules: list[Module],
    metadata: Metadata,
    trns: dict[str, TableTransformer],
    data: dict[str, LazyFrame],
):
    holder = ColumnMetricHolder(modules)
    holder.fit(meta=metadata, trns=trns, data=data)
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
            meta["raw"] = metadata
    else:
        if fs.encodings == "raw":
            meta = metadata
        else:
            assert isinstance(encoder, Encoder)
            meta = encoder.get_metadata()
    module.fit(meta=meta, data=data)
    return module


class SeqMetricWrapper(SeqColumnMetric):
    name = "seq"
    mode: Literal["dual", "single", "notrn"]

    def __init__(
        self,
        modules: list[Module],
        visual: dict[str, Any] | None = None,
        seq: dict[str, Any] | None = None,
        ctx: dict[str, Any] | None = None,
        seq_col: str | None = None,
        ctx_to_ref: dict[str, str] | None = None,
        order: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_col_ref = seq_col
        self.ctx_to_ref = ctx_to_ref
        self.order = order

        self.seq = []
        self.ctx = []
        self.visual = []

        # Load metrics
        # Use three modes
        if seq is not None and ctx is not None:
            self.mode = "dual"
        elif seq is not None:
            self.mode = "single"
        else:
            self.mode = "notrn"

        if seq is not None:
            seq_kwargs = seq.copy()
            seq_type = seq_kwargs.pop("type")
            seq_kwargs["nullable"] = True
            seq_frs = get_module_dict_multiple(ColumnMetricFactory, modules).get(
                cast(str, seq_type), []
            )
            self.seq = [f.build(**seq_kwargs) for f in seq_frs]

        if ctx is not None:
            ctx_kwargs = ctx.copy()
            ctx_type = ctx_kwargs.pop("type")
            ctx_frs = get_module_dict_multiple(ColumnMetricFactory, modules).get(
                cast(str, ctx_type), []
            )
            self.ctx = [f.build(**ctx_kwargs) for f in ctx_frs]

        if visual is not None:
            visual_kwargs = visual.copy()
            visual_type = visual_kwargs.pop("type")
            visual_frs = get_module_dict_multiple(ColumnMetricFactory, modules).get(
                cast(str, visual_type), []
            )
            self.visual = [f.build(**visual_kwargs) for f in visual_frs]

    def fit(
        self,
        table: str,
        col: str | tuple[str, ...],
        seq_val: SeqValue | None,
        data: SeqColumnData,
    ):
        seq = data["seq"]
        assert (
            seq_val is not None and seq is not None
        ), "Wrapping RefTransformers requires sequenced data, fill in `sequencer` for the table."

        self.table = table
        self.col = col
        self.max_len = cast(int, seq.max()) + 1
        self.parent = seq_val.table

        match self.mode:
            case "dual":
                if self.ctx:
                    ctx_in = _wrap_get_data_ctx(self.parent, **data)
                    for c in self.ctx:
                        c.fit(
                            self.table,
                            name_add_prefix(self.col, "ctx"),
                            ctx_in,
                        )

                # Data series is all rows where seq > 0 (skip initial)
                if self.seq:
                    seq_in = _wrap_get_data_seq_dual(self.parent, **data)
                    for c in self.seq:
                        c.fit(
                            self.table,
                            name_add_prefix(self.col, "seq"),
                            seq_in,
                        )
            case "single":
                if self.seq:
                    seq_in = _wrap_get_data_seq_single(
                        self.parent, **data, ctx_to_ref=self.ctx_to_ref
                    )
                    for c in self.seq:
                        c.fit(
                            self.table,
                            name_add_prefix(self.col, "seq"),
                            seq_in,
                        )
            case "notrn":
                pass

        for c in self.visual:
            if isinstance(c, SeqColumnMetric):
                c.fit(self.table, self.col, seq_val, data)
            else:
                c.fit(self.table, self.col, data)

    def preprocess(self, wrk: SeqColumnData, ref: SeqColumnData) -> Any | None:
        pre_viz = []
        for c in self.visual:
            pre_viz.append(c.preprocess(wrk, ref))

        match self.mode:
            case "dual":
                pre_ctx = []
                if self.ctx:
                    data_wrk = _wrap_get_data_ctx(self.parent, **wrk)
                    data_ref = _wrap_get_data_ctx(self.parent, **ref)
                    for c in self.ctx:
                        pre_ctx.append(c.preprocess(data_wrk, data_ref))

                pre_seq = []
                if self.seq:
                    data_wrk = _wrap_get_data_seq_dual(self.parent, **wrk)
                    data_ref = _wrap_get_data_seq_dual(self.parent, **ref)
                    for c in self.seq:
                        pre_seq.append(c.preprocess(data_wrk, data_ref))

                return (pre_viz, pre_ctx, pre_seq)
            case "single":
                pre_seq = []
                if self.seq:
                    data_wrk = _wrap_get_data_seq_single(
                        self.parent, **wrk, ctx_to_ref=self.ctx_to_ref
                    )
                    data_ref = _wrap_get_data_seq_single(
                        self.parent, **ref, ctx_to_ref=self.ctx_to_ref
                    )
                    for c in self.seq:
                        pre_seq.append(c.preprocess(data_wrk, data_ref))
                return (pre_viz, pre_seq)
            case "notrn":
                return (pre_viz,)

        assert False

    def process(
        self, wrk: SeqColumnData, ref: SeqColumnData, syn: SeqColumnData, pre: Any
    ) -> Any:
        proc_viz = []
        for c, p in zip(self.visual, pre[0]):
            proc_viz.append(c.process(wrk, ref, syn, p))

        match self.mode:
            case "dual":
                proc_ctx = []
                if self.ctx:
                    data_wrk = _wrap_get_data_ctx(self.parent, **wrk)
                    data_ref = _wrap_get_data_ctx(self.parent, **ref)
                    data_syn = _wrap_get_data_ctx(self.parent, **syn)
                    for c, p in zip(self.ctx, pre[1]):
                        proc_ctx.append(c.process(data_wrk, data_ref, data_syn, p))

                proc_seq = []
                if self.seq:
                    data_wrk = _wrap_get_data_seq_dual(self.parent, **wrk)
                    data_ref = _wrap_get_data_seq_dual(self.parent, **ref)
                    data_syn = _wrap_get_data_seq_dual(self.parent, **syn)
                    for c, p in zip(self.seq, pre[2]):
                        proc_seq.append(c.process(data_wrk, data_ref, data_syn, p))

                return (proc_viz, proc_ctx, proc_seq)
            case "single":
                proc_seq = []
                if self.seq:
                    data_wrk = _wrap_get_data_seq_single(
                        self.parent, **wrk, ctx_to_ref=self.ctx_to_ref
                    )
                    data_ref = _wrap_get_data_seq_single(
                        self.parent, **ref, ctx_to_ref=self.ctx_to_ref
                    )
                    data_syn = _wrap_get_data_seq_single(
                        self.parent, **syn, ctx_to_ref=self.ctx_to_ref
                    )
                    for c, p in zip(self.seq, pre[1]):
                        proc_seq.append(c.process(data_wrk, data_ref, data_syn, p))
                return (proc_viz, proc_seq)
            case "notrn":
                return (proc_viz,)

        assert False

    def combine(self, summaries: list) -> Any:
        sum_viz = []
        for i, c in enumerate(self.visual):
            sum_viz.append(c.combine([s[0][i] for s in summaries]))

        match self.mode:
            case "dual":
                sum_ctx = []
                for i, c in enumerate(self.ctx):
                    sum_ctx.append(c.combine([s[1][i] for s in summaries]))
                sum_seq = []
                for i, c in enumerate(self.seq):
                    sum_seq.append(c.combine([s[2][i] for s in summaries]))
                return (sum_viz, sum_ctx, sum_seq)
            case "single":
                sum_seq = []
                for i, c in enumerate(self.seq):
                    sum_seq.append(c.combine([s[1][i] for s in summaries]))
                return (sum_viz, sum_seq)
            case "notrn":
                return (sum_viz,)

        assert False

    def _distr(self, data: dict[str, Any], fun: str):
        for i, c in enumerate(self.visual):
            getattr(c, fun)({k: v[0][i] for k, v in data.items()})

        match self.mode:
            case "dual":
                for i, c in enumerate(self.ctx):
                    getattr(c, fun)({k: v[1][i] for k, v in data.items()})
                for i, c in enumerate(self.seq):
                    getattr(c, fun)({k: v[2][i] for k, v in data.items()})
            case "single":
                for i, c in enumerate(self.seq):
                    getattr(c, fun)({k: v[1][i] for k, v in data.items()})
            case "notrn":
                pass

    def visualise(self, data: dict[str, Any]):
        return self._distr(data, "visualise")

    def summarize(self, data: dict[str, Any]):
        return self._distr(data, "summarize")


def _wrap_get_data_ctx(
    parent: str,
    data: pd.Series | pd.DataFrame,
    ref: dict[str, pd.DataFrame],
    ids: pd.DataFrame,
    seq: pd.Series,
):
    ctx_data = (
        ids[[parent]]
        .join(data[seq == 0], how="right")
        .drop_duplicates(subset=[parent])
        .set_index(parent)
    )
    if isinstance(data, pd.Series):
        ctx_data = ctx_data[next(iter(ctx_data))]

    ctx_ref = None
    if ref:
        ctx_ref = ids.drop_duplicates(subset=[parent])
        for name, ref_table in ref.items():
            ctx_ref = ctx_ref.join(ref_table, on=name, how="left")
        ctx_ref = ctx_ref.set_index(parent).drop(
            columns=[d for d in ids.columns if d != parent]
        )
        if ctx_ref.shape[1] == 1:
            ctx_ref = ctx_ref[next(iter(ctx_ref))]

    return {"data": ctx_data, "ref": ctx_ref}


def _wrap_get_data_seq_dual(
    parent: str,
    data: pd.Series | pd.DataFrame,
    ref: dict[str, pd.DataFrame],
    ids: pd.DataFrame,
    seq: pd.Series,
):
    # Data series is all rows where seq > 0 (skip initial)
    ref_df = _backref_cols(ids, seq, data, parent)
    return {"data": data[seq > 0], "ref": ref_df}


def _wrap_get_data_seq_single(
    parent: str,
    data: pd.Series | pd.DataFrame,
    ref: dict[str, pd.DataFrame],
    ids: pd.DataFrame,
    seq: pd.Series,
    ctx_to_ref,
):
    ref_df = _backref_cols(ids, seq, data, parent)
    if ref:
        ctx_ref = ids[seq == 0].drop_duplicates(subset=[parent])
        for name, ref_table in ref.items():
            ctx_ref = ctx_ref.join(ref_table, on=name, how="left")
        ctx_ref = ctx_ref.drop(columns=ids.columns)

        if ctx_ref.shape[1] == 1:
            ctx_ref = ctx_ref[next(iter(ctx_ref))]

        if isinstance(ref_df, pd.Series) and isinstance(ctx_ref, pd.Series):
            ref_df = pd.concat([ctx_ref, ref_df])
        elif isinstance(ref_df, pd.DataFrame) and isinstance(ctx_ref, pd.DataFrame):
            if ctx_to_ref:
                ctx_ref = ctx_ref.rename(columns=ctx_to_ref)
            ref_df = pd.concat([ctx_ref, ref_df], axis=0)
            assert (
                ref_df.shape[1] == ctx_ref.shape[1]
            ), f"Parent columns not joined correctly to reference ones. If they have different names, pass in `ctx_to_ref` with names mapping them to parents"
        else:
            assert (
                False
            ), "fixme: mismatched reference column counts. If single column transformer, both should be series, otherwise both should be dataframes"

    return {"data": data, "ref": ref_df}


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
