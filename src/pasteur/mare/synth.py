from collections import defaultdict
import logging
from typing import Any, Type

import pandas as pd

from ..utils.data import LazyFrame

from .chains import (
    TableMeta,
    TablePartition,
    TableVersion,
    _calculate_stripped_meta,
    calculate_stripped_meta,
)

from ..attribute import Attributes, DatasetAttributes
from ..marginal import MarginalOracle
from ..marginal.numpy import TableSelector
from ..synth import Synth, make_deterministic
from ..utils import LazyDataset
from .unroll import ModelVersion, calculate_model_versions, gen_history

logger = logging.getLogger(__name__)


class MareModel:
    def fit(self, n: int, attrs: DatasetAttributes, oracle: MarginalOracle):
        ...

    def sample(
        self, index: pd.Index, hist: dict[TableSelector, pd.DataFrame]
    ) -> pd.DataFrame:
        ...


class MareSynth(Synth):
    name = "mare"
    type = "idx"

    def __init__(
        self,
        model_cls: Type[MareModel],
        marginal_mode: MarginalOracle.MODES = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        max_vers: int = 20,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.marginal_mode: MarginalOracle.MODES = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.max_vers = max_vers

        self.model_cls = model_cls

    def preprocess(self, meta: dict[str, Attributes], data: dict[str, LazyDataset]):
        logger.info(
            f"Calculating required model versions for dataset (max versions per model: {self.max_vers})..."
        )
        self.versions = calculate_model_versions(meta, data, self.max_vers)
        self.attrs = meta
        logger.info(f"Calculated {len(self.versions)} model versions.")

    def bake(self, data: dict[str, LazyDataset]):
        ...

    @make_deterministic
    def fit(self, data: dict[str, LazyDataset]):
        self.models: dict[ModelVersion, MareModel] = {}
        for i, (ver, (attrs, load)) in enumerate(self.versions.items()):
            logger.info(
                f"Fitting {i + 1:2d}/{len(self.versions)} '{'context' if ver.ctx else 'series'}' model for table '{ver.ver.name}'"
            )
            with MarginalOracle(data, attrs, load, mode=self.marginal_mode) as o:
                model = self.model_cls(**self.kwargs)
                model.fit(ver.ver.rows, attrs, o)
                self.models[ver] = model

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0):
        meta = calculate_stripped_meta(self.attrs)
        todo = list(self.models)

        inprogress_tables: dict[tuple[str, bool], list[pd.DataFrame]] = defaultdict(
            list
        )
        inprogress_ids: dict[tuple[str, bool], list[pd.DataFrame]] = defaultdict(list)
        tables: dict[tuple[str, bool], pd.DataFrame] = {}
        ids: dict[tuple[str, bool], pd.DataFrame] = {}

        while todo:
            # Find synthesizable candidate
            ver = None
            i = 0
            for i, candidate in enumerate(todo):
                if meets_requirements(candidate, todo):
                    ver = candidate
                    break
            assert ver is not None
            todo.pop(i)

            # Sample model
            table_ids, table = sample_model(ver, self.models[ver], n, meta, ids, tables)
            inprogress_tables[(ver.ver.name, ver.ctx)].append(table)
            inprogress_ids[(ver.ver.name, ver.ctx)].append(table_ids)

            # Concat finished tables
            for name, ctx in list(inprogress_tables):
                if is_finished(name, ctx, todo):
                    dfs = inprogress_tables.pop((name, ctx))
                    tables[(name, ctx)] = pd.concat(dfs, axis=1)
                    dfs = inprogress_ids.pop((name, ctx))
                    ids[(name, ctx)] = pd.concat(dfs, axis=1)

        return ids, tables


def sample_model(
    ver: ModelVersion,
    model: MareModel,
    n: int,
    meta: dict[str, TableMeta],
    ids: dict[tuple[str, bool], pd.DataFrame],
    tables: dict[tuple[str, bool], pd.DataFrame],
):

    if not ver.ver.parents:
        assert not ver.ctx, "Can't generate a context model for a primary relation"

        logger.info(
            f"Generating n={n} tuples for primary relation table '{ver.ver.name}'"
        )
        idx = pd.RangeIndex(n).rename(ver.ver.name)
        data = model.sample(idx, {})
        return pd.DataFrame(index=idx), data

    hist = gen_history(
        ver.ver.parents,
        LazyFrame.wrap(
            {name: table for (name, ctx), table in tables.items() if not ctx}
        ),
        LazyFrame.wrap({name: table for (name, ctx), table in ids.items() if not ctx}),
        meta,
    )
    tmeta = meta[ver.ver.name]
    if ver.ctx:
        # TODO: Fix id for multiple relations
        assert len(ver.ver.parents) == 1, "Multiple relations not supported for now"

        p = ver.ver.parents[0]
        if isinstance(p, TablePartition):
            p = p.table
        pname = p.name

        pids = ids[(pname, False)].reset_index()
        for name in hist:
            pids = pids[pids[name].isin(hist[name].index)]
        for name in list(hist):
            hist[name] = pids[[name]].join(hist[name], on=name, how='left').drop(columns=name)
        idx = pids.index

        table = model.sample(idx, hist)
        return pids, table
    else:
        pids = ids[(ver.ver.name, True)]

        if tmeta.unroll:
            pass
        elif tmeta.sequence:
            children = ver.ver.children
            assert children is not None

            

            for i in range(children):
                pass
            pids.index.name = ver.ver.name
        else:
            pass

    return pd.DataFrame(), pd.DataFrame()


def get_parents(version: TableVersion):
    out = set()
    for p in version.parents:
        if isinstance(p, TablePartition):
            out.add(p.table.name)
            out.update(get_parents(p.table))
        else:
            out.add(p.name)
            out.update(get_parents(p))
    return out


def is_finished(table: str, ctx: bool, todo: list[ModelVersion]):
    for ver in todo:
        if ctx == ver.ctx and table == ver.ver.name:
            return False

    return True


def meets_requirements(ver: ModelVersion, todo: list[ModelVersion]):
    parents = get_parents(ver.ver)

    # Find if a version that is not currently done is required
    for other in todo:

        # If the other version is a parent
        if other.ver.name in parents:
            return False

        # If this is a series model and a context model of this table is not synthesized
        if not ver.ctx and other.ver.name == ver.ver.name and other.ctx:
            return False

    return True
