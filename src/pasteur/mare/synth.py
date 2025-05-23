import logging
from collections import defaultdict
from typing import Any, Type, cast

import pandas as pd
import numpy as np

from ..attribute import (
    Attributes,
    CatValue,
    DatasetAttributes,
    SeqAttributes,
    get_dtype,
)
from ..hierarchy import rebalance_attributes
from ..marginal import MarginalOracle, counts_preprocess
from ..marginal.numpy import TableSelector
from ..synth import Synth, make_deterministic
from ..utils import LazyDataset
from ..utils.data import LazyFrame, tables_to_data
from .chains import (
    TableMeta,
    TablePartition,
    TableVersion,
    calculate_stripped_meta,
)
from .unroll import (
    ModelVersion,
    calculate_model_versions,
    gen_history,
    recurse_unroll_attr,
    get_parents as get_parents_upwards,
)
from .privacy import calc_privacy_budgets

logger = logging.getLogger(__name__)


class MareModel:
    def fit(
        self, n: int, table: str, attrs: DatasetAttributes, oracle: MarginalOracle
    ): ...

    def sample(
        self, index: pd.Index, hist: dict[TableSelector, pd.DataFrame]
    ) -> pd.DataFrame: ...


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
        rebalance: bool = False,
        etotal: float | None = None,
        no_seq: bool = False,
        no_hist: bool = False,
        max_sens: int | None = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.marginal_mode: MarginalOracle.MODES = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.max_vers = max_vers
        self.rebalance = rebalance
        self.etotal = etotal
        self.no_seq = no_seq
        self.no_hist = no_hist
        self.max_sens = max_sens

        self.model_cls = model_cls

    def preprocess(self, meta: dict[str, Attributes], data: dict[str, LazyDataset]):
        logger.info(
            f"Calculating required model versions for dataset (max versions per model: {self.max_vers})..."
        )

        if self.rebalance:
            with MarginalOracle(
                data,  # type: ignore
                meta,  # type: ignore
                mode=self.marginal_mode,
                min_chunk_size=self.marginal_min_chunk,
                max_worker_mult=self.marginal_worker_mult,
                preprocess=counts_preprocess,
            ) as o:
                counts = o.get_counts(desc="Calculating counts for column rebalancing")

            self.counts = counts
            new_meta = {
                k: rebalance_attributes(
                    counts[k],
                    v,
                    unbounded_dp=self.kwargs.get("unbounded_dp", True),
                    **self.kwargs,
                )
                for k, v in meta.items()
            }
        else:
            new_meta = meta

        if self.no_seq:
            logger.warning("Disabling order by dirty overwriting meta")
            for table in new_meta.values():
                for attr in table.values():
                    for val in attr.vals.values():
                        if hasattr(val, "order"):
                            setattr(val, "order", 0)

        self.versions = calculate_model_versions(
            new_meta, data, self.max_vers, no_hist=self.no_hist
        )
        # TODO: Find better sampling strategy. Pick a top level table and
        # use its stats for now.
        for ver in self.versions:
            if not ver.ver.parents:
                self._n = ver.ver.rows
                self._partitions = len(data[ver.ver.name])
        self.attrs = meta
        logger.info(f"Calculated {len(self.versions)} model versions.")

    def bake(self, data: dict[str, LazyDataset]): ...

    def __str__(self) -> str:
        out = ""
        out += (
            f"MARE synthesis algorithm encapsulating model type '{self.model_cls}'.\n"
        )
        out += f"Created {len(self.models)} models, which synthesize the following {len(set(v.ver.name for v in self.models))} tables:\n"
        out += str(sorted(set(v.ver.name for v in self.models))) + "\n"

        seen = defaultdict(lambda: 1)
        for i, (ver, model) in enumerate(self.models.items()):
            mtype = "context" if ver.ctx else "series"
            n_versions = len(
                [
                    other
                    for other in self.models
                    if other.ctx == ver.ctx and other.ver.name == ver.ver.name
                ]
            )
            out += f"\nModel {i+1:02d}/{len(self.versions):02d}: '{mtype}' model for table '{ver.ver.name}' (version {seen[(ver.ver.name, ver.ctx)]:2d}/{n_versions})\n"
            seen[(ver.ver.name, ver.ctx)] += 1
            out += str(model)
            out += "\n"

        return out

    @make_deterministic
    def fit(self, data: dict[str, LazyDataset]):
        self.models: dict[ModelVersion, MareModel] = {}
        if self.etotal:
            budgets, sensitivities = calc_privacy_budgets(
                self.etotal,
                self.versions,
                params={
                    "rake": self.kwargs.get("rake", True),
                    "no_hist": self.no_hist,
                    "no_seq": self.no_seq,
                    "max_sens": self.max_sens,
                },
            )
        else:
            budgets = None
            sensitivities = None

        for i, (ver, (attrs, load)) in enumerate(self.versions.items()):
            logger.info(
                f"Fitting {i + 1:2d}/{len(self.versions)} '{'context' if ver.ctx else 'series'}' model for table '{ver.ver.name}'"
            )
            with MarginalOracle(
                data,
                attrs,
                load,
                mode=self.marginal_mode,
                max_worker_mult=self.marginal_worker_mult,
                min_chunk_size=self.marginal_min_chunk,
            ) as o:
                kwargs = dict(self.kwargs)
                if budgets and sensitivities:
                    adj_budget = budgets[ver] / sensitivities[ver]
                    logger.info(
                        f"Using privacy budget {budgets[ver]:.5f}/{self.etotal:.5f} with sensitivity {sensitivities[ver]}, adjusted to e_adj = {adj_budget:.5f}"
                    )
                    kwargs["etotal"] = adj_budget
                # if sensitivities:
                #     prev = kwargs["minimum_cutoff"] if "minimum_cutoff" in kwargs else 3
                #     if prev:
                #         kwargs["minimum_cutoff"] = prev * sensitivities[ver]
                #         logger.info(
                #             f"Adjusting minimum cutoff from {prev} to {kwargs['minimum_cutoff']} due to sensitivity {sensitivities[ver]}."
                #         )

                model = self.model_cls(**kwargs)
                model.fit(ver.ver.rows, ver.ver.name, attrs, o)
                self.models[ver] = model

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0):
        part_id = i
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
            table_ids, table = sample_model(
                ver, self.models[ver], self.versions[ver][0], n, meta, ids, tables
            )
            inprogress_tables[(ver.ver.name, ver.ctx)].append(table)
            inprogress_ids[(ver.ver.name, ver.ctx)].append(table_ids)

            # Concat finished tables
            for name, ctx in list(inprogress_tables):
                if is_finished(name, ctx, todo):
                    # TODO: Cleanup botch.
                    # Check if there are inconsistencies in the columns
                    # If there are, convert to nullable integers
                    dfs = inprogress_tables.pop((name, ctx))
                    missing_cols = False
                    cols = set()
                    for df in dfs:
                        cols.update(df.columns)
                        if cols.difference(df.columns):
                            missing_cols = True
                    if missing_cols:
                        new_dfs = []
                        for df in dfs:
                            new_dfs.append(df.convert_dtypes())
                        dfs = new_dfs

                    new_table = (
                        pd.concat(dfs, axis=0, ignore_index=True)
                        .rename_axis(name)
                        .fillna(0)
                    )
                    # After creating new table, convert nullable integers back
                    # to non-nullable.
                    if missing_cols:
                        new_table = new_table.astype(
                            {
                                k: str(v).lower()
                                for k, v in new_table.dtypes.items()
                                if str(v).startswith("UI")
                            }
                        )

                    tables[(name, ctx)] = new_table
                    dfs = inprogress_ids.pop((name, ctx))
                    ids[(name, ctx)] = pd.concat(
                        dfs, axis=0, ignore_index=True
                    ).rename_axis(name)

                    # Sanity index check
                    assert tables[(name, ctx)].index.equals(ids[(name, ctx)].index)

        # Remove context tables
        out_ids = {name: df for (name, ctx), df in ids.items() if not ctx}
        out_tables = {name: df for (name, ctx), df in tables.items() if not ctx}

        return {
            k: {f"{part_id:03d}": v}
            for k, v in tables_to_data(out_ids, out_tables).items()
        }


def sample_model(
    ver: ModelVersion,
    model: MareModel,
    attrs: DatasetAttributes,
    n: int,
    meta: dict[str, TableMeta],
    ids: dict[tuple[str, bool], pd.DataFrame],
    tables: dict[tuple[str, bool], pd.DataFrame],
):
    PARENT_KEY = "_key_jhudfghkj"

    if not ver.ver.parents:
        assert not ver.ctx, "Can't generate a context model for a primary relation"

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
        # Check whether we are in a sequence
        seq_repeat = tmeta.seq_repeat
        parent, gparent, ggparent = get_parents_upwards(ver.ver)
        ctx_seq = ver.ctx and (
            seq_repeat and gparent and ggparent and meta[gparent].sequence
        )
        if ver.ctx and seq_repeat and gparent and ggparent and meta[gparent].sequence:
            stable = gparent
            ptable = ggparent
            ctx_seq = True
        elif (
            ver.ctx and not seq_repeat and parent and gparent and meta[parent].sequence
        ):
            stable = parent
            ptable = gparent
            ctx_seq = True
        else:
            stable = None
            ptable = None
            ctx_seq = False

        if ctx_seq:
            assert stable and ptable and parent
            # TODO: Fix id for multiple relations
            assert len(ver.ver.parents) == 1, "Multiple relations not supported for now"

            children = ver.ver.max_len or ver.ver.children
            assert children is not None

            seq_attrs = attrs[ver.ver.name]
            assert isinstance(seq_attrs, SeqAttributes)
            order = seq_attrs.order
            name = ver.ver.name

            seq = tables[(stable, False)][seq_attrs.seq.name]
            if stable == parent:
                tids = ids[(stable, False)]
            else:
                tids = (
                    ids[(parent, False)]
                    .drop_duplicates()
                    .reset_index()
                    .set_index(stable)
                )
            tids[stable] = tids.index  # dupe the key to have it as a column

            # if stable != parent:
            #     # grab some parent ids

            # Find values and their domains for placeholder history
            vals = {}
            if seq_attrs.hist:
                for attr in next(iter(seq_attrs.hist.values())).values():
                    # if attr.common:
                    #     vals[attr.common.name] = attr.common.get_domain(0)
                    for val_name, val in attr.vals.items():
                        if isinstance(val, CatValue):
                            vals[val_name] = val.get_domain(0)

            # Create sample history
            sampled = []
            for i in range(seq.max() + 1):
                idx = seq[seq == i].index
                if not len(idx):
                    break

                # Add seq value to history
                new_hist = {
                    n: h.loc[tids.loc[idx, n[0] if isinstance(n, tuple) else n]]
                    for n, h in hist.items()
                }
                new_hist[ver.ver.name] = pd.DataFrame(
                    {seq_attrs.seq.name: i},
                    index=idx,
                    dtype="uint8",
                )

                # Add placeholder if not enough history is sampled
                if i < order:
                    placeholder = pd.DataFrame(
                        {k: i for k in vals}, index=idx, dtype="uint8"
                    )
                    for j in range(i, order):
                        new_hist[(name, j)] = placeholder

                # Add the rest of history
                for j in range(min(len(sampled), order)):
                    new_hist[(name, j)] = (
                        tids.loc[idx, [ptable]]
                        .merge(
                            (
                                sampled[-j - 1].astype(
                                    {k: get_dtype(v + j + 1) for k, v in vals.items()}
                                )
                                + j
                                + 1
                            ).join(tids[[ptable]]),
                            on=ptable,
                        )
                        .drop(columns=ptable)
                    )

                table = model.sample(idx, new_hist)
                table[seq_attrs.seq.name] = i
                sampled.append(table)

            # Concat to single table with a context index column
            table = pd.concat(
                [s.reset_index(names=PARENT_KEY) for s in sampled], axis=0
            ).reset_index(drop=True)

            # Join context ids to the new index to form the new ids table
            pids = ids[(parent, False)]
            pids[parent] = pids.index

            # Reindex to add rows that are the same
            if stable != parent:
                table = (
                    ids[(parent, False)][[stable]].merge(
                        table, left_on=stable, right_on=PARENT_KEY
                    )
                ).drop(columns=[stable, PARENT_KEY])
            else:
                table = table.drop(columns=PARENT_KEY)

            return pids, table
        else:
            # TODO: Fix id for multiple relations
            assert len(ver.ver.parents) == 1, "Multiple relations not supported for now"

            p = ver.ver.parents[0]
            if isinstance(p, TablePartition):
                p = p.table
            pname = p.name

            og_ids = ids[(pname, False)].reset_index()
            for sel in hist:
                name = sel[0] if isinstance(sel, tuple) else sel
                og_ids = og_ids[og_ids[name].isin(hist[sel].index)]

            gen_first = tmeta.seq_repeat and meta[pname].sequence
            if gen_first:
                logger.info("Generating only first part of series")
                pids = og_ids[hist[pname][meta[pname].sequence] == 0]
            else:
                pids = og_ids

            for sel in list(hist):
                name = sel[0] if isinstance(sel, tuple) else sel
                hist[name] = (
                    pids[[name]]
                    .join(hist[name], on=name, how="left")
                    .drop(columns=name)
                )
            idx = pids.index

            table = model.sample(idx, hist)

            if gen_first:
                pp = p.parents[0]
                if isinstance(pp, TablePartition):
                    pp = pp.table
                ppname = pp.name
                midx = og_ids[[ppname]].merge(
                    pids.reset_index(names=PARENT_KEY), on=ppname
                )[[PARENT_KEY]]
                table = midx.merge(table, left_on=PARENT_KEY, right_index=True)

            return og_ids, table
    else:
        pids = ids[(ver.ver.name, True)]

        if tmeta.unroll:
            unrolls = ver.ver.unrolls
            along = tmeta.along
            assert unrolls and along, "TODO: Fix along being empty"

            # Reroll context rows for series history
            ptable = tables[(ver.ver.name, True)]
            _, _, col_maps, col_ofs = recurse_unroll_attr(
                unrolls, cast(Attributes, attrs[ver.ver.name])
            )
            ctx_dfs = []
            for unroll in unrolls:
                rev_maps = {v: k for k, v in col_maps[unroll].items()}
                df = (
                    ptable.loc[ptable[next(iter(rev_maps))] > 0, list(rev_maps)]
                    .reset_index(names=PARENT_KEY)
                    .rename(columns=rev_maps)
                )

                if not len(df):
                    continue

                sers = [
                    df[PARENT_KEY],
                    pd.Series(unroll, index=df.index, name=tmeta.unroll),
                ]
                for col, ofs in col_ofs[unroll].items():
                    sers.append(
                        df[col].astype(get_dtype(int(np.nanmax(df[col])) + ofs))
                        + ofs
                        - 1
                    )
                ctx_dfs.append(pd.concat(sers, axis=1))
            ctx_rerolled = pd.concat(ctx_dfs, axis=0, ignore_index=True)

            # Create ids by using the unrolled context index
            pkey = ctx_rerolled[[PARENT_KEY]]
            pids = pkey.join(pids, on=PARENT_KEY, how="left").drop(columns=PARENT_KEY)
            pids.index.name = ver.ver.name
            # Repeat for history dfs
            new_hist = {}
            for sel, df in hist.items():
                pname = sel[0] if isinstance(sel, tuple) else sel
                new_hist[sel] = (
                    pids[[pname]].join(df, on=pname, how="left").drop(columns=pname)
                )
            # Drop context index to form history
            ctx_cols = ctx_rerolled.drop(columns=PARENT_KEY)
            new_hist[ver.ver.name] = ctx_cols
            series_cols = model.sample(pkey.index, new_hist)
            table = pd.concat([ctx_cols, series_cols], axis=1)

            return pids, table
        elif tmeta.sequence:
            children = ver.ver.max_len or ver.ver.children
            assert children is not None

            seq_attrs = attrs[ver.ver.name]
            assert isinstance(seq_attrs, SeqAttributes)
            order = seq_attrs.order
            name = ver.ver.name
            num = tables[(ver.ver.name, True)][f"{name}_n"]
            tids = ids[(ver.ver.name, True)]

            # Find values and their domains for placeholder history
            vals = {}
            if seq_attrs.hist:
                for attr in next(iter(seq_attrs.hist.values())).values():
                    # if attr.common:
                    #     vals[attr.common.name] = attr.common.get_domain(0)
                    for val_name, val in attr.vals.items():
                        if isinstance(val, CatValue):
                            vals[val_name] = val.get_domain(0)

            # Create sample history
            sampled = []
            for i in range(children):
                idx = num[i < num].index
                if not len(idx):
                    break

                # Add seq value to history
                new_hist = {
                    n: h.loc[tids.loc[idx, n[0] if isinstance(n, tuple) else n]]
                    for n, h in hist.items()
                }
                new_hist[ver.ver.name] = pd.DataFrame(
                    {seq_attrs.seq.name: i},
                    index=idx,
                    dtype="uint8",
                )

                # Add placeholder if not enough history is sampled
                if i < order:
                    placeholder = pd.DataFrame(
                        {k: i for k in vals}, index=idx, dtype="uint8"
                    )
                    for j in range(i, order):
                        new_hist[(name, j)] = placeholder

                # Add the rest of history
                for j in range(min(len(sampled), order)):
                    new_hist[(name, j)] = (
                        sampled[-j - 1].astype(
                            {k: get_dtype(v + j + 1) for k, v in vals.items()}
                        )
                        + j
                        + 1
                    ).loc[idx]

                table = model.sample(idx, new_hist)
                table[seq_attrs.seq.name] = i
                sampled.append(table)

            # Concat to single table with a context index column
            table = pd.concat(
                [s.reset_index(names=PARENT_KEY) for s in sampled], axis=0
            ).reset_index(drop=True)

            # Join context ids to the new index to form the new ids table
            pids = (
                table[[PARENT_KEY]]
                .join(pids, on=PARENT_KEY, how="left")
                .drop(columns=PARENT_KEY)
            )
            pids.index.name = ver.ver.name
            # Drop context index
            table = table.drop(columns=PARENT_KEY)
            return pids, table
        else:
            # Reroll context rows for series history
            ptable = tables[(ver.ver.name, True)]
            children = ver.ver.children
            assert children

            key = ptable[[]].reset_index(names=PARENT_KEY)
            key_dfs = []
            for i in range(children):
                name = ver.ver.name
                num = tables[(ver.ver.name, True)][f"{name}_n"]
                key_dfs.append(key[num > i])
            key = pd.concat(key_dfs, axis=0, ignore_index=True)

            # Create ids by using the unrolled context index
            pids = key.join(pids, on=PARENT_KEY, how="left").drop(columns=PARENT_KEY)
            pids.index.name = ver.ver.name

            # Repeat for history dfs
            new_hist = {}
            for sel, df in hist.items():
                new_hist[sel] = key.join(df, on=PARENT_KEY, how="left").drop(
                    columns=PARENT_KEY
                )
            # Drop context index to form history
            table = model.sample(pids.index, new_hist)
            return pids, table


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
