from collections import defaultdict
from copy import copy
from functools import partial
from typing import Mapping, NamedTuple, cast

import numpy as np
import pandas as pd

from ..attribute import (
    Attribute,
    Attributes,
    CommonValue,
    DatasetAttributes,
    GenAttribute,
    get_gen_values,
    Grouping,
    SeqAttribute,
    SeqAttributes,
    SeqValue,
    StratifiedValue,
    get_dtype,
)
from ..hierarchy import RebalancedValue
from ..marginal.numpy import TableSelector
from ..marginal.oracle import PreprocessFun
from ..utils import LazyChunk
from ..utils.data import LazyDataset, LazyPartition, data_to_tables
from .chains import TablePartition, TableVersion, calculate_table_chains
from .reduce import (
    TableMeta,
    TablePartition,
    TableVersion,
    _calculate_stripped_meta,
    merge_versions,
    merge_versions_heuristic,
)


def _unroll_sequence(
    seq_name: str,
    order: int,
    ids: pd.DataFrame,
    data: pd.DataFrame,
    seq: pd.Series | None = None,
    stable: str | None = None,
    ptable: str | None = None,
):
    _IDX_NAME = "_id_lkjijk"
    _JOIN_NAME = "_id_zdjwk"

    if seq is not None:
        assert stable

        ids_seq = ids.merge(seq, left_on=stable, right_index=True, how="right")
        seq = ids_seq[seq_name]
        ids_seq = ids_seq.reset_index(names=_IDX_NAME)
    else:
        seq = data[seq_name]
        ids_seq = ids.join(seq, how="right").reset_index(names=_IDX_NAME)

    out = {}
    for i in range(order):
        # Create join with previous seq
        ids_seq_prev = ids.join(seq + i + 1, how="right").reset_index(names=_JOIN_NAME)

        if ptable:
            # Here, if we use a lower table, the merge below fails
            cols = [ptable]
        else:
            cols = ids.columns

        join_ids = ids_seq.merge(
            ids_seq_prev, on=[*cols, seq_name], how="inner"
        ).set_index(_IDX_NAME)[[_JOIN_NAME]]
        ref_df = join_ids.join(data, on=_JOIN_NAME).drop(columns=[_JOIN_NAME])

        if seq_name in ref_df:
            ref_df = ref_df.drop(columns=[seq_name])

        # Rebase discrete columns to new stratified structure with offset
        idx_cols = [
            c for c, t in ref_df.dtypes.items() if pd.api.types.is_integer_dtype(t)
        ]
        idx_dtypes = {c: get_dtype(ref_df[c].max() + 2 + i) for c in idx_cols}
        idx_df = (ref_df[idx_cols].astype(idx_dtypes) + 1 + i).reindex(
            ids.index, fill_value=0
        )

        # Fill value should be set depending on the history of each column
        for j in range(i):
            idx_df.loc[seq == j + 1] = i - j

        # Re-index continuous cols with NaN
        cnt_cols = [c for c in ref_df.columns if c not in idx_cols]
        cnt_df = ref_df[cnt_cols].reindex(ids.index)

        # Concat and fix
        ref_df = pd.concat([idx_df, cnt_df], axis=1)
        ref_df.index.name = data.index.name
        out[i] = ref_df

    return out


def _gen_history(
    ver: TableVersion | TablePartition,
    tables: dict[str, LazyChunk],
    ids: dict[str, LazyChunk],
    meta: dict[str, TableMeta],
    _out: dict[str | tuple[str, int], pd.DataFrame] | None = None,
) -> dict[str | tuple[str, int], pd.DataFrame]:
    if _out is None:
        _out = {}

    # Handle table partitions
    if isinstance(ver, TablePartition):
        if ver.table.name not in _out:
            _gen_history(ver.table, tables, ids, meta, _out)
        col = _out[ver.table.name][meta[ver.table.name].partition]
        if len(ver.partitions) == 1:
            # Faster than isin for single partition
            mask = col == ver.partitions[0]
        else:
            mask = col.isin(ver.partitions)

        _out[ver.table.name] = _out[ver.table.name][mask]
    elif ver.name not in _out:
        sequence = meta[ver.name].sequence
        order = meta[ver.name].order
        table = tables[ver.name]()
        _out[ver.name] = table.copy()
        if order and sequence:
            seq_hist = _unroll_sequence(sequence, order, ids[ver.name](), table)
            for o, data in seq_hist.items():
                _out[(ver.name, o)] = data
            _out[ver.name][sequence] = table[sequence].clip(upper=order)

        for parent in ver.parents:
            _gen_history(parent, tables, ids, meta, _out)

    return _out


def gen_history(
    parents: tuple[TableVersion | TablePartition, ...],
    tables: dict[str, LazyChunk],
    ids: dict[str, LazyChunk],
    meta: dict[str, TableMeta],
):
    out = {}
    for p in parents:
        _gen_history(p, tables, ids, meta, out)
    return out


def _recurse_unroll_groups(
    unroll_ofs: tuple[int, ...],
    cmn: Grouping,
    groups: dict[str | tuple[str, ...], Grouping],
    out,
    cmn_ofs=0,
    ofs=None,
    hv=None,
):
    if ofs is None:
        ofs = defaultdict(lambda: 0)
    if out is None:
        out = {}

    for i, v in enumerate(cmn):
        if isinstance(v, str):
            # If this common val is unrolled, update output
            if cmn_ofs in unroll_ofs:
                out[cmn_ofs] = {}
                for name, g in groups.items():
                    if isinstance(g[i], str):
                        out[cmn_ofs][name] = ofs[name]
                    else:
                        out[cmn_ofs][name] = (
                            ofs[name],
                            StratifiedValue(
                                (
                                    f"{hv[cmn_ofs]}.{name}"
                                    if hv
                                    else f"{name}.o{cmn_ofs:03d}"
                                ),
                                Grouping("cat", [None, g[i]]),
                            ),
                        )

            # Always update ofsets
            for name, g in groups.items():
                gi = g[i]
                if isinstance(gi, str):
                    ofs[name] += 1
                else:
                    ofs[name] += gi.get_domain(0)

            cmn_ofs += 1
        else:
            cmn_ofs = _recurse_unroll_groups(
                unroll_ofs,
                v,
                {n: cast(Grouping, g[i]) for n, g in groups.items()},
                out,
                cmn_ofs,
                ofs,
                hv=hv,
            )

    return cmn_ofs


def recurse_unroll_attr(unrolls: tuple[int, ...], attrs: Attributes):
    attr = None
    for attr in attrs.values():
        if attr.unroll:
            break
    assert attr and attr.unroll

    if attr.common is not None:
        cval = attr.common
        has_common = True
    else:
        assert (
            len(attr.vals) == 1
        ), f"If a common val is not provided, there should only be a single value."
        cval = next(iter(attr.vals.values()))
        has_common = False

    if isinstance(cval, RebalancedValue):
        cval = cval.original
    assert isinstance(
        cval, StratifiedValue
    ), "Unrolling is supported only for stratified and rebalanced values for now."
    cmn = cval.head

    groups = {}
    for along in [attr, *[attrs[n] for n in attr.along]]:
        for name, val in along.vals.items():
            if has_common and name == cval.name:
                continue  # skip common val if attr.common was none
            if isinstance(val, RebalancedValue):
                val = val.original
            assert isinstance(
                val, StratifiedValue
            ), "Unrolling is supported only for stratified and rebalanced values for now."
            groups[name] = val.head

    unrolled = {}
    hv = cval.get_human_readable()
    _recurse_unroll_groups(unrolls, cmn, groups, unrolled, hv=hv)

    new_attrs = {}
    cmn = {}
    cols = defaultdict(dict)
    ofs = defaultdict(dict)
    base_name = (attr.name,) if isinstance(attr.name, str) else attr.name

    for cmn_ofs, vals in unrolled.items():
        cmn_name = hv[cmn_ofs][:10]  # f"{'.'.join(base_name)}.o{cmn_ofs:03d}"

        new_name = (*base_name, cmn_ofs)
        new_vals = []
        for name, t in vals.items():
            if isinstance(t, tuple):
                ofs[cmn_ofs][name] = t[0]
                cols[cmn_ofs][name] = t[1].name
                new_vals.append(t[1])
            elif name != cval.name:
                ofs[cmn_ofs][name] = t

        cmn[cmn_ofs] = cmn_name
        if new_vals:
            new_attrs[new_name] = Attribute(
                new_name, new_vals, CommonValue(cmn_name, na=True)
            )
        else:
            new_attrs[new_name] = Attribute(new_name, [CommonValue(cmn_name, na=True)])

    return new_attrs, cmn, cols, ofs


def SeqCommonValue(name: str, order: int):
    g = f"O{order}"
    for ord in reversed(range(order)):
        title = f"O{ord}"
        g = Grouping("cat", [title, g], title=title)
    return StratifiedValue(name, cast(Grouping, g))


def convert_to_seq_val(s: StratifiedValue, order: int):
    g = s.head
    for ord in reversed(range(order + 1)):
        g = Grouping("cat", [f"H{ord}", g])
    return StratifiedValue(s.name, g)


def convert_rebalanced_to_seq_val(s: RebalancedValue, order: int):
    order += 1
    reb = copy(s)
    reb.counts = np.concatenate([[0] * order, s.counts])
    reb.grouping = np.concatenate(
        [
            [[0 for _ in range(order)] for _ in range(s.grouping.shape[0])],
            s.grouping.astype(np.uint16) + order,
        ],
        axis=1,
    )
    # TODO: verify this works
    reb.common_sizes = [[1 for _ in range(order)] + v for v in s.common_sizes]
    reb.common_groups = [list(v) for v in reb.grouping]
    reb.domains = [d + order for d in s.domains]
    return reb


def convert_to_seq_attr(attrs: Attributes, order: int) -> Attributes:
    out = {}
    for name, attr in attrs.items():
        vals = []
        for v in attr.vals.values():
            if isinstance(v, SeqValue):
                continue

            if isinstance(v, RebalancedValue):
                vals.append(convert_rebalanced_to_seq_val(v, order))
            elif isinstance(v, StratifiedValue):
                vals.append(convert_to_seq_val(v, order))
            else:
                assert (
                    False
                ), f"Attr. '{v.name}' is of type '{type(v)}', which is not supported."

        if attr.common:
            v = attr.common
            if isinstance(v, RebalancedValue):
                cmn = convert_rebalanced_to_seq_val(v, order)
            elif isinstance(v, StratifiedValue):
                cmn = convert_to_seq_val(v, order)
            else:
                assert (
                    False
                ), f"Attr. '{v.name}' is of type '{type(v)}', which is not supported."
        else:
            cmn = None
        if len(vals):
            out[name] = Attribute(attr.name, vals, cmn, attr.unroll, attr.along)

    return out


def strip_seq_vals(attrs: Attributes):
    out = {}
    for n, a in attrs.items():
        vals = []
        for v in a.vals.values():
            if not isinstance(v, SeqValue):
                vals.append(v)

        if len(vals):
            out[n] = Attribute(a.name, vals, a.common, a.unroll, a.along)
    return out


def _gen_history_attributes(
    ver: TableVersion | TablePartition,
    attrs: dict[str, Attributes],
    _out: dict[str, SeqAttributes | Attributes] | None = None,
):
    if _out is None:
        _out = {}

    if isinstance(ver, TablePartition):
        _gen_history_attributes(ver.table, attrs, _out)
    elif ver.name not in _out:
        seq = None
        for attr in attrs[ver.name].values():
            for val in attr.vals.values():
                if isinstance(val, SeqValue):
                    seq = val
                    break

        if seq is not None:
            order = seq.order
            new_attrs = strip_seq_vals(attrs[ver.name])
            hist = {}

            assert order is not None, f"Seq value's '{seq.name}' order is None"
            for i in range(order):
                hist[i] = convert_to_seq_attr(attrs[ver.name], i)

            _out[ver.name] = SeqAttributes(
                order, SeqCommonValue(seq.name, order), new_attrs, hist
            )
        else:
            _out[ver.name] = attrs[ver.name]

        for p in ver.parents:
            _gen_history_attributes(p, attrs, _out)
    return _out


def get_parents(ver):
    # Get parents for building sequences
    # This is a bit dirty coding-wise
    parent = None
    gparent = None
    ggparent = None
    if ver.parents:
        parent = ver.parents[0]
        if hasattr(parent, "table"):
            parent = getattr(parent, "table")

        if parent.parents:  # type: ignore
            gparent = parent.parents[0]  # type: ignore
            if hasattr(gparent, "table"):
                gparent = getattr(gparent, "table")

            if gparent.parents:  # type: ignore
                ggparent = gparent.parents[0]  # type: ignore
                if hasattr(ggparent, "table"):
                    ggparent = getattr(ggparent, "table")
                ggparent = getattr(ggparent, "name")

            gparent = getattr(gparent, "name")

        parent = getattr(parent, "name")

    return parent, gparent, ggparent


def gen_history_attributes(
    parents: tuple[TableVersion | TablePartition, ...],
    attrs: dict[str, Attributes],
):
    out = {}
    for p in parents:
        _gen_history_attributes(p, attrs, out)
    return out


def generate_fit_attrs(
    ver: TableVersion,
    attrs: dict[str, Attributes],
    ctx: bool,
    no_hist: bool = False,
    gen_len: int | None = None,
) -> DatasetAttributes | None:
    # Don't generate context tables for top level tables
    if not ver.parents and ctx:
        return None

    meta = _calculate_stripped_meta(attrs)

    unroll = None
    seq = None
    seq_repeat = False
    for attr in attrs[ver.name].values():
        if attr.unroll:
            unroll = attr.name
            seq_repeat = attr.seq_repeat
        for v in attr.vals.values():
            if isinstance(v, SeqValue):
                seq = v
    assert not (unroll and seq), f"Both unrolling and sequence found on the same table."

    if no_hist:
        hist = {}
    else:
        hist = gen_history_attributes(ver.parents, attrs)

    if unroll:
        if ctx:
            assert ver.unrolls
            synth = recurse_unroll_attr(ver.unrolls, attrs[ver.name])
            hist[None] = synth[0]
        else:
            unroll_attrs = {unroll: attrs[ver.name][unroll]}
            other_attrs = {}
            along = attrs[ver.name][unroll].along

            for name, attr in attrs[ver.name].items():
                if name in along or name == unroll:
                    unroll_attrs[name] = attr
                else:
                    other_attrs[name] = attr

            hist[ver.name] = unroll_attrs
            hist[None] = other_attrs
    elif ctx:
        # Context tables for normal tables and sequence tables same
        assert ver.children is not None
        hist[None] = {
            f"{ver.name}_n": GenAttribute(
                f"{ver.name}_n",
                seq.max if seq and seq.max else ver.children,
                gen_len=gen_len,
            )
        }
    elif seq:
        order = seq.order
        ahist = {}

        assert order is not None, f"Table '{ver.name}'s order is None"
        for i in range(order):
            ahist[i] = convert_to_seq_attr(attrs[ver.name], i)

        assert ver.children is not None
        hist[ver.name] = SeqAttributes(
            order,
            SeqCommonValue(seq.name, order),
            {seq.name: SeqAttribute(seq.name, order=order, max=ver.children)},
            ahist,
        )
        hist[None] = strip_seq_vals(attrs[ver.name])
    else:
        hist[None] = attrs[ver.name]

    # Add sequentiality to context tables through parent
    parent, gparent, ggparent = get_parents(ver)
    target = gparent if seq_repeat else parent
    ptarget = ggparent if seq_repeat else gparent
    if ctx and target and ptarget and meta[target].sequence:
        sequence = meta[target].sequence
        norder = meta[target].order or 1
        assert sequence

        ahist = {}
        for i in range(norder):
            ahist[i] = convert_to_seq_attr(hist[None], i)

        hist[ver.name] = SeqAttributes(
            norder,
            SeqCommonValue(sequence, norder),
            {sequence: SeqAttribute(sequence, order=norder, max=ver.children)},
            ahist,
        )

    return hist


def generate_fit_tables(
    data: Mapping[str, LazyPartition],
    attrs: dict[str, Attributes],
    ver: TableVersion,
    ctx: bool,
    new_attrs: DatasetAttributes,
) -> dict[TableSelector, pd.DataFrame]:
    ids, tables = data_to_tables(data)

    # Get history
    meta = _calculate_stripped_meta(attrs)
    hist = gen_history(ver.parents, tables, ids, meta)

    # Prune ids that are not used in this model, ex. due to partitioning
    fids = ids[ver.name]()
    for name, table in hist.items():
        if isinstance(name, tuple):
            name = name[0]
        fids = fids.join(table[[]], on=name, how="inner")

    _tmp = tables[ver.name]()
    try:
        table = _tmp.loc[fids.index]
    except Exception:
        # FIXME: Resolve id situation
        # If a context table is joined to a parent without child rows
        # that parent is pruned.
        table = _tmp.join(fids, how="inner")

    # If no parents, assume normal table and return it
    if not ver.parents:
        tmeta = meta[ver.name]
        assert not tmeta.unroll and not tmeta.sequence
        return {None: table}

    new_hist = {}
    unroll = meta[ver.name].unroll
    sequence = meta[ver.name].sequence
    order = meta[ver.name].order
    max_len = meta[ver.name].max_len
    seq_repeat = meta[ver.name].seq_repeat

    parent, gparent, ggparent = get_parents(ver)

    # Create new id that is unique per sequence
    SID_NAME = "nid_jsdi78"
    if seq_repeat and parent:
        # If seq_repeat, skip parent
        fltr = [c for c in fids.columns if c != parent]
    else:
        fltr = list(fids.columns)
    # Create index with just parents, add a column that acts as an
    # index, join back to fids
    sid = fids[fltr].groupby(fltr).first()[[]]
    sid[SID_NAME] = range(len(sid))
    sid = fids[fltr].join(sid, on=fltr).drop(columns=fltr)

    if ctx:
        # common operation for indexing to parent for context tables
        fids = fids.join(sid.drop_duplicates(), how="inner").set_index(SID_NAME)

    if unroll:
        if ctx:
            assert ver.unrolls
            _, cmn, cols, ofs = recurse_unroll_attr(ver.unrolls, attrs[ver.name])

            udfs = []
            for u in cmn.keys():
                udf = (
                    table.loc[table[unroll] == u, list(cols[u])]
                    .join(sid)
                    .drop_duplicates([SID_NAME])
                    .set_index(SID_NAME)
                    .convert_dtypes()
                )
                udf[cmn[u]] = pd.Series(1, dtype="UInt8", index=udf.index)
                for c, o in ofs[u].items():
                    udf[c] -= o - 1
                udfs.append(udf.rename(columns=cols[u]))

            utab = pd.concat(udfs, axis=1).fillna(0)
            synth = utab.astype(
                dtype={k: str(v).lower() for k, v in utab.dtypes.to_dict().items()}  # type: ignore
            )
        else:
            unroll_cols = [unroll]
            along = meta[ver.name].along
            if along:
                unroll_cols.extend(along)
            new_hist[ver.name] = table[unroll_cols]
            synth = table.drop(columns=unroll_cols)
    elif sequence:
        if ctx:
            lens = sid.groupby(SID_NAME).size()
            vn = f"{ver.name}_n"
            gen_val = new_attrs[None][vn].vals[vn]  # type: ignore
            if hasattr(gen_val, "gen_vals") and getattr(gen_val, "gen_vals"):
                vals = np.array(getattr(gen_val, "gen_vals"))
                lens = pd.Series(np.abs(lens.to_numpy()[:, None] - vals).argmin(axis=1),
                                 index=lens.index)
            elif max_len is not None:
                lens = lens.clip(upper=max_len)
            synth = pd.DataFrame(lens.rename(vn))
        else:
            assert order is not None

            seq_hist = _unroll_sequence(sequence, order, fids, table)
            for o, data in seq_hist.items():
                new_hist[(ver.name, o)] = data
            new_hist[ver.name] = pd.DataFrame(table[sequence].clip(upper=order))
            synth = table
    else:
        if ctx:
            lens = sid.groupby(SID_NAME).size()
            vn = f"{ver.name}_n"
            gen_val = new_attrs[None][vn].vals[vn]  # type: ignore
            if hasattr(gen_val, "gen_vals") and getattr(gen_val, "gen_vals"):
                vals = np.array(getattr(gen_val, "gen_vals"))
                lens = pd.Series(np.abs(lens.to_numpy()[:, None] - vals).argmin(axis=1),
                                 index=lens.index)

            synth = pd.DataFrame(lens.rename(vn))
        else:
            synth = table

    # With the new ids, prune the history
    for idx, table in hist.items():
        if isinstance(idx, tuple):
            name = idx[0]
        else:
            name = idx
        new_hist[idx] = (
            fids[[name]].join(table, on=name, how="inner").drop(columns=name)
        )

    # Apply sequentiality to context tables. If seq_repeat, to grand_parent. Else to parent
    target = gparent if seq_repeat else parent
    ptarget = ggparent if seq_repeat else gparent
    if ctx and target and ptarget and meta[target].sequence:
        seq = meta[target].sequence
        norder = meta[target].order or 1
        assert seq

        seq_hist = _unroll_sequence(
            seq,
            norder,
            fids,
            synth,
            seq=tables[target]()[seq],
            stable=target,
            ptable=ptarget,
        )
        for o, data in seq_hist.items():
            new_hist[(ver.name, o)] = data

        new_hist[ver.name] = new_hist[parent]

    return {**new_hist, None: synth}


class ModelVersion(NamedTuple):
    ver: TableVersion
    ctx: bool


def calculate_model_versions(
    attrs: dict[str, Attributes],
    data: Mapping[str, LazyDataset],
    max_vers: int,
    no_hist: bool = False,
    gen_len: int | None = None,
) -> dict[ModelVersion, tuple[DatasetAttributes, PreprocessFun]]:
    ids, tables = data_to_tables(data)
    chains = calculate_table_chains(attrs, ids, tables)
    meta = _calculate_stripped_meta(attrs)

    out: dict[ModelVersion, tuple[DatasetAttributes, PreprocessFun]] = {}
    for name, vers in chains.items():
        assert vers, f"Table {name} has 0 versions."
        tmeta = meta[name]

        if tmeta.unroll:
            preproc_fn = lambda v: set(v.unrolls) if v.unrolls else set()
            merge_fn = lambda a, b: a.union(b)
            score_fn = lambda a, b: len(a.symmetric_difference(b))

            # Unroll context models
            new_vers = merge_versions_heuristic(
                vers, max_vers, preproc_fn, merge_fn, score_fn
            )

            if not new_vers:
                new_vers = vers

            for ver in new_vers:
                new_attrs = generate_fit_attrs(
                    ver, attrs, True, no_hist=no_hist, gen_len=gen_len
                )
                assert new_attrs is not None

                load_fn = partial(
                    generate_fit_tables,
                    attrs=attrs,
                    ver=ver,
                    ctx=True,
                    new_attrs=new_attrs,
                )
                out[ModelVersion(ver, True)] = new_attrs, load_fn

            # Unroll series model
            ver = merge_versions(vers)
            new_attrs = generate_fit_attrs(
                ver, attrs, False, no_hist=no_hist, gen_len=gen_len
            )
            assert new_attrs is not None

            load_fn = partial(
                generate_fit_tables,
                attrs=attrs,
                ver=ver,
                ctx=False,
                new_attrs=new_attrs,
            )
            out[ModelVersion(ver, False)] = new_attrs, load_fn
        else:
            # Apart from unroll, create one ctx model and one series model
            # for each table
            ver = merge_versions(vers)
            for ctx in (True, False):
                new_attrs = generate_fit_attrs(
                    ver, attrs, ctx, no_hist=no_hist, gen_len=gen_len
                )
                if new_attrs is not None:
                    load_fn = partial(
                        generate_fit_tables,
                        attrs=attrs,
                        ver=ver,
                        ctx=ctx,
                        new_attrs=new_attrs,
                    )
                    out[ModelVersion(ver, ctx)] = new_attrs, load_fn

    return out
