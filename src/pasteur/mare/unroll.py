from collections import defaultdict
from typing import NamedTuple, cast

import pandas as pd

from ..attribute import (
    Attribute,
    Attributes,
    CommonValue,
    GenAttribute,
    Grouping,
    SeqValue,
    StratifiedValue,
    get_dtype,
)
from ..utils import LazyChunk
from .chains import TablePartition, TableVersion
from .reduce import TableMeta, TablePartition, TableVersion


def _unroll_sequence(seq_name: str, order: int, ids: pd.DataFrame, data: pd.DataFrame):
    _IDX_NAME = "_id_lkjijk"
    _JOIN_NAME = "_id_zdjwk"
    seq = data[seq_name]
    ids_seq = ids.join(seq, how="right").reset_index(names=_IDX_NAME)

    out = {}
    for i in range(order):
        # Create join with previous seq
        ids_seq_prev = ids.join(seq + i + 1, how="right").reset_index(names=_JOIN_NAME)
        join_ids = ids_seq.merge(
            ids_seq_prev, on=[*ids.columns, seq_name], how="inner"
        ).set_index(_IDX_NAME)[[_JOIN_NAME]]
        ref_df = join_ids.join(data, on=_JOIN_NAME).drop(columns=[_JOIN_NAME, seq_name])

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
            idx_df.loc[seq == j + 1] = j + 1

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
                                f"u{cmn_ofs:03d}_{name}", Grouping("cat", [None, g[i]])
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
    else:
        assert (
            len(attr.vals) == 1
        ), f"If a common val is not provided, there should only be a single value."
        cval = next(iter(attr.vals))
    assert isinstance(
        cval, StratifiedValue
    ), "Unrolling is supported only for stratified values for now."
    cmn = cval.head

    groups = {}
    for along in [attr, *[attrs[n] for n in attr.along]]:
        for name, val in along.vals.items():
            if name == cval.name:
                continue  # skip common val if attr.common was none
            assert isinstance(
                val, StratifiedValue
            ), "For unrolling, all values should be StratifiedValues"
            groups[name] = val.head

    unrolled = {}
    _recurse_unroll_groups(unrolls, cmn, groups, unrolled)

    new_attrs = {}
    cmn = {}
    cols = defaultdict(dict)
    ofs = defaultdict(dict)
    base_name = (attr.name,) if isinstance(attr.name, str) else attr.name

    for cmn_ofs, vals in unrolled.items():
        new_name = (*base_name, cmn_ofs)
        new_vals = []
        for name, t in vals.items():
            if isinstance(t, tuple):
                ofs[cmn_ofs][name] = t[0]
                cols[cmn_ofs][name] = t[1].name
                new_vals.append(t[1])
            else:
                ofs[cmn_ofs][name] = t

        cmn_name = f"u{cmn_ofs:03d}_cmn"
        cmn[cmn_ofs] = cmn_name
        if new_vals:
            new_attrs[new_name] = Attribute(
                new_name, new_vals, CommonValue(cmn_name, na=True)
            )
        else:
            new_attrs[new_name] = Attribute(new_name, [CommonValue(cmn_name, na=True)])

    return new_attrs, cmn, cols, ofs


class SeqAttributes(NamedTuple):
    order: int
    seq: StratifiedValue
    attrs: Attributes | None
    hist: dict[int, Attributes]


def SeqCommonValue(name: str, order: int):
    g = f"O{order}"
    for ord in reversed(range(order)):
        name = f"O{ord}"
        g = Grouping("cat", [name, g], title=name)
    return StratifiedValue(name, g)  # type: ignore


def convert_to_seq_val(s: StratifiedValue, order: int):
    g = s.head
    for ord in reversed(range(order + 1)):
        g = Grouping("cat", [f"H{ord}", g])
    return StratifiedValue(s.name, g)


def convert_to_seq_attr(attrs: Attributes, order: int):
    out = {}
    for name, attr in attrs.items():
        vals = []
        for v in attr.vals.values():
            if isinstance(v, SeqValue):
                continue

            assert isinstance(
                v, StratifiedValue
            ), f"Attr. '{v.name}' is of type '{type(v)}'. Not Stratified."
            vals.append(convert_to_seq_val(v, order))

        if attr.common:
            assert isinstance(attr.common, StratifiedValue)
            cmn = convert_to_seq_val(attr.common, order)
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


def gen_history_attributes(
    parents: tuple[TableVersion | TablePartition, ...],
    attrs: dict[str, Attributes],
):
    out = {}
    for p in parents:
        _gen_history_attributes(p, attrs, out)
    return out


def generate_synth_attrs(
    ver: TableVersion, attrs: dict[str, Attributes], ctx: bool
) -> tuple[dict[str, SeqAttributes | Attributes], Attributes] | None:
    unroll = None
    seq = None
    for attr in attrs[ver.name].values():
        if attr.unroll:
            unroll = attr.name
        for v in attr.vals.values():
            if isinstance(v, SeqValue):
                seq = v
    if not unroll and not seq:
        return
    assert not (unroll and seq), f"Both unrolling and sequence found on the same table."

    hist = gen_history_attributes(ver.parents, attrs)

    if unroll:
        if ctx:
            assert ver.unrolls
            synth = recurse_unroll_attr(ver.unrolls, attrs[ver.name])
            return hist, synth[0]
        else:
            unroll_attrs = {unroll: attrs[ver.name][unroll]}
            other_attrs = {}
            along = attrs[ver.name][unroll].along

            for name, attr in attrs[ver.name].items():
                if name in along:
                    unroll_attrs[name] = attr
                else:
                    other_attrs[name] = attr

            hist[ver.name] = unroll_attrs
            return hist, other_attrs
    elif seq:
        if ctx:
            assert seq.max is not None
            return hist, {seq.name: GenAttribute(seq.name, seq.max)}
        else:
            order = seq.order
            ahist = {}

            assert order is not None, f"Table '{ver.name}'s order is None"
            for i in range(order):
                ahist[i] = convert_to_seq_attr(attrs[ver.name], i)

            assert seq.max is not None
            hist[ver.name] = SeqAttributes(
                order,
                SeqCommonValue(seq.name, order),
                {seq.name: GenAttribute(seq.name, seq.max)},
                ahist,
            )
            synth = strip_seq_vals(attrs[ver.name])
    else:
        if ctx:
            return None
        else:
            return hist, attrs[ver.name]
