from collections import defaultdict
from typing import cast

import pandas as pd

from ..attribute import Attribute, Attributes, CommonValue, Grouping, StratifiedValue
from ..mare.reduce import TableMeta, TablePartition, TableVersion
from ..utils import LazyChunk


def _unroll_sequence(seq_name: str, order: int, ids: pd.DataFrame, data: pd.DataFrame):
    _IDX_NAME = "_id_lkjijk"
    _JOIN_NAME = "_id_zdjwk"
    seq = data[seq_name]
    ids_seq = ids.join(seq, how="right").reset_index(names=_IDX_NAME)

    out = {}
    for i in range(order):
        ids_seq_prev = ids.join(seq + i + 1, how="right").reset_index(names=_JOIN_NAME)
        join_ids = ids_seq.merge(
            ids_seq_prev, on=[*ids.columns, seq_name], how="inner"
        ).set_index(_IDX_NAME)[[_JOIN_NAME]]
        ref_df = (
            join_ids.join(data, on=_JOIN_NAME)
            .drop(columns=[_JOIN_NAME, seq_name])
            .reindex(ids.index, fill_value=0)
        )
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
        if order and sequence:
            seq_hist = _unroll_sequence(sequence, order, ids[ver.name](), table)
            for o, data in seq_hist.items():
                _out[(ver.name, o)] = data
        _out[ver.name] = table

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


def recurse_unroll_attr(
    unroll: str | tuple[str, ...], unrolls: tuple[int, ...], attrs: Attributes
):
    attr = attrs[unroll]
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
