""" This module contains functions for calculating the markov chain combinations 
of different tables in a dataset.
"""

from collections import defaultdict
from itertools import product
from typing import Generic, NamedTuple, Sequence, TypeVar

import pandas as pd

from ..attribute import Attributes, SeqValue
from ..utils import LazyFrame, lazy_load_tables


class TableVersion(NamedTuple):
    """Contains the parameters required to construct a markov chain for table
    with name `name`, by recursing on its parents."""

    name: str
    rows: int
    children: int | None
    max_len: int | None
    partitions: tuple[int, ...] | None
    unrolls: tuple[int, ...] | None
    parents: tuple["TableVersion | TablePartition", ...]


class TablePartition(NamedTuple):
    partitions: tuple[int, ...]
    table: TableVersion


class TableMeta(NamedTuple):
    sequence: str | None
    order: int | None
    partition: str | None
    unroll: str | None
    along: tuple[str, ...] | None
    parent: str | None
    max_len: int | None
    seq_repeat: bool


def get_parents(
    ids: dict[str, LazyFrame], table: str | None = None, _reverse: bool = False
):
    """Returns a graph of the parent relationships in the dataset, based on the id table.

    I.e. if there is a dataset with tables A, B, C, D, where C -> B -> A and D -> B,
    the function returns`{'D': {'B': {'A': {}}}, 'C': {'B': {'A': {}}}}`"""
    # Get parents
    parents = {name: set(id.sample().columns) for name, id in ids.items()}
    tables = list(parents)

    if _reverse:
        children = {t: set() for t in tables}
        for t, deps in parents.items():
            for p in deps:
                children[p].add(t)
        parents = children

    # Deduplicate parent relationships
    updated = True
    while updated:
        updated = False
        for k in tables:
            for l in tables:
                if k in parents[l] and parents[k] and parents[k].issubset(parents[l]):
                    parents[l].difference_update(parents[k])
                    updated = True

    # Create parent tree
    new_parents = {t: {} for t in tables}
    has_child = set()
    for t, dep in parents.items():
        for p in dep:
            new_parents[t][p] = new_parents[p]
            has_child.add(p)

    if table:
        return new_parents[table]

    # Keep only tables with no children in top level
    return {t: d for t, d in new_parents.items() if t not in has_child}


def get_children(ids: dict[str, LazyFrame]):
    """Returns a graph of the children relationships in the dataset, based on the id table.

    I.e. if there is a dataset with tables A, B, C, D, where C -> B -> A and D -> B,
    the function returns`{'A': {'B': {'C', 'D'}}}`"""
    return get_parents(ids, _reverse=True)


def _calculate_parent_mask(
    name: str,
    parents: tuple[TableVersion | TablePartition, ...],
    meta: dict[str, TableMeta],
    get_table,
    get_ids,
    cache,
):
    included_ids = {}
    for parent in parents:
        included_ids.update(
            _calculate_included_ids(parent, meta, get_table, get_ids, cache)
        )

    mask = True
    for parent, (pkey, incl_ids) in included_ids.items():
        key = (name, pkey)
        if key not in cache:
            cache[key] = get_ids(name)[parent].isin(incl_ids)
        mask &= cache[key]

    if mask is True:
        return None
    return mask


def _calculate_included_ids(
    ver: TableVersion | TablePartition,
    meta: dict[str, TableMeta],
    get_table,
    get_ids,
    cache,
):
    """Each table version excludes some rows based on partitioning and sequencing.
    And the partitioning and sequencing of its parents.

    This function returns a dictionary where for each parent table, the keys that
    should be included are provided as a pd.Series.

    `cache` is a dictionary that caches the ids for previous table versions."""

    # Out is a dictionary of keys used to calculate the mask
    out = {}
    if isinstance(ver, TablePartition):
        partitions = ver.partitions
        ver = ver.table
    else:
        partitions = None

    for parent in ver.parents:
        out.update(_calculate_included_ids(parent, meta, get_table, get_ids, cache))

    # Cache ids based on name, partition type, and sequence
    key = (ver.name, partitions)
    if key in cache:
        out[ver.name] = (key, cache[key])
        return out

    # Since we only filter when there is a partition, return without ids
    # If both don't exist
    if not partitions:
        return out

    # Apply mask based on sequence and partitions
    mask = True
    part_col = get_table(ver.name)[meta[ver.name].partition]
    if len(partitions) > 1:
        mask &= part_col.isin(partitions)
    else:
        mask &= part_col == partitions[0]

    # Calculate ids, update cache, and return them
    included_ids = get_ids(ver.name).index[mask].unique()
    cache[key] = included_ids
    out[ver.name] = (key, included_ids)
    return out


def calculate_stripped_meta(attrs: dict[str, Attributes]) -> dict[str, TableMeta]:
    out = {}
    for name in attrs:
        sequence = None
        order = None
        max_len = None
        partition = None
        unroll = None
        parent = None
        along = None
        seq_repeat = False

        for attr in attrs[name].values():
            for v in attr.vals.values():
                if isinstance(v, SeqValue) and v.order is not None:
                    assert parent is None
                    sequence = v.name
                    parent = v.table
                    order = v.order
                    max_len = v.max

            if attr.partition:
                if len(attr.vals) > 1:
                    assert attr.common
                    partition = attr.common.name
                else:
                    assert len(attr.vals) == 1
                    partition = next(iter(attr.vals))

            if attr.unroll:
                if len(attr.vals) > 1 or attr.common:
                    assert attr.common
                    unroll = attr.common.name
                else:
                    assert len(attr.vals) == 1
                    unroll = next(iter(attr.vals))

                along = []
                along.extend(attr.vals)
                seq_repeat = attr.seq_repeat

                for n in attr.along:
                    along.extend(attrs[name][n].vals)

        if along is not None:
            along = tuple(along)
        out[name] = TableMeta(
            sequence, order, partition, unroll, along, parent, max_len, seq_repeat
        )
    return out


_calculate_stripped_meta = calculate_stripped_meta


def _calculate_chains_of_table(
    name: str,
    meta: dict[str, TableMeta],
    ids: dict[str, LazyFrame],
    tables: dict[str, LazyFrame],
    parents: dict[str, tuple[TableVersion, ...]],
) -> tuple[TableVersion, ...]:
    """Calculates the possible markov chains for the table with `name`, by
    iterating over its parent combinations.

    The chains are returned as a tuple, alongside a dictionary of `chain -> row
    count` mappings, which may be used as auxiliary info when deciding
    which chains to keep."""

    def _expand_versions(vers: tuple[TableVersion, ...]):
        out = []
        for v in vers:
            if v.partitions:
                for p in v.partitions:
                    out.append(TablePartition((p,), v))
            else:
                out.append(v)
        return out

    combos = list(
        product(
            *[
                _expand_versions(v)
                for _, v in sorted(parents.items(), key=lambda x: x[0])
            ]
        )
    ) or [()]

    partition = meta[name].partition
    unroll = meta[name].unroll
    max_len = meta[name].max_len

    # Unrolling and Partitioning have values that are extracted from data so have
    # to run per partution
    rows_per_combo = defaultdict(lambda: 0)
    children_per_combo: dict[tuple, int | None] = defaultdict(lambda: None)
    unrolls_per_combo = defaultdict(set)
    partitions_per_combo = defaultdict(set)

    for ptables, pids in LazyFrame.zip_values([tables, ids]):
        get_table = lazy_load_tables(ptables)
        get_ids = lazy_load_tables(pids)
        included_ids_cache: dict[TablePartition | TableVersion, pd.Series] = {}

        for combo in combos:
            table = get_table(name)
            pmask = _calculate_parent_mask(
                name, combo, meta, get_table, get_ids, included_ids_cache
            )
            if pmask is not None:
                table = table.loc[pmask]

            rows_per_combo[combo] += len(table)

            if parents:
                SID_NAME = "nid_jsdi78"
                fids = get_ids(name)
                if pmask is not None:
                    fids = fids.loc[pmask]
                sid = fids.join(
                    fids.drop_duplicates()
                    .reset_index(drop=True)
                    .reset_index(names=SID_NAME)
                    .set_index(list(fids.columns)),
                    on=list(fids.columns),
                ).drop(columns=list(fids.columns))
                new_children = sid.groupby(SID_NAME).size().max()

                children = children_per_combo[combo]
                if children is None or children < new_children:
                    children_per_combo[combo] = new_children

            if unroll:
                unrolls_per_combo[combo].update(table[unroll].unique())

            if partition:
                counts = table[partition].value_counts().to_dict()
                partitions_per_combo[combo].update(counts.keys())

    versions = []
    for combo in combos:
        # Extract data from loading tables
        if partition:
            partitions = tuple(sorted(partitions_per_combo[combo]))
        else:
            partitions = None
        if unroll:
            unrolls = tuple(sorted(unrolls_per_combo[combo]))
        else:
            unrolls = None

        ver = TableVersion(
            name=name,
            rows=rows_per_combo[combo],
            children=children_per_combo[combo],
            max_len=max_len,
            partitions=partitions,
            unrolls=unrolls,
            parents=combo,
        )
        versions.append(ver)

    return tuple(versions)


def calculate_table_chains(
    attrs: dict[str, Attributes],
    ids: dict[str, LazyFrame],
    tables: dict[str, LazyFrame],
    return_all_tables=True,
    _parents=None,
    _cache=None,
) -> dict[str, tuple[TableVersion, ...]]:
    """Returns a tuple of all possible chain combinations for the tables in the
    provided view (as a dictionary of table -> chains) and a dictionary of chain
    to row count mappings."""
    if _parents is None:
        _parents = get_parents(ids)
    if _cache is None:
        _cache = {}
    meta = _calculate_stripped_meta(attrs)

    out = {}
    for name, parents in _parents.items():
        if name in _cache:
            out[name] = _cache[name]
        else:
            parent_versions = calculate_table_chains(
                attrs, ids, tables, False, parents, _cache
            )
            versions = _calculate_chains_of_table(
                name, meta, ids, tables, parent_versions
            )
            _cache[name] = versions
            out[name] = versions

    if return_all_tables:
        return _cache
    return out


__all__ = [
    "get_parents",
    "get_children",
    "calculate_table_chains",
]
