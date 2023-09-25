from collections import defaultdict
from itertools import product
from typing import Generic, NamedTuple, Sequence, TypeVar

import pandas as pd

from .attribute import Attributes, SeqValue
from .utils import LazyFrame, lazy_load_tables


class PartInfo(NamedTuple):
    name: str
    val: tuple[int, ...]
    along: tuple[str, ...]


class UnrollInfo(NamedTuple):
    name: str
    vals: tuple[int, ...]
    parent: str
    along: tuple[str, ...]


class SeqInfo(NamedTuple):
    name: str
    seq: int | None
    order: int
    parent: str


class TableVersion(NamedTuple):
    # Table Name
    name: str
    # Table pins
    partition: PartInfo | None
    sequence: SeqInfo | None
    unroll: UnrollInfo | None
    # Table parents
    parents: tuple["TableVersion", ...]


def get_parents(
    ids: dict[str, LazyFrame], table: str | None = None, _reverse: bool = False
):
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
    return get_parents(ids, _reverse=True)


def _calculate_parent_mask(
    name: str, parents: tuple[TableVersion, ...], get_table, get_ids, cache
):
    """Calculates the mask of table `name` that makes it include only the rows
    satisfied by its parents. `included_ids_cache` is updated.

    `cache` should be a dicti. Improves performance."""
    included_ids = {}
    for parent in parents:
        included_ids.update(_calculate_included_ids(parent, get_table, get_ids, cache))

    mask = True
    for parent, (pkey, incl_ids) in included_ids.items():
        key = (name, pkey)
        if key not in cache:
            cache[key] = get_ids(name)[parent].isin(incl_ids)
        mask &= cache[key]

    if mask is True:
        return None
    return mask


def _calculate_included_ids(ver: TableVersion, get_table, get_ids, cache):
    """Each table version excludes some rows based on partitioning and sequencing.
    And the partitioning and sequencing of its parents.

    This function returns a dictionary where for each parent table, the keys that
    should be included are provided as a pd.Series.

    `cache` is a dictionary that caches the ids for previous table versions."""

    # Out is a dictionary of keys used to calculate the mask
    out = {}
    for parent in ver.parents:
        out.update(_calculate_included_ids(parent, get_table, get_ids, cache))

    # Cache ids based on name, partition type, and sequence
    key = (ver.name, ver.partition, ver.sequence)
    if key in cache:
        out[ver.name] = (key, cache[key])
        return out

    # Since we only filter when there is a partition, return without ids
    # If both don't exist
    if not ver.partition and not ver.sequence:
        return out

    # Apply mask based on sequence and partitions
    mask = True
    if ver.partition:
        part_col = get_table(ver.name)[ver.partition.name]
        if len(ver.partition.val) > 1:
            mask &= part_col.isin(ver.partition.val)
        else:
            mask &= part_col == ver.partition.val[0]

    if ver.sequence:
        if ver.sequence.seq is not None:
            mask &= get_table(ver.name)[ver.sequence.name] == ver.sequence.seq
        else:
            mask &= get_table(ver.name)[ver.sequence.name] >= ver.sequence.order

    # Calculate ids, update cache, and return them
    included_ids = get_ids(ver.name).index[mask].unique()
    cache[key] = included_ids
    out[ver.name] = (key, included_ids)
    return out


def _calculate_versions_of_table(
    name: str,
    meta: dict[str, Attributes],
    ids: dict[str, LazyFrame],
    tables: dict[str, LazyFrame],
    parents: dict[str, Sequence[TableVersion]],
) -> tuple[tuple[TableVersion, ...], dict[TableVersion, int]]:

    sequence = None
    sequence_parent = None
    sequence_order = None
    partition = None
    partition_along = None
    unroll = None
    unroll_along = None
    unroll_parent = None

    attrs = meta[name]
    for attr in attrs.values():
        for v in attr.vals.values():
            if isinstance(v, SeqValue) and v.order is not None:
                sequence = v.name
                sequence_parent = v.table
                sequence_order = v.order

        if attr.partition:
            if len(attr.vals) > 1:
                assert attr.common
                partition = attr.common.name
            else:
                assert len(attr.vals) == 1
                partition = next(iter(attr.vals))

            along = []
            along.extend(attr.vals)
            if attr.common:
                along.append(attr.common.name)
            if attr.partition_with:
                for aname in attr.partition_with:
                    along.extend(attrs[aname].vals)
                    acmn = attrs[aname].common
                    if acmn:
                        along.append(acmn.name)
            partition_along = tuple(sorted(along))
        if attr.unroll:
            unroll_parent = attr.unroll
            if len(attr.vals) > 1:
                assert attr.common
                unroll = attr.common.name
            else:
                assert len(attr.vals) == 1
                unroll = next(iter(attr.vals))

            along = []
            along.extend(attr.vals)
            if attr.common:
                along.append(attr.common.name)
            if attr.unroll_with:
                for aname in attr.unroll_with:
                    along.extend(attrs[aname].vals)
                    acmn = attrs[aname].common
                    if acmn:
                        along.append(acmn.name)
            unroll_along = tuple(sorted(along))

    combos = list(
        product(*[v for _, v in sorted(parents.items(), key=lambda x: x[0])])
    ) or [()]

    # Unrolling and Partitioning have values that are extracted from data so have
    # to run per partution
    versions_per_combo = defaultdict(set)
    rows_per_combo = {}

    for ptables, pids in LazyFrame.zip_values([tables, ids]):
        get_table = lazy_load_tables(ptables)
        get_ids = lazy_load_tables(pids)
        included_ids_cache: dict[TableVersion, pd.Series] = {}

        for combo in combos:
            table = get_table(name)
            pmask = _calculate_parent_mask(
                name, combo, get_table, get_ids, included_ids_cache
            )
            if pmask is not None:
                table = table.loc[pmask]

            if partition:
                counts = table[partition].value_counts()
                counts_dict = counts.to_dict()
                if combo not in rows_per_combo:
                    rows_per_combo[combo] = {}
                for k, v in counts_dict.items():
                    rows_per_combo[combo][k] = rows_per_combo[combo].get(k, 0) + v
                versions_per_combo[combo].update(counts.index)
            elif unroll:
                rows_per_combo[combo] = rows_per_combo.get(combo, 0) + len(table)
                versions_per_combo[combo].update(table[unroll].unique())
            elif sequence:
                assert sequence_order is not None
                if combo not in rows_per_combo:
                    rows_per_combo[combo] = {}
                for i in range(sequence_order + 1):
                    if i < sequence_order:
                        inc = (table[sequence] == i).sum()
                    else:
                        inc = (table[sequence] >= sequence_order).sum()
                    rows_per_combo[combo][i] = rows_per_combo[combo].get(i, 0) + inc
            else:
                rows_per_combo[combo] = rows_per_combo.get(combo, 0) + len(table) 

    rows = {}
    versions = []
    for combo in combos:
        if partition and sequence:
            assert (
                partition_along is not None
                and sequence_order is not None
                and sequence_order > 0
                and sequence_parent
            )
            for i in range(sequence_order + 1):
                for val in sorted(versions_per_combo[combo]):
                    versions.append(
                        TableVersion(
                            name,
                            PartInfo(partition, (val,), partition_along),
                            SeqInfo(
                                sequence,
                                i if i < sequence_order else None,
                                sequence_order,
                                sequence_parent,
                            ),
                            None,
                            combo,
                        )
                    )
        if partition:
            assert partition_along is not None
            for val in sorted(versions_per_combo[combo]):
                versions.append(
                    TableVersion(
                        name,
                        PartInfo(partition, (val,), partition_along),
                        None,
                        None,
                        combo,
                    )
                )
                rows[versions[-1]] = rows_per_combo[combo][val]
        elif unroll:
            assert unroll_parent is not None and unroll_along is not None
            versions.append(
                TableVersion(
                    name,
                    None,
                    None,
                    UnrollInfo(
                        unroll,
                        tuple(sorted(versions_per_combo[combo])),
                        unroll_parent,
                        unroll_along,
                    ),
                    combo,
                )
            )
            rows[versions[-1]] = rows_per_combo[combo]
        elif sequence:
            assert sequence_order is not None and sequence_order > 0 and sequence_parent

            for i in range(sequence_order + 1):
                versions.append(
                    TableVersion(
                        name,
                        None,
                        SeqInfo(
                            sequence,
                            i if i < sequence_order else None,
                            sequence_order,
                            sequence_parent,
                        ),
                        None,
                        combo,
                    )
                )
                rows[versions[-1]] = rows_per_combo[combo][i]
        else:
            versions.append(TableVersion(name, None, None, None, combo))
            rows[versions[-1]] = rows_per_combo[combo]

    return tuple(versions), rows


def calculate_table_versions(
    meta: dict[str, Attributes],
    ids: dict[str, LazyFrame],
    tables: dict[str, LazyFrame],
    return_all_tables=True,
    _parents=None,
    _cache=None,
):
    if _parents is None:
        _parents = get_parents(ids)
    if _cache is None:
        _cache = {}

    out = {}
    rows = {}
    for name, parents in _parents.items():
        if name in _cache:
            out[name] = _cache[name]
        else:
            parent_versions, parent_rows = calculate_table_versions(
                meta, ids, tables, False, parents, _cache
            )
            versions, table_rows = _calculate_versions_of_table(
                name, meta, ids, tables, parent_versions
            )
            _cache[name] = versions
            rows.update(table_rows)
            rows.update(parent_rows)
            out[name] = versions

    if return_all_tables:
        return _cache, rows
    return out, rows
