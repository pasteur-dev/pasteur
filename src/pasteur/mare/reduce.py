"""This module contains heuristics for simplifying the chain combinations of a
dataset."""

from collections import defaultdict
from functools import reduce
from heapq import heappop, heappush
from itertools import combinations, product
from typing import (
    Generator,
    Generic,
    Protocol,
    Sequence,
    TypeVar,
    cast,
)

from ..attribute import Attributes
from .chains import TableMeta, TablePartition, TableVersion, _calculate_stripped_meta

A = TypeVar("A", covariant=True)
B = TypeVar("B")
C = TypeVar("C", contravariant=True)


class PreprocFn(Protocol, Generic[A]):
    def __call__(self, v: TableVersion) -> A: ...


class MergeFn(Protocol, Generic[B]):
    def __call__(self, a: B, b: B) -> B: ...


class ScoreFn(Protocol, Generic[C]):
    def __call__(self, a: C, b: C) -> int: ...


class Pair:
    def __init__(self, name, a, b, score):
        self.name = name
        self.a = a
        self.b = b
        self.score = score

    def __lt__(self, other: "Pair"):
        return self.score < other.score

    def __iter__(self):
        return iter((self.name, self.a, self.b))


def _get_partitions_names(ver: TableVersion):
    out = []
    for p in ver.parents:
        if isinstance(p, TablePartition):
            out.append(p.table.name)
            p = p.table
        out.extend(_get_partitions_names(p))

    return out


def _get_partitions(ver: TableVersion, out):
    for p in ver.parents:
        if isinstance(p, TablePartition):
            out.append(p.partitions)
            p = p.table
        _get_partitions(p, out)

    return out


def get_combos(
    vers: Sequence[TableVersion], tables: Sequence[str] | None = None
) -> tuple[Sequence[str], dict[tuple[tuple[int, ...], ...], TableVersion]]:
    names = _get_partitions_names(vers[0])
    unique_names = tables or sorted(set(names))

    if not unique_names:
        return [], {}

    name_idx = [names.index(u) for u in unique_names]

    candidates = {}
    for ver in vers:
        parts = []
        _get_partitions(ver, parts)
        candidates[tuple([parts[i] for i in name_idx])] = ver

    return unique_names, candidates


def merge_versions_heuristic(
    vers: Sequence[TableVersion],
    max_vers: int,
    preproc_fn: PreprocFn[B],
    merge_fn: MergeFn[B],
    score_fn: ScoreFn[B],
    tables: Sequence[str] | None = None,
):
    names, combos = get_combos(vers, tables)

    lookups = {}
    for i, name in enumerate(names):
        lookup = defaultdict(list)
        for combo, val in combos.items():
            lookup[combo[i]].append(val)

        lookups[name] = {k: preproc_fn(merge_versions(v)) for k, v in lookup.items()}

    heap = []
    used = {k: set() for k in lookups}

    for name, lookup in lookups.items():
        for a, b in combinations(lookup, 2):
            score = score_fn(lookups[name][a], lookups[name][b])
            heappush(heap, Pair(name, a, b, score))

    while True:
        if not heap or len(combos) <= max_vers:
            break

        name, a, b = heappop(heap)
        if a in used[name] or b in used[name]:
            continue

        c = tuple(sorted(set(a + b)))
        lookup = lookups[name]
        lookup[c] = merge_fn(lookup[a], lookup[b])
        lookup.pop(a)
        lookup.pop(b)
        used[name].add(a)
        used[name].add(b)

        for other in lookup:
            if other != c:
                score = score_fn(lookups[name][c], lookups[name][other])
                heappush(heap, Pair(name, c, other, score))

        i = names.index(name)
        merge_a = {}
        merge_b = {}
        for combo in combos:
            parts = combo[i]
            if a == parts:
                key = tuple(v for j, v in enumerate(combo) if j != i)
                merge_a[key] = combo
            elif b == parts:
                key = tuple(v for j, v in enumerate(combo) if j != i)
                merge_b[key] = combo

        for key in sorted(set(merge_a.keys()).union(merge_b.keys())):
            if key in merge_a and key in merge_b:
                template = merge_a[key]
                a_val = combos.pop(merge_a[key])
                b_val = combos.pop(merge_b[key])
                c_val = merge_versions([a_val, b_val])
            elif key in merge_a:
                template = merge_a[key]
                c_val = combos.pop(template)
            else:
                template = merge_b[key]
                c_val = combos.pop(template)

            new_c = tuple(k if j != i else c for j, k in enumerate(template))
            combos[new_c] = c_val

    return tuple(combos.values())


def _estimate_columns_for_chain_recurse(
    ver: TableVersion, smeta: dict[str, TableMeta], meta: dict[str, Attributes]
):
    base_cols = 0
    for attr in meta[ver.name].values():
        base_cols += attr.common is not None
        base_cols += len(attr.vals)

    if smeta[ver.name].order is not None:
        mult = smeta[ver.name].order
        if mult is not None:
            base_cols = mult * (base_cols - 1)

    along = smeta[ver.name].along
    if along and ver.unrolls:
        base_cols -= len(along) + 1
        base_cols += len(along) * len(ver.unrolls)

    out = {ver.name: base_cols}
    for p in ver.parents:
        if isinstance(p, TablePartition):
            p = p.table
        out.update(_estimate_columns_for_chain_recurse(p, smeta, meta))

    return out


def estimate_columns_for_chain(ver: TableVersion, meta: dict[str, Attributes]):
    """Estimates the number of columns for the provided chain based on metadata."""
    return sum(
        _estimate_columns_for_chain_recurse(
            ver, _calculate_stripped_meta(meta), meta
        ).values()
    )


def calc_model_num(combos: dict[str, tuple[set[int]]]):
    num = 1
    for c in combos.values():
        num *= len(c)

    return num


def get_parents(ver: TableVersion) -> Generator[TablePartition, None, None]:
    for p in ver.parents:
        if isinstance(p, TablePartition):
            yield p
            yield from get_parents(p.table)
        else:
            yield from get_parents(p)


def tuple_unique(a, b):
    if not a:
        return b
    if not b:
        return a
    return tuple(sorted(set(a + b)))


def merge_versions(vers: Sequence[TableVersion]):
    ref = vers[0]

    new_parents = []
    for i, p in enumerate(ref.parents):
        if isinstance(p, TablePartition):
            new_partitions = set()
            par_versions = []
            for ver in vers:
                partitions, table = cast(TablePartition, ver.parents[i])
                new_partitions.update(partitions)
                par_versions.append(table)
            new_parents.append(
                TablePartition(
                    tuple(sorted(new_partitions)), merge_versions(par_versions)
                )
            )
        else:
            new_parents.append(
                merge_versions([cast(TableVersion, ver.parents[i]) for ver in vers])
            )

    if ref.children is not None:
        children = max(cast(int, v.children) for v in vers)
    else:
        children = None

    if ref.partitions:
        partitions = set()
        for ver in vers:
            partitions.update(ver.partitions)  # type: ignore
        partitions = tuple(sorted(partitions))
    else:
        partitions = None

    if ref.unrolls:
        unrolls = set()
        for ver in vers:
            unrolls.update(ver.unrolls)  # type: ignore
        unrolls = tuple(sorted(unrolls))
    else:
        unrolls = None

    rows = 0
    for ver in vers:
        rows += ver.rows

    lens = [v.max_len for v in vers if v.max_len is not None]
    max_len = max(lens) if lens else None

    return TableVersion(
        name=ref.name,
        rows=rows,
        children=children,
        max_len=max_len,
        partitions=partitions,
        unrolls=unrolls,
        parents=tuple(new_parents),
    )


def calc_rows_cols(
    combo: dict[str, tuple[set[int]]],
    chains: tuple[TableVersion, ...],
    rows: dict[TableVersion | TablePartition, int],
    meta: dict[str, Attributes],
) -> list[tuple[TableVersion, int, int]]:
    out = []
    for partitions in product(*combo.values()):
        partitions = {k: v for k, v in zip(combo, partitions)}
        versions = []

        for ver in chains:
            reject = False
            for p in get_parents(ver):
                if p.partitions[0] not in partitions[p.table.name]:
                    reject = True
                    break
            if not reject:
                versions.append(ver)

        if not versions:
            continue

        new_version = merge_versions(versions)
        row_count = sum(rows[v] for v in versions)
        col_count = estimate_columns_for_chain(new_version, meta)
        out.append((new_version, row_count, col_count))
    return out
