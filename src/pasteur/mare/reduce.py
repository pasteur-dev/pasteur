"""This module contains heuristics for simplifying the chain combinations of a
dataset."""

from collections import defaultdict
from functools import reduce
from itertools import combinations, product
from typing import Any, Generator, Literal, NamedTuple, cast

from ..attribute import Attributes
from .chains import TableMeta, TablePartition, TableVersion, _calculate_stripped_meta


class TableNode(NamedTuple):
    vers: set[TableVersion]
    children: dict[str, "TableNode | PartitionedNode"]
    vals: set[int]


class PartitionedNode(NamedTuple):
    vers: set[TablePartition]
    children: dict[tuple[int, ...], TableNode]


def merge_partitions(a: PartitionedNode, b: PartitionedNode):
    new_vers = a.vers.union(b.vers)
    new_children = {}
    keys = list({**a.children, **b.children})
    for k in keys:
        if k in a.children and k in b.children:
            new_children[k] = merge_nodes(a.children[k], b.children[k])
        else:
            new_children[k] = a.children.get(k, b.children.get(k, None))

    return PartitionedNode(new_vers, new_children)


def merge_nodes(a: TableNode, b: TableNode):
    # merge vals
    new_vals = a.vals.union(b.vals)
    new_vers = a.vers.union(b.vers)

    # Merge children
    keys = list({**a.children, **b.children})
    new_children = {}
    for k in keys:
        if k not in a.children:
            new_children[k] = b.children[k]
        elif k not in b.children:
            new_children[k] = a.children[k]
        else:
            ac = a.children[k]
            bc = b.children[k]
            if isinstance(ac, TableNode):
                assert isinstance(bc, TableNode)
                new_children[k] = merge_nodes(ac, bc)
            else:
                assert isinstance(bc, PartitionedNode)
                new_children[k] = merge_partitions(ac, bc)

    return TableNode(new_vers, new_children, new_vals)


def merge_check(a: TableNode | PartitionedNode, b: TableNode | PartitionedNode):
    if isinstance(a, TableNode):
        assert isinstance(b, TableNode)
        return merge_nodes(a, b)
    else:
        assert isinstance(b, PartitionedNode)
        return merge_partitions(a, b)


def create_partition(
    p: TablePartition,
    children: dict[
        TableVersion | TablePartition, tuple[TableVersion | TablePartition, ...]
    ],
):
    if p in children:
        child_arr = []
        for c in children[p]:
            assert isinstance(c, TableVersion)
            child_arr.append(create_node(c, children))

        new_children = {p.partitions: reduce(merge_nodes, child_arr)}
    else:
        new_children = {}
    return PartitionedNode({p}, new_children)


def create_node(
    ver: TableVersion,
    children: dict[
        TableVersion | TablePartition, tuple[TableVersion | TablePartition, ...]
    ],
):
    vers = {ver}

    if ver in children:
        child_arr = defaultdict(list)
        for c in children[ver]:
            if isinstance(c, TableVersion):
                child_arr[c.name].append(create_node(c, children))
            else:
                child_arr[c.table.name].append(create_partition(c, children))

        new_children = {k: reduce(merge_check, v) for k, v in child_arr.items()}
    else:
        new_children = {}

    vals = set(ver.unrolls) if ver.unrolls else set()
    return TableNode(vers, new_children, vals)


def _compute_version_graph(
    ver: TableVersion | TablePartition,
    out: dict[TableVersion | TablePartition, dict[TableVersion | TablePartition, None]],
):
    if isinstance(ver, TableVersion):
        for parent in ver.parents:
            out[parent][ver] = None
            _compute_version_graph(parent, out)
    else:
        out[ver.table][ver] = None
        _compute_version_graph(ver.table, out)


def compute_version_graph(
    vers: tuple[TableVersion],
) -> dict[TableVersion | TablePartition, tuple[TableVersion | TablePartition, ...]]:
    out = defaultdict(dict)
    for v in vers:
        _compute_version_graph(v, out)

    return {k: tuple(v) for k, v in out.items()}


def print_node(node: TableNode, ofs=0, pre=""):
    out = ""
    pre = "\t" * ofs + pre
    table_name = next(iter(node.vers)).name
    out += f"{pre}> {table_name} V({len(node.vers)}) vals={node.vals}"

    if node.children:
        out += ":\n"
        for child in node.children.values():
            if isinstance(child, PartitionedNode):
                for vals, table_chld in child.children.items():
                    out += print_node(table_chld, ofs + 1, str(vals))
            else:
                out += print_node(child, ofs + 1)  # type: ignore

    if node.children:
        out += "\n"
    return out


def print_nodes(heads: dict[str, TableNode]):
    out = ""
    for k, v in heads.items():
        out += f"{k}\n-------------\n"
        out += print_node(v)
    return out


def calc_col_increase(a: TableNode, b: TableNode):
    # Removed checks for speed
    inc = 0
    if a.vals and b.vals:
        inc = len(a.vals.symmetric_difference(b.vals))

    for k in a.children:
        # if k not in b.children:
        #     continue

        l = a.children[k]
        m = b.children[k]

        if isinstance(l, TableNode):
            # assert isinstance(m, TableNode)
            inc += calc_col_increase(l, m)  # type: ignore
        else:
            # assert isinstance(m, PartitionedNode)
            for j in l.children:
                if j not in m.children:
                    continue
                inc += calc_col_increase(l.children[j], m.children[j])  # type: ignore
    return inc


def calculate_merge_order(node: TableNode) -> tuple[tuple[set[int], ...], ...]:
    data = defaultdict(dict)
    for table, part in node.children.items():
        assert isinstance(part, PartitionedNode)
        for pids, node in part.children.items():
            data[pids][table] = node

    resolution = []
    inc_cache = {}
    while len(data) > 2:
        mins = None
        mins_inc = -1
        for x in combinations(data, 2):
            if x in inc_cache:
                inc = inc_cache[x]
            else:
                inc = 0
                a, b = data[x[0]], data[x[1]]
                for name in a.keys():
                    if name not in b.keys():
                        continue
                    inc += calc_col_increase(a[name], b[name])
                inc_cache[x] = inc

            if inc == 0:
                mins = x
                break
            elif mins is None or inc < mins_inc:
                mins = x
                mins_inc = inc

        assert mins is not None
        a, b = mins
        c = tuple(sorted(set(a + b)))

        part_a = data[a]
        part_b = data[b]

        part_c = {}
        for k in {**part_a, **part_b}:
            if k in part_a and k in part_b:
                part_c[k] = merge_nodes(part_a[k], part_b[k])
            else:
                part_c[k] = part_a.get(k, part_b.get(k, None))

        resolution.append(tuple(map(set, data)))
        del data[a]
        del data[b]
        data[c] = part_c

    return tuple(resolution)


def calculate_variations(
    chains: tuple[TableVersion],
    rows: dict[TableVersion | TablePartition, int],
    meta: dict[str, Attributes],
) -> dict[str, tuple[tuple[set[int], ...], ...]]:
    smeta = _calculate_stripped_meta(meta)
    children = compute_version_graph(chains)
    partitioned = {}
    for v in children:
        if isinstance(v, TableVersion) and smeta[v.name].partition:
            new_node = create_node(v, children)

            if v.name in partitioned:
                partitioned[v.name] = merge_nodes(partitioned[v.name], (new_node))
            else:
                partitioned[v.name] = new_node

    out = {}
    for p, head in partitioned.items():
        out[p] = calculate_merge_order(head)

    return out


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


def _get_parents(ver: TableVersion) -> Generator[TablePartition, None, None]:
    for p in ver.parents:
        if isinstance(p, TablePartition):
            yield p
            yield from _get_parents(p.table)
        else:
            yield from _get_parents(p)


def tuple_unique(a, b):
    if not a:
        return b
    if not b:
        return a
    return tuple(sorted(set(a + b)))


def merge_versions(vers: list[TableVersion]):
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

    return TableVersion(ref.name, partitions, unrolls, tuple(new_parents))


def calc_rows_cols(
    combo: dict[str, tuple[set[int]]],
    chains: tuple[TableVersion],
    rows: dict[TableVersion | TablePartition, int],
    meta: dict[str, Attributes],
) -> list[tuple[TableVersion, int, int]]:
    out = []
    for partitions in product(*combo.values()):
        partitions = {k: v for k, v in zip(combo, partitions)}
        versions = []

        for ver in chains:
            reject = False
            for p in _get_parents(ver):
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


def choose_variation(
    variations: dict[str, tuple[tuple[set[int], ...], ...]],
    chains: tuple[TableVersion],
    rows: dict[TableVersion | TablePartition, int],
    meta: dict[str, Attributes],
    model_num: int,
) -> tuple[TableVersion, ...]:
    solutions = []
    valid = 0
    for combo in product(*variations.values()):
        combo = {k: v for k, v in zip(variations, combo)}

        if not calc_model_num(combo) <= model_num:
            continue

        solutions.append(calc_rows_cols(combo, chains, rows, meta)[0])
        # print(valid)
        valid += 1

    return tuple(max(solutions, key=lambda x: len(x)))

    # max_cols = [max(s[2] for s in sol) for sol in solutions]
    # min_max_col = min(max_cols)

    # for sol in solutions:
    #     print(f"N: {len(sol):4d} Max Cols: {max(s[2] for s in sol)}, Min rows: {min(s[1] for s in sol):5d}, AVG Cols: {sum(s[2] for s in sol) / len(sol):.1f}")

    # sol = solutions[30]
    # plt.hist([s[2] for s in sol], bins=20)
    # plt.title("Columns")

    # plt.figure()
    # plt.hist([s[1] for s in sol], bins=20, range=(0, 10_000))
    # plt.title("Rows (up to 10k)")

    # plt.figure()
    # plt.hist([s[1] for s in sol], bins=20, range=(0, 50_000))
    # plt.title("Rows (up to 50k)")

    # plt.figure()
    # plt.hist([s[1] for s in sol], bins=20)
    # plt.title("Rows Unbounded")


def choose_versions_heuristic(
    chains: tuple[TableVersion],
    rows: dict[TableVersion | TablePartition, int],
    meta: dict[str, Attributes],
    model_num: int,
):
    variations = calculate_variations(chains, rows, meta)
    return choose_variation(variations, chains, rows, meta, model_num)
