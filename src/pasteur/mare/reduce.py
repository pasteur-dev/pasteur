"""This module contains heuristics for simplifying the chain combinations of a
dataset."""

from collections import defaultdict
from functools import reduce
from itertools import combinations
from typing import NamedTuple

from ..attribute import Attributes
from .chains import TableChain


class PartitionedNode(NamedTuple):
    vers: set[TableChain]
    children: dict[tuple[int, ...], dict[str, "TableNode | PartitionedNode"]]
    vals: set[int]


class TableNode(NamedTuple):
    vers: set[TableChain]
    children: dict[str, "TableNode | PartitionedNode"]
    vals: set[int]


def merge_dicts(a, b):
    if not isinstance(a, dict):
        assert not isinstance(b, dict)
        return merge_nodes(a, b)

    keys = {**a, **b}
    out = {}
    for k in keys:
        if k in a and k in b:
            out[k] = merge_dicts(a[k], b[k])
        else:
            out[k] = a.get(k, b.get(k, None))
    return out


def merge_nodes(a, b):
    # merge vals
    new_vals = a.vals.union(b.vals)
    new_vers = a.vers.union(b.vers)

    # Merge children
    if isinstance(a, PartitionedNode):
        assert isinstance(b, PartitionedNode)
        new_children = merge_dicts(a.children, b.children)
        return PartitionedNode(new_vers, new_children, new_vals)
    elif isinstance(a, TableNode):
        assert isinstance(b, TableNode)
        new_children = merge_dicts(a.children, b.children)
        return TableNode(new_vers, new_children, new_vals)
    else:
        assert False


def create_node(ver: TableChain, children: dict[TableChain, tuple[TableChain, ...]]):
    vers = {ver}

    if children.get(ver, None):
        child_arr = defaultdict(list)
        for c in children[ver]:
            child_arr[c.name].append(create_node(c, children))

        new_children = {k: reduce(merge_nodes, v) for k, v in child_arr.items()}
    else:
        new_children = {}

    vals = set(ver.unroll.vals) if ver.unroll else set()
    if ver.partition:
        return PartitionedNode(vers, {ver.partition.val: new_children}, vals)
    else:
        return TableNode(vers, new_children, vals)


def _compute_version_graph(
    ver: TableChain, out: dict[TableChain, dict[TableChain, None]]
):
    for parent in ver.parents:
        out[parent][ver] = None
        _compute_version_graph(parent, out)


def compute_version_graph(
    vers: tuple[TableChain],
) -> dict[TableChain, tuple[TableChain, ...]]:
    out = defaultdict(dict)
    for v in vers:
        _compute_version_graph(v, out)

    return {k: tuple(v) for k, v in out.items()}


def print_node(node: TableNode | PartitionedNode, ofs=0, pre=""):
    out = ""
    pre = "\t" * ofs + pre
    table_name = next(iter(node.vers)).name
    out += f"{pre}> {table_name} V({len(node.vers)}) vals={node.vals}"

    if node.children:
        out += ":\n"
    if isinstance(node, PartitionedNode):
        for vals, table_chld in node.children.items():
            for child in table_chld.values():
                out += print_node(child, ofs + 1, str(vals))
    else:
        for child in node.children.values():
            out += print_node(child, ofs + 1)  # type: ignore

    if node.children:
        out += "\n"
    return out


def print_nodes(heads: dict[str, TableNode | PartitionedNode]):
    out = ""
    for k, v in heads.items():
        out += f"{k}\n-------------\n"
        out += print_node(v)
    return out


def calc_col_increase(a: TableNode | PartitionedNode, b: TableNode | PartitionedNode):
    inc = len(a.vals.symmetric_difference(b.vals))

    if isinstance(a, TableNode):
        assert isinstance(b, TableNode)

        for k in a.children.keys():
            if k not in b.children:
                continue

            inc += calc_col_increase(a.children[k], b.children[k])
    elif isinstance(a, PartitionedNode):
        assert isinstance(b, PartitionedNode)

        for k in a.children.keys():
            if k not in b.children:
                continue

            for l in a.children[k]:
                if l not in b.children[k]:
                    continue

                inc += calc_col_increase(a.children[k][l], b.children[k][l])

    return inc


def calculate_merge_order(node: PartitionedNode):
    partitions = dict(node.children)

    resolution = []
    inc_cache = {}
    while len(partitions) > 2:
        mins = None
        mins_inc = -1
        for x in combinations(partitions, 2):
            if x in inc_cache:
                inc = inc_cache[x]
            else:
                inc = 0
                a, b = partitions[x[0]], partitions[x[1]]
                for k in a.keys():
                    if k not in b:
                        continue
                    inc += calc_col_increase(a[k], b[k])
                inc_cache[x] = inc

            if inc == 0:
                mins = x
                break
            elif mins is None or inc < mins_inc:
                mins = x
                mins_inc = inc

        assert mins is not None
        a, b = mins
        c = tuple(sorted(set(a).union(set(b))))

        part_a = partitions[a]
        part_b = partitions[b]

        part_c = {}
        for k in {**part_a, **part_b}:
            if k in part_a and k in part_b:
                part_c[k] = merge_nodes(part_a[k], part_b[k])
            else:
                part_c[k] = part_a.get(k, part_b.get(k, None))

        resolution.append(tuple(partitions))
        del partitions[a]
        del partitions[b]
        partitions[c] = part_c

    return tuple(resolution)


def calculate_variations(chains: tuple[TableChain], rows: dict[TableChain, int]):
    children = compute_version_graph(chains)
    partitioned = {}
    sequential = {}
    for v in children:
        if v.partition:
            new_node = create_node(v, children)

            if v.name in partitioned:
                partitioned[v.name] = merge_nodes(partitioned[v.name], (new_node))
            else:
                partitioned[v.name] = new_node
        elif v.sequence:
            if v.name in sequential:
                assert v.sequence.order == sequential[v.name]
            else:
                sequential[v.name] = v.sequence.order

    out = {}
    for p, head in partitioned.items():
        out[p] = ("partitioned", calculate_merge_order(head))

    for s, order in sequential.items():
        out[s] = ("sequential", tuple(range(order)))

    return out


def _estimate_columns_for_chain_recurse(ver: TableChain, meta: dict[str, Attributes]):
    base_cols = 0
    for attr in meta[ver.name].values():
        base_cols += attr.common is not None
        base_cols += len(attr.vals)

    if ver.sequence:
        mult = ver.sequence.seq or ver.sequence.order
        base_cols = mult * (base_cols - 1)

    if ver.unroll:
        base_cols -= len(ver.unroll.along) + 1
        base_cols += len(ver.unroll.along) * len(ver.unroll.vals)

    out = {ver.name: base_cols}
    for p in ver.parents:
        out.update(_estimate_columns_for_chain_recurse(p, meta))

    return out


def estimate_columns_for_chain(ver: TableChain, meta: dict[str, Attributes]):
    """Estimates the number of columns for the provided chain based on metadata."""
    return sum(_estimate_columns_for_chain_recurse(ver, meta).values())
