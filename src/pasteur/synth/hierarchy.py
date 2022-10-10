from itertools import combinations
from math import ceil, log

import numpy as np

from ..transform.attribute import IdxColumn, Level, LevelColumn, get_dtype


def get_group_for_x(x: int, n: int) -> tuple[bool]:
    """Returns a tuple `g` with length `n`, where `g[x]=True` and others `False`.

    A tuple `g` represents a bucket group, where it states which buckets it contains with True/False.

    Tuples are used because they are immutable and can be used in sets/dict keys,
    allowing for dynamic programming caches."""
    return tuple(i == x for i in range(n))


def merge_groups(a: tuple[bool], b: tuple[bool]) -> tuple[bool]:
    """Merges groups a, b into group c containing the superset of both groups."""
    return tuple(i or j for i, j in zip(a, b))


class OrdNode(list):
    pass


class CatNode(list):
    pass


def create_tree(node: Level, ofs: int = 0, n: int = None) -> list:
    """Receives the top node of the tree of a hierarchical attribute and
    converts it into the same tree structure, where the leaves have been
    replaced by bucket groups, with each bucket group containing the node it replaced."""
    if n is None:
        n = node.get_domain(0)
    out = CatNode() if node.type == "cat" else OrdNode()

    for child in node:
        if isinstance(child, Level):
            out.append(create_tree(child, ofs, n))
        else:
            out.append(get_group_for_x(ofs, n))

        ofs += 1

    return out


def get_group_size(
    counts: list[float], g: tuple, group_sizes: dict[tuple, float] = {}
) -> float:
    """Returns the sum of the sizes of the buckets included in tuple `g`.

    `counts` is a list with the size of each bucket. `group_sizes` is a dynamic
    programming cache, which is used to save the size of each group."""
    if g in group_sizes:
        return group_sizes[g]

    s = 0
    for i, present in enumerate(g):
        if present:
            s += counts[i]

    group_sizes[g] = s
    return s


def print_group(counts: np.array, g: tuple) -> str:
    """Converts the group into a string representation:
    `(False, False, True, True, False)` -> `(2,3)`"""
    s = "("
    for i, present in enumerate(g):
        if present:
            s += f"{i},"
    return s[:-1] + "):" + str(counts, get_group_size(g))


def print_tree(node: list):
    """Converts a tree into a human ledgible representation.
    Groups are converted using `print_group`, shown together with list format.

    `{}` are used for nodes that have categorical children,
    `[]` are used for nodes with ordinal children.

    Mirroring the set (unordered), list (ordered) format of python."""
    s = "[" if isinstance(node, OrdNode) else "{"
    for n in node:
        if isinstance(n, list):
            s += print_tree(n)
        else:
            s += print_group(n)
        s += ","
    s = s[:-1] + ("]" if isinstance(node, OrdNode) else "}")
    return s


def merge_groups_in_node(node: list, a: tuple, b: tuple):
    """Computes `c = a intersection b` and inserts it into `node`, by
    replacing `a` with `c` and removing `b`.

    This is valid for both ordinal nodes, where `a`, `b`, are next to each other,
    and categorical nodes, where `b` might be in any place.

    If `len(node) == 2`, node won't be replaced its child due to python limitations.
    `prune_tree()` can be called to replace all nodes where `len(node) == 1` with
    their children.
    """

    tmp_node = node.copy()
    node.clear()

    for child in tmp_node:
        if child == a:
            node.append(merge_groups(a, b))
        elif child == b:
            pass
        else:
            node.append(child)


def prune_tree(tree: list):
    """Replaces all nodes in the tree where `len(node) == 1` with their children."""
    for i, child in enumerate(tree):
        if not isinstance(child, list):
            continue

        if len(child) == 1:
            tree[i] = child[0]
        else:
            prune_tree(child)


def find_smallest_group(counts: np.array, parent: list):
    """Finds groups `a`, `b` which when combined form `c`, where `c` is the smallest
    group that can be formed by any two nodes in the tree, which are valid to merge.

    Returns the parent `node` of `a` and `b`, `a`, `b`, and the size of the resulting group.

    Can be used with `merge_groups_in_node()` and `prune_tree()` to merge the two smallest
    groups in the tree."""
    s_node = None
    s_a = None
    s_b = None
    s_size = -1

    # First do a recursive pass
    for child in parent:
        if isinstance(child, list):
            node, a, b, size = find_smallest_group(child)
            if s_size == -1 or size < s_size:
                s_node = node
                s_a = a
                s_b = b
                s_size = size

    # Secondly, consider children
    if isinstance(parent, OrdNode):
        # For ordinal nodes we only check nearby nodes
        for i in range(len(parent) - 1):
            a = parent[i]
            if isinstance(a, list):
                continue
            b = parent[i + 1]
            if isinstance(b, list):
                continue

            size = get_group_size(counts, merge_groups(a, b))
            if s_size == -1 or size < s_size:
                s_node = parent
                s_a = a
                s_b = b
                s_size = size
    else:
        # For categorical nodes we check all pairs
        for i, j in combinations(range(len(parent)), 2):
            a = parent[i]
            if isinstance(a, list):
                continue
            b = parent[j]
            if isinstance(b, list):
                continue

            size = get_group_size(counts, merge_groups(a, b))
            if s_size == -1 or size < s_size:
                s_node = parent
                s_a = a
                s_b = b
                s_size = size

    return s_node, s_a, s_b, s_size


def create_node_to_group_map(tree: list, n: int, ofs: int = 0):
    """Traverses `tree` and creates an array which maps nodes to groups such that:

    `arr[x] = y`, where `x` is the node and `y` is its group.

    `n` is the number of discrete values of the attribute and `ofs` is
    used to keep track of the current group number during recursion."""

    grouping = np.empty((n,))

    for child in tree:
        if isinstance(child, tuple):
            for i, present in enumerate(child):
                if present:
                    grouping[i] = ofs
            ofs += 1
        else:
            ofs = create_node_to_group_map(child, grouping, ofs)

    return ofs


def make_grouping(counts: np.array, head: Level) -> np.ndarray:
    """Converts the hierarchical attribute level tree provided into a node-to-group
    mapping, where `group[i][j] = z`, where `i` is the height of the mapping
    `j` is node `j` and `z` is the group the node is associated at that height.
    """

    tree = create_tree(head)
    n = head.get_domain(0)
    grouping = np.empty((n - 1, n), dtype=get_dtype(n))

    for i in range(0, n - 1):
        node, a, b, _ = find_smallest_group(counts, tree)
        merge_groups_in_node(node, a, b)
        prune_tree(tree)
        grouping[i, :] = create_node_to_group_map(tree, n)

    return grouping


def generate_domain_list(
    max_domain: int, u: float = 1.3, fixed: list[int] = [2, 4, 5, 8, 12]
):
    """Takes in a `max_domain` value and uses it to produce a new increasing domain
    list, based on `u` and `fixed`.

    The strategy used is increasing the domain every time by the ratio `u`,
    where `u > 1`. For example:
    if `u = 1.3`, domain 10 becomes 13 etc.

    For low domain values, however, this leads to repeating values and low increase,
    so the `fixed` domain list is used to specify the starting domain values.

    If the `fixed` list goes higher than `max_domain`, only the values up to `max_domain`
    are kept, and `max_domain` is placed at the end.

    Otherwise, the last `fixed` value is multiplied by `u` and ceiled. This repeats
    until surpassing `max_domain`, and the last value is replaced by `max_domain`."""

    # Start by applying the fixed domain values
    # If the fixed domain list goes higher than the domain of the attribute
    # use the fixed list values that are lower, and append the maximum value at the end
    new_domains = []
    for i, dom in enumerate(fixed):
        if dom >= max_domain:
            new_domains.append(max_domain)
            break
        else:
            new_domains.append(dom)

    # If the fixed values don't go that high, continue by adding values that increase
    # by u, yielding log(max_domain, u) levels
    fixed_max_dom = fixed[-1]
    if fixed_max_dom < max_domain:
        new_level_n = ceil(log(max_domain / fixed_max_dom, u))

        for i in range(1, new_level_n):
            dom = ceil(fixed_max_dom * u**i)
            new_domains.append(int(dom))

        new_domains.append(max_domain)

    return new_domains


class RebalancedColumn(IdxColumn):
    def __init__(
        self,
        counts: np.array,
        col: LevelColumn,
        reshape_domain: bool = True,
        u: float = 1.3,
        fixed: list[int] = [2, 4, 5, 8, 12],
    ) -> None:
        self.grouping = make_grouping(counts, col.head)

        if reshape_domain:
            max_domain = self.grouping.shape[1]
            domains = generate_domain_list(max_domain, u, fixed)

            self.height_to_grouping = [max_domain - dom for dom in reversed(domains)]

        else:
            self.height_to_grouping = list(range(len(self.grouping)))

    def get_domain(self, height: int) -> int:
        return self.grouping[self.height_to_grouping[height], :].max()

    def get_mapping(self, height: int) -> np.array:
        return self.grouping[self.height_to_grouping[height], :]

    @property
    def height(self) -> int:
        return len(self.height_to_grouping)

    def is_ordinal(self) -> bool:
        return False
