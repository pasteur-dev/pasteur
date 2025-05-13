""" Highly experimental and unpublished class for rebalancing Stratified Values
with Differential Privacy.

@TODO: Documentation."""

import logging
from itertools import combinations, product
from math import ceil, log
from typing import Any, Sequence, TypeVar, cast

import numpy as np

from .attribute import (
    Attributes,
    CatValue,
    Grouping,
    StratifiedValue,
    get_dtype,
    Attribute,
)

logger = logging.getLogger(__name__)

ZERO_FILL = 1e-24


class CommonNode(list["CommonNode | OrdNode | CatNode | set[int]"]):
    pass


class OrdNode(list["CommonNode | OrdNode | CatNode | set[int]"]):
    pass


class CatNode(list["CommonNode | OrdNode | CatNode | set[int]"]):
    pass


def create_tree(
    node: Grouping, common: Grouping | Any, ofs: int = 0, n: int | None = None
) -> CommonNode | OrdNode | CatNode:
    """Receives the top node of the tree of a hierarchical attribute and
    converts it into the same tree structure, where the leaves have been
    replaced by bucket groups, with each bucket group containing the node it replaced.
    """
    if n is None:
        n = node.get_domain(0)

    if isinstance(common, Grouping):
        out = CommonNode()
        assert len(node) == len(common)
    elif node.type == "cat":
        out = CatNode()
    else:
        out = OrdNode()

    for i, child in enumerate(node):
        cmn = common[i] if isinstance(common, Grouping) else None

        if isinstance(child, Grouping):
            out.append(create_tree(child, cmn, ofs, n))
        else:
            out.append(set([ofs]))

        ofs += 1

    return out


def get_group_size(counts: np.ndarray, g: set[int]) -> float:
    """Returns the sum of the sizes of the buckets included in set `g`.

    `counts` is a list with the size of each bucket. `group_sizes` is a dynamic
    programming cache, which is used to save the size of each group."""
    s = 0
    for i in g:
        s += int(counts[i])
    return s


def print_group(counts: np.ndarray | None, g: set[int]) -> str:
    """Converts the group into a string representation:
    `(False, False, True, True, False)` -> `(2,3)`"""
    s = "("
    for i in sorted(g):
        s += f"{i},"
    base = s[:-1] + ")"
    if counts:
        return base + f":{get_group_size(counts, g)}"
    return base


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
            s += print_group(None, n)
        s += ","
    s = s[:-1] + ("]" if isinstance(node, OrdNode) else "}")
    return s


def merge_groups_in_node(node: list, a: set, b: set):
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
            node.append(a.union(b))
        elif child == b:
            pass
        else:
            node.append(child)


def prune_tree(tree: list):
    """Replaces all nodes in the tree where `len(node) == 1` with their children."""
    for i, child in enumerate(tree):
        if not isinstance(child, list):
            continue

        if len(child) == 1 and not isinstance(child, CommonNode):
            tree[i] = child[0]
        else:
            prune_tree(child)


def get_common_sizes(tree):
    if not isinstance(tree, CommonNode):
        return [len(tree)]

    out = []
    for n in tree:
        out.extend(get_common_sizes(n))
    return out


N = TypeVar("N", CatNode, OrdNode, set)


def find_smallest_group(
    counts: np.ndarray, parent: CommonNode | OrdNode | CatNode
) -> tuple[list, set, set, float]:
    """Finds groups `a`, `b` which when combined form `c`, where `c` is the smallest
    group that can be formed by any two nodes in the tree, which are valid to merge.

    Returns the parent `node` of `a` and `b`, `a`, `b`, and the size of the resulting group.

    Can be used with `merge_groups_in_node()` and `prune_tree()` to merge the two smallest
    groups in the tree.

    `parent` represents a tree over the hierarchy of the attribute.
    Children can either be lists (nodes), sets (leafs, groups), or None
    (placeholders, common values, shouldn't be merged).
    """
    s_node = []
    s_a = set()
    s_b = set()
    s_size = -1

    # First do a recursive pass
    for child in parent:
        if isinstance(child, list):
            node, a, b, size = find_smallest_group(counts, child)
            if s_size == -1 or size < s_size:
                s_node = node
                s_a = a
                s_b = b
                s_size = size

    # Secondly, consider children
    if isinstance(parent, CommonNode):
        # Do not modify Common Node contents
        pass
    elif isinstance(parent, OrdNode):
        # For ordinal nodes we only check nearby nodes
        for i in range(len(parent) - 1):
            a = parent[i]
            if not isinstance(a, set):
                continue
            b = parent[i + 1]
            if not isinstance(b, set):
                continue

            size = get_group_size(counts, a.union(b))
            if s_size == -1 or size < s_size:
                s_node = parent
                s_a = a
                s_b = b
                s_size = size
    else:
        # Find two smallest parents and pair
        sizes = np.array([get_group_size(counts, a) if isinstance(a, set) else np.nan for a in parent])
        i, j = np.argsort(sizes)[:2]
        size = float(sizes[i] + sizes[j])
        if s_size == -1 or size < s_size:
            s_node = parent
            s_a = parent[i]
            s_b = parent[j]
            s_size = size

        # # For categorical nodes we check all pairs
        # for i, j in combinations(range(len(parent)), 2):
        #     a = parent[i]
        #     if not isinstance(a, set):
        #         continue
        #     b = parent[j]
        #     if not isinstance(b, set):
        #         continue

        #     size = get_group_size(counts, a.union(b))
        #     if s_size == -1 or size < s_size:
        #         s_node = parent
        #         s_a = a
        #         s_b = b
        #         s_size = size

    return s_node, s_a, s_b, s_size


def create_node_to_group_map(
    tree: CommonNode | OrdNode | CatNode, grouping: np.ndarray, ofs: int = 0
):
    """Traverses `tree` and creates an array which maps nodes to groups such that:

    `arr[x] = y`, where `x` is the node and `y` is its group.

    `grouping` is updated in place to form arr recursively and `ofs` is
    used to keep track of the current group number during recursion."""

    for child in tree:
        if child is None:
            # unclean, assumes when `None`, `ofs <= common`, ie the first `common`
            # buckets will be None
            grouping[ofs] = ofs
            ofs += 1
        elif isinstance(child, set):
            for i in child:
                grouping[i] = ofs
            ofs += 1
        else:
            ofs = create_node_to_group_map(child, grouping, ofs)

    return ofs


def get_len(tree):
    if isinstance(tree, set):
        return len(tree)
    return sum([get_len(c) for c in tree])


def get_children(tree):
    if isinstance(tree, set):
        for c in tree:
            yield c
    else:
        for c in tree:
            yield from get_children(c)


def get_common_groups(tree, ofs: int = 0):
    if isinstance(tree, CommonNode):
        out = []
        for n in tree:
            c, ofs = get_common_groups(n, ofs)
            out.extend(c)
        return out, ofs
    else:
        if isinstance(tree, set):
            return [[0 for _ in range(len(tree))]], ofs + len(tree)
        else:
            arr = [0 for _ in range(get_len(tree))]
            tmp = 0
            for i, c in enumerate(tree):
                for child in get_children(c):
                    arr[child - ofs] = i
                    tmp += 1

            return [arr], ofs + tmp


def make_grouping(counts: np.ndarray, head: Grouping, common: Grouping | None):
    cmn = len(common) if common else 2
    tree = create_tree(head, common)
    n = head.get_domain(0)
    grouping = np.empty((n - cmn + 1, n), dtype=get_dtype(n))

    common_sizes = [get_common_sizes(tree)]
    common_groups = [get_common_groups(tree)[0]]
    create_node_to_group_map(tree, grouping[0, :])

    for i in range(1, n - cmn + 1):
        node, a, b, _ = find_smallest_group(counts, tree)
        merge_groups_in_node(node, a, b)
        prune_tree(tree)
        common_sizes.append(get_common_sizes(tree))
        common_groups.append(get_common_groups(tree)[0])
        create_node_to_group_map(tree, grouping[i, :])

    return grouping, common_sizes, common_groups


def generate_domain_list(
    max_domain: int, common: int, u: float = 1.3, fixed: list[int] = [2, 4, 5, 8, 12]
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
    until surpassing `max_domain`, and the last value is replaced by `max_domain`.

    `common` is a correction that ensures the minimum domain is bigger or equal
    to `common` + 1."""

    # Start by applying the fixed domain values
    # If the fixed domain list goes higher than the domain of the attribute
    # use the fixed list values that are lower, and append the maximum value at the end
    new_domains = []

    if common != 0:
        new_domains.append(common)

    for i, dom in enumerate(fixed):
        if dom >= max_domain:
            new_domains.append(max_domain)
            break
        elif dom > common + 1:
            new_domains.append(dom)

    # If the fixed values don't go that high, continue by adding values that increase
    # by u, yielding log(max_domain, u) levels
    fixed_max_dom = new_domains[-1]
    if fixed_max_dom < max_domain:
        new_level_n = ceil(log(max_domain / fixed_max_dom, u))

        for i in range(1, new_level_n):
            dom = ceil(fixed_max_dom * u**i)
            new_domains.append(int(dom))

        new_domains.append(max_domain)

    return new_domains


class RebalancedValue(CatValue):
    def __init__(
        self,
        counts: np.ndarray,
        col: StratifiedValue,
        *,
        reshape_domain: bool = True,
        u: float = 1.3,
        fixed: list[int] = [2, 4, 5, 8, 12],
        c: float | None = None,
        **_,
    ) -> None:
        self.original = col

        self.name = col.name
        self.ignore_nan = col.ignore_nan
        self.counts = counts
        self.common = col.common
        self.grouping, self.common_sizes, self.common_groups = make_grouping(
            counts, col.head, self.common
        )
        self.c = c
        self.reshape_domain = reshape_domain

        if reshape_domain:
            max_domain = self.grouping.shape[1]
            domains = generate_domain_list(
                max_domain, self.common.domain if self.common else 0, u, fixed
            )
            self.domains = domains
            self.height_to_grouping = [max_domain - dom for dom in reversed(domains)]

        else:
            self.height_to_grouping = list(range(len(self.grouping)))

    def get_domain(self, height: int) -> int:
        return int(self.grouping[self.height_to_grouping[height], :].max() + 1)

    def get_mapping(self, height: int) -> np.ndarray:
        return self.grouping[self.height_to_grouping[height], :]

    def select_height(self) -> int:
        assert (
            not self.reshape_domain
        ), "Fixme: selected_height function is not adjusted to the lowered domain list (neither should it)"

        if self.c is None:
            return 0

        domains = np.max(self.grouping, axis=1) + 1

        count_list = []
        for i in range(len(domains)):
            d = domains[i]
            count = np.zeros((d,))

            for j in range(len(self.counts)):
                count[self.grouping[i, j]] += self.counts[j]

            count_list.append(count)

        entropies = []
        for count in count_list:
            p = count / count.max() + ZERO_FILL
            entropies.append(-np.sum(p * np.log2(p)))

        entropies = np.array(entropies)
        h = (1 + self.c / np.log2(domains)) * entropies

        return int(np.nanargmax(h))

    @property
    def height(self) -> int:
        return len(self.height_to_grouping)

    def is_ordinal(self) -> bool:
        return False

    def upsample(self, column: np.ndarray, height: int, deterministic: bool = True):
        if height == 0:
            return column
        if deterministic:
            return super().upsample(column, height)

        d = self.get_domain(height)
        mapping = self.get_mapping(height)

        # create reverse mapping
        upsampled = np.empty_like(column)
        for g in range(d):
            mask = column == g
            group_size = int(np.sum(mask))

            group_mask = mapping == g
            group_counts = self.counts[group_mask]
            group_idx = np.argwhere(group_mask)[:, 0]
            p = group_counts.clip(0)
            p /= p.sum()
            if np.any(np.isnan(p)):
                logger.debug(f"Found na column during upsampling: {self.name}")
                # Sampling uniformely instead of crashing
                p = 1 / (len(p))

            upsampled[mask] = np.random.choice(group_idx, p=p, size=(group_size,))

        return upsampled

    @staticmethod
    def get_domain_multiple(heights: Sequence[int], vals: Sequence[CatValue]):
        assert len(vals) == len(heights)

        dom = 0
        for l in range(len(cast(RebalancedValue, vals[0]).common_sizes[0])):
            l_dom = 1
            for v, h in zip(vals, heights):
                assert isinstance(v, RebalancedValue)
                l_dom *= v.common_sizes[v.height_to_grouping[h]][l]
            dom += l_dom
        return dom

    @staticmethod
    def get_mapping_multiple(
        heights: Sequence[int] | int,
        common: "RebalancedValue",
        vals: Sequence["RebalancedValue"],
    ) -> np.ndarray:
        if isinstance(heights, int):
            assert common
            all_ofs = common.grouping[common.height_to_grouping[heights]]
            ofs = 0

            out = []
            for l, ofs in enumerate(all_ofs):
                combos = [v.common_sizes[0][l] for v in vals]
                nums = list(combos)
                finished = False
                while not finished:
                    out.append(ofs)
                    for i in range(len(nums)):
                        if nums[i] > 1:
                            nums[i] -= 1
                            break
                        elif i == len(nums) - 1:
                            finished = True
                            break
                        else:
                            nums[i] = combos[i]
        else:
            assert len(vals) == len(heights)

            out = []
            ofs = 0
            for l in range(len(vals[0].common_groups[0])):
                groupings = [
                    (
                        v.common_groups[v.height_to_grouping[h]][l]
                        if h != -1
                        else [0 for _ in range(v.common_sizes[0][l])]
                    )
                    for v, h in reversed(list(zip(vals, heights)))
                ]
                domains = [max(g) + 1 for g in groupings]

                pass

                for combos in product(*groupings):
                    l_dom = 1
                    num = ofs
                    for dom, comb in zip(domains, combos):
                        num += l_dom * comb
                        l_dom *= dom
                    out.append(num)

                ofs = max(out) + 1

        return np.array(out, dtype=get_dtype(ofs))

    @staticmethod
    def get_naive_mapping_multiple(
        heights: Sequence[int] | int,
        common: "RebalancedValue",
        vals: Sequence["RebalancedValue"],
    ):
        if isinstance(heights, int):
            assert common
            all_ofs = common.grouping[common.height_to_grouping[heights]]
            ofs = 0

            out = []
            for l, ofs in enumerate(all_ofs):
                combos = [v.common_sizes[0][l] for v in vals]
                nums = list(combos)
                finished = False
                while not finished:
                    out.append(ofs)
                    for i in range(len(nums)):
                        if nums[i] > 1:
                            nums[i] -= 1
                            break
                        elif i == len(nums) - 1:
                            finished = True
                            break
                        else:
                            nums[i] = combos[i]

            return np.array(out, dtype=get_dtype(max(out)))
        else:
            assert len(vals) == len(heights)

            out = []
            new_ofs = [0 for _ in range(len(vals))]
            domains = [v.grouping.shape[1] for v in vals]

            for l in range(len(vals[0].common_groups[0])):
                groupings = [
                    (
                        v.common_groups[v.height_to_grouping[h]][l]
                        if h != -1
                        else [0 for _ in range(v.common_sizes[0][l])]
                    )
                    for v, h in reversed(list(zip(vals, heights)))
                ]

                ofs = list(new_ofs)
                for combos in product(*groupings):
                    l_dom = 1
                    num = 0
                    for i, c in enumerate(combos):
                        val = ofs[i] + c
                        new_ofs[i] = max(val, new_ofs[i])
                        num += val * l_dom + ofs[i]
                        l_dom *= domains[i]
                    out.append(num)

                for i in range(len(new_ofs)):
                    new_ofs[i] += 1

            return np.array(out, dtype=get_dtype(max(out)))


def rebalance_attributes(
    counts: dict[str, np.ndarray],
    attrs: Attributes,
    warn: bool = True,
    **kwargs,
):
    new_attrs = {}
    for name, attr in attrs.items():
        acommon = attr.common
        if acommon:
            assert isinstance(acommon, StratifiedValue)
            if acommon.name not in counts:
                # If common is not sampled, substitute with the value with the
                # smallest domain
                sub_val = min(
                    [v for v in attr.vals.values() if isinstance(v, CatValue)],
                    key=lambda v: v.domain,
                )
                acounts = sub_val.get_mapping(sub_val.height - 1)[counts[sub_val.name]]
            else:
                acounts = counts[acommon.name]

            common = RebalancedValue(acounts, acommon, **kwargs)
        else:
            common = None

        cols = []
        for col_name, col in attr.vals.items():
            if isinstance(col, StratifiedValue):
                cols.append(
                    RebalancedValue(
                        counts[col_name],
                        col,
                        **kwargs,
                    )
                )
            else:
                cols.append(col)

        new_attr = Attribute(name, cols, common)
        new_attrs[name] = new_attr

    return new_attrs
