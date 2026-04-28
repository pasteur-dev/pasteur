import logging
from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations

from typing import Collection, Literal, NamedTuple, Sequence, cast

import networkx as nx
import numpy as np

from ..attribute import Attributes, CatValue, DatasetAttributes, SeqAttributes
from .utils import display_induced_graph

logger = logging.getLogger(__name__)


def _none_safe_key(v):
    if v is None:
        return (0,)
    return (1, v)


class AttrMeta(NamedTuple):
    table: str | None
    order: int | None
    attr: str | tuple[str, ...]
    sel: int | tuple[tuple[str, int], ...]

    def _cmp_key(self):
        return (
            _none_safe_key(self.table),
            _none_safe_key(self.order),
            _none_safe_key(self.attr),
            _none_safe_key(self.sel),
        )

    def __lt__(self, other):
        if not isinstance(other, AttrMeta):
            return NotImplemented
        return self._cmp_key() < other._cmp_key()

    def __le__(self, other):
        if not isinstance(other, AttrMeta):
            return NotImplemented
        return self._cmp_key() <= other._cmp_key()

    def __gt__(self, other):
        if not isinstance(other, AttrMeta):
            return NotImplemented
        return self._cmp_key() > other._cmp_key()

    def __ge__(self, other):
        if not isinstance(other, AttrMeta):
            return NotImplemented
        return self._cmp_key() >= other._cmp_key()


CliqueMeta = tuple[AttrMeta, ...]


def get_attrs(
    attrs: DatasetAttributes, table: str | None, order: int | None
) -> Attributes:
    if order is not None:
        tattrs = cast(SeqAttributes, attrs[table]).hist[order]
    else:
        tattrs = attrs[table]
        if isinstance(tattrs, SeqAttributes):
            tattrs = cast(Attributes, tattrs.attrs)
    return tattrs


def create_clique_meta(
    cl: Collection[str], g: nx.Graph, attrs: DatasetAttributes, compress: bool = True
) -> CliqueMeta:
    """Creates a hashable metadata holder for tuples with a fixed ordering."""

    sels = defaultdict(dict)
    for var in cl:
        table = g.nodes[var]["table"]
        order = g.nodes[var]["order"]
        attr = g.nodes[var]["attr"]
        val = g.nodes[var]["value"]
        height = g.nodes[var]["height"]

        if not compress:
            height = 0
        if (table, order, attr) in sels and val in sels[(table, order, attr)]:
            height = min(sels[(table, order, attr)][val], height)
        sels[(table, order, attr)][val] = height

    out = []
    for (table, order, attr_name), sel in sels.items():
        attr = get_attrs(attrs, table, order)[attr_name]
        if len(sel) == 1 and attr.common and next(iter(sel)) == attr.common.name:
            new_sel = sel[attr.common.name]
        else:
            cmn = attr.common.name if attr.common else None
            # Order by attr.vals insertion order so sel matches
            # get_mapping_multiple's reversed-product ordering.
            val_order = {vn: i for i, vn in enumerate(attr.vals)}
            new_sel = []
            for val, h in sel.items():
                if val == cmn:
                    continue  # skip common
                new_sel.append((val, h))
            new_sel = tuple(sorted(new_sel, key=lambda x: val_order.get(x[0], 0)))
        out.append(AttrMeta(table, order, attr_name, new_sel))

    return tuple(sorted(out, key=lambda x: tuple((0, "") if v is None else (1, v) for v in x[:-1])))


def to_moral(g: nx.DiGraph, to_undirected=True):
    h = deepcopy(g.to_undirected() if to_undirected else g)
    for descendent, preds in g.pred.items():
        for a, b in combinations(preds, r=2):
            if h.has_edge(a, b):
                continue

            h.edges[descendent, a]["immoral"] = True
            h.edges[descendent, b]["immoral"] = True
            h.add_edge(a, b, immorality=True)
    return h


def get_clique_domain(clique: CliqueMeta, attrs: DatasetAttributes):
    dom = 1
    for table, order, attr_name, sel in clique:
        attr = get_attrs(attrs, table, order)[attr_name]
        cmn = attr.common

        if isinstance(sel, int):
            assert cmn
            dom *= cmn.get_domain(sel)
        else:
            dom *= CatValue.get_domain_multiple(
                [v[1] for v in sel],
                [cast(CatValue, attr[v[0]]) for v in sel],
            )

    return dom


def get_factor_domain(factor: Collection[str], g: nx.Graph, attrs: DatasetAttributes):
    meta = create_clique_meta(factor, g, attrs)
    return get_clique_domain(meta, attrs)


def _factor_domain_direct(factor, g_nodes, attrs):
    """Compute factor domain directly from graph node data, skipping CliqueMeta."""
    # Group by (table, order, attr) -> {value: min_height}
    sels: dict[tuple, dict] = {}
    for var in factor:
        data = g_nodes[var]
        key = (data["table"], data["order"], data["attr"])
        val = data["value"]
        height = data["height"]
        if key in sels:
            s = sels[key]
            if val in s:
                if height < s[val]:
                    s[val] = height
            else:
                s[val] = height
        else:
            sels[key] = {val: height}

    dom = 1
    for (table, order, attr_name), sel in sels.items():
        attr = get_attrs(attrs, table, order)[attr_name]
        cmn = attr.common

        if len(sel) == 1 and cmn and cmn.name in sel:
            dom *= cmn.get_domain(sel[cmn.name])
        else:
            cmn_name = cmn.name if cmn else None
            heights = []
            vals = []
            for v, h in sel.items():
                if v == cmn_name:
                    continue
                heights.append(h)
                vals.append(cast(CatValue, attr[v]))
            if vals:
                dom *= CatValue.get_domain_multiple(heights, vals)

    return dom


def _node_cost(node, g, attrs, elim_factor_cost):
    factor = [node] + list(g[node])
    return _factor_domain_direct(factor, g.nodes, attrs) ** elim_factor_cost


def elimination_order_greedy(
    g: nx.Graph,
    attrs: DatasetAttributes,
    stochastic: bool = False,
    display: bool = False,
    condensed: bool = True,
    elim_factor_cost: float = 1,
    seed: int | None = None,
):
    orig = g
    g = deepcopy(g)
    rng = np.random.RandomState(seed) if stochastic else None

    # Initialize costs for all nodes
    cost_map = {}
    for node in g:
        cost_map[node] = _node_cost(node, g, attrs, elim_factor_cost)

    order = []
    fill_edges = []
    total_cost = 0
    for _ in range(len(cost_map)):
        unmarked = list(cost_map)
        costs = np.array([cost_map[a] for a in unmarked])

        if rng is not None:
            c = 1 / costs
            p = c / c.sum()
            idx = rng.choice(len(p), p=p)
        else:
            idx = np.argmin(costs)
        total_cost += costs[idx]

        popped = unmarked[idx]
        neighbors = set(g[popped])

        # Add fill edges to working graph, collect for triangulated
        for a, b in combinations(neighbors, 2):
            if not g.has_edge(a, b):
                g.add_edge(a, b, triangulated=True)
                fill_edges.append((a, b))

        if display:
            logger.info(f"Removing node `{popped}` with cost: {costs[idx]:_d}")
            g.nodes[popped]["marked"] = True
            display_induced_graph(g, condensed=condensed)
        g.remove_node(popped)
        del cost_map[popped]
        order.append(popped)

        # Only recompute costs for affected neighbors
        for nb in neighbors:
            if nb in cost_map:
                cost_map[nb] = _node_cost(nb, g, attrs, elim_factor_cost)

    if display:
        triangulated = deepcopy(orig)
        for a, b in fill_edges:
            triangulated.add_edge(a, b, triangulated=True)
        logger.info(f"Final cordal graph with cost {total_cost}:")
        display_induced_graph(triangulated, condensed=condensed)
        logger.info(f"Elimination order:\n{order}")

    return order, fill_edges, total_cost


def _triangulate(g: nx.Graph, fill_edges: list[tuple[str, str]]):
    """Build the triangulated graph from original graph + fill edges."""
    triangulated = deepcopy(g)
    for a, b in fill_edges:
        triangulated.add_edge(a, b, triangulated=True)
    return triangulated


def _elim_order_search(
    g: nx.Graph, attrs: DatasetAttributes, seeds: Sequence[int], elim_factor_cost: float = 1
):
    """Run stochastic elimination order search over the given seeds."""
    best = None
    for seed in seeds:
        order, fill, cost = elimination_order_greedy(
            g, attrs, True, elim_factor_cost=elim_factor_cost, seed=seed
        )
        if best is None or cost < best[2]:
            best = (order, fill, cost)
    return best


def find_elim_order(
    g: nx.Graph, attrs: DatasetAttributes, max_attempts: int = 5000, elim_factor_cost: float = 1
):
    from ..utils.progress import MULTIPROCESS_ENABLE, IS_SUBPROCESS, process_async
    from os import cpu_count

    # Greedy (deterministic) baseline
    min_order, min_fill, min_cost = elimination_order_greedy(
        g, attrs, False, elim_factor_cost=elim_factor_cost
    )

    if max_attempts <= 0:
        return min_order, _triangulate(g, min_fill), min_cost

    # Precompute deterministic seeds for all stochastic attempts
    seed_rng = np.random.RandomState(0)
    all_seeds = seed_rng.randint(0, 2**31, size=max_attempts)

    if MULTIPROCESS_ENABLE and not IS_SUBPROCESS:
        n_workers = max((cpu_count() or 1) // 2, 1)
        # Split seeds evenly across workers
        seed_chunks = np.array_split(all_seeds, n_workers)
        futures = [
            process_async(_elim_order_search, g, attrs, chunk, elim_factor_cost)
            for chunk in seed_chunks
        ]

        for f in futures:
            order, fill, cost = f.get()
            if cost < min_cost:
                min_order, min_fill, min_cost = order, fill, cost
    else:
        result = _elim_order_search(g, attrs, all_seeds, elim_factor_cost)
        if result[2] < min_cost:
            min_order, min_fill, min_cost = result

    return min_order, _triangulate(g, min_fill), min_cost


def get_junction_tree_from_cliques(
    cliques: Sequence[CliqueMeta],
):
    """Build a junction tree directly from a set of CliqueMeta tuples.

    Uses maximum spanning tree on the number of shared attributes.
    No triangulation or fill cliques — the cliques ARE the tree nodes."""
    full_tree = nx.Graph()
    for cl in cliques:
        full_tree.add_node(cl)

    for a, b in combinations(cliques, 2):
        shared = sum(
            1
            for ai in a
            for bi in b
            if ai.table == bi.table and ai.order == bi.order and ai.attr == bi.attr
        )
        if shared > 0:
            full_tree.add_edge(a, b, common=shared)

    return nx.maximum_spanning_tree(full_tree, weight="common")


def get_junction_tree(
    triangulated: nx.Graph,
    attrs: DatasetAttributes,
    metric: Literal["domain", "common"] = "domain",
    compress: bool = True,
):
    full_tree = nx.Graph()

    for a, b in combinations(nx.find_cliques(triangulated), 2):
        full_tree.add_edge(
            create_clique_meta(a, triangulated, attrs, compress=compress),
            create_clique_meta(b, triangulated, attrs, compress=compress),
            common=len(set(a) & set(b)),
            domain=get_factor_domain(set(a) & set(b), triangulated, attrs),
        )

    return nx.maximum_spanning_tree(full_tree, weight=metric)


def get_message_passing_order(
    junction: nx.Graph,
) -> Sequence[Sequence[tuple[CliqueMeta, CliqueMeta]]]:
    # The messages that need to be sent are
    # all directed versions of the junction tree edges
    messages = nx.DiGraph()
    for a, b in junction.edges:
        messages.add_node((a, b))
        messages.add_node((b, a))

    # In order for the message a -> b to be sent,
    # all n -> a messages need to have been received (other than b -> a)
    for a, b in messages.nodes:
        for n in junction.neighbors(a):
            if b != n:
                messages.add_edge((n, a), (a, b))

    return tuple(nx.topological_generations(messages))


def cap_heights(g: nx.Graph, mode: str = "hugin_unvalley"):
    """Cap node heights in-place to reduce resolution bottlenecks.

    ``hugin_uncomp``: for each (attr, val), keep at most the 2 lowest
    heights and collapse everything above to the 2nd-lowest.
      [0, 2]       → [0, 2]       (no change)
      [0, 2, 3, 4] → [0, 2, 2, 2] (cap at 2nd-lowest)

    ``hugin_unvalley`` (default): only collapse heights above the first
    non-consecutive gap in the sorted chain.
      [0, 2]       → [0, 2]       (no change)
      [0, 2, 3, 4] → [0, 2, 3, 4] (consecutive, keep)
      [0, 3, 5]    → [0, 3, 3]    (3→5 gap, cap at 3)
    """
    height_sets: dict[tuple, list[int]] = {}
    for _, data in g.nodes(data=True):
        key = (data["table"], data["order"], data["attr"], data["value"])
        height_sets.setdefault(key, []).append(data["height"])

    for key, hs in height_sets.items():
        unique = sorted(set(hs))
        if len(unique) <= 2:
            continue

        if mode == "hugin_uncomp":
            cap = unique[1]
        else:
            # hugin_unvalley: walk chain, cap at first non-consecutive gap
            cap = unique[1]
            for i in range(2, len(unique)):
                if unique[i] != unique[i - 1] + 1:
                    break
                cap = unique[i]
            else:
                continue  # fully consecutive, no gap

        for _, data in g.nodes(data=True):
            k = (data["table"], data["order"], data["attr"], data["value"])
            if k == key and data["height"] > cap:
                data["height"] = cap
