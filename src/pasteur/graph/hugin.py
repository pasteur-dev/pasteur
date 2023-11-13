import logging
from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations
from time import perf_counter
from typing import Collection, Literal, NamedTuple, cast

import networkx as nx
import numpy as np

from ..attribute import Attributes, CatValue, DatasetAttributes, SeqAttributes
from .utils import display_induced_graph

logger = logging.getLogger(__name__)


class AttrMeta(NamedTuple):
    table: str | None
    order: int | None
    attr: str | tuple[str, ...]
    sel: int | tuple[tuple[str, int], ...]


CliqueMeta = tuple[AttrMeta, ...]


def get_attrs(attrs: DatasetAttributes, table: str | None, order: int | None) -> Attributes:
    if order is not None:
        tattrs = cast(SeqAttributes, attrs[table]).hist[order]
    else:
        tattrs = attrs[table]
        if isinstance(tattrs, SeqAttributes):
            tattrs = cast(Attributes, tattrs.attrs)
    return tattrs


def create_clique_meta(
    cl: Collection[str], g: nx.Graph, attrs: DatasetAttributes
) -> CliqueMeta:
    """Creates a hashable metadata holder for tuples with a fixed ordering."""

    sels = defaultdict(dict)
    for var in cl:
        table = g.nodes[var]["table"]
        order = g.nodes[var]["order"]
        attr = g.nodes[var]["attr"]
        val = g.nodes[var]["value"]
        height = g.nodes[var]["height"]

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
            new_sel = []
            for val, h in sel.items():
                if val == cmn:
                    continue  # skip common
                new_sel.append((val, h))
            new_sel = tuple(sorted(new_sel))
        out.append(AttrMeta(table, order, attr_name, new_sel))

    return tuple(sorted(out, key=lambda x: x[:-1]))


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


def get_factor_domain(factor: Collection[str], g: nx.Graph, attrs: DatasetAttributes):
    meta = create_clique_meta(factor, g, attrs)

    dom = 1
    for table, order, attr_name, sel in meta:
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


def elimination_order_greedy(
    g: nx.Graph,
    attrs: DatasetAttributes,
    stochastic: bool = False,
    display: bool = False,
    condensed: bool = True,
):
    triangulated = deepcopy(g)
    g = deepcopy(g)

    order = []
    total_cost = 0
    for _ in range(len(g)):
        costs = []

        unmarked = list(g)
        for a in unmarked:
            cls = nx.find_cliques(g, nodes=[a])
            # @Warning: traversing set, code might not be reproducible
            new_factor = set(chain.from_iterable(cls))
            costs.append(get_factor_domain(new_factor, g, attrs))
        costs = np.array(costs)

        if stochastic:
            c = 1 / costs
            p = c / c.sum()
            idx = np.random.choice(len(p), p=p)
        else:
            idx = np.argmin(costs)
        total_cost += costs[idx]

        popped = unmarked[idx]
        for a, b in combinations(g[popped], 2):
            if not g.has_edge(a, b):
                # Apply operations in both the triangulated graph
                # and standin graph
                for k in (g, triangulated):
                    k.add_edge(a, b, triangulated=True)

        if display:
            logger.info(f"Removing node `{popped}` with cost: {costs[idx]:_d}")
            g.nodes[popped]["marked"] = True
            display_induced_graph(g, condensed=condensed)
        g.remove_node(popped)
        order.append(popped)

    if display:
        logger.info(f"Final cordal graph with cost {total_cost}:")
        display_induced_graph(triangulated, condensed=condensed)
        logger.info(f"Elimination order:\n{order}")

    return order, triangulated, total_cost


def find_elim_order(g: nx.Graph, attrs: DatasetAttributes, max_time: float = 10):
    start = perf_counter()
    min_order, min_triag, min_cost = elimination_order_greedy(g, attrs, False)

    while perf_counter() - start < max_time:
        order, triag, cost = elimination_order_greedy(g, attrs, True)
        if cost < min_cost:
            min_order = order
            min_cost = cost
            min_triag = triag

    return min_order, min_triag, min_cost


def get_junction_tree(
    triangulated: nx.Graph,
    attrs: DatasetAttributes,
    metric: Literal["domain", "common"] = "domain",
):
    full_tree = nx.Graph()

    for a, b in combinations(nx.find_cliques(triangulated), 2):
        full_tree.add_edge(
            create_clique_meta(a, triangulated, attrs),
            create_clique_meta(b, triangulated, attrs),
            common=len(set(a) & set(b)),
            domain=get_factor_domain(set(a) & set(b), triangulated, attrs),
        )

    return nx.maximum_spanning_tree(full_tree, weight=metric)


def get_message_passing_order(junction: nx.Graph):
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
