import logging
from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations
from time import perf_counter
from typing import cast

import networkx as nx
import numpy as np

from ..attribute import CatValue, DatasetAttributes
from .utils import enchanced_display

logger = logging.getLogger(__name__)


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


def elimination_order_greedy(
    g: nx.Graph,
    attrs: DatasetAttributes,
    stochastic: bool = False,
    display: bool = False,
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

            sels = defaultdict(dict)
            for var in new_factor:
                table = g.nodes[var]["table"]
                attr = g.nodes[var]["attr"]
                val = g.nodes[var]["value"]
                height = g.nodes[var]["height"]
                
                if (table, attr) in sels and val in sels[(table, attr)]:
                    height = min(sels[(table, attr)][val], height)
                sels[(table, attr)][val] = height

            dom = 1
            for (table, attr), sel in sels.items():
                cmn = attrs[table][attr].common
                if cmn and cmn.name in sel and len(sel) > 1:
                    sel = dict(sel)
                    sel.pop(cmn.name)
                    dom *= CatValue.get_domain_multiple(
                        list(sel.values()),
                        [cast(CatValue, attrs[table][attr][v]) for v in sel.keys()],
                    )
                elif cmn and cmn.name in sel:
                    dom *= cmn.get_domain(sel[cmn.name])
                else:
                    dom *= CatValue.get_domain_multiple(
                        list(sel.values()),
                        [cast(CatValue, attrs[table][attr][v]) for v in sel.keys()],
                    )

            costs.append(dom)
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
                # Apply operations in both the
                for k in (g, triangulated):
                    k.add_edge(a, b, triangulated=True)

        if display:
            logger.info(f"Removing node `{popped}` with cost: {costs[idx]:_d}")
            g.nodes[popped]["marked"] = True
            enchanced_display(g)
        g.remove_node(popped)
        order.append(popped)

    if display:
        logger.info(f"Final cordal graph with cost {total_cost}:")
        enchanced_display(triangulated)
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
