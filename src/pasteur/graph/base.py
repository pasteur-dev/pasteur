from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations
from typing import cast

import networkx as nx
import numpy as np

from ..attribute import CatValue, DatasetAttributes
from time import perf_counter


def to_moral(g: nx.DiGraph, to_undirected=True):
    h = deepcopy(g.to_undirected() if to_undirected else g)
    for descendent, preds in g.pred.items():
        for a, b in combinations(preds, r=2):
            if h.has_edge(a, b):
                continue

            h.edges[descendent, a]["immoral"] = True
            h.edges[descendent, b]["immoral"] = True
            h.add_edge(a, b, immorality=True)
            h.nodes[a]["heights"][b] = h.nodes[a]["heights"][descendent]
            h.nodes[b]["heights"][a] = h.nodes[b]["heights"][descendent]
    return h


def elimination_order_greedy(
    g: nx.Graph, attrs: DatasetAttributes, stochastic: bool = False
):
    g = g.copy()
    order = []
    total_cost = 0

    for _ in range(len(g)):
        costs = []

        unmarked = list(g)
        for a in unmarked:
            cls = nx.find_cliques(g, nodes=[a])
            # @Warning: traversing set, code might not be reproducible
            new_factor = set(chain.from_iterable(cls))

            heights = {}
            for var in new_factor:
                for n, h in g.nodes[var]["heights"].items():
                    if n in new_factor and (var not in heights or h < heights[var]):
                        heights[var] = h

            sels = defaultdict(dict)
            for var in new_factor:
                table = g.nodes[var]["table"]
                attr = g.nodes[var]["attr"]
                val = g.nodes[var]["value"]
                h = heights.get(var, 0)
                sels[(table, attr)][val] = h

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
                g.add_edge(a, b)
                g.nodes[a]["heights"][b] = g.nodes[a]["heights"][popped]
                g.nodes[b]["heights"][a] = g.nodes[b]["heights"][popped]
        g.remove_node(popped)
        # enchanced_display(g)
        order.append(popped)

    return order, total_cost


def find_elim_order(g: nx.Graph, attrs: DatasetAttributes, max_time: float = 10):
    start = perf_counter()
    min_order, min_cost = elimination_order_greedy(g, attrs, False)

    while perf_counter() - start < max_time:
        order, cost = elimination_order_greedy(g, attrs, True)
        if cost < min_cost:
            min_order = order
            min_cost = cost

    return min_order, min_cost
