"""Adjuvant: DP structure learning via greedy edge addition with height-chain nodes.

Core implementation: cached marginal computation, noisy TVD scoring,
height-chain graph construction, greedy edge addition with exponential
mechanism, and measurement/observation building.
"""

import itertools
import logging
from math import sqrt
from typing import NamedTuple, Sequence, cast

import networkx as nx
import numpy as np

from ....attribute import Attributes, CatValue, DatasetAttributes, SeqAttributes
from ....marginal import MarginalOracle

logger = logging.getLogger(__name__)


# ============================================================
# Data structures
# ============================================================
class CachedMarginals(NamedTuple):
    """Pre-computed 1-way and 2-way true marginals from a single data pass."""

    one_way: dict[str, np.ndarray]  # attr_name -> count vector (flat)
    two_way: dict[tuple[str, str], np.ndarray]  # (attr_a, attr_b) -> joint counts


# ============================================================
# Step 0: Single data pass
# ============================================================
def compute_all_marginals(
    oracle: MarginalOracle,
    attrs: DatasetAttributes,
    all_attrs: list[str],
) -> CachedMarginals:
    """Batch-query all 1-way and 2-way true marginals in one oracle call."""
    from ..sota.common import _attr_sel

    requests_1 = [[(a, _attr_sel(a, attrs))] for a in all_attrs]
    pairs = list(itertools.combinations(all_attrs, 2))
    requests_2 = [
        [(a, _attr_sel(a, attrs)), (b, _attr_sel(b, attrs))] for a, b in pairs
    ]

    results = oracle.process(requests_1 + requests_2, postprocess=None)

    one_way: dict[str, np.ndarray] = {}
    for a, r in zip(all_attrs, results[: len(all_attrs)]):
        one_way[a] = r.ravel().astype(np.float64)

    two_way: dict[tuple[str, str], np.ndarray] = {}
    for (a, b), r in zip(pairs, results[len(all_attrs) :]):
        two_way[a, b] = r.astype(np.float64)

    return CachedMarginals(one_way, two_way)


# ============================================================
# Step 1: Noisy 1-way marginals
# ============================================================
def add_noise_1way(
    cached: CachedMarginals,
    rho1: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Add Gaussian noise to 1-way marginals with budget rho1.

    Returns (noisy_marginals, sigma1)."""
    d = len(cached.one_way)
    if rho1 <= 0 or d == 0:
        return {k: v.copy() for k, v in cached.one_way.items()}, 0.0

    sigma = sqrt(d / (2 * rho1))
    noisy = {
        name: mar + np.random.normal(0, sigma, size=mar.size)
        for name, mar in cached.one_way.items()
    }
    return noisy, sigma


# ============================================================
# Step 2b: Noisy TVD
# ============================================================
def compute_tvd(
    cached: CachedMarginals,
) -> dict[tuple[str, str], float]:
    """Compute exact pairwise TVD = ||P(A,B) - P(A)P(B)||_1 / 2.

    No noise added — the exponential mechanism provides privacy for selection."""
    # Normalize 1-way to probabilities
    p1: dict[str, np.ndarray] = {}
    for a, m in cached.one_way.items():
        s = m.sum()
        p1[a] = m / s if s > 0 else m

    tvd: dict[tuple[str, str], float] = {}
    for (a, b), joint in cached.two_way.items():
        flat = joint.ravel()
        s = flat.sum()
        p_ab = flat / s if s > 0 else flat
        indep = np.outer(p1[a], p1[b]).ravel()
        val = float(np.sum(np.abs(p_ab - indep)) / 2)
        tvd[a, b] = val
        tvd[b, a] = val

    return tvd


# ============================================================
# Step 2a: Height-chain graph
# ============================================================
def _node_name(table, order, attr, value, height) -> str:
    out = ""
    if table:
        out += str(table)
        if order is not None:
            out += f"[{order}]"
        out += "_"
    return out + f"{attr}.{value}[{height}]"


def build_height_chain_graph(attrs: DatasetAttributes) -> nx.DiGraph:
    """Build directed height-chain graph (chain edges only, no cross-attribute).

    Nodes are (attr, value, height) with metadata for table/order.
    Chain edges connect successive heights within the same (attr, value).
    Common values connect to child values at the boundary."""
    g = nx.DiGraph()

    for table, tattrs in attrs.items():
        if isinstance(tattrs, SeqAttributes):
            attr_sets: dict = {**tattrs.hist, None: tattrs.attrs}
        else:
            attr_sets = {None: tattrs}

        for order, attr_set in attr_sets.items():
            if not attr_set:
                continue
            for name, attr in attr_set.items():
                cmn = attr.common
                if cmn:
                    for h in range(cmn.height):
                        g.add_node(
                            _node_name(table, order, name, cmn.name, h),
                            table=table,
                            order=order,
                            attr=name,
                            value=cmn.name,
                            height=h,
                            is_common=True,
                        )
                        if h > 0:
                            g.add_edge(
                                _node_name(table, order, name, cmn.name, h),
                                _node_name(table, order, name, cmn.name, h - 1),
                            )

                for v in attr.vals.values():
                    if not isinstance(v, CatValue):
                        continue
                    h_range = v.height if cmn is None else v.height - 1
                    for h in range(h_range):
                        g.add_node(
                            _node_name(table, order, name, v.name, h),
                            table=table,
                            order=order,
                            attr=name,
                            value=v.name,
                            height=h,
                            is_common=False,
                        )
                        if h > 0:
                            g.add_edge(
                                _node_name(table, order, name, v.name, h),
                                _node_name(table, order, name, v.name, h - 1),
                            )
                    if cmn and h_range > 0:
                        g.add_edge(
                            _node_name(table, order, name, cmn.name, 0),
                            _node_name(table, order, name, v.name, h_range - 1),
                        )

    return g


# ============================================================
# Step 2c: Edge candidates
# ============================================================
def generate_candidates(
    g: nx.DiGraph,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], list[int]]]:
    """Generate cross-attribute edge candidates at all height combinations.

    Only non-common value nodes are candidates for cross-attribute edges.
    Candidates are grouped by attribute pair for batch removal after selection.

    Returns:
        candidates: List of (node_a, node_b) edge candidates.
        attr_pair_map: Maps sorted (attr_a, attr_b) -> list of indices into candidates.
    """
    # Group non-common nodes by attribute name
    attr_nodes: dict[str, list[str]] = {}
    for node, data in g.nodes(data=True):
        if data.get("is_common", False):
            continue
        attr_nodes.setdefault(data["attr"], []).append(node)

    candidates: list[tuple[str, str]] = []
    attr_pair_map: dict[tuple[str, str], list[int]] = {}
    attr_names = sorted(attr_nodes.keys())

    for i, attr_a in enumerate(attr_names):
        for attr_b in attr_names[i + 1 :]:
            pair_key = (attr_a, attr_b)
            attr_pair_map[pair_key] = []
            for na in attr_nodes[attr_a]:
                for nb in attr_nodes[attr_b]:
                    attr_pair_map[pair_key].append(len(candidates))
                    candidates.append((na, nb))

    return candidates, attr_pair_map


# ============================================================
# Step 2d: Graph scoring helpers
# ============================================================
def _triangulate_simple(g: nx.Graph) -> nx.Graph:
    """Min-degree elimination triangulation (fast, for scoring during search)."""
    work = g.copy()
    fill: list[tuple[str, str]] = []
    while work.number_of_nodes() > 0:
        v = min(work.nodes(), key=lambda n: work.degree(n))
        for u, w in itertools.combinations(work.neighbors(v), 2):
            if not work.has_edge(u, w):
                work.add_edge(u, w)
                fill.append((u, w))
        work.remove_node(v)
    result = g.copy()
    result.add_edges_from(fill)
    return result


def _build_adj(g: nx.Graph) -> dict[str, set[str]]:
    """Build adjacency sets from a networkx graph (once, then copy for reuse)."""
    return {v: set(g.neighbors(v)) for v in g.nodes()}


def _score_with_one_edge(
    base_adj: dict[str, set[str]],
    na: str,
    nb: str,
    node_data: dict[str, dict],
    attrs: DatasetAttributes,
    max_clique_size: float,
) -> tuple[float, bool]:
    """Add one edge to cached adjacency, triangulate, compute total domain.

    Works entirely on dicts/sets — no networkx calls. Extracts maximal cliques
    from the elimination order directly.

    Returns (total_domain, valid). valid=False if any clique exceeds max_clique_size."""
    from ....graph.hugin import get_clique_domain, create_clique_meta, AttrMeta

    # Shallow-copy adjacency and add the candidate edge
    adj = {v: s.copy() for v, s in base_adj.items()}
    adj[na].add(nb)
    adj[nb].add(na)

    # Min-degree elimination: collect cliques as we go
    remaining = set(adj)
    cliques: list[frozenset[str]] = []

    while remaining:
        # Pick min-degree node
        v = min(remaining, key=lambda n: len(adj[n] & remaining))
        neighbors = adj[v] & remaining

        # The factor {v} ∪ neighbors is a potential clique
        factor = frozenset(neighbors | {v})

        # Add fill edges between neighbors
        nb_list = list(neighbors)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                u, w = nb_list[i], nb_list[j]
                if w not in adj[u]:
                    adj[u].add(w)
                    adj[w].add(u)

        remaining.discard(v)

        # Factor is maximal if no previously collected clique is a superset
        is_maximal = True
        for c in cliques:
            if factor <= c:
                is_maximal = False
                break
        if is_maximal:
            cliques.append(factor)

    # Compute total domain, check constraints
    total_domain = 0
    for cl in cliques:
        # Build CliqueMeta inline from node_data (avoid networkx lookups)
        from collections import defaultdict

        sels: dict[tuple, dict[str, int]] = defaultdict(dict)
        for var in cl:
            d = node_data[var]
            key = (d["table"], d["order"], d["attr"])
            val = d["value"]
            h = d["height"]
            if key in sels and val in sels[key]:
                h = min(sels[key][val], h)
            sels[key][val] = h

        meta = []
        from ....graph.hugin import get_attrs as _get_attrs

        for (table, order, attr_name), sel in sels.items():
            attr = _get_attrs(attrs, table, order)[attr_name]
            if len(sel) == 1 and attr.common and next(iter(sel)) == attr.common.name:
                new_sel = sel[attr.common.name]
            else:
                cmn = attr.common.name if attr.common else None
                new_sel = tuple(sorted((v, h) for v, h in sel.items() if v != cmn))
            meta.append(AttrMeta(table, order, attr_name, new_sel))
        clique_meta = tuple(sorted(meta, key=lambda x: x[:-1]))

        dom = get_clique_domain(clique_meta, attrs)
        if dom > max_clique_size:
            return 0, False
        total_domain += dom

    return total_domain, True


def _fmt_node(node: str, g, attrs) -> str:
    """Format a graph node as 'attr.val[h] (dom=X)'."""
    from ....graph.hugin import get_attrs as _get_attrs

    d = g.nodes[node]
    val = cast(CatValue, _get_attrs(attrs, d["table"], d["order"])[d["attr"]][d["value"]])
    dom = val.get_domain(d["height"])
    return f"{d['attr']}.{d['value']}[{d['height']}] (dom={dom})"


def _fmt_edge(na: str, nb: str, g, attrs) -> str:
    """Format a graph edge as 'attr.val[h] x attr.val[h] (domA, domB)'."""
    from ....graph.hugin import get_attrs as _get_attrs

    def _info(node):
        d = g.nodes[node]
        val = cast(CatValue, _get_attrs(attrs, d["table"], d["order"])[d["attr"]][d["value"]])
        dom = val.get_domain(d["height"])
        return f"{d['attr']}.{d['value']}[{d['height']}]", dom

    a_str, a_dom = _info(na)
    b_str, b_dom = _info(nb)
    return f"{a_str} x {b_str} ({a_dom}x{b_dom}={a_dom*b_dom})"


def compute_edge_weight(
    node_a: str,
    node_b: str,
    g: nx.Graph,
    attrs: DatasetAttributes,
    size_penalty: float = 0.3,
) -> float:
    from ....graph.hugin import get_attrs as _get_attrs

    def _weight_for_node(node: str) -> float:
        d = g.nodes[node]
        a = _get_attrs(attrs, d["table"], d["order"])[d["attr"]]
        val = cast(CatValue, a[d["value"]])
        h = d["height"]
        dom_h = val.get_domain(h)
        return 1 / dom_h if dom_h > 0 else 1.0

    return 1 / (1 + (_weight_for_node(node_a) * _weight_for_node(node_b))**size_penalty)


def score_graph(
    moral: nx.Graph,
    tvd: dict[tuple[str, str], float],
    attrs: DatasetAttributes,
    structure_edges: set[frozenset[str]],
    max_clique_size: float,
    size_penalty: float,
) -> float:
    """Score: sum(TVD * boost for structure edges) - penalty * total_domain.

    Returns -inf if any clique exceeds max_clique_size."""
    from ....graph.hugin import get_factor_domain

    tri = moral if nx.is_chordal(moral) else _triangulate_simple(moral)

    total_domain = 0
    for clique in nx.find_cliques(tri):
        dom = get_factor_domain(set(clique), tri, attrs)
        if dom > max_clique_size:
            return float("-inf")
        total_domain += dom

    tvd_sum = 0.0
    for edge in structure_edges:
        na, nb = tuple(edge)
        da, db = moral.nodes[na], moral.nodes[nb]
        base = tvd.get((da["attr"], db["attr"]), 0.0)
        boost = compute_edge_weight(na, nb, moral, attrs, size_penalty)
        tvd_sum += base * boost

    return tvd_sum - size_penalty * total_domain


# ============================================================
# Step 2d: Structure learning main loop
# ============================================================
def structure_learn(
    directed_graph: nx.DiGraph,
    attrs: DatasetAttributes,
    tvd: dict[tuple[str, str], float],
    n: int,
    size_penalty: float,
    rho2_exp: float,
    min_tvd: float,
) -> tuple[nx.Graph, set[frozenset[str]], float]:
    """Greedy edge addition with exponential mechanism.

    Uses local penalty (domain ratio of affected cliques, not total graph size),
    per-attribute edge limits (1 to sqrt(d)), and forces at least one edge per
    attribute.

    Returns:
        moral: Undirected moralized graph with structure-learning edges added.
        structure_edges: Set of frozenset node pairs for structure-learning edges.
        rho2_exp_remaining: Unspent exponential mechanism budget.
    """
    from ..sota.common import exponential_mechanism
    from ....graph.hugin import to_moral, get_factor_domain
    from ....utils.progress import piter, check_exit

    # Moralize the directed height-chain graph -> undirected base
    moral = to_moral(directed_graph)

    # Generate candidates and group by attribute pair
    candidates, attr_pair_map = generate_candidates(directed_graph)
    connected_pairs: set[tuple[str, str]] = set()
    structure_edges: set[frozenset[str]] = set()

    # Per-attribute edge limits: 1 to sqrt(d)
    d = len(set(a for pair in attr_pair_map for a in pair))
    max_edges_per_attr = min(d, 2 * int(sqrt(d) + 0.5))
    attr_edge_count: dict[str, int] = {}
    saturated_attrs: set[str] = set()

    max_steps = d * (max_edges_per_attr // 2 + 1)
    eps_per_step = sqrt(8 * rho2_exp / max_steps) if max_steps > 0 and rho2_exp > 0 else 0
    rho_spent = 0.0


    pbar = piter(
        None,
        total=max_steps,
        desc="Adjuvant structure [0 edges, score=0.0000]",
        unit="pair",
        bar_format=" " * 11
        + ">>>>>>>  {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
        + " [{elapsed}<{remaining}]",
    )

    # Precompute TVD*boost for all candidates (doesn't change across iterations)
    cand_tvd_boost = np.empty(len(candidates))
    for idx, (na, nb) in enumerate(candidates):
        da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
        base = tvd.get((da["attr"], db["attr"]), 0.0)
        boost = compute_edge_weight(na, nb, moral, attrs, size_penalty)
        cand_tvd_boost[idx] = base * boost

    for it in range(max_steps):
        try:
            check_exit()
        except Exception as e:
            pbar.close()
            raise e

        # Filter to active candidates:
        # - attribute pair not yet connected
        # - neither attribute saturated (reached max edges)
        active: list[tuple[int, str, str]] = []
        for idx, (na, nb) in enumerate(candidates):
            da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
            attr_a, attr_b = da["attr"], db["attr"]
            pair = tuple(sorted([attr_a, attr_b]))
            if pair in connected_pairs:
                continue
            if attr_a in saturated_attrs or attr_b in saturated_attrs:
                continue
            active.append((idx, na, nb))

        if not active:
            n_connected = len(connected_pairs)
            n_saturated = len(saturated_attrs)
            n_remaining_pairs = len(attr_pair_map) - n_connected
            logger.info(
                f"Adjuvant: exit (no active candidates) at iter {it}. "
                f"pairs={n_connected}, saturated={n_saturated}/{d}, "
                f"remaining_pairs={n_remaining_pairs} (all connected or saturated)"
            )
            break

        # Cache base triangulation and clique info for this iteration
        base_tri = moral if nx.is_chordal(moral) else _triangulate_simple(moral)
        base_cliques = list(nx.find_cliques(base_tri))

        # Node-to-cliques index (for same-clique fast path)
        node_to_cliques: dict[str, list[int]] = {}
        for ci, cl in enumerate(base_cliques):
            for node in cl:
                node_to_cliques.setdefault(node, []).append(ci)

        # Score each candidate: TVD * edge_weight (1/sqrt(dom_a*dom_b))
        # Same-clique and cross-clique get the same score since we measure
        # 2-way edge marginals directly (no clique domain penalty needed).
        scores = np.full(len(active), float("-inf"))
        for i, (idx, na, nb) in enumerate(active):
            scores[i] = cand_tvd_boost[idx]

        # Check if any valid candidate exists
        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            n_same = sum(1 for i, (idx, na, nb) in enumerate(active)
                         if set(node_to_cliques.get(na, [])) & set(node_to_cliques.get(nb, [])))
            n_cross = len(active) - n_same
            logger.info(
                f"Adjuvant: exit (no valid candidates) at iter {it}. "
                f"active={len(active)} (same_clq={n_same}, cross_clq={n_cross}), "
                f"edges={len(structure_edges)}"
            )
            break

        # Exponential mechanism selection among valid candidates + "stop" option.
        # The stop option has score = min_tvd. If EM picks it, we exit:
        # all remaining candidates are below the noise floor.
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        # Append stop option at the end, boosted by log(N) to compensate
        # for competing against N candidates in the softmax
        sensitivity = 2.0 / n
        log_n_boost = 2 * sensitivity * np.log(max(len(valid_scores), 1)) / eps_per_step if eps_per_step > 0 else 0
        em_scores = np.append(valid_scores, min_tvd + log_n_boost if min_tvd else 0)
        stop_idx = len(valid_scores)

        if eps_per_step > 0:
            sensitivity = 2.0 / n  # TVD sensitivity
            sel = exponential_mechanism(em_scores, eps_per_step, sensitivity)
            rho_spent += eps_per_step ** 2 / 8
        else:
            sel = int(np.argmax(em_scores))

        if sel == stop_idx:
            logger.info(
                f"Adjuvant: exit (EM picked stop option, min_tvd={min_tvd}) "
                f"at iter {it}, edges={len(structure_edges)}"
            )
            pbar.update(1)
            break

        _, na, nb = active[valid_indices[sel]]

        # Accept edge
        moral.add_edge(na, nb, structure=True)
        structure_edges.add(frozenset([na, nb]))

        # Update attribute pair tracking and edge counts
        da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
        attr_a, attr_b = da["attr"], db["attr"]
        pair = tuple(sorted([attr_a, attr_b]))
        connected_pairs.add(pair)

        attr_edge_count[attr_a] = attr_edge_count.get(attr_a, 0) + 1
        attr_edge_count[attr_b] = attr_edge_count.get(attr_b, 0) + 1
        if attr_edge_count[attr_a] >= max_edges_per_attr:
            saturated_attrs.add(attr_a)
        if attr_edge_count[attr_b] >= max_edges_per_attr:
            saturated_attrs.add(attr_b)

        logger.info(
            f"Adj. iter {it+1:3d}/{max_steps} (score={scores[valid_indices[sel]]:.4f}): "
            f"{_fmt_edge(na, nb, moral, attrs)}"
        )

        pbar.set_description(
            f"Adjuvant structure [{len(structure_edges)} edges, "
            f"score={scores[valid_indices[sel]]:.4f}]"
        )
        pbar.update(1)

    pbar.close()

    # Force at least one edge per attribute: for any attribute with 0 edges,
    # use exponential mechanism to pick an edge (budget from leftover steps)
    all_attrs_in_graph = set()
    for node, data in directed_graph.nodes(data=True):
        if not data.get("is_common", False):
            all_attrs_in_graph.add(data["attr"])

    disconnected = all_attrs_in_graph - set(attr_edge_count.keys())
    if disconnected:
        logger.info(
            f"Adjuvant: forcing edges for {len(disconnected)} disconnected attrs: "
            f"{disconnected}"
        )
        for attr in disconnected:
            # Gather candidates involving this attribute
            attr_cands: list[tuple[int, str, str]] = []
            attr_scores: list[float] = []
            for idx, (na, nb) in enumerate(candidates):
                da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
                if da["attr"] != attr and db["attr"] != attr:
                    continue
                pair = tuple(sorted([da["attr"], db["attr"]]))
                if pair in connected_pairs:
                    continue
                attr_cands.append((idx, na, nb))
                attr_scores.append(cand_tvd_boost[idx])

            if not attr_cands:
                continue

            # Add stop option to forced-edge EM too
            scores_arr = np.append(np.array(attr_scores), min_tvd)
            forced_stop_idx = len(attr_scores)

            if eps_per_step > 0:
                sel = exponential_mechanism(scores_arr, eps_per_step, 2.0 / n)
                rho_spent += eps_per_step ** 2 / 8
            else:
                sel = int(np.argmax(scores_arr))

            if sel == forced_stop_idx:
                logger.info(f"Adj. skip forced edge for {attr} (below min_tvd)")
                continue

            _, na, nb = attr_cands[sel]
            moral.add_edge(na, nb, structure=True)
            structure_edges.add(frozenset([na, nb]))
            da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
            pair = tuple(sorted([da["attr"], db["attr"]]))
            connected_pairs.add(pair)
            attr_edge_count[da["attr"]] = attr_edge_count.get(da["attr"], 0) + 1
            attr_edge_count[db["attr"]] = attr_edge_count.get(db["attr"], 0) + 1
            logger.info(f"Adj. forced edge: {_fmt_edge(na, nb, moral, attrs)}")

    rho_remaining = rho2_exp - rho_spent

    # Diagnostic: show top unconnected pairs by TVD
    all_pairs_tvd: list[tuple[float, str, str]] = []
    for (a, b), val in tvd.items():
        if a < b:  # avoid duplicates
            all_pairs_tvd.append((val, a, b))
    all_pairs_tvd.sort(reverse=True)

    logger.info(
        f"Adjuvant: structure learning done, "
        f"{len(structure_edges)} edges, "
        f"{len(connected_pairs)} attribute pairs connected, "
        f"rho2_exp spent={rho_spent:.4f}, remaining={rho_remaining:.4f}"
    )
    logger.info("Adjuvant: Connected pairs (by TVD):")
    for val, a, b in all_pairs_tvd:
        pair = tuple(sorted([a, b]))
        if pair in connected_pairs:
            for edge in structure_edges:
                ena, enb = tuple(edge)
                da, db = directed_graph.nodes[ena], directed_graph.nodes[enb]
                if tuple(sorted([da["attr"], db["attr"]])) == pair:
                    logger.info(f"  CONNECTED TVD={val:.4f} {_fmt_edge(ena, enb, moral, attrs)}")
                    break
        else:
            logger.info(f"    MISSING TVD={val:.4f} {a} x {b}")

    return moral, structure_edges, rho_remaining


# ============================================================
# Step 3: Measurement helpers
# ============================================================
def _clique_to_request(clique):
    """Convert CliqueMeta to oracle request format."""
    from ....graph.beliefs import convert_sel

    return [
        (attr_name, convert_sel(sel))
        for _, _, attr_name, sel in clique
    ]


def select_cliques_to_measure(
    junction_cliques: list,
    triangulated: nx.Graph,
    structure_edges: set[frozenset[str]],
) -> list:
    """Filter junction tree cliques to those containing structure-learning edges.

    Uses the junction tree's own CliqueMeta tuples, so observations are
    guaranteed to have parent cliques in the tree.

    Args:
        junction_cliques: List of CliqueMeta from the junction tree.
        triangulated: The triangulated graph (for node lookups).
        structure_edges: Set of frozenset node pairs from structure learning.
    """
    # Build reverse map: for each (attr, value) -> set of graph nodes
    attr_val_to_nodes: dict[tuple, set[str]] = {}
    for node, data in triangulated.nodes(data=True):
        key = (data["table"], data["order"], data["attr"], data["value"])
        attr_val_to_nodes.setdefault(key, set()).add(node)

    measured = []
    for clique_meta in junction_cliques:
        # Expand CliqueMeta back to the set of graph nodes it covers
        clique_nodes: set[str] = set()
        for am in clique_meta:
            if isinstance(am.sel, int):
                # Common value at height h: find matching nodes
                for node in attr_val_to_nodes.get((am.table, am.order, am.attr, am.attr), set()):
                    clique_nodes.add(node)
            else:
                for val, h in am.sel:
                    for node in attr_val_to_nodes.get((am.table, am.order, am.attr, val), set()):
                        clique_nodes.add(node)

        # Check if any structure-learning edge has both endpoints in this clique
        has_structure_edge = any(edge <= clique_nodes for edge in structure_edges)
        if has_structure_edge:
            measured.append(clique_meta)

    return measured


def measure_cliques(
    oracle: MarginalOracle,
    cliques_to_measure: list,
    attrs: DatasetAttributes,
    n: int,
    rho3: float,
) -> tuple[list, float]:
    """Measure selected clique marginals with Gaussian noise.

    Returns (list of LinearObservation, sigma3)."""
    from ....graph.hugin import get_clique_domain, get_attrs as _get_attrs
    from ....graph.loss import LinearObservation
    from ....graph.beliefs import convert_sel

    K = len(cliques_to_measure)
    if K == 0 or rho3 <= 0:
        return [], 0.0

    sigma3 = sqrt(K / (2 * rho3))

    # Build oracle requests from CliqueMeta
    requests = [_clique_to_request(cl) for cl in cliques_to_measure]
    results = oracle.process(requests, postprocess=None)

    obs_list = []
    for clique, result in zip(cliques_to_measure, results):
        dom = get_clique_domain(clique, attrs)

        # Build the expected shape in source (CliqueMeta) order
        src_dims = []
        for _, _, attr_name, sel in clique:
            a = _get_attrs(attrs, None, None)[attr_name]
            src_dims.append(a.get_domain(convert_sel(sel)))

        raw = result.astype(np.float64).ravel()
        if sigma3 > 0:
            raw = raw + np.random.normal(0, sigma3, size=raw.size)
        raw = raw.clip(0)
        s = raw.sum()
        prob = (raw / s if s > 0 else raw).astype(np.float32)
        prob = prob.reshape(src_dims)

        confidence = n / (n + sigma3 * dom)
        obs_list.append(LinearObservation(clique, None, prob, confidence))

    return obs_list, sigma3


def measure_edges(
    oracle: MarginalOracle,
    structure_edges: set[frozenset[str]],
    graph: nx.Graph,
    attrs: DatasetAttributes,
    n: int,
    rho3: float,
) -> tuple[list, float]:
    """Measure 2-way marginals for structure-learning edges with Gaussian noise.

    Returns (list of LinearObservation, sigma3)."""
    from ....graph.hugin import AttrMeta, get_clique_domain, get_attrs as _get_attrs
    from ....graph.loss import LinearObservation
    from ....graph.beliefs import convert_sel
    from ..sota.common import _attr_sel

    edges = list(structure_edges)
    K = len(edges)
    if K == 0 or rho3 <= 0:
        return [], 0.0

    sigma3 = sqrt(K / (2 * rho3))

    # Build oracle requests and CliqueMeta for each edge.
    # Use the graph node metadata to match create_clique_meta's convention.
    from collections import defaultdict

    requests = []
    edge_metas = []
    for edge in edges:
        na, nb = tuple(edge)

        # Build AttrMeta from graph nodes (same logic as create_clique_meta)
        sels: dict[tuple, dict[str, int]] = defaultdict(dict)
        for node in (na, nb):
            d = graph.nodes[node]
            key = (d["table"], d["order"], d["attr"])
            val = d["value"]
            h = d["height"]
            if key in sels and val in sels[key]:
                h = min(sels[key][val], h)
            sels[key][val] = h

        source = []
        for (table, order, attr_name), sel_dict in sels.items():
            attr = _get_attrs(attrs, table, order)[attr_name]
            if len(sel_dict) == 1 and attr.common and next(iter(sel_dict)) == attr.common.name:
                new_sel: int | tuple = sel_dict[attr.common.name]
            else:
                cmn = attr.common.name if attr.common else None
                new_sel = tuple(sorted((v, h) for v, h in sel_dict.items() if v != cmn))
            source.append(AttrMeta(None, None, attr_name, new_sel))
        source_tuple = tuple(sorted(source, key=lambda x: x[:-1]))
        edge_metas.append(source_tuple)

        # Oracle request: use the sel from the source
        requests.append([
            (attr_name, convert_sel(sel))
            for _, _, attr_name, sel in source_tuple
        ])

    results = oracle.process(requests, postprocess=None)

    obs_list = []
    for source_tuple, result in zip(edge_metas, results):
        dom = get_clique_domain(source_tuple, attrs)

        # Build shape in source order
        src_dims = []
        for _, _, attr_name, sel in source_tuple:
            a = _get_attrs(attrs, None, None)[attr_name]
            src_dims.append(a.get_domain(convert_sel(sel)))

        raw = result.astype(np.float64).ravel()
        if sigma3 > 0:
            raw = raw + np.random.normal(0, sigma3, size=raw.size)
        raw = raw.clip(0)
        s = raw.sum()
        prob = (raw / s if s > 0 else raw).astype(np.float32)
        prob = prob.reshape(src_dims)

        confidence = n / (n + sigma3 * dom)
        obs_list.append(LinearObservation(source_tuple, None, prob, confidence))

    return obs_list, sigma3


def build_1way_observations(
    noisy_1way: dict[str, np.ndarray],
    attrs: DatasetAttributes,
    n: int,
    sigma1: float,
) -> list:
    """Build LinearObservation objects from noisy 1-way marginals."""
    from ....graph.hugin import AttrMeta, get_attrs as _get_attrs
    from ....graph.loss import LinearObservation
    from ..sota.common import _attr_sel

    obs_list = []
    for attr_name, noisy_mar in noisy_1way.items():
        # Build source AttrMeta (same as what _attr_sel produces)
        attr = _get_attrs(attrs, None, None)[attr_name]
        if attr.common:
            sel: int | tuple = 0
        else:
            sel = tuple(sorted((v, 0) for v in attr.vals))
        source = (AttrMeta(None, None, attr_name, sel),)

        raw = noisy_mar.copy().clip(0)
        s = raw.sum()
        prob = (raw / s if s > 0 else raw).astype(np.float32)
        dom = len(raw)
        confidence = n / (n + sigma1 * dom) if sigma1 > 0 else 1.0

        obs_list.append(LinearObservation(source, None, prob, confidence))

    return obs_list
