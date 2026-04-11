"""Adjuvant: DP structure learning via greedy edge addition with height-chain nodes.

Core implementation: cached marginal computation, noisy TVD scoring,
height-chain graph construction, greedy edge addition with exponential
mechanism, and measurement/observation building.
"""

from __future__ import annotations

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
def compute_noisy_tvd(
    cached: CachedMarginals,
    n: int,
    rho2_tvd: float,
) -> dict[tuple[str, str], float]:
    """Compute noisy pairwise TVD = ||P(A,B) - P(A)P(B)||_1 / 2."""
    num_pairs = len(cached.two_way)
    tvd_sens = 2.0 / n
    sigma = tvd_sens * sqrt(num_pairs / (2 * rho2_tvd)) if rho2_tvd > 0 else 0

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
        if sigma > 0:
            val += np.random.normal(0, sigma)
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


def compute_height_boost(
    node_a: str,
    node_b: str,
    g: nx.Graph,
    attrs: DatasetAttributes,
) -> float:
    """Height boost: sqrt(dom_h0 / dom_h) per side. Factor=1 at (0,0)."""
    from ....graph.hugin import get_attrs as _get_attrs

    def _boost_for_node(node: str) -> float:
        d = g.nodes[node]
        a = _get_attrs(attrs, d["table"], d["order"])[d["attr"]]
        val = cast(CatValue, a[d["value"]])
        h = d["height"]
        if h == 0:
            return 1.0
        dom_h0 = val.get_domain(0)
        dom_h = val.get_domain(h)
        return sqrt(dom_h0 / dom_h) if dom_h > 0 else 1.0

    return _boost_for_node(node_a) * _boost_for_node(node_b)


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
        boost = compute_height_boost(na, nb, moral, attrs)
        tvd_sum += base * boost

    return tvd_sum - size_penalty * total_domain


# ============================================================
# Step 2d: Structure learning main loop
# ============================================================
def structure_learn(
    directed_graph: nx.DiGraph,
    attrs: DatasetAttributes,
    tvd: dict[tuple[str, str], float],
    max_clique_size: float,
    size_penalty: float,
    max_no_improve: int,
    rho2_exp: float,
) -> tuple[nx.Graph, set[frozenset[str]], float]:
    """Greedy edge addition with exponential mechanism.

    Returns:
        moral: Undirected moralized graph with structure-learning edges added.
        structure_edges: Set of frozenset node pairs for structure-learning edges.
        rho2_exp_remaining: Unspent exponential mechanism budget.
    """
    from ..sota.common import exponential_mechanism
    from ....graph.hugin import to_moral, get_factor_domain
    from ....utils.progress import piter

    # Moralize the directed height-chain graph -> undirected base
    moral = to_moral(directed_graph)

    # Generate candidates and group by attribute pair
    candidates, attr_pair_map = generate_candidates(directed_graph)
    connected_pairs: set[tuple[str, str]] = set()
    structure_edges: set[frozenset[str]] = set()

    # Budget: one exponential mechanism selection per attribute pair (upper bound)
    # Each selection costs eps^2 / 8 in rho-CDP
    max_steps = len(attr_pair_map)
    eps_per_step = sqrt(8 * rho2_exp / max_steps) if max_steps > 0 and rho2_exp > 0 else 0
    rho_spent = 0.0

    no_improve_count = 0

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
        boost = compute_height_boost(na, nb, moral, attrs)
        cand_tvd_boost[idx] = base * boost

    for it in range(max_steps):
        # Filter to active candidates (attribute pair not yet connected)
        active: list[tuple[int, str, str]] = []
        for idx, (na, nb) in enumerate(candidates):
            da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
            pair = tuple(sorted([da["attr"], db["attr"]]))
            if pair not in connected_pairs:
                active.append((idx, na, nb))

        if not active:
            break

        # Cache base triangulation and clique info for this iteration
        base_tri = moral if nx.is_chordal(moral) else _triangulate_simple(moral)
        base_cliques = list(nx.find_cliques(base_tri))

        # Compute base domain total and check validity
        base_domain_total = 0
        base_valid = True
        for cl in base_cliques:
            dom = get_factor_domain(set(cl), base_tri, attrs)
            if dom > max_clique_size:
                base_valid = False
                break
            base_domain_total += dom

        # Build node-to-cliques index: for each node, which cliques contain it
        node_to_cliques: dict[str, list[int]] = {}
        for ci, cl in enumerate(base_cliques):
            for node in cl:
                node_to_cliques.setdefault(node, []).append(ci)

        # Compute base TVD sum
        base_tvd_sum = 0.0
        for edge in structure_edges:
            ena, enb = tuple(edge)
            da, db = moral.nodes[ena], moral.nodes[enb]
            base_val = tvd.get((da["attr"], db["attr"]), 0.0)
            boost = compute_height_boost(ena, enb, moral, attrs)
            base_tvd_sum += base_val * boost

        base_score = (base_tvd_sum - size_penalty * base_domain_total) if base_valid else float("-inf")

        # Score each candidate
        scores = np.full(len(active), float("-inf"))
        for i, (idx, na, nb) in enumerate(active):
            # Check if both endpoints share a base clique
            cliques_a = set(node_to_cliques.get(na, []))
            cliques_b = set(node_to_cliques.get(nb, []))
            shared = cliques_a & cliques_b

            if shared and base_valid:
                # Both in same clique: triangulation unchanged, just add TVD contribution
                scores[i] = base_score + cand_tvd_boost[idx]
            else:
                # Different cliques or base invalid: need full re-triangulation
                moral.add_edge(na, nb)
                trial_edges = structure_edges | {frozenset([na, nb])}
                scores[i] = score_graph(
                    moral, tvd, attrs, trial_edges, max_clique_size, size_penalty
                )
                moral.remove_edge(na, nb)

        # Check if any valid candidate exists
        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
            pbar.update(1)
            continue

        no_improve_count = 0

        # Exponential mechanism selection among valid candidates
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        if eps_per_step > 0:
            sensitivity = 1.0  # TVD bounded in [0, 1]
            sel = exponential_mechanism(valid_scores, eps_per_step, sensitivity)
            rho_spent += eps_per_step ** 2 / 8
        else:
            sel = int(np.argmax(valid_scores))

        _, na, nb = active[valid_indices[sel]]

        # Accept edge
        moral.add_edge(na, nb, structure=True)
        structure_edges.add(frozenset([na, nb]))

        # Remove all height permutations for this attribute pair
        da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
        pair = tuple(sorted([da["attr"], db["attr"]]))
        connected_pairs.add(pair)

        logger.info(
            f"Adj. iter {it+1: 3d}/{max_steps} (score={scores[valid_indices[sel]]:.4f}): ({na}, {nb})"
        )

        pbar.set_description(
            f"Adjuvant structure [{len(structure_edges)} edges, "
            f"score={scores[valid_indices[sel]]:.4f}]"
        )
        pbar.update(1)

    pbar.close()

    rho_remaining = rho2_exp - rho_spent
    logger.info(
        f"Adjuvant: structure learning done, "
        f"{len(structure_edges)} edges, "
        f"{len(connected_pairs)} attribute pairs connected, "
        f"rho2_exp spent={rho_spent:.4f}, remaining={rho_remaining:.4f}"
    )

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
    triangulated: nx.Graph,
    attrs: DatasetAttributes,
    structure_edges: set[frozenset[str]],
) -> list:
    """Select maximal cliques that contain at least one structure-learning edge.

    Returns list of CliqueMeta tuples for cliques worth measuring."""
    from ....graph.hugin import create_clique_meta

    measured = []
    for clique_nodes in nx.find_cliques(triangulated):
        node_set = set(clique_nodes)
        # Check if any structure-learning edge has both endpoints in this clique
        has_structure_edge = any(
            edge <= node_set for edge in structure_edges
        )
        if has_structure_edge:
            meta = create_clique_meta(clique_nodes, triangulated, attrs)
            measured.append(meta)

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
