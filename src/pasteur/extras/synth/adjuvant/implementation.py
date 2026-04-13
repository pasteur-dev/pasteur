"""Adjuvant: DP structure learning via greedy edge addition with height-chain nodes.

Core implementation: cached marginal computation, noisy TVD scoring,
height-chain graph construction, greedy edge addition with exponential
mechanism, and measurement/observation building.
"""

import itertools
import logging
from math import exp, log1p, sqrt
from typing import NamedTuple, Sequence, cast

import networkx as nx
import numpy as np
from scipy.special import softmax

from ....attribute import Attributes, CatValue, DatasetAttributes, SeqAttributes
from ....marginal import MarginalOracle

logger = logging.getLogger(__name__)

# Column identifier: (table, order, attribute_name, value_name)
# For standalone (single-table) use, table and order are both None.
Col = tuple[str | None, int | None, str, str]


def _none_safe_key(c):
    """Sort key that handles None and mixed str/int fields."""
    return tuple((0, "") if x is None else (1, x) for x in c)


def _col_sort_key(c: Col):
    return _none_safe_key(c)


def _attr_meta_sort_key(m):
    """Sort key for AttrMeta tuples, handling None in table/order fields."""
    return _none_safe_key(m[:-1])


# ============================================================
# Local helpers (no dependency on sota/common)
# ============================================================
def _cdp_delta(rho: float, eps: float) -> float:
    if rho == 0:
        return 0.0
    amin, amax = 1.01, (eps + 1) / (2 * rho) + 2
    for _ in range(1000):
        alpha = (amin + amax) / 2
        derivative = (2 * alpha - 1) * rho - eps + log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha
    delta = exp((alpha - 1) * (alpha * rho - eps) + alpha * log1p(-1 / alpha)) / (
        alpha - 1.0
    )
    return min(delta, 1.0)


def cdp_rho(eps: float, delta: float) -> float:
    """Find smallest rho such that rho-CDP implies (eps, delta)-DP."""
    if delta >= 1:
        return 0.0
    rhomin, rhomax = 0.0, eps + 1
    for _ in range(1000):
        rho = (rhomin + rhomax) / 2
        if _cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return rhomin


def compute_rho_for_theta(
    dom: int, n: int | float, theta: float, sigma_floor: float
) -> float | None:
    """Compute rho budget to achieve confidence theta/(theta+1) for a marginal of domain `dom`.

    The target is sigma_eff * dom = n / theta, where
    sigma_eff^2 = sigma_dp^2 + n/(dom * sigma_floor).

    Returns rho = 1/(2*sigma_dp^2), or None if unachievable
    (sampling noise alone exceeds the target)."""
    if theta <= 0 or dom <= 0:
        return 0.0
    target_seff2 = (n / (theta * dom)) ** 2
    sampling_var = n / (dom * sigma_floor) if sigma_floor > 0 else 0.0
    sigma_dp2 = target_seff2 - sampling_var
    if sigma_dp2 <= 0:
        return None
    return 1.0 / (2.0 * sigma_dp2)


def exponential_mechanism(
    qualities: dict | np.ndarray,
    eps: float,
    sensitivity: float = 1.0,
) -> object:
    """Sample from the exponential mechanism. Returns the selected key."""
    if isinstance(qualities, dict):
        keys = list(qualities.keys())
        q = np.array([qualities[k] for k in keys])
    else:
        keys = np.arange(len(qualities))
        q = np.array(qualities)
    q = q - q.max()
    p = softmax(0.5 * eps / sensitivity * q)
    return keys[np.random.choice(p.size, p=p)]


def get_col_names(
    attrs: DatasetAttributes,
) -> list[Col]:
    """Get all (table, order, attr_name, val_name) Col tuples for CatValue columns.

    Iterates all tables in attrs (including hist tables for MARE).
    For standalone single-table use, table and order are both None."""
    result: list[Col] = []
    for table, tattrs in attrs.items():
        if isinstance(tattrs, SeqAttributes):
            attr_sets: dict = {**tattrs.hist, None: tattrs.attrs}
        else:
            attr_sets = {None: tattrs}

        for order, attr_set in attr_sets.items():
            if not attr_set:
                continue
            for attr_name, attr in attr_set.items():
                if not attr.vals:
                    continue
                if attr.common:
                    result.append((table, order, attr_name, attr.common.name))
                for val_name, val in attr.vals.items():
                    if isinstance(val, CatValue) and (
                        attr.common is None or val.name != attr.common.name
                    ):
                        result.append((table, order, attr_name, val_name))
    return result


def get_hist_cols(cols: list[Col]) -> set[Col]:
    """Return the subset of cols that are from hist/parent tables (table is not None)."""
    return {c for c in cols if c[0] is not None}


def _col_sel(col: Col, attrs: DatasetAttributes):
    """Oracle selector for a single column at height 0.

    Returns a 2-tuple (attr_name, sel) for main table columns, or a
    3-tuple (table_sel, attr_name, sel) for hist/parent table columns.

    Common value -> sel = 0. Regular value -> sel = {val_name: 0}."""
    table, order, attr_name, val_name = col
    from ....graph.hugin import get_attrs

    attr_set = get_attrs(attrs, table, order)
    attr = attr_set[attr_name]
    if attr.common and val_name == attr.common.name:
        sel = 0
    else:
        sel = {val_name: 0}

    if table is not None:
        table_sel = (table, order) if order is not None else table
        return (table_sel, attr_name, sel)
    return (attr_name, sel)


def calc_confidence(
    n: int | float, sigma: float, dom: int, sigma_floor: float = 1.0
) -> float:
    """Calculate observation confidence from sample size, noise scale, and domain size.

    Uses a uniform-bin model for sampling noise: each cell has probability
    p = 1/(dom * sigma_floor), so variance = n * p * (1-p) ≈ n / (dom * sigma_floor).
    The effective sigma combines DP noise and sampling noise in quadrature:
    sigma_eff = sqrt(sigma_dp^2 + n / (dom * sigma_floor)).

    When sigma_floor is 0, sampling noise is ignored (pure DP confidence)."""
    dom = max(dom, 1)
    s2 = sigma * sigma
    if sigma_floor > 0:
        s2 += n / (dom * sigma_floor)
    if s2 == 0:
        return 1.0
    return n / (n + sqrt(s2) * dom)


# ============================================================
# Data structures
# ============================================================
class CachedMarginals(NamedTuple):
    """Pre-computed 1-way and 2-way true marginals from a single data pass."""

    one_way: dict[Col, np.ndarray]  # (attr_name, val_name) -> count vector (flat)
    two_way: dict[tuple[Col, Col], np.ndarray]  # (col_a, col_b) -> joint counts


# ============================================================
# Step 0: Single data pass
# ============================================================
def compute_all_marginals(
    oracle: MarginalOracle,
    attrs: DatasetAttributes,
    all_cols: list[Col],
) -> CachedMarginals:
    """Batch-query all 1-way and 2-way true marginals in one oracle call."""
    requests_1 = [[_col_sel(c, attrs)] for c in all_cols]
    pairs = list(itertools.combinations(all_cols, 2))
    requests_2 = [[_col_sel(c1, attrs), _col_sel(c2, attrs)] for c1, c2 in pairs]

    results = oracle.process(requests_1 + requests_2, postprocess=None)

    one_way: dict[Col, np.ndarray] = {}
    for col, r in zip(all_cols, results[: len(all_cols)]):
        one_way[col] = r.ravel().astype(np.float64)

    two_way: dict[tuple[Col, Col], np.ndarray] = {}
    for (ca, cb), r in zip(pairs, results[len(all_cols) :]):
        two_way[ca, cb] = r.astype(np.float64)

    return CachedMarginals(one_way, two_way)


# ============================================================
# Step 1: Noisy 1-way marginals
# ============================================================
def compute_1way_budget(
    cached: CachedMarginals,
    n: int | float,
    theta_1w: float,
    sigma_floor: float,
    rho_max: float | None = None,
) -> tuple[dict[Col, float], float, float]:
    """Compute per-column DP noise sigma and total rho for 1-way marginals.

    Each column gets noise calibrated so that its confidence achieves theta_1w.
    If total rho exceeds rho_max, theta_1w is reduced via binary search.

    Returns (sigma_per_col, total_rho1, effective_theta_1w)."""

    def _total_rho(theta):
        total = 0.0
        for mar in cached.one_way.values():
            rho = compute_rho_for_theta(mar.size, n, theta, sigma_floor)
            if rho is not None:
                total += rho
        return total

    # Check if requested theta_1w fits within budget
    rho1 = _total_rho(theta_1w)
    if rho_max is not None and rho1 > rho_max and rho_max > 0:
        # Binary search for max achievable theta
        lo, hi = 1.0, theta_1w
        for _ in range(64):
            mid = (lo + hi) / 2
            if _total_rho(mid) <= rho_max:
                lo = mid
            else:
                hi = mid
        theta_1w = lo
        rho1 = _total_rho(theta_1w)
        logger.info(
            f"Adjuvant: theta_1w capped to {theta_1w:.1f} "
            f"(rho1={rho1:.6f}, rho_max={rho_max:.6f})"
        )

    # Compute per-column sigma_dp
    sigmas: dict[Col, float] = {}
    for col, mar in cached.one_way.items():
        rho = compute_rho_for_theta(mar.size, n, theta_1w, sigma_floor)
        if rho is not None and rho > 0:
            sigmas[col] = sqrt(1.0 / (2.0 * rho))
        else:
            sigmas[col] = 0.0

    return sigmas, rho1, theta_1w


def add_noise_1way(
    cached: CachedMarginals,
    sigmas: dict[Col, float],
) -> dict[Col, np.ndarray]:
    """Add Gaussian noise to 1-way marginals with per-column sigma.

    Returns noisy_marginals."""
    noisy = {}
    for col, mar in cached.one_way.items():
        sigma = sigmas.get(col, 0.0)
        if sigma > 0:
            noisy[col] = mar + np.random.normal(0, sigma, size=mar.size)
        else:
            noisy[col] = mar.copy()
    return noisy


# ============================================================
# Step 2b: Noisy TVD (per height combination)
# ============================================================
def _build_transition_mappings(val: "CatValue", h_range: int) -> list[np.ndarray]:
    """Build transition mappings from height h to height h+1.

    Returns a list of length h_range-1, where transitions[i] maps
    group indices at height i to group indices at height i+1."""
    transitions: list[np.ndarray] = []
    dom_0 = val.get_domain(0)
    prev_map: np.ndarray | None = None
    for h in range(1, h_range):
        cur_map = np.asarray(val.get_mapping(h))
        if h == 1:
            # h=0 groups are individual leaves, transition is just the mapping
            transitions.append(cur_map)
        else:
            assert prev_map is not None
            dom_prev = val.get_domain(h - 1)
            trans = np.empty(dom_prev, dtype=cur_map.dtype)
            seen = np.zeros(dom_prev, dtype=bool)
            for leaf in range(dom_0):
                g = prev_map[leaf]
                if not seen[g]:
                    trans[g] = cur_map[leaf]
                    seen[g] = True
            transitions.append(trans)
        prev_map = np.asarray(val.get_mapping(h))
    return transitions


def compute_tvd(
    cached: CachedMarginals,
    attrs: DatasetAttributes,
    all_cols: list[Col],
) -> dict[tuple[Col, Col], np.ndarray]:
    """Compute exact pairwise TVD at every height combination.

    Grabs 2-way histograms at full resolution (height 0), then iteratively
    aggregates using transition mappings to compute TVD at all (ha, hb) combos.

    No noise added — the exponential mechanism provides privacy for selection.

    Returns dict mapping (col_a, col_b) -> 2D array of shape (Ha, Hb)
    where Ha, Hb are the number of graph heights for each column.
    Both (ca, cb) and (cb, ca) are stored (the latter transposed)."""
    from ....graph.hugin import get_attrs as _get_attrs

    # Build per-column metadata
    col_meta: dict[Col, tuple[CatValue, int]] = {}
    col_transitions: dict[Col, list[np.ndarray]] = {}

    for col in all_cols:
        table, order, attr_name, val_name = col
        attr = _get_attrs(attrs, table, order)[attr_name]
        cmn = attr.common
        if cmn and val_name == cmn.name:
            val = cmn
            h_range = cmn.height
        else:
            val = cast(CatValue, attr[val_name])
            h_range = val.height if cmn is None else val.height - 1
        col_meta[col] = (val, h_range)
        col_transitions[col] = _build_transition_mappings(val, h_range)

    tvd: dict[tuple[Col, Col], np.ndarray] = {}

    for (ca, cb), joint_raw in cached.two_way.items():
        val_a, ha_range = col_meta[ca]
        val_b, hb_range = col_meta[cb]
        dom_a0 = val_a.get_domain(0)
        dom_b0 = val_b.get_domain(0)
        joint_00 = joint_raw.reshape(dom_a0, dom_b0).astype(np.float64)
        n_total = joint_00.sum()

        if n_total == 0:
            tvd[ca, cb] = np.zeros((ha_range, hb_range))
            tvd[cb, ca] = np.zeros((hb_range, ha_range))
            continue

        result = np.zeros((ha_range, hb_range))
        trans_a = col_transitions[ca]
        trans_b = col_transitions[cb]

        # Iteratively aggregate dim_a, then for each ha iterate dim_b
        joint_ha_0 = joint_00
        for ha in range(ha_range):
            if ha > 0:
                t = trans_a[ha - 1]
                new_joint = np.zeros((val_a.get_domain(ha), joint_ha_0.shape[1]))
                np.add.at(new_joint, (t, slice(None)), joint_ha_0)
                joint_ha_0 = new_joint

            joint_ha_hb = joint_ha_0
            for hb in range(hb_range):
                if hb > 0:
                    t = trans_b[hb - 1]
                    new_joint = np.zeros((joint_ha_hb.shape[0], val_b.get_domain(hb)))
                    np.add.at(new_joint, (slice(None), t), joint_ha_hb)
                    joint_ha_hb = new_joint

                p_ab = joint_ha_hb / n_total
                p_a = p_ab.sum(axis=1)
                p_b = p_ab.sum(axis=0)
                indep = np.outer(p_a, p_b)
                result[ha, hb] = float(np.abs(p_ab - indep).sum() / 2)

        tvd[ca, cb] = result
        tvd[cb, ca] = result.T

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
    frozen_nodes: set[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[tuple[Col, Col], list[int]]]:
    """Generate edge candidates between non-common value nodes.

    Candidates are grouped by column pair (table, order, attr, value) for tracking.
    Includes same-attribute pairs (different columns), excludes same-column pairs.

    If ``frozen_nodes`` is provided, edges between two frozen nodes are excluded
    (hist-hist edges are blocked as they represent the prior).

    Returns:
        candidates: List of (node_a, node_b) edge candidates.
        col_pair_map: Maps sorted Col pair -> list of indices.
    """
    # Group non-common nodes by column (table, order, attr, value)
    col_nodes: dict[Col, list[str]] = {}
    for node, data in g.nodes(data=True):
        if data.get("is_common", False):
            continue
        col: Col = (data.get("table"), data.get("order"), data["attr"], data["value"])
        col_nodes.setdefault(col, []).append(node)

    candidates: list[tuple[str, str]] = []
    col_pair_map: dict[tuple[Col, Col], list[int]] = {}
    col_names = sorted(col_nodes.keys(), key=_col_sort_key)

    for i, col_a in enumerate(col_names):
        for col_b in col_names[i + 1 :]:
            pair_key = (col_a, col_b)
            col_pair_map[pair_key] = []
            for na in col_nodes[col_a]:
                for nb in col_nodes[col_b]:
                    # Block edges between two frozen (hist) nodes
                    if frozen_nodes and na in frozen_nodes and nb in frozen_nodes:
                        continue
                    col_pair_map[pair_key].append(len(candidates))
                    candidates.append((na, nb))

    return candidates, col_pair_map


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
        clique_meta = tuple(sorted(meta, key=_attr_meta_sort_key))

        dom = get_clique_domain(clique_meta, attrs)
        if dom > max_clique_size:
            return 0, False
        total_domain += dom

    return total_domain, True


def _fmt_node(node: str, g, attrs) -> str:
    """Format a graph node as 'attr.val[h] (dom=X)'."""
    from ....graph.hugin import get_attrs as _get_attrs

    d = g.nodes[node]
    val = cast(
        CatValue, _get_attrs(attrs, d["table"], d["order"])[d["attr"]][d["value"]]
    )
    dom = val.get_domain(d["height"])
    return f"{d['attr'] + '.' if d['attr'] != d['value'] else ''}{d['value']}[{d['height']}] (dom={dom})"


def _fmt_edge(na: str, nb: str, g, attrs) -> str:
    """Format a graph edge as 'attr.val[h] x attr.val[h] (domA, domB)'."""
    from ....graph.hugin import get_attrs as _get_attrs

    def _info(node):
        d = g.nodes[node]
        val = cast(
            CatValue, _get_attrs(attrs, d["table"], d["order"])[d["attr"]][d["value"]]
        )
        dom = val.get_domain(d["height"])
        return (
            f"{d['attr'] + '.' + d['value'].replace(d['attr'] + '_', '') if d['attr'] != d['value'] else d['value']}[{d['height']}]",
            dom,
        )

    a_str, a_dom = _info(na)
    b_str, b_dom = _info(nb)
    return f"{a_str} x {b_str} ({a_dom}x{b_dom}={a_dom*b_dom})"


def compute_edge_weight(
    node_a: str,
    node_b: str,
    g: nx.Graph,
    attrs: DatasetAttributes,
    size_penalty: float,
) -> float:
    from ....graph.hugin import get_attrs as _get_attrs

    def _dom_for_node(node: str) -> float:
        d = g.nodes[node]
        a = _get_attrs(attrs, d["table"], d["order"])[d["attr"]]
        val = cast(CatValue, a[d["value"]])
        return float(val.get_domain(d["height"]))

    return 1 / ((_dom_for_node(node_a) * _dom_for_node(node_b)) ** size_penalty)


# ============================================================
# Step 2d: Structure learning main loop
# ============================================================
def _compute_cand_edge_rho(
    idx: int,
    candidates: list[tuple[str, str]],
    directed_graph: nx.DiGraph,
    attrs: DatasetAttributes,
    n: int | float,
    theta_2w: float,
    sigma_floor: float,
) -> float:
    """Compute the measurement rho cost for candidate edge at index idx.

    Returns float('inf') if the edge cannot achieve theta_2w confidence."""
    from ....graph.hugin import get_attrs as _ga

    na, nb = candidates[idx]
    da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
    val_a = cast(
        CatValue, _ga(attrs, da["table"], da["order"])[da["attr"]][da["value"]]
    )
    val_b = cast(
        CatValue, _ga(attrs, db["table"], db["order"])[db["attr"]][db["value"]]
    )
    dom = val_a.get_domain(da["height"]) * val_b.get_domain(db["height"])
    rho = compute_rho_for_theta(dom, n, theta_2w, sigma_floor)
    return rho if rho is not None else float("inf")


def structure_learn(
    directed_graph: nx.DiGraph,
    attrs: DatasetAttributes,
    tvd: dict[tuple[Col, Col], np.ndarray],
    n: int,
    size_penalty: float,
    rho_avail: float,
    min_tvd: float,
    em_z: float,
    theta_2w: float,
    sigma_floor: float = 1.0,
    frozen_nodes: set[str] | None = None,
    n_hist_cols: int = 0,
    max_clique_size: float = 1e5,
) -> tuple[nx.Graph, set[frozenset[str]], float]:
    """Greedy edge addition with exponential mechanism and budget tracking.

    Each EM step costs a fixed rho derived from em_z.  Each selected edge
    commits measurement budget derived from theta_2w and the edge's domain.
    The loop exits when the remaining budget cannot cover the next step.

    Forces at least one edge per column (budget permitting).

    Returns:
        moral: Undirected moralized graph with structure-learning edges added.
        structure_edges: Set of frozenset node pairs for structure-learning edges.
        rho_remaining: Unspent budget (for measurement + leftover).
    """
    from ....graph.hugin import to_moral, get_factor_domain
    from ....utils.progress import piter, check_exit

    # Moralize the directed height-chain graph -> undirected base
    moral = to_moral(directed_graph)

    # Generate candidates and group by column pair
    candidates, col_pair_map = generate_candidates(directed_graph, frozen_nodes)
    connected_pairs: set[tuple[Col, Col]] = set()
    structure_edges: set[frozenset[str]] = set()

    # EM uses the real TVD sensitivity (2/n) for the mechanism call,
    # giving a high discrimination factor (em_z * n / N_cands).
    # Budget cost uses eps = em_z * 4/N_cands, scaling with candidate count.
    sensitivity = 2.0 / n

    # Pre-compute per-candidate measurement rho and theta-filter
    cand_rho_edge = np.zeros(len(candidates))
    cand_valid = np.ones(len(candidates), dtype=bool)
    if rho_avail > 0:
        n_filtered = 0
        for idx in range(len(candidates)):
            rho = _compute_cand_edge_rho(
                idx, candidates, directed_graph, attrs, n, theta_2w, sigma_floor
            )
            cand_rho_edge[idx] = rho
            if np.isinf(rho):
                cand_valid[idx] = False
                n_filtered += 1
        if n_filtered:
            logger.info(
                f"Adjuvant: theta_2w filter excluded {n_filtered}/{len(candidates)} "
                f"candidates (unachievable at theta_2w={theta_2w})"
            )

    # Per-column edge limits
    d = len(set(c for pair in col_pair_map for c in pair))
    h = n_hist_cols
    if h > 0:
        max_edges_per_col = min(d, 2 * int(sqrt(h + d) + 0.5))
    else:
        max_edges_per_col = min(d, 2 * int(sqrt(d) + 0.5))
    col_edge_count: dict[Col, int] = {}
    saturated_cols: set[Col] = set()

    max_steps = d * (max_edges_per_col // 2 + 1)
    rho_em = 0.0  # cumulative EM selection budget spent
    rho_committed = 0.0  # cumulative edge measurement budget committed

    def _em_cost(n_cands: int) -> tuple[float, float]:
        """Compute (eps, rho) for EM over n_cands candidates.

        eps = em_z * 4 / n_cands; the EM call uses sensitivity = 2/n
        giving discrimination factor em_z * n / n_cands."""
        if em_z <= 0 or rho_avail <= 0 or n_cands <= 0:
            return 0.0, 0.0
        eps = em_z * 4.0 / n_cands
        return eps, eps * eps / 8.0

    pbar = piter(
        None,
        total=max_steps,
        desc="Adjuvant structure [0 edges, score=0.0000]",
        unit="col_pair",
        bar_format=" " * 11
        + ">>>>>>>  {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
        + " [{elapsed}<{remaining}]",
    )

    # Precompute TVD*boost for all candidates (doesn't change across iterations)
    cand_tvd_boost = np.empty(len(candidates))
    for idx, (na, nb) in enumerate(candidates):
        da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
        col_a: Col = (da.get("table"), da.get("order"), da["attr"], da["value"])
        col_b: Col = (db.get("table"), db.get("order"), db["attr"], db["value"])
        tvd_arr = tvd.get((col_a, col_b))
        base = (
            float(tvd_arr[da["height"], db["height"]]) if tvd_arr is not None else 0.0
        )
        boost = compute_edge_weight(na, nb, moral, attrs, size_penalty)
        cand_tvd_boost[idx] = base * boost

    # Cache node data for clique-size checks (immutable, built once)
    node_data = {n: d for n, d in directed_graph.nodes(data=True)}
    # Maintained adjacency dict (updated incrementally, avoids rebuilding from nx)
    base_adj = _build_adj(moral)
    # Track which candidates have been validated against the current graph.
    # When an edge is added, candidates touching either endpoint's extended
    # neighborhood are invalidated and must be re-checked.
    cand_clique_checked = np.zeros(len(candidates), dtype=bool)
    cand_clique_ok = np.ones(len(candidates), dtype=bool)

    for it in range(max_steps):
        try:
            check_exit()
        except Exception as e:
            pbar.close()
            raise e

        # Filter to active candidates:
        # - not excluded by theta filter
        # - column pair not yet connected
        # - neither column saturated (reached max edges)
        active: list[tuple[int, str, str]] = []
        for idx, (na, nb) in enumerate(candidates):
            if not cand_valid[idx]:
                continue
            da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
            col_a: Col = (da.get("table"), da.get("order"), da["attr"], da["value"])
            col_b: Col = (db.get("table"), db.get("order"), db["attr"], db["value"])
            pair = tuple(sorted([col_a, col_b], key=_col_sort_key))
            if pair in connected_pairs:
                continue
            if col_a in saturated_cols or col_b in saturated_cols:
                continue
            active.append((idx, na, nb))

        if not active:
            logger.info(
                f"Adjuvant: exit (no active candidates) at iter {it}. "
                f"pairs={len(connected_pairs)}, saturated={len(saturated_cols)}/{d}, "
                f"remaining_pairs={len(col_pair_map) - len(connected_pairs)} "
                f"(all connected or saturated)"
            )
            break

        # Clique-size filter: check candidates via exact triangulation.
        # Cached: only re-check candidates whose neighborhoods changed
        # since last validation (touched by a newly added edge).
        n_clique_filtered = 0
        for idx, na, nb in active:
            if cand_clique_checked[idx] and cand_clique_ok[idx]:
                continue  # still valid from a previous iteration
            _, valid = _score_with_one_edge(
                base_adj, na, nb, node_data, attrs, max_clique_size
            )
            cand_clique_checked[idx] = True
            cand_clique_ok[idx] = valid
            if not valid:
                cand_valid[idx] = False
                n_clique_filtered += 1
        if n_clique_filtered:
            # Rebuild active list after filtering
            active = [(idx, na, nb) for idx, na, nb in active if cand_valid[idx]]
            if not active:
                logger.info(
                    f"Adjuvant: exit (no candidates after clique filter) at iter {it}."
                )
                break

        # --- Two-pass EM cost + affordability filter ---
        # Pass 1: estimate EM cost from full active set
        _, rho_em_est = _em_cost(len(active))
        rho_after_em = rho_avail - rho_em - rho_committed - rho_em_est

        if rho_avail > 0 and rho_after_em < 0:
            logger.info(
                f"Adjuvant: exit (budget exhausted) at iter {it}. "
                f"rho_avail={rho_avail:.6f}, rho_em={rho_em:.6f}, "
                f"rho_committed={rho_committed:.6f}"
            )
            break

        # Drop candidates whose edge measurement exceeds the remaining budget
        if rho_avail > 0:
            affordable = [
                (idx, na, nb)
                for idx, na, nb in active
                if cand_rho_edge[idx] <= rho_after_em
            ]
        else:
            affordable = active

        if not affordable:
            logger.info(
                f"Adjuvant: exit (no affordable candidates) at iter {it}. "
                f"active={len(active)}, rho_after_em={rho_after_em:.6f}"
            )
            break

        # Pass 2: recompute EM cost with the filtered candidate count
        eps_step, rho_em_step = _em_cost(len(affordable))

        # Score affordable candidates
        scores = np.array([cand_tvd_boost[idx] for idx, _, _ in affordable])

        # Exponential mechanism selection among affordable candidates + "stop" option.
        log_n_boost = (
            2 * sensitivity * np.log(max(len(scores), 1)) / eps_step
            if eps_step > 0
            else 0
        )
        em_scores = np.append(scores, min_tvd + log_n_boost if min_tvd else 0)
        stop_idx = len(scores)

        if eps_step > 0:
            sel = exponential_mechanism(em_scores, eps_step, sensitivity)
            rho_em += rho_em_step
        else:
            sel = int(np.argmax(em_scores))

        if sel == stop_idx:
            # Selected one of the N stop slots
            logger.info(
                f"Adjuvant: exit (EM picked stop option, min_tvd={min_tvd}) "
                f"at iter {it}, edges={len(structure_edges)}"
            )
            pbar.update(1)
            break

        cand_idx, na, nb = affordable[sel]
        edge_rho = cand_rho_edge[cand_idx]

        # Accept edge
        moral.add_edge(na, nb, structure=True)
        base_adj[na].add(nb)
        base_adj[nb].add(na)
        structure_edges.add(frozenset([na, nb]))
        rho_committed += edge_rho

        # Invalidate clique cache for candidates touching either endpoint's
        # extended neighborhood (nodes whose triangulation could change)
        affected = base_adj[na] | base_adj[nb] | {na, nb}
        for idx, (ca, cb) in enumerate(candidates):
            if cand_clique_checked[idx] and (ca in affected or cb in affected):
                cand_clique_checked[idx] = False

        # Update column pair tracking and edge counts
        da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
        col_a: Col = (da.get("table"), da.get("order"), da["attr"], da["value"])
        col_b: Col = (db.get("table"), db.get("order"), db["attr"], db["value"])
        pair = tuple(sorted([col_a, col_b], key=_col_sort_key))
        connected_pairs.add(pair)

        col_edge_count[col_a] = col_edge_count.get(col_a, 0) + 1
        col_edge_count[col_b] = col_edge_count.get(col_b, 0) + 1
        if col_edge_count[col_a] >= max_edges_per_col:
            saturated_cols.add(col_a)
        if col_edge_count[col_b] >= max_edges_per_col:
            saturated_cols.add(col_b)

        logger.info(
            f"-> {it+1:3d}/{max_steps} "
            + f"(score={scores[sel]:.4f}"
            + (
                f", budget={rho_avail - rho_em - rho_committed:.6f}"
                if rho_avail > 0
                else ""
            )
            + (
                f", rm {n_clique_filtered}/{len(active)}"
                if n_clique_filtered
                else ""
            )
            + f"): {_fmt_edge(na, nb, moral, attrs)}"
        )

        pbar.set_description(
            f"Adjuvant structure [{len(structure_edges)} edges, "
            f"score={scores[sel]:.4f}]"
        )
        pbar.update(1)

    pbar.close()
    rho_remaining = rho_avail - rho_em - rho_committed

    logger.info(
        f"Adjuvant: structure learning done, "
        + f"{len(structure_edges)} edges, "
        + f"{len(connected_pairs)} column pairs connected, "
        + (
            f"rho_em={rho_em:.6f}, rho_measure={rho_committed:.6f}, "
            + f"rho_remaining={rho_remaining:.6f}"
            if rho_avail > 0
            else "no budget tracking"
        )
    )

    diag = format_tvd_diagnostic(
        tvd, structure_edges, connected_pairs, col_pair_map,
        directed_graph, moral, attrs, min_tvd,
    )
    for line in diag.splitlines():
        logger.info(line)

    return moral, structure_edges, rho_remaining


def format_tvd_diagnostic(
    tvd: dict[tuple[Col, Col], np.ndarray],
    structure_edges: set[frozenset[str]],
    connected_pairs: set[tuple[Col, Col]],
    col_pair_map: dict,
    directed_graph: "nx.DiGraph",
    moral: "nx.Graph",
    attrs: DatasetAttributes,
    min_tvd: float,
) -> str:
    """Format TVD diagnostic showing connected and missing column pairs."""
    candidate_cols = set(c for pair in col_pair_map for c in pair)
    all_pairs_tvd: list[tuple[float, Col, Col]] = []
    for (ca, cb), val_arr in tvd.items():
        if _col_sort_key(ca) < _col_sort_key(cb) and ca in candidate_cols and cb in candidate_cols:
            all_pairs_tvd.append((float(val_arr[0, 0]), ca, cb))
    all_pairs_tvd.sort(key=lambda x: (-x[0], _col_sort_key(x[1]), _col_sort_key(x[2])))

    lines = ["Connected column pairs (by TVD):"]
    for val, ca, cb in all_pairs_tvd:
        pair = tuple(sorted([ca, cb], key=_col_sort_key))
        if pair in connected_pairs:
            for edge in structure_edges:
                ena, enb = tuple(edge)
                da, db = directed_graph.nodes[ena], directed_graph.nodes[enb]
                edge_pair = tuple(
                    sorted(
                        [
                            (da.get("table"), da.get("order"), da["attr"], da["value"]),
                            (db.get("table"), db.get("order"), db["attr"], db["value"]),
                        ],
                        key=_col_sort_key,
                    )
                )
                if edge_pair == pair:
                    ha, hb = da["height"], db["height"]
                    col_a_t: Col = (
                        da.get("table"),
                        da.get("order"),
                        da["attr"],
                        da["value"],
                    )
                    col_b_t: Col = (
                        db.get("table"),
                        db.get("order"),
                        db["attr"],
                        db["value"],
                    )
                    tvd_arr = tvd.get((col_a_t, col_b_t))
                    tvd_at_h = float(tvd_arr[ha, hb]) if tvd_arr is not None else 0.0
                    lines.append(
                        f"  CONNECTED TVD={tvd_at_h:.4f}{f'/{val:.4f}' if ha != 0 or hb != 0 else ''} "
                        f"{_fmt_edge(ena, enb, moral, attrs)}"
                    )
                    break
        elif val >= min_tvd:
            ca_tbl, ca_ord, ca_attr, ca_val = ca
            cb_tbl, cb_ord, cb_attr, cb_val = cb
            # Skip hist-table pairs (they are frozen, not candidates)
            if ca_tbl is not None or cb_tbl is not None:
                continue
            lines.append(
                f"    MISSING TVD={val:.4f} "
                f"{ca_attr + '.' if ca_attr != ca_val else ''}{ca_val} x "
                f"{cb_attr + '.' if cb_attr != cb_val else ''}{cb_val}"
            )

    return "\n".join(lines)


def print_adjuvant(
    attrs: DatasetAttributes,
    moral: "nx.Graph",
    rho: float,
    rho_remaining: float,
    theta_1w: float,
    theta_2w: float,
    em_z: float,
    n_obs: int,
) -> str:
    """Format a summary string for an Adjuvant model."""
    from ....graph.hugin import get_attrs as _get_attrs

    s = "Adjuvant Graphical Model:\n"
    s += (
        f"(rho={rho:.6f}, rho_remaining={rho_remaining:.6f}, "
        f"theta_1w={theta_1w:.1f}, theta_2w={theta_2w:.1f}, em_z={em_z:.1f}, "
        f"{n_obs} observations)\n"
    )

    # Collect structure-learning edges
    structure_edges: list[tuple[str, str]] = []
    for na, nb, data in moral.edges(data=True):
        if data.get("structure"):
            structure_edges.append((na, nb))

    if not structure_edges:
        s += "No structure-learning edges.\n"
        return s

    # Format edge table
    edge_len = 55
    s += f"┌{'─' * 8}┬{'─' * edge_len}┐\n"
    s += f"│{'TVD':>7s} │ {'Edge':{edge_len - 2}s} │\n"
    s += f"├{'─' * 8}┼{'─' * edge_len}┤\n"

    edge_info: list[tuple[float, str]] = []
    for na, nb in structure_edges:
        da, db = moral.nodes[na], moral.nodes[nb]

        val_a = cast(
            CatValue, _get_attrs(attrs, da.get("table"), da.get("order"))[da["attr"]][da["value"]]
        )
        val_b = cast(
            CatValue, _get_attrs(attrs, db.get("table"), db.get("order"))[db["attr"]][db["value"]]
        )
        dom_a = val_a.get_domain(da["height"])
        dom_b = val_b.get_domain(db["height"])

        def _node_str(d):
            if d["attr"] != d["value"]:
                return f"{d['attr']}.{d['value'].replace(d['attr'] + '_', '')}[{d['height']}]"
            return f"{d['value']}[{d['height']}]"

        edge_str = f"{_node_str(da)} x {_node_str(db)} ({dom_a}x{dom_b}={dom_a * dom_b})"
        edge_info.append((0.0, edge_str))

    edge_info.sort(key=lambda x: -x[0])
    for tvd_val, edge_str in edge_info:
        s += f"│ {tvd_val:6.4f} │ {edge_str:{edge_len - 2}s} │\n"

    s += f"└{'─' * 8}┴{'─' * edge_len}┘\n"

    # Multi-value attrs
    tattrs = cast(Attributes, attrs[None])
    if any(len(attr.vals) > 1 for attr in tattrs.values()):
        tlen = max(len(name) for name in tattrs) + 1
        s += f"┌{'─' * tlen}┬{'─' * 6}┬{'─' * 40}┐\n"
        s += f"│{'Multi-Val Attrs':>{tlen - 1}s} │  Cmn │ {'Values':<38s} │\n"
        s += f"├{'─' * tlen}┼{'─' * 6}┼{'─' * 40}┤\n"
        for name, attr in tattrs.items():
            if len(attr.vals) <= 1:
                continue
            cmn = str(attr.common.domain) if attr.common else "NIL"
            vals_str = " ".join(attr.vals.keys())
            s += f"│{name:>{tlen - 1}s} │ {cmn:>4s} │ {vals_str:<38s} │\n"
        s += f"└{'─' * tlen}┴{'─' * 6}┴{'─' * 40}┘\n"

    return s


# ============================================================
# Step 3: Measurement helpers
# ============================================================
def _clique_to_request(clique):
    """Convert CliqueMeta to oracle request format."""
    from ....graph.beliefs import convert_sel

    return [(attr_name, convert_sel(sel)) for _, _, attr_name, sel in clique]


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
                for node in attr_val_to_nodes.get(
                    (am.table, am.order, am.attr, am.attr), set()
                ):
                    clique_nodes.add(node)
            else:
                for val, h in am.sel:
                    for node in attr_val_to_nodes.get(
                        (am.table, am.order, am.attr, val), set()
                    ):
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
    sigma_floor: float = 1.0,
) -> tuple[list, float]:
    """Measure selected clique marginals with Gaussian noise.

    Returns (list of LinearObservation, sigma3)."""
    from ....graph.hugin import get_clique_domain, get_attrs as _get_attrs
    from ....graph.loss import LinearObservation
    from ....graph.beliefs import convert_sel

    K = len(cliques_to_measure)
    if K == 0:
        return [], 0.0

    sigma3 = 0.0 if rho3 <= 0 else sqrt(K / (2 * rho3))

    # Build oracle requests from CliqueMeta
    requests = [_clique_to_request(cl) for cl in cliques_to_measure]
    results = oracle.process(requests, postprocess=None)

    obs_list = []
    for clique, result in zip(cliques_to_measure, results):
        # Oracle returns data in naive shape (product of per-value domains)
        naive_dims = []
        for _, _, attr_name, sel in clique:
            a = _get_attrs(attrs, None, None)[attr_name]
            sel_d = convert_sel(sel)
            if isinstance(sel_d, int):
                naive_dims.append(a.common.get_domain(sel_d))
            else:
                nd = 1
                for vn, h in sel_d.items():
                    nd *= cast(CatValue, a.vals[vn]).get_domain(h)
                naive_dims.append(nd)

        raw = result.astype(np.float64).ravel()
        if sigma3 > 0:
            raw = raw + np.random.normal(0, sigma3, size=raw.size)
        raw = raw.clip(0)
        prob = raw.reshape(naive_dims)

        # Compress naive→compressed for each dim
        for dim_i, (_, _, attr_name, sel) in enumerate(clique):
            sel_d = convert_sel(sel)
            if isinstance(sel_d, int):
                continue
            a = _get_attrs(attrs, None, None)[attr_name]
            naive_dom = prob.shape[dim_i]
            compressed_dom = a.get_domain(sel_d)
            if naive_dom == compressed_dom:
                continue

            raw_naive = a.get_naive_mapping(sel_d)
            raw_compressed = a.get_mapping(sel_d)
            _, unique_idx = np.unique(raw_naive, return_index=True)
            naive_idx = raw_naive[unique_idx]
            compressed_idx = raw_compressed[unique_idx]

            i_map = tuple(
                naive_idx if j == dim_i else slice(None) for j in range(len(prob.shape))
            )
            o_map = tuple(
                compressed_idx if j == dim_i else slice(None)
                for j in range(len(prob.shape))
            )
            tmp = np.zeros(
                [compressed_dom if j == dim_i else d for j, d in enumerate(prob.shape)],
                dtype=prob.dtype,
            )
            np.add.at(tmp, o_map, prob[i_map])
            prob = tmp

        s = prob.sum()
        prob = (prob / s if s > 0 else prob).astype(np.float32)

        dom = get_clique_domain(clique, attrs)
        confidence = calc_confidence(n, sigma3, dom, sigma_floor)
        obs_list.append(LinearObservation(clique, None, prob, confidence))

    return obs_list, sigma3


def measure_edges(
    oracle: MarginalOracle,
    structure_edges: set[frozenset[str]],
    graph: nx.Graph,
    attrs: DatasetAttributes,
    n: int,
    theta_2w: float,
    sigma_floor: float = 1.0,
    rho_extra: float = 0.0,
) -> tuple[list, float]:
    """Measure 2-way marginals with per-edge noise calibrated to theta_2w.

    Each edge gets its own sigma_dp derived from its domain and theta_2w.
    If rho_extra > 0, the effective theta is raised (via binary search)
    to exhaust the leftover budget, improving all edge measurements.

    Returns (list of LinearObservation, max_sigma)."""
    from ....graph.hugin import AttrMeta, get_clique_domain, get_attrs as _get_attrs
    from ....graph.loss import LinearObservation
    from ....graph.beliefs import convert_sel

    edges = list(structure_edges)
    K = len(edges)
    if K == 0:
        return [], 0.0

    # Build oracle requests and CliqueMeta for each edge.
    # Columns from the same attribute are merged into a single AttrMeta
    # with a combined selector (required by junction tree parent matching).
    from collections import defaultdict

    requests = []
    edge_metas = []
    for edge in edges:
        na, nb = tuple(edge)

        # Group by (table, order, attr), merge values from same attribute
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
            if (
                len(sel_dict) == 1
                and attr.common
                and next(iter(sel_dict)) == attr.common.name
            ):
                new_sel: int | tuple = sel_dict[attr.common.name]
            else:
                cmn = attr.common.name if attr.common else None
                new_sel = tuple(sorted((v, h) for v, h in sel_dict.items() if v != cmn))
            source.append(AttrMeta(table, order, attr_name, new_sel))
        source_tuple = tuple(sorted(source, key=_attr_meta_sort_key))
        edge_metas.append(source_tuple)

        # Oracle request: use the sel from the source
        # For hist table columns, include (table, order) selector prefix.
        req = []
        for tbl, ord_, attr_name, sel in source_tuple:
            sel_d = convert_sel(sel)
            if tbl is not None:
                table_sel = (tbl, ord_) if ord_ is not None else tbl
                req.append((table_sel, attr_name, sel_d))
            else:
                req.append((attr_name, sel_d))
        requests.append(req)

    results = oracle.process(requests, postprocess=None)

    # Collect per-edge domains for sigma computation
    edge_doms = [get_clique_domain(st, attrs) for st in edge_metas]

    # If there is leftover budget, boost the effective theta via binary search
    eff_theta = theta_2w
    if rho_extra > 0 and K > 0:

        def _total_rho_2w(theta):
            total = 0.0
            for dom in edge_doms:
                r = compute_rho_for_theta(dom, n, theta, sigma_floor)
                if r is not None:
                    total += r
            return total

        base_rho = _total_rho_2w(theta_2w)
        target_rho = base_rho + rho_extra
        # Binary search for max theta that fits within target_rho
        lo, hi = theta_2w, theta_2w * 1000
        # First check if hi is enough
        if _total_rho_2w(hi) <= target_rho:
            eff_theta = hi
        else:
            for _ in range(64):
                mid = (lo + hi) / 2
                if _total_rho_2w(mid) <= target_rho:
                    lo = mid
                else:
                    hi = mid
            eff_theta = lo
        if eff_theta > theta_2w:
            logger.info(
                f"Adjuvant: boosted theta_2w {theta_2w:.1f} -> {eff_theta:.1f} "
                f"(rho_extra={rho_extra:.6f})"
            )

    # Compute per-edge sigma from effective theta
    edge_sigmas: list[float] = []
    for dom in edge_doms:
        rho_edge = compute_rho_for_theta(dom, n, eff_theta, sigma_floor)
        if rho_edge is not None and rho_edge > 0:
            edge_sigmas.append(sqrt(1.0 / (2.0 * rho_edge)))
        else:
            edge_sigmas.append(0.0)

    obs_list = []
    max_sigma = 0.0
    for source_tuple, result, sigma_edge in zip(edge_metas, results, edge_sigmas):
        max_sigma = max(max_sigma, sigma_edge)

        # Oracle returns data with per-attribute domain that accounts for
        # common values: product(d_i - common) + common for multi-value
        # selectors.  For single-value selectors, oracle dim = val domain.
        naive_dims = []
        for tbl, ord_, attr_name, sel in source_tuple:
            a = _get_attrs(attrs, tbl, ord_)[attr_name]
            sel_d = convert_sel(sel)
            if isinstance(sel_d, int):
                naive_dims.append(a.common.get_domain(sel_d))
            elif len(sel_d) == 1:
                vn, h = next(iter(sel_d.items()))
                naive_dims.append(cast(CatValue, a.vals[vn]).get_domain(h))
            else:
                from ....marginal.numpy import _calc_common
                cmn = min(
                    _calc_common(cast(CatValue, a.vals[vn]), a.common)
                    for vn in sel_d
                )
                nd = 1
                for vn, h in sel_d.items():
                    nd *= cast(CatValue, a.vals[vn]).get_domain(h) - cmn
                nd += cmn
                naive_dims.append(nd)

        raw = result.astype(np.float64).ravel()
        if sigma_edge > 0:
            raw = raw + np.random.normal(0, sigma_edge, size=raw.size)
        raw = raw.clip(0)
        prob = raw.reshape(naive_dims)

        # Compress oracle→compressed for each dim
        for dim_i, (tbl, ord_, attr_name, sel) in enumerate(source_tuple):
            sel_d = convert_sel(sel)
            if isinstance(sel_d, int) or len(sel_d) <= 1:
                continue
            a = _get_attrs(attrs, tbl, ord_)[attr_name]
            oracle_dom = prob.shape[dim_i]
            compressed_dom = a.get_domain(sel_d)
            if oracle_dom == compressed_dom:
                continue

            # Build oracle→compressed mapping by iterating over raw
            # combinations.  The oracle uses common-aware linearization:
            #   bin = v_last_full + v_prev_noncommon * stride ...
            from ....marginal.numpy import _calc_common
            cmn_c = min(
                _calc_common(cast(CatValue, a.vals[vn]), a.common)
                for vn in sel_d
            )
            val_items = list(sel_d.items())
            val_doms = [
                cast(CatValue, a.vals[vn]).get_domain(h)
                for vn, h in val_items
            ]
            raw_total = int(np.prod(val_doms))

            # Per-value indices for every raw combination
            per_val = []
            remaining = np.arange(raw_total, dtype=np.int64)
            for d in reversed(val_doms):
                per_val.append(remaining % d)
                remaining //= d
            per_val.reverse()

            # Oracle flat index for each raw combination
            oracle_bins = np.zeros(raw_total, dtype=np.int64)
            o_stride = 1
            for vi_rev in range(len(val_items)):
                vi = len(val_items) - 1 - vi_rev
                gv = per_val[vi]
                if cmn_c == 0 or vi_rev == 0:
                    oracle_bins += gv * o_stride
                else:
                    oracle_bins += np.maximum(0, gv - cmn_c) * o_stride
                o_stride *= val_doms[vi] - cmn_c

            comp_mapping = np.array(a.get_mapping(sel_d), dtype=np.int64)
            o2c = np.zeros(oracle_dom, dtype=np.int64)
            o2c[oracle_bins] = comp_mapping

            i_map = tuple(
                slice(None) for _ in range(len(prob.shape))
            )
            o_map = tuple(
                o2c if j == dim_i else slice(None)
                for j in range(len(prob.shape))
            )
            tmp = np.zeros(
                [compressed_dom if j == dim_i else d for j, d in enumerate(prob.shape)],
                dtype=prob.dtype,
            )
            np.add.at(tmp, o_map, prob[i_map])
            prob = tmp

        s = prob.sum()
        prob = (prob / s if s > 0 else prob).astype(np.float32)

        dom = get_clique_domain(source_tuple, attrs)
        confidence = calc_confidence(n, sigma_edge, dom, sigma_floor)
        obs_list.append(LinearObservation(source_tuple, None, prob, confidence))

    return obs_list, max_sigma


def build_1way_observations(
    noisy_1way: dict[Col, np.ndarray],
    attrs: DatasetAttributes,
    n: int,
    sigmas: dict[Col, float],
    sigma_floor: float = 1.0,
) -> list:
    """Build per-column LinearObservation objects from noisy 1-way marginals.

    Each column already has its own marginal and sigma, so no marginalization needed."""
    from ....graph.hugin import AttrMeta, get_attrs as _get_attrs
    from ....graph.loss import LinearObservation

    obs_list = []
    for col, noisy_mar in noisy_1way.items():
        table, order, attr_name, val_name = col
        attr = _get_attrs(attrs, table, order)[attr_name]

        if attr.common and val_name == attr.common.name:
            source = (AttrMeta(table, order, attr_name, 0),)
        else:
            source = (AttrMeta(table, order, attr_name, ((val_name, 0),)),)

        raw = noisy_mar.copy().clip(0)
        s = raw.sum()
        prob = (raw / s if s > 0 else raw).astype(np.float32)
        dom = len(raw)
        sigma_col = sigmas.get(col, 0.0)
        confidence = calc_confidence(n, sigma_col, dom, sigma_floor)
        obs_list.append(LinearObservation(source, None, prob, confidence))

    return obs_list


# ============================================================
# High-level pipeline functions
# ============================================================
def adjuvant_fit(
    oracle: MarginalOracle,
    attrs: DatasetAttributes,
    n: int,
    *,
    rho: float = 0.0,
    theta_1w: float = 50,
    theta_2w: float = 4,
    em_z: float = 2.0,
    ew_ratio: float = 0.8,
    size_penalty: float = 0.0,
    min_tvd: float = 0.05,
    sigma_floor: float = 1.0,
    frozen_nodes: set[str] | None = None,
    n_hist_cols: int = 0,
    max_clique_size: float = 1e5,
    rescale: bool = True,
) -> tuple[list, "nx.Graph", float]:
    """Run the full Adjuvant pipeline: marginals, noise, structure learn, measure.

    Returns (all_obs, moral, rho_remaining) where all_obs is a list of
    LinearObservation, moral is the moralized graph with structure-learning
    edges, and rho_remaining is the unspent CDP budget."""

    all_cols = get_col_names(attrs)
    hist_cols = get_hist_cols(all_cols)
    d = len(all_cols)
    h = n_hist_cols or len(hist_cols)

    # Step 0: Compute all 1-way and 2-way marginals
    logger.info(f"Adjuvant Step 0: Computing marginals ({d} cols, {h} hist, {n} rows)")
    cached = compute_all_marginals(oracle, attrs, all_cols)

    # Step 1: Noisy 1-way marginals (budget from theta_1w)
    rho1_max = ew_ratio * rho if rho > 0 else None
    sigmas_1w, rho1, eff_theta_1w = compute_1way_budget(
        cached, n, theta_1w, sigma_floor, rho1_max
    )
    logger.info(
        f"Adjuvant Step 1: Noisy 1-way marginals "
        f"(theta_1w={eff_theta_1w:.1f}, rho1={rho1:.6f})"
    )
    noisy_1way = add_noise_1way(cached, sigmas_1w)

    # Step 2: Structure learning (remaining budget)
    rho_avail = rho - rho1
    logger.info(
        f"Adjuvant Step 2: Structure learning "
        f"(rho_avail={rho_avail:.6f}, em_z={em_z}, theta_2w={theta_2w})"
    )
    tvd = compute_tvd(cached, attrs, all_cols)
    directed_graph = build_height_chain_graph(attrs)
    logger.info(
        f"Adjuvant: height-chain graph has {directed_graph.number_of_nodes()} "
        f"nodes, {directed_graph.number_of_edges()} chain edges"
    )

    moral, structure_edges, rho_remaining = structure_learn(
        directed_graph,
        attrs,
        tvd,
        n,
        size_penalty,
        rho_avail,
        min_tvd,
        em_z=em_z,
        theta_2w=theta_2w,
        sigma_floor=sigma_floor,
        frozen_nodes=frozen_nodes,
        n_hist_cols=h,
        max_clique_size=max_clique_size,
    )

    # Step 3: Measure edge marginals (per-edge sigma from theta_2w)
    logger.info(
        f"Adjuvant Step 3: Measuring {len(structure_edges)} edge marginals "
        f"(theta_2w={theta_2w}, rho_remaining={rho_remaining:.6f})"
    )
    edge_obs, max_sigma = measure_edges(
        oracle,
        structure_edges,
        moral,
        attrs,
        n,
        theta_2w,
        sigma_floor,
        rho_extra=rho_remaining if rescale else 0.0,
    )
    oneway_obs = build_1way_observations(noisy_1way, attrs, n, sigmas_1w, sigma_floor)
    all_obs = edge_obs + oneway_obs
    logger.info(
        f"Adjuvant: {len(edge_obs)} edge obs + {len(oneway_obs)} 1-way obs, "
        f"max_sigma_2w={max_sigma:.2f}"
    )

    return all_obs, moral, rho_remaining


def adjuvant_run_md(
    all_obs: list,
    attrs: DatasetAttributes,
    moral: "nx.Graph | None",
    md_params: dict,
) -> tuple:
    """Build junction tree and run mirror descent.

    Returns (junction, cliques, potentials)."""
    from ....graph.hugin import get_clique_domain
    from ....graph.mirror_descent import (
        MIRROR_DESCENT_DEFAULT,
        build_junction_tree,
        mirror_descent,
    )

    md = {**MIRROR_DESCENT_DEFAULT, **md_params}
    md.pop("compress", None)
    md.pop("sample", None)
    tree_mode = md.pop("tree", "hugin")
    elim_factor_cost = md.pop("elim_factor_cost", 1)
    elim_max_attempts = md.pop("elim_max_attempts", 5000)
    device = md.pop("device", "auto")
    device = None if device == "auto" else device

    mg = moral if tree_mode != "maximal" else None
    logger.info(f"Adjuvant: building junction tree (mode={tree_mode})")
    junction, cliques, messages = build_junction_tree(
        all_obs,
        attrs,
        tree_mode=tree_mode,
        moral_graph=mg,
        elim_factor_cost=elim_factor_cost,
        elim_max_attempts=elim_max_attempts,
    )
    total_params = sum(get_clique_domain(cl, attrs) for cl in cliques)
    logger.info(
        f"Adjuvant: junction tree has {len(cliques)} cliques, "
        f"{total_params:_} parameters"
    )

    logger.info(
        f"Adjuvant: running mirror descent (max_iters={md.get('max_iters', 1000)})"
    )
    potentials, *_ = mirror_descent(
        cliques,
        messages,
        all_obs,
        attrs,
        device=device,
        **md,
    )

    return junction, cliques, potentials
