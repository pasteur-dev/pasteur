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

from ....attribute import CatValue, DatasetAttributes, SeqAttributes
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


def _sigma_for_theta(dom: int, n: int | float, theta: float) -> float:
    """Compute DP noise std dev to achieve confidence theta/(theta+1)."""
    if theta <= 0 or dom <= 0:
        return 0.0
    return n / (theta * dom)


def _sigma_to_budget(sigma: float, dp_type: str = "cdp") -> float:
    """Convert noise std dev to budget cost (rho for CDP, epsilon for DP)."""
    if sigma <= 0:
        return 0.0
    if dp_type == "cdp":
        return 1.0 / (2.0 * sigma * sigma)
    else:
        return sqrt(2.0) / sigma


def _budget_to_sigma(budget: float, dp_type: str = "cdp") -> float:
    """Convert budget (rho for CDP, epsilon for DP) to noise std dev."""
    if budget <= 0:
        return 0.0
    if dp_type == "cdp":
        return sqrt(1.0 / (2.0 * budget))
    else:
        return sqrt(2.0) / budget


def _add_dp_noise(data: np.ndarray, sigma: float, dp_type: str = "cdp") -> np.ndarray:
    """Add Gaussian (CDP) or Laplace (DP) noise with std dev sigma."""
    if sigma <= 0:
        return data.copy()
    if dp_type == "cdp":
        return data + np.random.normal(0, sigma, size=data.shape)
    else:
        return data + np.random.laplace(0, sigma / sqrt(2), size=data.shape)


def _em_budget_cost(eps: float, dp_type: str = "cdp") -> float:
    """Convert exponential mechanism epsilon to budget cost."""
    if dp_type == "cdp":
        return eps * eps / 2.0
    else:
        return eps


def compute_budget_for_theta(
    dom: int,
    n: int | float,
    theta: float,
    dp_type: str = "cdp",
) -> float:
    """Compute budget to achieve confidence theta/(theta+1) for a marginal of domain `dom`.

    Returns rho (CDP) or epsilon (DP)."""
    sigma = _sigma_for_theta(dom, n, theta)
    if sigma == 0.0:
        return 0.0
    return _sigma_to_budget(sigma, dp_type)


# Keep backward-compatible alias
compute_rho_for_theta = compute_budget_for_theta


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


def calc_confidence(n: int | float, sigma: float, dom: int) -> float:
    dom = max(dom, 1)
    return n / (n + dom * sigma * sigma)


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
    skip_pair_cols: set[Col] | None = None,
) -> CachedMarginals:
    """Batch-query all 1-way and 2-way true marginals in one oracle call.

    Pairs where both columns are in ``skip_pair_cols`` are omitted entirely —
    used to avoid computing hist-hist (evidence-evidence) marginals under MARE,
    since those edges are frozen and never consulted downstream."""
    requests_1 = [[_col_sel(c, attrs)] for c in all_cols]
    pairs = [
        (c1, c2)
        for c1, c2 in itertools.combinations(all_cols, 2)
        if not (skip_pair_cols and c1 in skip_pair_cols and c2 in skip_pair_cols)
    ]
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
    budget_max: float | None = None,
    dp_type: str = "cdp",
    skip_cols: set[Col] | None = None,
    budget_min: float | None = None,
) -> tuple[dict[Col, float], float, float]:
    """Compute per-column DP noise sigma and total budget for 1-way marginals.

    Each column gets noise calibrated so that its confidence achieves theta_1w.
    The actual budget is clamped to [budget_min, budget_max] by binary-searching
    theta_1w: too-high → reduce theta, too-low → raise theta.
    Columns in skip_cols (e.g. hist columns) get sigma=0 and cost no budget.

    Returns (sigma_per_col, total_budget1, effective_theta_1w)."""

    def _total_budget(theta):
        total = 0.0
        for col, mar in cached.one_way.items():
            if skip_cols and col in skip_cols:
                continue
            b = compute_budget_for_theta(mar.size, n, theta, dp_type)
            if b is not None:
                total += b
        return total

    # Check if requested theta_1w fits within budget
    budget1 = _total_budget(theta_1w)
    if budget_max is not None and budget1 > budget_max and budget_max > 0:
        # Binary search for max achievable theta
        lo, hi = 0, theta_1w
        for _ in range(64):
            mid = (lo + hi) / 2
            if _total_budget(mid) <= budget_max:
                lo = mid
            else:
                hi = mid
        theta_1w = lo
        budget1 = _total_budget(theta_1w)
        logger.info(
            f"Adjuvant: theta_1w capped to {theta_1w:.1f} "
            f"(budget1={budget1:.6f}, budget_max={budget_max:.6f})"
        )
    elif budget_min is not None and budget1 < budget_min and budget_min > 0:
        # Binary search for min theta that reaches the floor (bounded above by budget_max)
        lo, hi = theta_1w, max(theta_1w * 2, 1.0)
        # Expand hi until it overshoots budget_min (or budget_max if tighter)
        cap = budget_max if budget_max is not None else budget_min * 10
        while _total_budget(hi) < min(budget_min, cap) and hi < 1e9:
            lo, hi = hi, hi * 2
        for _ in range(64):
            mid = (lo + hi) / 2
            if _total_budget(mid) < budget_min:
                lo = mid
            else:
                hi = mid
        theta_1w = hi
        budget1 = _total_budget(theta_1w)
        if budget_max is not None and budget1 > budget_max:
            # Floor conflicts with cap — cap wins
            lo, hi = 0, theta_1w
            for _ in range(64):
                mid = (lo + hi) / 2
                if _total_budget(mid) <= budget_max:
                    lo = mid
                else:
                    hi = mid
            theta_1w = lo
            budget1 = _total_budget(theta_1w)
        logger.info(
            f"Adjuvant: theta_1w raised to {theta_1w:.1f} to meet floor "
            f"(budget1={budget1:.6f}, budget_min={budget_min:.6f})"
        )

    # Compute per-column sigma_dp (sigma is mechanism-independent, determined by theta)
    sigmas: dict[Col, float] = {}
    for col, mar in cached.one_way.items():
        if skip_cols and col in skip_cols:
            sigmas[col] = 0.0
            continue
        sigmas[col] = _sigma_for_theta(mar.size, n, theta_1w)

    return sigmas, budget1, theta_1w


def add_noise_1way(
    cached: CachedMarginals,
    sigmas: dict[Col, float],
    dp_type: str = "cdp",
    skip_cols: set[Col] | None = None,
) -> dict[Col, np.ndarray]:
    """Add noise to 1-way marginals with per-column sigma.

    Columns in skip_cols are omitted from the result entirely.
    Uses Gaussian (CDP) or Laplace (DP) noise. Returns noisy_marginals."""
    noisy = {}
    for col, mar in cached.one_way.items():
        if skip_cols and col in skip_cols:
            continue
        sigma = sigmas.get(col, 0.0)
        noisy[col] = _add_dp_noise(mar, sigma, dp_type)
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


def compute_mi(
    cached: CachedMarginals,
    attrs: DatasetAttributes,
    all_cols: list[Col],
) -> dict[tuple[Col, Col], np.ndarray]:
    """Compute exact pairwise mutual information I(X;Y) at every height combination.

    Same layout as ``compute_tvd``. Uses log2 for compatibility with PrivBayes
    sensitivity ``sens_mutual_info`` and normalizes with a small ZERO_FILL to
    avoid log(0). Empty joints return zeros."""
    from ....graph.hugin import get_attrs as _get_attrs

    ZERO_FILL = 1e-24

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

    mi: dict[tuple[Col, Col], np.ndarray] = {}

    for (ca, cb), joint_raw in cached.two_way.items():
        val_a, ha_range = col_meta[ca]
        val_b, hb_range = col_meta[cb]
        dom_a0 = val_a.get_domain(0)
        dom_b0 = val_b.get_domain(0)
        joint_00 = joint_raw.reshape(dom_a0, dom_b0).astype(np.float64)
        n_total = joint_00.sum()

        if n_total == 0:
            mi[ca, cb] = np.zeros((ha_range, hb_range))
            mi[cb, ca] = np.zeros((hb_range, ha_range))
            continue

        result = np.zeros((ha_range, hb_range))
        trans_a = col_transitions[ca]
        trans_b = col_transitions[cb]

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
                indep = np.outer(p_a, p_b) + ZERO_FILL
                result[ha, hb] = float(
                    np.sum(p_ab * np.log2((p_ab + ZERO_FILL) / indep))
                )

        mi[ca, cb] = result
        mi[cb, ca] = result.T

    return mi


def sens_mutual_info(n: int) -> float:
    """log2 sensitivity of mutual information for dataset size n (PrivBayes Lemma 3)."""
    return 2 / n * np.log2((n + 1) / 2) + (n - 1) / n * np.log2((n + 1) / (n - 1))


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
    rake: bool = True,
    max_order: int | None = None,
) -> tuple[list[tuple[str, str]], dict[tuple[Col, Col], list[int]]]:
    """Generate edge candidates between non-common value nodes.

    Candidates are grouped by column pair (table, order, attr, value) for tracking.
    Includes same-attribute pairs (different columns), excludes same-column pairs.

    If ``frozen_nodes`` is provided, edges between two frozen nodes are excluded
    (hist-hist edges are blocked as they represent the prior).

    If ``rake`` is True, sequential hist columns (table != None and order != None)
    only connect to endpoints sharing the same attribute name — mirroring the
    PrivBayes rake filter that restricts temporal dependencies to the same column
    across time steps.

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
        a_seq = col_a[0] is not None and col_a[1] is not None
        for col_b in col_names[i + 1 :]:
            b_seq = col_b[0] is not None and col_b[1] is not None
            if rake and (a_seq or b_seq) and col_a[2] != col_b[2]:
                # Sequential hist columns only connect to same-attribute endpoints
                continue
            if max_order is not None and (
                (col_a[1] is not None and col_a[1] >= max_order)
                or (col_b[1] is not None and col_b[1] >= max_order)
            ):
                continue
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


def _factor_domain(
    factor,
    node_data: dict[str, dict],
    attrs: DatasetAttributes,
) -> int:
    """Product of variable domains for the given factor (a set of graph nodes).

    Port of ``_factor_domain_direct`` from ``graph/hugin.py`` operating on our
    ``node_data`` dict — used so the viability check uses the same clique-cost
    metric as hugin's junction-tree builder.  Result equals
    ``get_clique_domain(create_clique_meta(factor, ...), attrs)``."""
    from ....graph.hugin import get_attrs as _get_attrs

    sels: dict[tuple, dict[str, int]] = {}
    for var in factor:
        d = node_data[var]
        key = (d["table"], d["order"], d["attr"])
        val = d["value"]
        height = d["height"]
        s = sels.get(key)
        if s is None:
            sels[key] = {val: height}
        else:
            cur = s.get(val)
            if cur is None or height < cur:
                s[val] = height

    dom = 1
    for (table, order, attr_name), sel in sels.items():
        attr = _get_attrs(attrs, table, order)[attr_name]
        cmn = attr.common
        if len(sel) == 1 and cmn and cmn.name in sel:
            dom *= cmn.get_domain(sel[cmn.name])
        else:
            cmn_name = cmn.name if cmn else None
            heights: list[int] = []
            vals: list[CatValue] = []
            for v, h in sel.items():
                if v == cmn_name:
                    continue
                heights.append(h)
                vals.append(cast(CatValue, attr[v]))
            if vals:
                dom *= CatValue.get_domain_multiple(heights, vals)
    return dom


def _hugin_eliminate(
    adj: dict[str, set[str]],
    cost_map: dict[str, int],
    node_data: dict[str, dict],
    attrs: DatasetAttributes,
    max_clique_size: float,
) -> tuple[list[frozenset[str]], bool]:
    """Run min-factor-domain greedy elimination on ``adj`` (mutated) and
    collect the maximal cliques of the resulting chordal graph.

    Same algorithm as ``elimination_order_greedy`` in ``graph/hugin.py``: at
    each step picks the remaining node whose factor (= node + remaining
    neighbors) has the smallest domain, fills in edges between its
    neighbors, and recomputes costs only for the affected neighbors.

    Mirrors the evidence-aware policy in ``elimination_order_greedy``:
    evidence (hist) nodes whose elimination would *marginalize them over
    ≥ 2 main-table neighbours* are processed before main-table nodes.
    Evidence with 0–1 main neighbours is "passthrough" and falls through
    to the regular min-degree heuristic — prioritizing it adds no useful
    main–main fill-in.  Keeping this estimator in lockstep with the JT
    builder is necessary so structure_learn's ``max_clique_size`` cap
    accepts/rejects exactly the edges the actual JT will accept/reject.

    Early-rejects (returns ``(_, False)``) the moment any factor's domain
    exceeds ``max_clique_size`` — sound because every clique of the
    resulting triangulated graph is a subset of some step's factor, so an
    overflow factor proves at least one clique is over the limit.

    ``cost_map`` is consumed (mutated); pass a fresh copy if you need to
    keep the original."""
    remaining = set(adj)
    cliques: list[frozenset[str]] = []

    def _is_marginalize_out(node):
        if node_data[node].get("table") is None:
            return False
        n_main = 0
        for nb in adj[node]:
            if nb in remaining and node_data[nb].get("table") is None:
                n_main += 1
                if n_main >= 2:
                    return True
        return False

    while remaining:
        candidates = [v for v in remaining if _is_marginalize_out(v)] or remaining
        v = min(candidates, key=cost_map.__getitem__)
        factor_dom = cost_map[v]

        if factor_dom > max_clique_size:
            return cliques, False

        neighbors = adj[v] & remaining
        factor = frozenset(neighbors | {v})

        nb_list = list(neighbors)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                u, w = nb_list[i], nb_list[j]
                if w not in adj[u]:
                    adj[u].add(w)
                    adj[w].add(u)

        remaining.discard(v)
        del cost_map[v]

        # Hugin's incremental update: only v's surviving neighbors had their
        # induced factor change (v left, plus new fills among them).
        for nb in neighbors:
            if nb in cost_map:
                cost_map[nb] = _factor_domain(
                    (adj[nb] & remaining) | {nb}, node_data, attrs
                )

        is_maximal = True
        for c in cliques:
            if factor <= c:
                is_maximal = False
                break
        if is_maximal:
            cliques.append(factor)

    return cliques, True


def _triangulate_base(
    base_adj: dict[str, set[str]],
    node_data: dict[str, dict],
    attrs: DatasetAttributes,
) -> tuple[list[frozenset[str]], dict[str, list[int]], dict[str, int]]:
    """Triangulate ``base_adj`` with hugin's min-factor-domain elimination.

    Returns:
      - cliques: maximal cliques of the triangulated graph.
      - cliques_by_node[v]: indices into ``cliques`` containing v.
      - base_cost_map[v]: initial factor domain for v in base_adj.  Reused
        by ``_score_with_one_edge`` so each candidate only has to update
        the two endpoints' costs instead of recomputing all V costs.
    """
    adj = {v: s.copy() for v, s in base_adj.items()}

    base_cost_map: dict[str, int] = {}
    for v in adj:
        base_cost_map[v] = _factor_domain(adj[v] | {v}, node_data, attrs)

    cliques, _ok = _hugin_eliminate(
        adj, dict(base_cost_map), node_data, attrs, float("inf")
    )

    cliques_by_node: dict[str, list[int]] = {v: [] for v in base_adj}
    for idx, c in enumerate(cliques):
        for v in c:
            cliques_by_node[v].append(idx)
    return cliques, cliques_by_node, base_cost_map


def _score_with_one_edge(
    base_adj: dict[str, set[str]],
    na: str,
    nb: str,
    node_data: dict[str, dict],
    attrs: DatasetAttributes,
    max_clique_size: float,
    base_cliques: list[frozenset[str]] | None = None,
    cliques_by_node: dict[str, list[int]] | None = None,
    base_cost_map: dict[str, int] | None = None,
) -> tuple[float, bool]:
    """Decide whether adding edge (na, nb) keeps every clique ≤ max_clique_size.

    Exact under hugin's min-factor-domain triangulation — no heuristic
    shortcuts:
      - Fast path: if ``na`` and ``nb`` already share a base clique, the
        edge is already present in the triangulated base graph, so the
        new triangulation equals the old.  Provably no clique grows.
      - Slow path: fresh full hugin elimination on ``base_adj ∪ {(na, nb)}``
        with early rejection on first oversized factor.

    The only reuse across candidates is the initial cost map — adding one
    edge changes only ``na`` and ``nb``'s induced factors, so all other
    nodes start with the same cost as in the base triangulation.

    Returns (0, valid). The first element is unused by the caller."""
    # Fast path: edge internal to an existing base clique → no new clique.
    if cliques_by_node is not None and base_cliques is not None:
        ca = cliques_by_node.get(na, [])
        cb = cliques_by_node.get(nb, [])
        if ca and cb:
            sa = set(ca)
            for i in cb:
                if i in sa:
                    return 0.0, True

    # Slow path: fresh hugin elimination on base_adj + (na, nb).
    adj = {v: s.copy() for v, s in base_adj.items()}
    adj[na].add(nb)
    adj[nb].add(na)

    if base_cost_map is not None:
        cost_map = dict(base_cost_map)
    else:
        cost_map = {v: _factor_domain(adj[v] | {v}, node_data, attrs) for v in adj}
    # Only na and nb gained a neighbor relative to base.
    cost_map[na] = _factor_domain(adj[na] | {na}, node_data, attrs)
    cost_map[nb] = _factor_domain(adj[nb] | {nb}, node_data, attrs)

    _, valid = _hugin_eliminate(adj, cost_map, node_data, attrs, max_clique_size)
    return 0.0, valid


def _fmt_node(node: str, g, attrs) -> str:
    """Format a graph node as 'attr.val[h] (dom=X)'."""
    from ....graph.hugin import get_attrs as _get_attrs

    d = g.nodes[node]
    val = cast(
        CatValue, _get_attrs(attrs, d["table"], d["order"])[d["attr"]][d["value"]]
    )
    dom = val.get_domain(d["height"])
    return f"{d['attr'] + '.' if d['attr'] != d['value'] else ''}{d['value']}[{d['height']}] (dom={dom})"

def _fmt_attr(s: str | tuple[str, int]) -> str:
    """Format an attribute name, which may be a string or a (name, order) tuple."""
    if isinstance(s, str):
        return s
    else:
        return f"{s[0]}[{s[1]}]"

def _fmt_node(d: dict) -> str:
    """Format a graph node as '[table.]attr.val[h]', with table prefix for evidence vars."""
    prefix = ""
    if d.get("table") is not None:
        table = d["table"]
        order = d.get("order")
        if order is not None:
            prefix = f"{table}[-{1 + order}]."
        else:
            prefix = f"{table}."
    if d["attr"] != d["value"]:
        return f"{prefix}{_fmt_attr(d["attr"])}.{d['value'].replace(_fmt_attr(d["attr"]) + '_', '')}[{d['height']}]"
    return f"{prefix}{d['value']}[{d['height']}]"


def _fmt_edge(na: str, nb: str, g, attrs) -> str:
    """Format a graph edge as 'attr.val[h] x attr.val[h] (domA, domB)'."""
    from ....graph.hugin import get_attrs as _get_attrs

    def _info(node):
        d = g.nodes[node]
        val = cast(
            CatValue, _get_attrs(attrs, d["table"], d["order"])[d["attr"]][d["value"]]
        )
        dom = val.get_domain(d["height"])
        return _fmt_node(d), dom

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

    return 1 / (1 + np.log2(_dom_for_node(node_a) * _dom_for_node(node_b)) * size_penalty)


# ============================================================
# Step 2d: Structure learning main loop
# ============================================================
def _edge_clique_domain(
    na: str,
    nb: str,
    graph: nx.DiGraph,
    attrs: DatasetAttributes,
) -> int:
    """Compute the clique domain for an edge using merged AttrMeta.

    Merges nodes from the same attribute into a single AttrMeta with a
    combined selector, matching the domain calculation in measure_edges."""
    from collections import defaultdict
    from ....graph.hugin import AttrMeta, get_clique_domain, get_attrs as _ga

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
        attr = _ga(attrs, table, order)[attr_name]
        if (
            len(sel_dict) == 1
            and attr.common
            and next(iter(sel_dict)) == attr.common.name
        ):
            new_sel: int | tuple = sel_dict[attr.common.name]
        else:
            cmn = attr.common.name if attr.common else None
            val_order = {vn: i for i, vn in enumerate(attr.vals)}
            new_sel = tuple(
                sorted(
                    ((v, h) for v, h in sel_dict.items() if v != cmn),
                    key=lambda x: val_order.get(x[0], 0),
                )
            )
        source.append(AttrMeta(table, order, attr_name, new_sel))
    source_tuple = tuple(sorted(source, key=_attr_meta_sort_key))
    return get_clique_domain(source_tuple, attrs)


def _compute_cand_edge_budget(
    idx: int,
    candidates: list[tuple[str, str]],
    directed_graph: nx.DiGraph,
    attrs: DatasetAttributes,
    n: int | float,
    theta_2w: float,
    dp_type: str = "cdp",
) -> float:
    """Compute the measurement budget cost for candidate edge at index idx."""
    na, nb = candidates[idx]
    dom = _edge_clique_domain(na, nb, directed_graph, attrs)
    return compute_budget_for_theta(dom, n, theta_2w, dp_type)


def structure_learn(
    directed_graph: nx.DiGraph,
    attrs: DatasetAttributes,
    tvd: dict[tuple[Col, Col], np.ndarray],
    n: int,
    size_penalty: float,
    rho_avail: float,
    min_score: float,
    em_z: float,
    theta_2w: float,
    frozen_nodes: set[str] | None = None,
    n_hist_cols: int = 0,
    max_clique_size: float = 1e5,
    max_em_budget: float = float("inf"),
    min_em_budget: float = 0.0,
    em_max: float = 50.0,
    rake: bool = True,
    max_order: int | None = None,
    dp_type: str = "cdp",
    scoring: str = "tvd",
    min_safety_factor: float = 3.0,
) -> tuple[nx.Graph, set[frozenset[str]], float]:
    """Greedy edge addition with exponential mechanism and budget tracking.

    Each EM step costs budget derived from em_z.  Each selected edge
    commits measurement budget derived from theta_2w and the edge's domain.
    The loop exits when the remaining budget cannot cover the next step.

    Budget is rho (CDP) or epsilon (DP) depending on dp_type.

    Returns:
        moral: Undirected moralized graph with structure-learning edges added.
        structure_edges: Set of frozenset node pairs for structure-learning edges.
        budget_remaining: Unspent budget (for measurement + leftover).
    """
    from ....graph.hugin import to_moral, get_factor_domain
    from ....utils.progress import piter, check_exit

    # Moralize the directed height-chain graph -> undirected base
    moral = to_moral(directed_graph)

    # Generate candidates and group by column pair
    candidates, col_pair_map = generate_candidates(
        directed_graph, frozen_nodes, rake=rake, max_order=max_order
    )
    connected_pairs: set[tuple[Col, Col]] = set()
    structure_edges: set[frozenset[str]] = set()

    # EM sensitivity depends on the score function. The TVD score here is
    # ½|P(X,Y) − P(X)P(Y)|₁: one row perturbs P(X,Y) by ≤2/n and P(X)P(Y)
    # by ≤4/n in L1, so sens ≤ 3/n. MI uses PrivBayes Lemma 3 (log2).
    if scoring == "mi":
        sensitivity = float(sens_mutual_info(n))
    else:
        sensitivity = 3.0 / n

    # Pre-compute per-candidate measurement budget and theta-filter
    cand_bdg_edge = np.zeros(len(candidates))
    cand_valid = np.ones(len(candidates), dtype=bool)
    if rho_avail > 0:
        n_filtered = 0
        for idx in range(len(candidates)):
            b = _compute_cand_edge_budget(
                idx,
                candidates,
                directed_graph,
                attrs,
                n,
                theta_2w,
                dp_type,
            )
            cand_bdg_edge[idx] = b
            if np.isinf(b):
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
    bdg_em = 0.0  # cumulative EM selection budget spent
    bdg_committed = 0.0  # cumulative edge measurement budget committed

    def _em_cost(n_cands: int) -> tuple[float, float]:
        """Compute (eps, budget_cost) for EM over n_cands candidates.

        eps = em_z * 4 / n_cands; budget cost is rho=eps²/2 (CDP) or eps (DP).
        If max_em_budget is finite, eps is clamped so the cost does not exceed it."""
        if em_z <= 0 or rho_avail <= 0 or n_cands <= 0:
            return 0.0, 0.0, 0.0
        eps = em_z * 4.0 / n_cands
        cost = _em_budget_cost(eps, dp_type)
        if cost > max_em_budget:
            # Clamp eps so cost = max_em_budget
            if dp_type == "cdp":
                eps_clamped = (2.0 * max_em_budget) ** 0.5
            else:
                eps_clamped = max_em_budget
            em_z_eff = eps_clamped * n_cands / 4.0
            eps = eps_clamped
            cost = _em_budget_cost(eps, dp_type)
        elif cost < min_em_budget and min_em_budget > 0:
            # Raise eps so cost = min_em_budget (spend more per round, sharper EM)
            if dp_type == "cdp":
                eps_clamped = (2.0 * min_em_budget) ** 0.5
            else:
                eps_clamped = min_em_budget
            em_z_eff = eps_clamped * n_cands / 4.0
            # Cap em_z_eff so a low-candidate round doesn't drive EM arbitrarily sharp
            if em_max > 0 and em_z_eff > em_max:
                em_z_eff = em_max
                eps_clamped = em_max * 4.0 / n_cands
            eps = eps_clamped
            cost = _em_budget_cost(eps, dp_type)
        else:
            em_z_eff = em_z
        return eps, cost, em_z_eff

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
    # Cached base triangulation — (re)built lazily each iteration after edges
    # get accepted.  Lets the post-EM clique check short-circuit when the
    # picked edge is already internal to an existing clique, and reuses the
    # per-node initial factor-domain cost map so each check only updates
    # the two endpoints' costs instead of re-initializing all V costs.
    base_cliques: list[frozenset[str]] | None = None
    cliques_by_node: dict[str, list[int]] | None = None
    base_cost_map: dict[str, int] | None = None
    # Post-EM clique rejection mask — set when the inner loop tests a
    # candidate and finds it would create an oversized clique.  Kept
    # separate from ``cand_valid`` (which is *data-independent* exclusion:
    # theta-filter, saturated, connected) because the set of clique-rejected
    # candidates depends on the data-dependent EM draws inside past inner
    # loops.  Mixing the two would make ``_em_cost``'s ``n_cands`` and
    # therefore ``eps_step`` data-dependent and break composition.
    cand_invalid_clique = np.zeros(len(candidates), dtype=bool)

    for it in range(max_steps):
        try:
            check_exit()
        except Exception as e:
            pbar.close()
            raise e

        # Filter to active candidates using *only* data-independent state
        # (cand_valid covers the theta filter; saturated/connected are
        # derived from past accepted edges, which are public).  This count
        # drives _em_cost below and must not depend on any data-dependent
        # past draws.
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

        # --- EM cost + affordability filter (iterate until stable) ---
        # Filtering candidates raises EM cost (fewer candidates → larger eps),
        # which may make more candidates unaffordable.  Loop until the
        # affordable set and EM cost are consistent.
        #
        # Both inputs to this fixed-point are data-independent (active is
        # filtered only by public state; cand_bdg_edge depends on attrs/n
        # only), so eps_step and bdg_em_step are themselves data-independent.
        if rho_avail > 0:
            affordable = list(active)
            while True:
                eps_step, bdg_em_step, _ = _em_cost(len(affordable))
                bdg_after_em = rho_avail - bdg_em - bdg_committed - bdg_em_step
                if bdg_after_em < 0:
                    affordable = []
                    break
                new_affordable = [
                    (idx, na, nb)
                    for idx, na, nb in affordable
                    if cand_bdg_edge[idx] <= bdg_after_em
                ]
                if len(new_affordable) == len(affordable):
                    break  # stable
                affordable = new_affordable
            eps_step, bdg_em_step, em_z_eff = _em_cost(len(affordable))
        else:
            eps_step = bdg_em_step = em_z_eff = 0
            affordable = list(active)

        if not affordable:
            logger.info(
                f"Adjuvant: exit (no affordable candidates) at iter {it}. "
                f"active={len(active)}, budget_avail={rho_avail:.6f}, "
                f"bdg_em={bdg_em:.6f}, bdg_committed={bdg_committed:.6f}"
            )
            break

        # Apply the data-dependent clique-rejection mask only to the EM pool.
        # eps_step / bdg_em_step were already locked in above on the
        # data-independent ``affordable`` set, so excluding past-rejected
        # candidates here doesn't feed back into ε allocation.
        em_pool = [c for c in affordable if not cand_invalid_clique[c[0]]]
        if not em_pool:
            logger.info(
                f"Adjuvant: exit (no candidates left after clique rejections) "
                f"at iter {it}."
            )
            break

        # Build the cached base triangulation up-front; reused across all
        # rejection retries below (the graph only changes when we accept).
        if base_cliques is None:
            base_cliques, cliques_by_node, base_cost_map = _triangulate_base(
                base_adj, node_data, attrs
            )

        # Charge EM budget once for this selection event.  Each rejection
        # below re-runs EM with the same eps_step on a shrinking pool, but
        # because the validity predicate is data-independent the joint
        # output distribution equals EM applied directly to the valid set
        # (rejection sampling on a fixed predicate), which is ε_step-DP.
        if rho_avail > 0:
            assert eps_step
            log_n_boost = min_safety_factor * 2 * sensitivity / eps_step
        else:
            log_n_boost = 0
        bdg_em += bdg_em_step

        n_invalid_this_iter = 0
        accepted = False
        stopped = False
        while em_pool:
            scores = np.array([cand_tvd_boost[idx] for idx, _, _ in em_pool])
            stop_idx = len(scores)

            if rho_avail > 0:
                em_scores = np.append(
                    scores, min_score + log_n_boost if min_score else 0
                )
                sel = exponential_mechanism(em_scores, eps_step, sensitivity)
            else:
                em_scores = np.append(scores, min_score)
                sel = int(np.argmax(em_scores))

            if sel == stop_idx:
                stopped = True
                break

            cand_idx, na, nb = em_pool[sel]
            _, valid = _score_with_one_edge(
                base_adj,
                na,
                nb,
                node_data,
                attrs,
                max_clique_size,
                base_cliques=base_cliques,
                cliques_by_node=cliques_by_node,
                base_cost_map=base_cost_map,
            )
            if valid:
                accepted = True
                break
            # Invalid: mark in the data-dependent rejection mask.  This is
            # used only to shrink the EM pool — never to size eps_step or
            # bdg_em_step, both of which are computed from data-independent
            # state above.
            cand_invalid_clique[cand_idx] = True
            em_pool.pop(sel)
            n_invalid_this_iter += 1

        if stopped:
            logger.info(
                f"Adjuvant: exit (EM picked stop option, min_score={min_score}) "
                f"at iter {it}, edges={len(structure_edges)}, "
                f"rejected={n_invalid_this_iter}"
            )
            pbar.update(1)
            break

        if not accepted:
            # All affordable candidates failed the clique check this iter.
            # No point continuing — the graph hasn't changed, so the same
            # candidates would fail again next iter at extra budget cost.
            logger.info(
                f"Adjuvant: exit (no valid candidate among affordable) at iter {it}. "
                f"rejected={n_invalid_this_iter}"
            )
            break

        edge_bdg = cand_bdg_edge[cand_idx]

        # Accept edge
        moral.add_edge(na, nb, structure=True)
        base_adj[na].add(nb)
        base_adj[nb].add(na)
        # base_adj changed — invalidate the triangulation cache
        base_cliques = None
        cliques_by_node = None
        base_cost_map = None
        structure_edges.add(frozenset([na, nb]))
        bdg_committed += edge_bdg

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
                f", budget={rho_avail - bdg_em - bdg_committed:.6f}, em_z={em_z_eff:.2f}"
                if rho_avail > 0
                else ""
            )
            + (
                f", rejected={n_invalid_this_iter}"
                if n_invalid_this_iter
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
    bdg_remaining = rho_avail - bdg_em - bdg_committed

    bdg_label = "rho" if dp_type == "cdp" else "eps"
    logger.info(
        f"Adjuvant: structure learning done, "
        + f"{len(structure_edges)} edges, "
        + f"{len(connected_pairs)} column pairs connected, "
        + (
            f"{bdg_label}_em={bdg_em:.6f}, {bdg_label}_measure={bdg_committed:.6f}, "
            + f"{bdg_label}_remaining={bdg_remaining:.6f}"
            if rho_avail > 0
            else "no budget tracking"
        )
    )

    diag = format_tvd_diagnostic(
        tvd,
        structure_edges,
        connected_pairs,
        col_pair_map,
        directed_graph,
        moral,
        attrs,
        min_score,
        label=scoring.upper(),
    )
    for line in diag.splitlines():
        logger.info(line)

    return moral, structure_edges, bdg_remaining, diag


def format_tvd_diagnostic(
    tvd: dict[tuple[Col, Col], np.ndarray],
    structure_edges: set[frozenset[str]],
    connected_pairs: set[tuple[Col, Col]],
    col_pair_map: dict,
    directed_graph: "nx.DiGraph",
    moral: "nx.Graph",
    attrs: DatasetAttributes,
    min_tvd: float,
    label: str = "TVD",
) -> str:
    """Format score diagnostic showing connected and missing column pairs."""
    candidate_cols = set(c for pair in col_pair_map for c in pair)
    all_pairs_tvd: list[tuple[float, Col, Col]] = []
    for (ca, cb), val_arr in tvd.items():
        if (
            _col_sort_key(ca) < _col_sort_key(cb)
            and ca in candidate_cols
            and cb in candidate_cols
        ):
            all_pairs_tvd.append((float(val_arr[0, 0]), ca, cb))
    all_pairs_tvd.sort(key=lambda x: (-x[0], _col_sort_key(x[1]), _col_sort_key(x[2])))

    lines = [f"Connected column pairs (by {label}):"]
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
                        f"  CONNECTED {label}={tvd_at_h:.4f}{f'/{val:.4f}' if ha != 0 or hb != 0 else ''} "
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
                f"    MISSING {label}={val:.4f} "
                f"{_fmt_attr(ca_attr) + '.' if ca_attr != ca_val else ''}{ca_val} x "
                f"{_fmt_attr(cb_attr) + '.' if cb_attr != cb_val else ''}{cb_val}"
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
    dp_type: str = "cdp",
    tvd_diag: str = "",
) -> str:
    """Format a summary string for an Adjuvant model."""
    bdg_label = "rho" if dp_type == "cdp" else "eps"
    s = f"Adjuvant Graphical Model ({dp_type.upper()}):\n"
    s += (
        f"({bdg_label}={rho:.6f}, {bdg_label}_remaining={rho_remaining:.6f}, "
        f"theta_1w={theta_1w:.1f}, theta_2w={theta_2w:.1f}, em_z={em_z:.1f}, "
        f"{n_obs} observations)\n"
    )

    if tvd_diag:
        s += tvd_diag + "\n"
    else:
        n_edges = sum(1 for _, _, d in moral.edges(data=True) if d.get("structure"))
        s += f"{n_edges} structure-learning edges (no TVD diagnostic available).\n"

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
    dp_type: str = "cdp",
) -> tuple[list, float]:
    """Measure selected clique marginals with DP noise.

    Returns (list of LinearObservation, sigma3)."""
    from ....graph.hugin import get_clique_domain, get_attrs as _get_attrs
    from ....graph.loss import LinearObservation
    from ....graph.beliefs import convert_sel

    K = len(cliques_to_measure)
    if K == 0:
        return [], 0.0

    sigma3 = 0.0 if rho3 <= 0 else _budget_to_sigma(rho3 / K, dp_type)

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
            raw = _add_dp_noise(raw, sigma3, dp_type)
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
        confidence = calc_confidence(n, sigma3, dom)
        obs_list.append(LinearObservation(clique, None, prob, confidence))

    return obs_list, sigma3


def measure_edges(
    oracle: MarginalOracle,
    structure_edges: set[frozenset[str]],
    graph: nx.Graph,
    attrs: DatasetAttributes,
    n: int,
    theta_2w: float,
    rho_extra: float = 0.0,
    dp_type: str = "cdp",
    no_noise: bool = False,
) -> tuple[list, float]:
    """Measure 2-way marginals with per-edge noise calibrated to theta_2w.

    Each edge gets its own sigma_dp derived from its domain and theta_2w.
    If rho_extra > 0, the effective theta is raised (via binary search)
    to exhaust the leftover budget, improving all edge measurements.
    If no_noise is True, all sigmas are forced to 0 (no DP budget).

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
                # Order values by attr.vals insertion order (not alphabetical)
                # so that the sel matches the ordering used by
                # get_mapping_multiple and calc_marginal consistently.
                val_order = {vn: i for i, vn in enumerate(attr.vals)}
                new_sel = tuple(
                    sorted(
                        ((v, h) for v, h in sel_dict.items() if v != cmn),
                        key=lambda x: val_order.get(x[0], 0),
                    )
                )
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

        def _total_bdg_2w(theta):
            return sum(
                compute_budget_for_theta(dom, n, theta, dp_type) for dom in edge_doms
            )

        base_bdg = _total_bdg_2w(theta_2w)
        target_bdg = base_bdg + rho_extra
        hi = theta_2w * 1000
        lo = 0
        # Binary search for max theta that fits within target budget
        if _total_bdg_2w(hi) <= target_bdg:
            eff_theta = hi
        else:
            for _ in range(64):
                mid = (lo + hi) / 2
                if _total_bdg_2w(mid) <= target_bdg:
                    lo = mid
                else:
                    hi = mid
            eff_theta = lo
        if eff_theta > theta_2w:
            logger.info(
                f"Adjuvant: boosted theta_2w {theta_2w:.1f} -> {eff_theta:.1f} "
                f"(budget_extra={rho_extra:.6f})"
            )

    # Compute per-edge sigma from effective theta (sigma is mechanism-independent)
    edge_sigmas: list[float] = []
    if no_noise:
        edge_sigmas = [0.0] * K
    else:
        for dom in edge_doms:
            edge_sigmas.append(_sigma_for_theta(dom, n, eff_theta))

    obs_list = []
    max_sigma = 0.0
    for source_tuple, result, sigma_edge in zip(edge_metas, results, edge_sigmas):
        max_sigma = max(max_sigma, sigma_edge)

        # Oracle returns data in packed shape.  For multi-value selectors
        # with common overlap c the per-attr dim is (d1-c)*(d2-c)+c,
        # otherwise it equals the product of individual value domains.
        from ....marginal.numpy import _calc_common

        oracle_dims = []
        attr_commons = []  # per-attr common count (0 for int/single-val)
        for tbl, ord_, attr_name, sel in source_tuple:
            a = _get_attrs(attrs, tbl, ord_)[attr_name]
            sel_d = convert_sel(sel)
            if isinstance(sel_d, int):
                oracle_dims.append(a.common.get_domain(sel_d))
                attr_commons.append(0)
            elif len(sel_d) == 1:
                vn, h = next(iter(sel_d.items()))
                oracle_dims.append(cast(CatValue, a.vals[vn]).get_domain(h))
                attr_commons.append(0)
            else:
                cmn = min(
                    _calc_common(cast(CatValue, a.vals[vn]), a.common) for vn in sel_d
                )
                nd = 1
                for vn, h in sel_d.items():
                    nd *= cast(CatValue, a.vals[vn]).get_domain(h) - cmn
                nd += cmn
                oracle_dims.append(nd)
                attr_commons.append(cmn)

        raw = result.astype(np.float64).ravel()
        if sigma_edge > 0:
            raw = _add_dp_noise(raw, sigma_edge, dp_type)
        raw = raw.clip(0)
        prob = raw.reshape(oracle_dims)

        # Compress oracle→compressed for each multi-value dim with common.
        # Single-value dims and dims with common=0 have oracle == compressed.
        for dim_i, (tbl, ord_, attr_name, sel) in enumerate(source_tuple):
            sel_d = convert_sel(sel)
            if isinstance(sel_d, int) or len(sel_d) <= 1:
                continue
            cmn = attr_commons[dim_i]
            a = _get_attrs(attrs, tbl, ord_)[attr_name]
            oracle_dom = prob.shape[dim_i]
            compressed_dom = a.get_domain(sel_d)
            if oracle_dom == compressed_dom:
                continue

            if cmn == 0:
                # No common: oracle == naive, use standard naive→compressed
                raw_naive = a.get_naive_mapping(sel_d)
                raw_compressed = a.get_mapping(sel_d)
                _, unique_idx = np.unique(raw_naive, return_index=True)
                naive_idx = raw_naive[unique_idx]
                compressed_idx = raw_compressed[unique_idx]

                i_map = tuple(
                    naive_idx if j == dim_i else slice(None) for j in range(prob.ndim)
                )
                o_map = tuple(
                    compressed_idx if j == dim_i else slice(None)
                    for j in range(prob.ndim)
                )
                tmp = np.zeros(
                    [
                        compressed_dom if j == dim_i else d
                        for j, d in enumerate(prob.shape)
                    ],
                    dtype=prob.dtype,
                )
                np.add.at(tmp, o_map, prob[i_map])
                prob = tmp
            else:
                # Common > 0: oracle uses packed encoding.
                # Build oracle_bin → compressed_bin mapping by iterating
                # over all per-value group combinations.
                val_items = list(sel_d.items())
                val_doms = [
                    cast(CatValue, a.vals[vn]).get_domain(h) for vn, h in val_items
                ]
                raw_total = int(np.prod(val_doms))

                # Per-value indices for every raw combination
                per_val = []
                rem = np.arange(raw_total, dtype=np.int64)
                for d in reversed(val_doms):
                    per_val.append(rem % d)
                    rem //= d
                per_val.reverse()

                # Oracle flat index for each raw combination
                oracle_bins = np.zeros(raw_total, dtype=np.int64)
                o_stride = 1
                for vi_rev in range(len(val_items)):
                    vi = len(val_items) - 1 - vi_rev
                    gv = per_val[vi]
                    if cmn == 0 or vi_rev == 0:
                        oracle_bins += gv * o_stride
                    else:
                        oracle_bins += np.maximum(0, gv - cmn) * o_stride
                    o_stride *= val_doms[vi] - cmn

                comp_mapping = np.array(a.get_mapping(sel_d), dtype=np.int64)
                o2c = np.zeros(oracle_dom, dtype=np.int64)
                o2c[oracle_bins] = comp_mapping

                o_map = tuple(
                    o2c if j == dim_i else slice(None) for j in range(prob.ndim)
                )
                tmp = np.zeros(
                    [
                        compressed_dom if j == dim_i else d
                        for j, d in enumerate(prob.shape)
                    ],
                    dtype=prob.dtype,
                )
                np.add.at(tmp, o_map, prob)
                prob = tmp

        s = prob.sum()
        prob = (prob / s if s > 0 else prob).astype(np.float32)

        dom = get_clique_domain(source_tuple, attrs)
        confidence = calc_confidence(n, sigma_edge, dom)
        obs_list.append(LinearObservation(source_tuple, None, prob, confidence))

    return obs_list, max_sigma


def build_1way_observations(
    noisy_1way: dict[Col, np.ndarray],
    attrs: DatasetAttributes,
    n: int,
    sigmas: dict[Col, float],
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
        confidence = calc_confidence(n, sigma_col, dom)
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
    e_w1_max_ratio: float = 0.8,
    e_w1_min_ratio: float = 0.0,
    e_em_max_ratio: float | None = None,
    e_em_min_ratio: float | None = None,
    em_max: float = 50.0,
    size_penalty: float = 0.0,
    min_tvd: float = 0.05,
    min_mi: float = 0.0,
    min_safety_factor: float = 3.0,
    frozen_nodes: set[str] | None = None,
    n_hist_cols: int = 0,
    max_clique_size: float = 1e5,
    rescale: bool = True,
    rake: bool = True,
    max_order: int | None = None,
    dp_type: str = "cdp",
    scoring: str = "tvd",
) -> tuple[list, "nx.Graph", float]:
    """Run the full Adjuvant pipeline: marginals, noise, structure learn, measure.

    Budget parameter `rho` is rho (CDP) or epsilon (DP) depending on dp_type.

    Returns (all_obs, moral, budget_remaining) where all_obs is a list of
    LinearObservation, moral is the moralized graph with structure-learning
    edges, and budget_remaining is the unspent budget."""

    bdg_label = "rho" if dp_type == "cdp" else "eps"
    all_cols = get_col_names(attrs)
    hist_cols = get_hist_cols(all_cols)
    d = len(all_cols)
    h = n_hist_cols or len(hist_cols)

    # Step 0: Compute all 1-way and 2-way marginals
    logger.info(
        f"Adjuvant Step 0: Computing marginals ({d} cols, {h} hist, {n} rows, {dp_type})"
    )
    cached = compute_all_marginals(
        oracle, attrs, all_cols, skip_pair_cols=hist_cols if hist_cols else None
    )

    # Step 1: Noisy 1-way marginals (budget from theta_1w)
    # Skip hist columns — they are provided as evidence, not generated
    # When rho=0 (no DP), skip noise entirely — theta-based sigma would
    # add spurious noise independent of the privacy budget.
    if rho > 0:
        bdg1_max = e_w1_max_ratio * rho
        bdg1_min = e_w1_min_ratio * rho
        sigmas_1w, bdg1, eff_theta_1w = compute_1way_budget(
            cached,
            n,
            theta_1w,
            bdg1_max,
            dp_type,
            skip_cols=hist_cols if hist_cols else None,
            budget_min=bdg1_min,
        )
        noisy_1way = add_noise_1way(
            cached, sigmas_1w, dp_type, skip_cols=hist_cols if hist_cols else None
        )
    else:
        bdg1 = 0.0
        eff_theta_1w = theta_1w
        sigmas_1w = {col: 0.0 for col in cached.one_way}
        noisy_1way = {
            col: mar.copy()
            for col, mar in cached.one_way.items()
            if not (hist_cols and col in hist_cols)
        }
    logger.info(
        f"Adjuvant Step 1: Noisy 1-way marginals "
        f"(theta_1w={eff_theta_1w:.1f}, {bdg_label}1={bdg1:.6f})"
    )

    # Step 2: Structure learning (remaining budget)
    bdg_avail = rho - bdg1
    assert (
        bdg_avail >= 0
    ), f"Available budget went negative: {bdg_avail}, 0 for disable and positive for enabled"
    logger.info(
        f"Adjuvant Step 2: Structure learning "
        f"({bdg_label}_avail={bdg_avail:.6f}, em_z={em_z}, theta_2w={theta_2w})"
    )
    if scoring == "mi":
        scores = compute_mi(cached, attrs, all_cols)
        min_score = min_mi
    else:
        scores = compute_tvd(cached, attrs, all_cols)
        min_score = min_tvd
    directed_graph = build_height_chain_graph(attrs)
    logger.info(
        f"Adjuvant: height-chain graph has {directed_graph.number_of_nodes()} "
        f"nodes, {directed_graph.number_of_edges()} chain edges "
        f"(scoring={scoring}, min_score={min_score})"
    )

    max_em_budget = e_em_max_ratio * rho if rho > 0 and e_em_max_ratio else float("inf")
    min_em_budget = e_em_min_ratio * rho if rho > 0 and e_em_min_ratio else 0.0
    moral, structure_edges, bdg_remaining, tvd_diag = structure_learn(
        directed_graph,
        attrs,
        scores,
        n,
        size_penalty,
        bdg_avail,
        min_score,
        em_z=em_z,
        theta_2w=theta_2w,
        frozen_nodes=frozen_nodes,
        n_hist_cols=h,
        max_clique_size=max_clique_size,
        rake=rake,
        max_order=max_order,
        max_em_budget=max_em_budget,
        min_em_budget=min_em_budget,
        em_max=em_max,
        dp_type=dp_type,
        scoring=scoring,
        min_safety_factor=min_safety_factor,
    )

    # Step 3: Measure edge marginals (per-edge sigma from theta_2w)
    # When rho=0, measure without noise (theta_2w would add spurious noise).
    logger.info(
        f"Adjuvant Step 3: Measuring {len(structure_edges)} edge marginals "
        f"(theta_2w={theta_2w}, {bdg_label}_remaining={bdg_remaining:.6f})"
    )
    edge_obs, max_sigma = measure_edges(
        oracle,
        structure_edges,
        moral,
        attrs,
        n,
        theta_2w,
        rho_extra=bdg_remaining if rescale else 0.0,
        dp_type=dp_type,
        no_noise=rho <= 0,
    )
    oneway_obs = build_1way_observations(noisy_1way, attrs, n, sigmas_1w)
    all_obs = edge_obs + oneway_obs

    max_conf = max((o.confidence for o in all_obs), default=0.0)
    if max_conf > 0:
        all_obs = [o._replace(confidence=o.confidence / max_conf) for o in all_obs]

    logger.info(
        f"Adjuvant: {len(edge_obs)} edge obs + {len(oneway_obs)} 1-way obs, "
        f"max_sigma_2w={max_sigma:.2f}"
    )

    return all_obs, moral, bdg_remaining, tvd_diag


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

    # If all observations are 1-way (no edge obs), skip MD entirely.
    # Each 1-way obs is already a valid potential for its single-column clique.
    has_multiway = any(len(obs.source) > 1 for obs in all_obs)
    if not has_multiway:
        logger.info(
            f"Adjuvant: only 1-way observations ({len(all_obs)}), "
            f"skipping junction tree and mirror descent"
        )
        cliques = [obs.source for obs in all_obs]
        potentials = [obs.obs for obs in all_obs]
        # Junction tree: isolated nodes (one per clique, no edges)
        junction = nx.Graph()
        for cl in cliques:
            junction.add_node(cl)
        return junction, cliques, potentials

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

    MAX_PARAMS = 150_000_000
    if total_params > MAX_PARAMS:
        logger.error(
            f"Total params too high: {total_params} > {MAX_PARAMS}. Clique Information:\n"
            f"{','.join(f'{get_clique_domain(cl, attrs):_d}' for cl in cliques)}\n"
            "Exitting..."
        )
        assert False

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
