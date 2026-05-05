"""Adjuvant: DP structure learning via greedy edge addition with height-chain nodes.

Core implementation: cached marginal computation, noisy TVD scoring,
height-chain graph construction, greedy edge addition with exponential
mechanism, and measurement/observation building.
"""

import itertools
import logging
from math import exp, log1p, pi, sqrt
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


def _tvd_floor_2w(theta: float, dp_type: str = "cdp") -> float:
    """Expected TVD between a noisy and true 2-way marginal at confidence theta.

    Domain-independent: sigma = n / (theta * dom) so the noise contribution
    summed over dom cells leaves only a 1/theta factor."""
    if theta <= 0:
        return float("inf")
    if dp_type == "cdp":
        return sqrt(2.0 / pi) / (2.0 * theta)
    # Laplace: noise is Laplace(0, sigma/sqrt(2)) so E|.| = sigma/sqrt(2)
    return 1.0 / (2.0 * sqrt(2.0) * theta)


def _tvd_floor_1w_pair(theta_1w: float, dp_type: str = "cdp") -> float:
    """Upper-bound TVD contribution of P̃(X)P̃(Y) − P(X)P(Y) at theta_1w.

    First-order bound ½(Σ|ε_x| + Σ|ε_y|), with both 1-ways at confidence
    theta_1w (so dom·sigma/n = 1/theta_1w on each side)."""
    if theta_1w <= 0:
        return float("inf")
    if dp_type == "cdp":
        return sqrt(2.0 / pi) / theta_1w
    return 1.0 / (sqrt(2.0) * theta_1w)


def _solve_eff_theta_2w(
    doms: list[int],
    n: int | float,
    target_budget: float,
    theta_min: float,
    dp_type: str = "cdp",
) -> float:
    """Find max theta s.t. Σ budget(d, theta) ≤ target_budget across `doms`.

    Mirrors the binary search used by ``measure_edges`` to boost theta_2w
    when there is leftover budget. Returns +inf if doms is empty (no
    measurements would consume budget).

    Closed form: budget(d, θ) is monotone in θ with a single power per
    mechanism, so Σ budget(d_i, θ) = c·θ^p·S where S = Σd_iᵖ. CDP: p=2,
    c=1/(2n²); DP: p=1, c=√2/n."""
    if not doms:
        return float("inf")
    if target_budget <= 0:
        return theta_min

    if dp_type == "cdp":
        s = sum(d * d for d in doms)
        if s == 0:
            return float("inf")
        theta = n * sqrt(2.0 * target_budget / s)
    else:
        s = sum(doms)
        if s == 0:
            return float("inf")
        theta = n * target_budget / (sqrt(2.0) * s)

    hi_cap = max(theta_min * 1000.0, theta_min + 1.0)
    return min(hi_cap, theta)


def _marginal_floor_cost_2w(
    accepted_doms: list[int],
    cand_dom_arr: np.ndarray,
    n: int | float,
    target_budget: float,
    theta_min: float,
    dp_type: str = "cdp",
) -> np.ndarray:
    """Per-candidate total damage in 2-way noise floor (TVD units).

    The candidate is charged for (a) its own measurement noise and (b) the
    noise-floor rise it inflicts on already-accepted edges:

        penalty = floor_with + K_acc · (floor_with − floor_now)
                = (K_acc + 1)·floor_with − K_acc·floor_now

    Vectorized closed-form: precomputes the accepted-set aggregate once,
    adds each candidate's contribution in O(1)."""
    K = len(cand_dom_arr)
    if K == 0 or target_budget <= 0:
        return np.zeros(K)

    cand_dom = cand_dom_arr.astype(np.float64)
    K_acc = len(accepted_doms)
    if dp_type == "cdp":
        s_acc = float(sum(d * d for d in accepted_doms))
        s_with = s_acc + cand_dom * cand_dom
        const = 2.0 * target_budget
        theta_with = n * np.sqrt(const / s_with)
        theta_now = n * sqrt(const / s_acc) if s_acc > 0 else float("inf")
        floor_const = sqrt(2.0 / pi) / 2.0
    else:
        s_acc = float(sum(accepted_doms))
        s_with = s_acc + cand_dom
        const = target_budget / sqrt(2.0)
        theta_with = n * const / s_with
        theta_now = n * const / s_acc if s_acc > 0 else float("inf")
        floor_const = 1.0 / (2.0 * sqrt(2.0))

    hi_cap = max(theta_min * 1000.0, theta_min + 1.0)
    theta_with = np.minimum(theta_with, hi_cap)
    floor_with = floor_const / theta_with

    if theta_now == float("inf"):
        floor_now = 0.0
    else:
        floor_now = floor_const / min(theta_now, hi_cap)

    return np.maximum(0.0, (K_acc + 1) * floor_with - K_acc * floor_now)


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
    noisy_1way: dict[Col, np.ndarray] | None = None,
    n: int | None = None,
) -> dict[tuple[Col, Col], np.ndarray]:
    """Compute exact pairwise TVD-from-independence at every height combination.

    Score: ½|P(X,Y) − P(X)P(Y)|₁. Grabs 2-way histograms at full resolution
    (height 0), then iteratively aggregates using transition mappings to
    compute the score at all (ha, hb) combos.

    When ``noisy_1way`` (and ``n``) are provided, the independence baseline
    P(X)P(Y) is replaced by the *noisy* 1-ways already released earlier in
    the pipeline: P̃(X)P̃(Y). Because that side is post-processing of an
    already-released DP query, ‖Δ‖₁ = 0 there and the EM sensitivity drops
    from 3/n to 1/n — caller is responsible for picking the right
    sensitivity in ``structure_learn``. The noisy 1-way at height 0 is
    aggregated up to higher heights via the same transition mappings as the
    joint, so per-(ha, hb) cells use the correctly-aggregated baseline.

    No noise added to the score itself — the exponential mechanism provides
    privacy for selection.

    Returns dict mapping (col_a, col_b) -> 2D array of shape (Ha, Hb)
    where Ha, Hb are the number of graph heights for each column.
    Both (ca, cb) and (cb, ca) are stored (the latter transposed)."""
    from ....graph.hugin import get_attrs as _get_attrs

    use_noisy = noisy_1way is not None
    if use_noisy and n is None:
        raise ValueError("compute_tvd: n is required when noisy_1way is set")

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

    # When using a noisy-1-way baseline, pre-aggregate it to every height of
    # each column. Linear aggregation is post-processing of the same released
    # vector — no extra DP cost — and gives us ``noisy_per_height[col][h]``
    # directly for the (ha, hb) score below.
    noisy_per_height: dict[Col, list[np.ndarray]] = {}
    if use_noisy:
        assert noisy_1way is not None and n is not None  # for type checkers
        for col in all_cols:
            if col not in noisy_1way:
                continue
            val, h_range = col_meta[col]
            chain = [noisy_1way[col].astype(np.float64) / n]
            trans = col_transitions[col]
            for h in range(1, h_range):
                t = trans[h - 1]
                new = np.zeros(val.get_domain(h), dtype=np.float64)
                np.add.at(new, t, chain[-1])
                chain.append(new)
            noisy_per_height[col] = chain

    tvd: dict[tuple[Col, Col], np.ndarray] = {}

    for (ca, cb), joint_raw in cached.two_way.items():
        val_a, ha_range = col_meta[ca]
        val_b, hb_range = col_meta[cb]
        dom_a0 = val_a.get_domain(0)
        dom_b0 = val_b.get_domain(0)
        joint_00 = joint_raw.reshape(dom_a0, dom_b0).astype(np.float64)
        n_total = joint_00.sum()

        if n_total == 0 or (
            use_noisy
            and (ca not in noisy_per_height or cb not in noisy_per_height)
        ):
            tvd[ca, cb] = np.zeros((ha_range, hb_range))
            tvd[cb, ca] = np.zeros((hb_range, ha_range))
            continue

        result = np.zeros((ha_range, hb_range))
        trans_a = col_transitions[ca]
        trans_b = col_transitions[cb]
        chain_a = noisy_per_height.get(ca)
        chain_b = noisy_per_height.get(cb)

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
                if use_noisy:
                    indep = np.outer(chain_a[ha], chain_b[hb])
                else:
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


def _value_col(d: dict) -> Col:
    """Column key (table, order, attr, value) for a node's data dict."""
    return (d.get("table"), d.get("order"), d["attr"], d["value"])


def init_active_heights(
    directed_graph: "nx.DiGraph",
    moral: "nx.Graph",
) -> tuple[dict[Col, list[int]], dict[tuple[Col, int], str]]:
    """Initial active heights and (col, height) -> node lookup.

    A height is "active" when its node has at least one cross-value edge in
    the moral graph (e.g. the cmn[0] -> v[h_range-1] cross-attribute edge)
    or when it's the protected h=0 of a main-table value (where 1-way obs
    attach).  Returned per-column lists are kept sorted ascending."""
    active: dict[Col, set[int]] = {}
    lookup: dict[tuple[Col, int], str] = {}
    for n, d in directed_graph.nodes(data=True):
        lookup[(_value_col(d), d["height"])] = n
        if d.get("table") is None and d["height"] == 0:
            active.setdefault(_value_col(d), set()).add(0)
    for u, v in moral.edges():
        du = directed_graph.nodes[u]
        dv = directed_graph.nodes[v]
        ku, kv = _value_col(du), _value_col(dv)
        if ku != kv:
            active.setdefault(ku, set()).add(du["height"])
            active.setdefault(kv, set()).add(dv["height"])
    sorted_active: dict[Col, list[int]] = {k: sorted(v) for k, v in active.items()}
    return sorted_active, lookup


def activate_height(
    col: Col,
    h: int,
    base_active: dict[Col, list[int]],
) -> None:
    """Mark height ``h`` as active for ``col`` (sorted insertion).

    Pure book-keeping — chain edges are *not* committed to the moral
    graph or base adjacency here.  Instead they are added temporarily
    during the clique-size check (via ``provisional_chain_edges``) and
    permanently materialised on the moral graph just before junction-tree
    construction (via ``finalize_moral_graph``)."""
    import bisect

    hs = base_active.setdefault(col, [])
    if h in hs:
        return
    idx = bisect.bisect_left(hs, h)
    hs.insert(idx, h)


def provisional_chain_edges(
    touched_nodes: set[str],
    base_active: dict[Col, list[int]],
    height_lookup: dict[tuple[Col, int], str],
    node_data: dict[str, dict],
    adj: dict[str, set[str]],
) -> list[tuple[str, str]]:
    """Chain edges to add for the clique-size check's tentative graph.

    Iterates over *every* column with ≥2 active heights (the touched
    candidate's column plus all previously-activated columns) and returns
    the consecutive-pair chain edges not already in ``adj``.  Covering
    every active column matters: chain edges in unrelated columns can fold
    structure-edge endpoints into larger cliques during triangulation, so
    omitting them would underestimate clique sizes and let oversized
    candidates slip past the check."""
    touched_active: dict[Col, set[int]] = {}
    for n in touched_nodes:
        d = node_data.get(n)
        if d is None:
            continue
        touched_active.setdefault(_value_col(d), set()).add(d["height"])

    all_active: dict[Col, set[int]] = {}
    for col, hs in base_active.items():
        if len(hs) >= 1:
            all_active[col] = set(hs)
    for col, hs in touched_active.items():
        all_active.setdefault(col, set()).update(hs)

    edges: list[tuple[str, str]] = []
    for col, hs in all_active.items():
        if len(hs) < 2:
            continue
        sorted_h = sorted(hs)
        # Chain consecutive *active* heights only — intermediate heights
        # that never got activated are skipped over, e.g. [1, 3, 7] →
        # h_1—h_3 and h_3—h_7 (no h_2/h_4/h_5/h_6).
        for h_lo, h_hi in zip(sorted_h, sorted_h[1:]):
            n_lo = height_lookup.get((col, h_lo))
            n_hi = height_lookup.get((col, h_hi))
            if n_lo is None or n_hi is None:
                continue
            if n_hi not in adj.get(n_lo, ()):
                edges.append((n_lo, n_hi))
    return edges


def finalize_moral_graph(
    moral: "nx.Graph",
    base_active: dict[Col, list[int]],
    height_lookup: dict[tuple[Col, int], str],
) -> tuple[int, int]:
    """Commit chain edges and prune chain-only height nodes.

    Chain edges connect consecutive *active* heights per column — "active"
    here means within the sorted ``base_active`` set, *not* consecutive
    integers — so intermediate heights that never gained a structure edge
    stay isolated and the chain skips directly across them.  For active
    heights ``[1, 3, 7]`` this yields exactly two edges (h_1—h_3 and
    h_3—h_7) with no h_2/h_4/h_5/h_6 involvement.

    After committing, any node whose only neighbours (if any) sit at
    other heights of the same ``(table, order, attr, value)`` is dropped:
    no structure edge, no cross-attribute edge, nothing for triangulation
    to attach to.  Heights are deterministic refinements of each other,
    so removing such a node and bridging its two chain neighbours
    preserves the joint exactly (sum_v[mid] P(v[low], v[mid], v[high]) =
    P(v[low], v[high])).  Isolated singletons (no neighbours at all) and
    orphan ``cmn[0]–v[boundary]`` pairs left over from unused evidence
    values are covered by the same rule, since the cmn-glue spans
    different ``(attr, value)`` keys.

    Main-table h=0 nodes are protected — that's where 1-way observations
    attach.

    Returns ``(n_chain_edges_added, n_nodes_pruned)``."""
    n_added = 0
    for col, hs in base_active.items():
        if len(hs) < 2:
            continue
        sorted_h = sorted(hs)
        for h_lo, h_hi in zip(sorted_h, sorted_h[1:]):
            n_lo = height_lookup.get((col, h_lo))
            n_hi = height_lookup.get((col, h_hi))
            if n_lo is None or n_hi is None:
                continue
            if not moral.has_edge(n_lo, n_hi):
                moral.add_edge(n_lo, n_hi, chain=True)
                n_added += 1

    def _key(node):
        d = moral.nodes[node]
        return (d.get("table"), d.get("order"), d["attr"], d["value"])

    def _protected(node):
        d = moral.nodes[node]
        # 1-way obs cover every main-table column at h=0.
        return d.get("table") is None and d.get("height") == 0

    n_pruned = 0
    while True:
        prunable: list[str] = []
        for node in moral.nodes():
            if _protected(node):
                continue
            key = _key(node)
            chain_only = True
            for nb in moral.neighbors(node):
                if _key(nb) != key:
                    chain_only = False
                    break
            if chain_only:
                prunable.append(node)

        if not prunable:
            break

        for node in prunable:
            if node not in moral:
                continue
            nbs = list(moral.neighbors(node))
            moral.remove_node(node)
            n_pruned += 1
            if len(nbs) == 2:
                a, b = nbs
                if a in moral and b in moral and not moral.has_edge(a, b):
                    moral.add_edge(a, b, chain_bridged=True)

    return n_added, n_pruned


def build_height_chain_graph(attrs: DatasetAttributes) -> nx.DiGraph:
    """Build directed height-chain graph (cross-attribute edges only).

    Nodes are (attr, value, height) with metadata for table/order.  Chain
    edges within the same (attr, value) are *not* added here — they would
    inflate cliques for intermediate heights that no structure edge ever
    touches.  Necessary chain edges are added later by ``structure_learn``
    (incrementally as heights become active) and provisionally during the
    clique-size check, so the final moral graph chains only the heights
    actually used.

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
    evidence_vars: frozenset[str] | None = None,
    max_root_clique_size: float = float("inf"),
) -> tuple[list[frozenset[str]], bool]:
    """Run min-factor-domain greedy elimination on ``adj`` (mutated) and
    collect the maximal cliques of the resulting chordal graph.

    Same algorithm as ``elimination_order_greedy`` in ``graph/hugin.py``: at
    each step picks the remaining node whose factor (= node + remaining
    neighbors) has the smallest domain, fills in edges between its
    neighbors, and recomputes costs only for the affected neighbors.

    Early-rejects (returns ``(_, False)``) the moment any factor's domain
    exceeds the looser of the two caps — sound because every clique of the
    resulting triangulated graph is a subset of some step's factor, so an
    overflow factor proves at least one clique is over the loosest limit.

    Per-clique validation runs as a post-pass: only factors that survive
    subsumption (= max cliques) are checked, against ``max_root_clique_size``
    if they contain all of ``evidence_vars`` and ``max_clique_size``
    otherwise.  Subset factors arising while the root clique dissolves
    during elimination would otherwise be falsely rejected.

    ``cost_map`` is consumed (mutated); pass a fresh copy if you need to
    keep the original."""
    remaining = set(adj)
    cliques: list[frozenset[str]] = []
    clique_doms: list[float] = []
    eff_max = max_clique_size
    if evidence_vars and max_root_clique_size > max_clique_size:
        eff_max = max_root_clique_size

    while remaining:
        v = min(remaining, key=cost_map.__getitem__)
        factor_dom = cost_map[v]

        if factor_dom > eff_max:
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
            clique_doms.append(factor_dom)

    if evidence_vars:
        for c, dom in zip(cliques, clique_doms):
            cap = max_root_clique_size if evidence_vars.issubset(c) else max_clique_size
            if dom > cap:
                return cliques, False

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
    evidence_vars: frozenset[str] | None = None,
    max_root_clique_size: float = float("inf"),
    extra_edges: list[tuple[str, str]] | None = None,
    base_active: dict[Col, list[int]] | None = None,
    height_lookup: dict[tuple[Col, int], str] | None = None,
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

    ``extra_edges`` are added to the adjacency along with (na, nb) — used
    by structure_learn to extend the evidence clique when na or nb is an
    evidence var that hasn't been selected yet.  When present, the fast
    path is bypassed (the new edges may create or grow cliques).

    ``base_active`` and ``height_lookup`` enable provisional chain edges:
    chain edges between consecutive active heights are added to the local
    adjacency before elimination (covering both the candidate's column
    and every previously-activated column), since base_adj/moral don't
    persist chain edges between iterations.

    Returns (0, valid). The first element is unused by the caller."""
    # Fast path: edge internal to an existing base clique → no new clique.
    if not extra_edges and cliques_by_node is not None and base_cliques is not None:
        ca = cliques_by_node.get(na, [])
        cb = cliques_by_node.get(nb, [])
        if ca and cb:
            sa = set(ca)
            for i in cb:
                if i in sa:
                    return 0.0, True

    # Slow path: fresh hugin elimination on base_adj + (na, nb) + extra_edges.
    adj = {v: s.copy() for v, s in base_adj.items()}
    adj[na].add(nb)
    adj[nb].add(na)
    touched: set[str] = {na, nb}
    if extra_edges:
        for a, b in extra_edges:
            adj[a].add(b)
            adj[b].add(a)
            touched.add(a)
            touched.add(b)

    # Provisional chain edges over every active column (not just the
    # candidate's): chain edges are not persisted on base_adj/moral, so
    # the slow-path triangulation needs them temporarily reconstructed
    # here to see the same graph the final junction-tree builder will.
    if base_active is not None and height_lookup is not None:
        for ca, cb in provisional_chain_edges(
            touched, base_active, height_lookup, node_data, adj
        ):
            adj[ca].add(cb)
            adj[cb].add(ca)
            touched.add(ca)
            touched.add(cb)

    # Only touched nodes gained neighbors relative to base, so reuse the
    # cached cost_map and recompute just those.
    if base_cost_map is not None:
        cost_map = dict(base_cost_map)
        for v in touched:
            cost_map[v] = _factor_domain(adj[v] | {v}, node_data, attrs)
    else:
        cost_map = {v: _factor_domain(adj[v] | {v}, node_data, attrs) for v in adj}

    _, valid = _hugin_eliminate(
        adj,
        cost_map,
        node_data,
        attrs,
        max_clique_size,
        evidence_vars=evidence_vars,
        max_root_clique_size=max_root_clique_size,
    )
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


def _edge_dom_log2(
    node_a: str,
    node_b: str,
    g: nx.Graph,
    attrs: DatasetAttributes,
) -> float:
    """log2 of the candidate edge's clique-domain product (dom_a · dom_b)."""
    from ....graph.hugin import get_attrs as _get_attrs

    def _dom_for_node(node: str) -> float:
        d = g.nodes[node]
        a = _get_attrs(attrs, d["table"], d["order"])[d["attr"]]
        val = cast(CatValue, a[d["value"]])
        return float(val.get_domain(d["height"]))

    return float(np.log2(_dom_for_node(node_a) * _dom_for_node(node_b)))


def compute_edge_weight(
    node_a: str,
    node_b: str,
    g: nx.Graph,
    attrs: DatasetAttributes,
    size_penalty: float,
    d_ref_log2: float = 0.0,
) -> float:
    """Hyperbolic size-penalty boost in score units (boost ∈ (0, 1]).

    boost = 1 / (1 + size_penalty · max(0, log2(D / D_ref)))

    Single-parameter, no exponent: a candidate at the reference size
    scores at boost=1; at size_penalty=0.10 a 2x candidate boosts to
    ~0.91, a 4x to ~0.83.  ``d_ref_log2`` should be precomputed as the
    minimum ``log2(dom_a · dom_b)`` across all candidates so the
    smallest candidate sits at the reference."""
    excess = _edge_dom_log2(node_a, node_b, g, attrs) - d_ref_log2
    if excess <= 0:
        return 1.0
    return 1.0 / (1.0 + size_penalty * excess)


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
    min_score: "float | tuple[str, float]",
    em_z: float,
    theta_2w: float,
    frozen_nodes: set[str] | None = None,
    n_hist_cols: int = 0,
    max_clique_size: float = 1e5,
    max_root_clique_size: float = float("inf"),
    max_em_budget: float = float("inf"),
    min_em_budget: float = 0.0,
    em_max: float = 50.0,
    rake: bool = True,
    max_order: int | None = None,
    dp_type: str = "cdp",
    scoring: str = "tvd",
    min_safety_factor: float = 3.0,
    theta_1w_eff: float = 0.0,
    real_tvd: "dict[tuple[Col, Col], np.ndarray] | None" = None,
    cost_penalty: bool = True,
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

    # Evidence (hist) vars only need to share a single root clique once
    # they are *connected* to the rest of the model — i.e. the moment EM
    # picks an edge touching them.  Pre-adding all-pair edges among every
    # frozen node would create huge intermediate cliques that have nothing
    # to do with the model and force every candidate edge to fail the
    # max_clique_size check.  We instead grow the evidence clique
    # incrementally as evidence vars get selected (see ``selected_evidence``
    # below).
    evidence_pool: frozenset[str] = (
        frozenset(v for v in frozen_nodes if v in moral) if frozen_nodes else frozenset()
    )
    selected_evidence: set[str] = set()

    # Generate candidates and group by column pair
    candidates, col_pair_map = generate_candidates(
        directed_graph, frozen_nodes, rake=rake, max_order=max_order
    )
    connected_pairs: set[tuple[Col, Col]] = set()
    structure_edges: set[frozenset[str]] = set()

    # EM sensitivity depends on the score function. The TVD score is
    # ½|P(X,Y) − P(X)P(Y)|₁: one row perturbs P(X,Y) by ≤2/n and P(X)P(Y)
    # by ≤4/n in L1, so sens ≤ 3/n. With ``tvd_n`` the independence baseline
    # is the *noisy* 1-ways released in Step 1 — post-processing, so its
    # contribution is 0 and only ‖ΔP(X,Y)‖₁ ≤ 2/n remains, sens ≤ 1/n.
    # MI uses PrivBayes Lemma 3 (log2).
    if scoring == "mi":
        sensitivity = float(sens_mutual_info(n))
    elif scoring == "tvd_n":
        sensitivity = 1.0 / n
    else:
        sensitivity = 3.0 / n

    # Pre-compute per-candidate measurement budget and theta-filter
    cand_bdg_edge = np.zeros(len(candidates))
    cand_doms = np.zeros(len(candidates), dtype=np.int64)
    cand_valid = np.ones(len(candidates), dtype=bool)
    for idx, (na, nb) in enumerate(candidates):
        cand_doms[idx] = _edge_clique_domain(na, nb, directed_graph, attrs)
    if rho_avail > 0:
        n_filtered = 0
        for idx in range(len(candidates)):
            b = compute_budget_for_theta(
                int(cand_doms[idx]), n, theta_2w, dp_type
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
    accepted_doms: list[int] = []  # doms of accepted edges, for "auto" min_score
    edge_budgets: dict[frozenset[str], float] = {}  # per-edge budget reserved at theta_2w

    if isinstance(min_score, tuple):
        auto_min_score = min_score[0] == "auto"
        min_score = min_score[1]
    else:
        auto_min_score = False
    floor_1w = (
        _tvd_floor_1w_pair(theta_1w_eff, dp_type)
        if auto_min_score and scoring == "tvd_n"
        else 0.0
    )

    def _current_min_score() -> float:
        """Effective per-iter stop threshold.

        For "auto", the threshold is the noise floor we'd see if we stopped
        here: theta_2w boosted by rescaling the leftover budget across the
        edges accepted so far. For ``tvd_n`` we additionally add the noise
        contribution that the noisy 1-way baseline already injects into the
        score itself, so a candidate must clear both noise components."""
        if not auto_min_score:
            return float(min_score)
        if rho_avail <= 0:
            return 0.0
        target_bdg = max(0.0, rho_avail - bdg_em)
        eff_theta = _solve_eff_theta_2w(
            accepted_doms, n, target_bdg, theta_2w, dp_type
        )
        return _tvd_floor_2w(eff_theta, dp_type) + floor_1w + min_score

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

    # Precompute TVD*boost for all candidates (doesn't change across iterations).
    # d_ref_log2 = log2 of the smallest candidate's clique-domain product so
    # the smallest candidate sits at boost=1 and all others scale relative
    # to it (a 2x candidate boosts to 0.9, 4x to 0.85, etc.).
    cand_tvd_boost = np.empty(len(candidates))
    if candidates:
        d_ref_log2 = min(
            _edge_dom_log2(na, nb, directed_graph, attrs) for na, nb in candidates
        )
    else:
        d_ref_log2 = 0.0
    for idx, (na, nb) in enumerate(candidates):
        da, db = directed_graph.nodes[na], directed_graph.nodes[nb]
        col_a: Col = (da.get("table"), da.get("order"), da["attr"], da["value"])
        col_b: Col = (db.get("table"), db.get("order"), db["attr"], db["value"])
        tvd_arr = tvd.get((col_a, col_b))
        base = (
            float(tvd_arr[da["height"], db["height"]]) if tvd_arr is not None else 0.0
        )
        boost = compute_edge_weight(
            na, nb, moral, attrs, size_penalty, d_ref_log2=d_ref_log2
        )
        cand_tvd_boost[idx] = base * boost

    # Cache node data for clique-size checks (immutable, built once)
    node_data = {n: d for n, d in directed_graph.nodes(data=True)}
    # Maintained adjacency dict (updated incrementally, avoids rebuilding from nx)
    base_adj = _build_adj(moral)
    # Per-column sorted active heights + (col, height) -> node lookup.
    # base_adj/moral never carry chain edges during the loop; activate_height
    # just records which heights have a structure or evidence edge, and the
    # consecutive-pair chain edges are reconstructed temporarily inside
    # _score_with_one_edge (clique check) and committed once on moral via
    # finalize_moral_graph after the loop ends.
    base_active, height_lookup = init_active_heights(directed_graph, moral)
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
        eff_min_score = _current_min_score()

        # Per-candidate marginal-cost penalty (TVD units).
        # Picking a high-domain edge drops the rescale-effective theta_2w
        # for every edge, raising the post-measurement noise floor.
        # We charge each candidate the *marginal* rise in that floor,
        # i.e. floor(accepted ∪ {cand}) − floor(accepted).  Both terms
        # depend only on data-independent quantities (doms, budgets), so
        # EM sensitivity is unchanged.  Disabled when rho_avail<=0 so
        # selection degenerates exactly to the pre-cost behaviour.
        cand_cost: dict[int, float] | None = None
        if cost_penalty and rho_avail > 0 and em_pool:
            target_bdg = max(0.0, rho_avail - bdg_em)
            pool_idxs = [idx for idx, _, _ in em_pool]
            costs = _marginal_floor_cost_2w(
                accepted_doms,
                cand_doms[pool_idxs],
                n,
                target_bdg,
                theta_2w,
                dp_type,
            )
            # Anchor the cheapest candidate at zero penalty so we don't
            # tilt EM toward the stop option (which carries no cost).
            costs = costs - costs.min()
            cand_cost = dict(zip(pool_idxs, costs.tolist()))

        while em_pool:
            scores = np.array([cand_tvd_boost[idx] for idx, _, _ in em_pool])
            if cand_cost is not None:
                scores = scores - np.array(
                    [cand_cost[idx] for idx, _, _ in em_pool]
                )
            stop_idx = len(scores)

            if rho_avail > 0:
                if eff_min_score > 0:
                    em_scores = np.append(
                        scores, eff_min_score + log_n_boost if eff_min_score else 0
                    )
                else:
                    em_scores = scores
                sel = exponential_mechanism(em_scores, eps_step, sensitivity)
            else:
                em_scores = np.append(scores, eff_min_score)
                sel = int(np.argmax(em_scores))

            if sel == stop_idx:
                stopped = True
                break

            cand_idx, na, nb = em_pool[sel]

            # Dynamic evidence-clique extension: if either endpoint is
            # an evidence var not yet selected, accepting this edge pulls
            # it into the root clique, which means we must add edges
            # between it and every already-selected evidence var (and
            # between the two endpoints if both are newly entering).
            new_ev: set[str] = set()
            if na in evidence_pool and na not in selected_evidence:
                new_ev.add(na)
            if nb in evidence_pool and nb not in selected_evidence:
                new_ev.add(nb)

            extra_edges: list[tuple[str, str]] = []
            for v in new_ev:
                for u in selected_evidence:
                    extra_edges.append((v, u))
            if len(new_ev) == 2:
                a, b = tuple(new_ev)
                if a != na or b != nb:  # already added as the candidate
                    extra_edges.append((a, b))

            tentative_selected = selected_evidence | new_ev
            tentative_ev = (
                frozenset(tentative_selected) if len(tentative_selected) >= 2 else None
            )

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
                evidence_vars=tentative_ev,
                max_root_clique_size=max_root_clique_size,
                extra_edges=extra_edges if extra_edges else None,
                base_active=base_active,
                height_lookup=height_lookup,
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
            auto_tag = " [auto]" if auto_min_score else ""
            logger.info(
                f"Adjuvant: exit (EM picked stop option, "
                f"min_score={eff_min_score:.4f}{auto_tag}) "
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
        accepted_doms.append(int(cand_doms[cand_idx]))
        edge_budgets[frozenset([na, nb])] = float(edge_bdg)

        # Accept edge
        moral.add_edge(na, nb, structure=True)
        base_adj[na].add(nb)
        base_adj[nb].add(na)
        # Persist any evidence-clique extension edges committed by this
        # candidate, so future candidate checks see them in base_adj.
        for ea, eb in extra_edges:
            if not moral.has_edge(ea, eb):
                moral.add_edge(ea, eb, evidence=True)
            base_adj[ea].add(eb)
            base_adj[eb].add(ea)
        selected_evidence |= new_ev
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

        # Mark heights as active for the candidate endpoints (and any new
        # evidence vars pulled in via extra_edges).  Pure book-keeping —
        # chain edges are added temporarily during the next clique check
        # via provisional_chain_edges, and committed to the moral graph
        # once after the loop via finalize_moral_graph.
        activate_height(col_a, da["height"], base_active)
        activate_height(col_b, db["height"], base_active)
        for ea, eb in extra_edges:
            for ev in (ea, eb):
                dev = directed_graph.nodes[ev]
                col_ev: Col = (
                    dev.get("table"),
                    dev.get("order"),
                    dev["attr"],
                    dev["value"],
                )
                activate_height(col_ev, dev["height"], base_active)
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

    # Materialise the chain edges between consecutive active heights on
    # the moral graph now that structure learning has settled.  Up to
    # this point chain edges only existed transiently inside the
    # clique-size check; the junction-tree builder needs them present in
    # the actual graph it triangulates.  Same call also drops components
    # that carry no observations (unused evidence values, isolated
    # intermediate heights) so triangulation doesn't waste cliques on
    # them.
    n_chain, n_pruned = finalize_moral_graph(moral, base_active, height_lookup)
    if n_chain or n_pruned:
        logger.info(
            f"Adjuvant: committed {n_chain} chain edges, "
            f"pruned {n_pruned} unused nodes from moral graph"
        )

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
        _current_min_score(),
        label=scoring.upper(),
        real_tvd=real_tvd,
        edge_budgets=edge_budgets,
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
    real_tvd: "dict[tuple[Col, Col], np.ndarray] | None" = None,
    edge_budgets: "dict[frozenset[str], float] | None" = None,
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

    selection_budget = (
        sum(edge_budgets.values()) if edge_budgets else 0.0
    )

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
                    real_tag = ""
                    if real_tvd is not None:
                        r_arr = real_tvd.get((col_a_t, col_b_t))
                        if r_arr is not None:
                            r_at_h = float(r_arr[ha, hb])
                            r_at_0 = float(r_arr[0, 0])
                            real_tag = (
                                f" (real={r_at_h:.4f}"
                                + (f"/{r_at_0:.4f}" if ha != 0 or hb != 0 else "")
                                + ")"
                            )
                    bdg_tag = ""
                    if edge_budgets is not None and selection_budget > 0:
                        b = edge_budgets.get(edge)
                        if b is not None:
                            bdg_tag = f" bdg={100.0 * b / selection_budget:.2f}%"
                    lines.append(
                        f"  CONNECTED {label}={tvd_at_h:.4f}"
                        f"{f'/{val:.4f}' if ha != 0 or hb != 0 else ''}"
                        f"{real_tag}{bdg_tag} "
                        f"{_fmt_edge(ena, enb, moral, attrs)}"
                    )
                    break
        elif val >= min_tvd:
            ca_tbl, ca_ord, ca_attr, ca_val = ca
            cb_tbl, cb_ord, cb_attr, cb_val = cb
            # Skip hist-table pairs (they are frozen, not candidates)
            if ca_tbl is not None or cb_tbl is not None:
                continue
            real_tag = ""
            if real_tvd is not None:
                r_arr = real_tvd.get((ca, cb))
                if r_arr is not None:
                    real_tag = f" (real={float(r_arr[0, 0]):.4f})"
            lines.append(
                f"    MISSING {label}={val:.4f}{real_tag} "
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

        prob = (prob / n).astype(np.float32)

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

        prob = (noisy_mar / n).astype(np.float32)
        dom = len(prob)
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
    min_tvd: "float | tuple[str, float]" = 0.05,
    min_mi: float = 0.0,
    min_safety_factor: float = 3.0,
    frozen_nodes: set[str] | None = None,
    n_hist_cols: int = 0,
    max_clique_size: float = 1e5,
    max_root_clique_size: float = float("inf"),
    rescale: bool = True,
    rake: bool = True,
    max_order: int | None = None,
    dp_type: str = "cdp",
    scoring: str = "tvd",
    skip_structure: bool = False,
    no_confidence: bool = False,
    cost_penalty: bool = True,
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
        if skip_structure:
            # Ablation: route the entire budget to 1-way marginals.
            bdg1_max = rho
            bdg1_min = rho
        else:
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
        if scoring == "tvd_n":
            scoring = "tvd"
    logger.info(
        f"Adjuvant Step 1: Noisy 1-way marginals "
        f"(theta_1w={eff_theta_1w:.1f}, {bdg_label}1={bdg1:.6f})"
    )

    # Step 2: Structure learning (remaining budget)
    bdg_avail = rho - bdg1
    assert (
        bdg_avail >= 0
    ), f"Available budget went negative: {bdg_avail}, 0 for disable and positive for enabled"

    if skip_structure:
        # Ablation: bypass structure learning and edge measurement; produce
        # only 1-way observations.
        logger.info(
            f"Adjuvant Step 2: skipped (ablation=1-way), "
            f"{bdg_label}_remaining={bdg_avail:.6f}"
        )
        moral = nx.Graph()
        edge_obs: list = []
        max_sigma = 0.0
        bdg_remaining = bdg_avail
        tvd_diag = ""
    else:
        logger.info(
            f"Adjuvant Step 2: Structure learning "
            f"({bdg_label}_avail={bdg_avail:.6f}, em_z={em_z}, theta_2w={theta_2w})"
        )
        real_tvd_for_diag: "dict[tuple[Col, Col], np.ndarray] | None" = None
        if scoring == "mi":
            scores = compute_mi(cached, attrs, all_cols)
            # "auto" only makes sense for TVD-based scoring; fall back to 0.
            min_score: "float | str" = (
                0.0 if isinstance(min_mi, str) else min_mi
            )
        elif scoring == "tvd_n" and bdg_avail:
            scores = compute_tvd(
                cached, attrs, all_cols, noisy_1way=noisy_1way, n=n
            )
            # Real (true-baseline) TVD for diagnostics only — selection still
            # uses the noisy-baseline scores.
            real_tvd_for_diag = compute_tvd(cached, attrs, all_cols)
            min_score = min_tvd
        else:
            scores = compute_tvd(cached, attrs, all_cols)
            min_score = min_tvd
        directed_graph = build_height_chain_graph(attrs)
        logger.info(
            f"Adjuvant: height-chain graph has {directed_graph.number_of_nodes()} "
            f"nodes (scoring={scoring}, min_score={min_score})"
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
            max_root_clique_size=max_root_clique_size,
            rake=rake,
            max_order=max_order,
            max_em_budget=max_em_budget,
            min_em_budget=min_em_budget,
            em_max=em_max,
            dp_type=dp_type,
            scoring=scoring,
            min_safety_factor=min_safety_factor,
            theta_1w_eff=eff_theta_1w,
            real_tvd=real_tvd_for_diag,
            cost_penalty=cost_penalty,
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

    if no_confidence:
        all_obs = [o._replace(confidence=1.0) for o in all_obs]
    else:
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
    evidence_vars: set[str] | None = None,
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

    mg = moral.copy() if tree_mode != "maximal" and moral is not None else None
    logger.info(f"Adjuvant: building junction tree (mode={tree_mode})")
    junction, cliques, messages = build_junction_tree(
        all_obs,
        attrs,
        tree_mode=tree_mode,
        moral_graph=mg,
        elim_factor_cost=elim_factor_cost,
        elim_max_attempts=elim_max_attempts,
        evidence_vars=evidence_vars,
    )
    total_params = sum(get_clique_domain(cl, attrs) for cl in cliques)
    logger.info(
        f"Adjuvant: junction tree has {len(cliques)} cliques, "
        f"{total_params:_} parameters"
    )

    MAX_PARAMS = 500_000_000
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
