import logging
from itertools import combinations
from typing import NamedTuple, Sequence, cast

import numpy as np
import pandas as pd

from ....attribute import Attributes, IdxValue, get_dtype
from ....marginal import (
    ZERO_FILL,
    AttrSelector,
    AttrSelectors,
    MarginalOracle,
    MarginalRequest,
)
from ....utils.progress import piter, prange, process_in_parallel

logger = logging.getLogger(__name__)

MAX_EPSILON = 1e3 - 10
MAX_T = 1e5
MAX_COLS = 128
MAX_ATTR_COLS = 8

# Represents a parent set, immutable, hashable, fast
# index is col, val is height. -1 means not included
EMPTY_PSET = tuple(-1 for _ in range(MAX_COLS))

EMPTY_LIST = [-1 for _ in range(MAX_ATTR_COLS)]
EMPTY_ACTIVE = [False for _ in range(MAX_ATTR_COLS)]


class Node(NamedTuple):
    attr: str
    col: str
    domain: int
    partial: bool
    p: dict[str, AttrSelector]


Nodes = list[Node]


def sens_mutual_info(n: int):
    """Provides the the log2 sensitivity of the mutual information function for a given
    dataset size (n)."""
    return 2 / n * np.log2((n + 1) / 2) + (n - 1) / n * np.log2((n + 1) / (n - 1))


def calc_mutual_info(j_mar, x_mar, p_mar):
    """Calculates mutual information I(X,P) for the provided data using log2."""
    mi = np.sum(j_mar * np.log2(j_mar / (np.outer(x_mar, p_mar) + ZERO_FILL)))
    return mi


def sens_r_function(n: int):
    """Provides the the R function sensitivity for a given dataset size (n)."""
    return 3 / n + 2 / (n**2)


def calc_r_function(j_mar, x_mar, p_mar):
    """Calculates the R(X,P) function for the provided data."""
    r = np.sum(np.abs(j_mar - np.outer(x_mar, p_mar))) / 2
    return r


def calc_entropy(mar):
    """Calculates the entropy for the provided data."""
    # TODO: check this is correct using scipy
    return -np.sum(mar * np.log2(mar))


def sens_entropy(n: int):
    """Provides the sensitivity for the entropy function for a given dataset size (n)."""
    # TODO: Mathematically prove this is correct.
    return np.log2(n) / n - (n - 1) / n * np.log2((n - 1) / n)


def add_to_pset(s: tuple, x: int, h: int):
    """Given parent set `s`, adds attribute `x` with height `h`.

    Making a list is faster than ex. using an iterator."""
    y = list(s)
    y[x] = h
    return tuple(y)


def add_multiple_to_pset(s: tuple, x: Sequence[int], h: list[int]):
    """Given parent set `s`, adds attributes `x` with heights `h`.

    `x` is checked for length, so `h` may be a larger array."""

    y = list(s)

    for i in range(len(x)):
        y[x[i]] = h[i]
    return tuple(s)

def find_maximal_parents(heights, domain, common, A, tau, partial):

    def maximal_parents(
        A: tuple[tuple[int], ...], tau: float, partial: bool = False
    ) -> list[tuple[int]]:
        """Given a set V containing hierarchical attributes (by int) and a tau
        score that is divided by the size of the domain, return a set of all
        possible combinations of attributes, such that if t > 1 there isn't an
        attribute that can be indexed in a higher level

        If `partial` is true the first set domain will be reduced by common"""

        if not A:
            return [EMPTY_PSET]

        a_full = A[0]
        cmn = common[a_full[0]]

        A = A[1:]
        S = []
        U_global = set()
        U = set()

        # First do single combinations, they are simplified
        not_first = False
        for x in a_full:
            U = set()
            # Only the first variable can have a only common domain (last height)
            # (all last heights are equivalent for the same attribute; skip to prevent bias)
            for h in range(heights[x] - not_first):
                # Find domain
                l_dom = domain[x][h] - (cmn if partial else 0)

                # Run check to skip recursion
                if tau < l_dom:
                    continue

                for z in maximal_parents(A, tau / l_dom):
                    if z not in U:
                        U_global.add(z)
                        U.add(z)
                        S.append(add_to_pset(z, x, h))
            not_first = True

        a_full_n = len(a_full)
        if a_full_n > 1:
            for l in range(2, a_full_n + 1):
                for a in combinations(a_full, r=l):
                    U = set()

                    # Compensating for multi-domain attrs is more complicated
                    curr_attrs = list(EMPTY_LIST)
                    has_combs = True

                    while has_combs:
                        # Find domain
                        l_dom = 1
                        for i in range(l):
                            dom = domain[a[i]][curr_attrs[i]]
                            l_dom *= dom - cmn
                        if not partial:
                            l_dom += cmn
                        # print(curr_attrs[:a_n], l_dom)

                        # Run check to skip recursion
                        if tau > l_dom:
                            for z in maximal_parents(A, tau / l_dom):
                                if z not in U:
                                    U_global.add(z)
                                    U.add(z)
                                    S.append(add_multiple_to_pset(z, a, curr_attrs))

                        # Simple counter structure without iterators that will iterate over
                        # all attribute height combinations
                        for i in range(l):
                            # In multi-attr mode, none of the variables cah have
                            # a common only domain (last height)
                            if curr_attrs[i] < heights[a[i]] - 2:
                                curr_attrs[i] += 1
                                break
                            else:
                                curr_attrs[i] = 0
                                # Detect overflow and break
                                # Placing check on with condition would make it run every time
                                if i == l - 1:
                                    has_combs = False

        # As in the default privbayes maximal_parents, add all combinations that don't include
        # parents
        for z in maximal_parents(A, tau):
            if z not in U_global:
                S.append(z)

        return S
    
    return maximal_parents(A, tau, partial)

def greedy_bayes(
    oracle: MarginalOracle,
    attrs: Attributes,
    e1: float | None,
    e2: float | None,
    theta: float,
    use_r: bool,
    unbounded_dp: bool,
    random_init: bool,
    skip_zero_counts: bool,
) -> tuple[Nodes, float]:
    """Performs the greedy bayes algorithm for variable domain data.

    Supports variable e1, e2, where in the paper they are defined as
    `e1 = b * e` and `e2 = (1 - b) * e`, variable theta, and both
    mutual information and R functions.

    Binary domains are not supported due to computational intractability."""

    n, d = oracle.get_shape()
    n_chosen = 1 if random_init else 0
    calc_fun = calc_r_function if use_r else calc_mutual_info
    sens_fun = sens_r_function if use_r else sens_mutual_info

    #
    # Set up maximal parents algorithm as shown in paper
    # (recursive implementation is a bit slow)
    #
    col_names = []  # col -> str name
    groups = []  # col -> attribute parent
    group_names = []  # attr (group) -> str name
    heights = []  # col -> max height
    common = []  # col -> common val number
    domain = []  # col, height -> domain
    for i, (an, a) in enumerate(attrs.items()):
        group_names.append(an)
        for c_n, c in a.vals.items():
            c = cast(IdxValue, c)
            col_names.append(c_n)
            groups.append(i)
            heights.append(c.height)
            common.append(a.common)
            domain.append([c.get_domain(h) for h in range(c.height)])

    # If skip_zero_counts is on, calculate domain based on non-zero values
    domain_orig = domain
    if skip_zero_counts:
        domain = []
        counts = oracle.get_counts(f"Calculating counts for nonzero domain")

        for i, (an, a) in enumerate(attrs.items()):
            for c_n, c in a.vals.items():
                c = cast(IdxValue, c)

                doms = []
                for i in range(c.height):
                    mapping = c.get_mapping(i)
                    d = c.get_domain(i)
                    count = np.zeros((d,))

                    for j in range(len(counts[c_n])):
                        count[mapping[j]] += counts[c_n][j]

                    doms.append(np.count_nonzero(count))
                domain.append(doms)

    def group_nodes(V: Sequence[int], x: int):
        """Groups nodes in set `V` into tuples based on their group.

        This allows the `maximal_parent` algorithm to skip calculating the domain
        every time.

        Then shifts the group of `x` to be first so that `maximal_parent` can
        calculate the partial domain. The rest of the tuples are sorted based on
        length, due to the higher complexity of calculating multi-domain attributes."""
        A_dict = {}
        for c in V:
            group = groups[c]
            if group in A_dict:
                A_dict[groups[c]].append(c)
            else:
                A_dict[groups[c]] = [c]

        x_group = A_dict.pop(groups[x], None)
        A = [tuple(group) for group in A_dict.values()]
        A.sort(key=len, reverse=True)

        if x_group:
            A = (x_group, *A)
        else:
            A = tuple(A)

        return A

    #
    # Implement misc functions for summating the scores
    #
    score_cache = {}

    def pset_to_attr_sel(pset: tuple[int]) -> AttrSelectors:
        """Converts a parent set to the format required by the marginal calculation."""
        p_groups: dict[int, dict[int, int]] = {}
        for p, h in enumerate(pset):
            if h == -1:
                continue

            g = groups[p]
            if g in p_groups:
                p_groups[g][p] = h
            else:
                p_groups[g] = {p: h}

        p_attrs: AttrSelectors = {}
        for i, g in p_groups.items():
            cmn = common[next(iter(g))]
            p_attrs[group_names[i]] = AttrSelector(
                name=group_names[i],
                common=cmn,
                cols={col_names[p]: h for p, h in g.items()},
            )

        return p_attrs

    def calc_candidate_scores(candidates: list[tuple[int, bool, tuple[int]]]):
        """Calculates the mutual information approximation score for each candidate
        marginal based on `calc_fun`"""

        # Split marginals into already processed and to be processed
        scores = np.empty((len(candidates)), dtype="float")
        cached = np.zeros((len(candidates)), dtype="bool")

        requests = []
        for i, candidate in enumerate(candidates):
            if candidate in score_cache:
                scores[i] = score_cache[candidate]
                cached[i] = True
                continue

            x, partial, pset = candidate

            # Create selector for x
            x_attr = AttrSelector(group_names[groups[x]], common[x], {col_names[x]: 0})

            # Create selectors for parents by first merging into attribute groups
            p_attrs = pset_to_attr_sel(pset)
            requests.append(MarginalRequest(x_attr, p_attrs, partial))

        # Process new ones
        new_mar = np.sum(~cached)
        all_mar = len(candidates)
        if new_mar > 0:
            new_scores: list[float] = oracle.process(
                requests,
                desc=f"Calculating {new_mar}/{all_mar} ({all_mar/new_mar:.1f}x w/ cache) marginals",
                postprocess=calc_fun,
            )
        else:
            new_scores = []
            
        # Update cache
        scores[~cached] = new_scores
        for i, candidate in enumerate(candidates):
            if not cached[i]:
                score_cache[candidate] = scores[i]

        return scores

    def pick_candidate(
        candidates: list[tuple[int, bool, tuple[int]]]
    ) -> tuple[int, bool, tuple[int]]:
        """Selects a candidate based on the exponential mechanism by calculating
        all of their scores first."""
        candidates = list(candidates)
        vals = np.array(calc_candidate_scores(candidates))

        # If e1 is bigger than max_epsilon, assume it's infinite.
        if e1 is None or e1 > MAX_EPSILON:
            return candidates[np.argmax(vals)]

        # np.exp is unstable for large vals
        # subtract max (taken from original source)
        # doesn't affect probabilities
        vals -= vals.max()

        delta = (d - n_chosen) * sens_fun(n) / e1
        p = np.exp(vals / 2 / delta)
        p /= p.sum()

        choice = np.random.choice(len(candidates), size=1, p=p)[0]

        return candidates[choice]

    #
    # Implement greedy bayes (as shown in the paper)
    #
    n, d = oracle.get_shape()
    if random_init:
        x1 = np.random.randint(d)
    else:
        # Pick x1 based on entropy
        # consumes some privacy budget, but starting with a bad choice can lead
        # to a bad network.
        reqs = [
            {
                group_names[groups[x]]: AttrSelector(
                    group_names[groups[x]], 0, {col_names[x]: 0}
                )
            }
            for x in range(d)
        ]

        vals = list(
            map(
                calc_entropy,
                oracle.process(
                    reqs,
                    desc="Choosing first node based on entropy",
                ),
            )
        )
        vals = np.array(vals)
        vals -= vals.max()

        delta = d * sens_entropy(n) / (e1 if e1 is not None else 1)
        p = np.exp(vals / 2 / delta)
        p /= p.sum()

        if e1 is None or e1 > MAX_EPSILON:
            x1 = np.array(range(d))[np.argmax(p)]
        else:
            x1: int = np.random.choice(range(d), size=1, p=p)[0]

    A = [a for a in range(d) if a != x1]
    t = (n * (e2 if e2 is not None else 1)) / ((1 if unbounded_dp else 2) * d * theta)

    # Allow for "unbounded" privacy budget without destroying the computer.
    if e1 is None or e1 > MAX_EPSILON:
        logger.warning("Baking without DP (e1=inf).")
    if t > n / 10 or e2 is None or e2 > MAX_EPSILON or t > MAX_T:
        t = min(n / 10, MAX_T)
        logger.warning(
            f"Considering e2={e2} unbounded, t will be bound to min(n/10, {MAX_T:.0e})={t:.2f} for computational reasons."
        )

    V = [x1]
    V_groups = set()
    N: list[tuple[int, bool, tuple[int, ...]]] = [(cast(int, x1), False, EMPTY_PSET)]

    for _ in prange(1, d, desc="Finding Nodes: "):
        O = list()
        
        base_args = {
            "heights": heights,
            "domain": domain,
            "common": common
        } 
        per_call_args = []
        info = []

        for x in A:
            partial = groups[x] in V_groups
            Vg = group_nodes(V, x)
            new_tau = t / (domain[x][0] - (common[x] if partial else 0))

            per_call_args.append({
                "A": Vg,
                "partial": partial,
                "tau": new_tau
            })
            info.append((x, partial))
        
        node_psets = process_in_parallel(
            find_maximal_parents, per_call_args, base_args, desc="Finding Maximal Parent sets"
        )

        for (x, partial), psets in zip(info, node_psets):
            for pset in psets:
                O.append((x, partial, pset))
            if not psets:
                O.append((x, partial, EMPTY_PSET))

        node = pick_candidate(O)
        V.append(node[0])
        V_groups.add(groups[node[0]])
        A.remove(node[0])
        N.append(node)

    nodes = []
    for x, partial, pset in N:
        node = Node(
            attr=group_names[groups[x]],
            col=col_names[x],
            domain=domain_orig[x][0],
            partial=partial,
            p=pset_to_attr_sel(pset),
        )
        nodes.append(node)

    return nodes, t


def print_tree(
    attrs: Attributes,
    nodes: Nodes,
    e1: float,
    e2: float,
    theta: float,
    t: float,
):
    s = f"Bayesian Network Tree:\n"
    e1 = e1 or -1
    e2 = e2 or -1
    s += f"(PrivBayes e1={e1:.5f}, e2={e2:.5f}, theta={theta:.2f}, available t={t:.2f})"

    pset_len = 57

    tlen = len("Column Nodes")
    for _, x, _, _, _ in nodes:
        if len(x) > tlen:
            tlen = len(x)

    for name in attrs:
        if len(name) > tlen:
            tlen = len(name)

    tlen += 1

    s += f"\n┌{'─'*(tlen+1)}┬──────┬──────────┬{'─'*pset_len}┐"
    s += f"\n│{'Column Nodes'.rjust(tlen)} │  Dom │ Avail. t │ Attribute Parents{' '*(pset_len - 18)}│"
    s += f"\n├{'─'*(tlen+1)}┼──────┼──────────┼{'─'*pset_len}┤"
    for x_attr, x, domain, partial, p in nodes:
        # Show * when using a reduced marginal + correct domain
        common = attrs[x_attr].common
        if partial and common:
            dom = f"{domain-common:>4d}*"
        else:
            dom = f"{domain:>4d} "

        # Print Line start
        s += f"\n│{x.rjust(tlen)} │ {dom}│ {t/domain:>8.2f} │"

        # Print Parents
        line_str = ""
        for p_name, attr_sel in p.items():
            p_str = f" {p_name}["
            for col in attrs[p_name].vals:
                if col in attr_sel.cols:
                    p_str += str(attr_sel.cols[col])
                else:
                    p_str += "."
            p_str += "]"

            if len(p_str) + len(line_str) >= pset_len:
                s += f"{line_str:57s}│"
                s += f"\n│{' '*(tlen+1)}│      │          │"
                line_str = f" >{p_str}"
            else:
                line_str += p_str

        s += f"{line_str:57s}│"

    # Skip multi-col attr printing if there aren't any of them.
    if not any(len(attr.vals) > 1 for attr in attrs.values()):
        s += f"\n└{'─'*(tlen+1)}┴──────┴──────────┴{'─'*pset_len}┘"
        return s

    # Print mutli-column attrs
    s += f"\n├{'─'*(tlen+1)}┼──────┼──────────┴{'─'*pset_len}┤"
    s += f"\n│{'Multi-Col Attrs'.rjust(tlen)} │  Cmn │ Columns   {' '*pset_len}│"
    s += f"\n├{'─'*(tlen+1)}┼──────┼───────────{'─'*pset_len}┤"

    for name, attr in attrs.items():
        cols = attr.vals
        if len(cols) <= 1:
            continue

        s += f"\n│{name.rjust(tlen)} │ {attr.common:>4d} │"
        line_str = ""
        for i, col in enumerate(cols):
            c_str = f" {col}"

            if len(c_str) + len(line_str) >= pset_len + 11:
                s += f"{line_str:57s}│"
                s += f"\n│{' '*(tlen+1)}│     │"
                line_str = f" >{c_str}"
            else:
                line_str += c_str

        s += f"{line_str:68s}│"

    s += f"\n└{'─'*(tlen+1)}┴──────┴───────────{'─'*pset_len}┘"
    return s


def calc_noisy_marginals(
    oracle: MarginalOracle,
    attrs: Attributes,
    nodes: Nodes,
    noise_scale: float,
    skip_zero_counts: bool,
):
    """Calculates the marginals and adds laplacian noise with scale `noise_scale`."""
    requests = []
    for x_attr, x, _, partial, p in nodes:
        requests.append(
            MarginalRequest(
                AttrSelector(x_attr, attrs[x_attr].common, {x: 0}),
                p,
                partial,
            )
        )

    marginals = oracle.process(
        requests, desc="Calculating noisy marginals.", zero_fill=None
    )

    noised_marginals = []
    for (x_attr, x, _, partial, p), (marginal, _, _) in zip(nodes, marginals):
        noise = cast(
            np.ndarray, np.random.laplace(scale=noise_scale, size=marginal.shape)
        )

        if skip_zero_counts:
            # Skip adding noise to zero counts
            noise[marginal.sum(axis=1) == 0] = 0

        noised_marginal = (marginal + noise).clip(0)
        noised_marginal /= noised_marginal.sum()
        noised_marginals.append(noised_marginal)

    return noised_marginals


def sample_rows(
    attrs: Attributes, nodes: list[Node], marginals: np.ndarray, n: int
) -> pd.DataFrame:
    out = pd.DataFrame()

    attr_sampled_cols: dict[str, str] = {}
    for (x_attr, x, x_domain, partial, p), marginal in piter(
        zip(nodes, marginals), total=len(nodes), desc="Sampling values sequentially", leave=False
    ):
        if len(p) == 0:
            # No parents = use 1-way marginal
            # Concatenate m to avoid N dimensions and use lookup table to recover
            m = marginal.reshape(-1)
            common = attrs[x_attr].common if partial else 0
            out_col = np.random.choice(x_domain - common, size=n, p=m) + common

            if common:
                col = out[attr_sampled_cols[x_attr]]
                out_col[col < common] = col[col < common]

            out_col = out_col.astype(get_dtype(x_domain))
        else:
            # Use conditional probability
            # Get groups for marginal
            mul = 1
            dtype = get_dtype(marginal.shape[0] * marginal.shape[1])
            _sum_nd = np.zeros((n,), dtype=dtype)
            _tmp_nd = np.zeros((n,), dtype=dtype)
            for attr_name, attr in p.items():
                common = attr.common
                l_mul = 1
                p_partial = partial and attr_name == x_attr
                for i, (col_name, h) in enumerate(attr.cols.items()):
                    col = attrs[attr_name].vals[col_name]
                    col = cast(IdxValue, col)
                    mapping = np.array(col.get_mapping(h), dtype=dtype)
                    domain = col.get_domain(h)

                    col_lvl = mapping[out[col_name]]
                    if common != 0 and (i != 0 or p_partial):
                        col_lvl = np.where(col_lvl > common, col_lvl - common, 0)
                    np.multiply(col_lvl, mul * l_mul, out=_tmp_nd, dtype=dtype)
                    np.add(_sum_nd, _tmp_nd, out=_sum_nd, dtype=dtype)
                    l_mul *= domain - common

                if p_partial:
                    mul *= l_mul
                else:
                    mul *= l_mul + common

            # Sample groups
            out_col = np.zeros((n,), dtype=get_dtype(x_domain))
            groups = _sum_nd
            # Use reduced marginal if column has been sampled before
            # if `common=0` behavior is identical
            common = attrs[x_attr].common if partial else 0

            for group in np.unique(groups):
                m = marginal
                m_g = m[:, group]
                m_sum = m_g.sum()

                # FIXME: find sampling strategy for this
                if m_sum < 1e-6:
                    # If the sum of the group is zero, there are no samples
                    # Use the average probability of the variable
                    m_avg = marginal.sum(axis=1) / marginal.sum()
                    m_g = m_avg
                else:
                    # Otherwise normalize
                    m_g = m_g / m_sum

                size = np.count_nonzero(groups == group)
                out_col[groups == group] = (
                    np.random.choice(x_domain - common, size=size, p=m_g) + common
                )

            # Pin common values to equal the parent
            if common:
                col = out[attr_sampled_cols[x_attr]]
                out_col[col < common] = col[col < common]

        # Output column
        out[x] = out_col
        attr_sampled_cols[x_attr] = x

    return out
