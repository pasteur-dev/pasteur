import logging
from functools import reduce
from itertools import chain
from typing import NamedTuple

import numpy as np
import pandas as pd

from pasteur.transform.attribute import IdxColumn

from ..progress import piter, prange, process_in_parallel
from ..transform import Attribute, Attributes
from .base import Synth, make_deterministic
from .math import (
    AttrSelector,
    AttrSelectors,
    calc_marginal,
    calc_marginal_1way,
    expand_table,
)

logger = logging.getLogger(__name__)

MAX_EPSILON = 1e3 - 10
MAX_T = 1e5
ZERO_FILL = 1e-24


class Node(NamedTuple):
    attr: str
    col: str
    domain: int
    p: dict[str, AttrSelector]


Nodes = list[Node]


def sens_mutual_info(n: int):
    """Provides the the log2 sensitivity of the mutual information function for a given
    dataset size (n)."""
    return 2 / n * np.log2((n + 1) / 2) + (n - 1) / n * np.log2((n + 1) / (n - 1))


def calc_mutual_info(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelector,
    p: AttrSelectors,
):
    """Calculates mutual information I(X,P) for the provided data using log2."""
    j_mar, x_mar, p_mar = calc_marginal(cols, cols_noncommon, domains, x, p, ZERO_FILL)
    mi = np.sum(j_mar * np.log2(j_mar / (np.outer(x_mar, p_mar) + ZERO_FILL)))
    return mi


def sens_r_function(n: int):
    """Provides the the R function sensitivity for a given dataset size (n)."""
    return 3 / n + 2 / (n**2)


def calc_r_function(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelector,
    p: AttrSelectors,
):
    """Calculates the R(X,P) function for the provided data."""
    j_mar, x_mar, p_mar = calc_marginal(cols, cols_noncommon, domains, x, p)
    r = np.sum(np.abs(j_mar - np.outer(x_mar, p_mar))) / 2
    return r


def calc_entropy(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelectors,
):
    """Calculates the entropy for the provided data."""
    # TODO: check this is correct using scipy
    mar = calc_marginal_1way(cols, cols_noncommon, domains, x, ZERO_FILL)
    ent = -np.sum(mar * np.log2(mar))
    return ent


def sens_entropy(n: int):
    """Provides the sensitivity for the entropy function for a given dataset size (n)."""
    # TODO: Mathematically prove this is correct.
    return np.log2(n) / n - (n - 1) / n * np.log2((n - 1) / n)


def greedy_bayes(
    table: pd.DataFrame,
    attrs: Attributes,
    e1: float,
    e2: float,
    theta: float,
    use_r: bool,
    random_init: bool,
) -> tuple[Nodes, float]:
    """Performs the greedy bayes algorithm for variable domain data.

    Supports variable e1, e2, where in the paper they are defined as
    `e1 = b * e` and `e2 = (1 - b) * e`, variable theta, and both
    mutual information and R functions.

    Binary domains are not supported due to computational intractability."""

    n, d = table.shape
    n_chosen = 1 if random_init else 0
    calc_fun = calc_r_function if use_r else calc_mutual_info
    sens_fun = sens_r_function if use_r else sens_mutual_info

    #
    # Set up maximal parents algorithm as shown in paper
    # (recursive implementation is a bit slow)
    # (wrapped in a closure due to fancy syntactic sugar lambdas)
    #
    col_names = []
    groups = []
    group_names = []
    heights = []
    common = []
    domain = []
    for i, (an, a) in enumerate(attrs.items()):
        group_names.append(an)
        for n, c in a.cols.items():
            col_names.append(n)
            groups.append(i)
            heights.append(c.lvl.height)
            common.append(a.common)
            domain.append([c.get_domain(h) for h in range(c.height)])

    empty_pset = tuple(-1 for _ in range(d))

    def add_to_pset(s, x, h):
        s = list(s)
        s[x] = h
        return tuple(s)

    def maximal_parents(
        V: tuple[int], tau: float, P: dict[int, dict[int, int]] = {}
    ) -> list[tuple[int]]:
        """Given a set V containing hierarchical attributes (by int) and a tau
        score that is divided by the size of the domain, return a set of all
        possible combinations of attributes, such that if t > 1 there isn't an
        attribute that can be indexed in a higher level"""

        # Calculate domain manually to correct it
        dom = 1
        for g, a in P.items():
            l_dom = 1
            for x, h in a.items():
                cmn = common[x]  # same for x of a
                l_dom *= domain[x][h] - cmn
            dom *= l_dom + cmn

        if tau / dom < 1:
            return []
        if not V:
            return [empty_pset]

        x = V[0]
        V = V[1:]

        S = []
        U = set()

        # Create copy of P for the version with x that has
        # a dict for x's group. Avoid deep copying
        P_x = P.copy()
        g = groups[x]
        gx = P.get(g, {}).copy()
        P_x[g] = gx

        for h in range(heights[x]):
            # Only changes local copy
            P_x[g][x] = h
            for z in maximal_parents(V, tau, P_x):
                if z not in U:
                    U.add(z)
                    S.append(add_to_pset(z, x, h))

        for z in maximal_parents(V, tau, P):
            if z not in U:
                S.append(z)

        return S

    #
    # Implement misc functions for summating the scores
    #
    cols, cols_noncommon, domains = expand_table(attrs, table)
    score_cache = {}

    def pset_to_attr_sel(pset: tuple[int]) -> AttrSelectors:
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
                common=cmn, cols={col_names[p]: h for p, h in g.items()}
            )

        return p_attrs

    def calc_candidate_scores(candidates: list[tuple[int, tuple[int]]]):
        """Calculates the mutual information approximation score for each candidate
        marginal based on `calc_fun`"""

        # Split marginals into already processed and to be processed
        scores = np.empty((len(candidates)), dtype="float")
        cached = np.zeros((len(candidates)), dtype="bool")

        to_be_processed = []
        for i, candidate in enumerate(candidates):
            if candidate in score_cache:
                scores[i] = score_cache[candidate]
                cached[i] = True
                continue

            x, pset = candidate

            # Create selector for x
            x_attr = AttrSelector(common[x], {col_names[x]: 0})

            # Create selectors for parents by first merging into attribute groups
            p_attrs = pset_to_attr_sel(pset)
            to_be_processed.append({"x": x_attr, "p": p_attrs})

        # Process new ones
        base_args = {"cols": cols, "cols_noncommon": cols_noncommon, "domains": domains}
        new_mar = np.sum(~cached)
        all_mar = len(candidates)
        new_scores = process_in_parallel(
            calc_fun,
            to_be_processed,
            base_args,
            500,
            f"Calculating {new_mar}/{all_mar} ({all_mar/new_mar:.1f}x w/ cache) marginals",
        )

        # Update cache
        scores[~cached] = new_scores
        for i, candidate in enumerate(candidates):
            if not cached[i]:
                score_cache[candidate] = scores[i]

        return scores

    def pick_candidate(
        candidates: list[tuple[int, tuple[int]]]
    ) -> tuple[int, tuple[int]]:
        """Selects a candidate based on the exponential mechanism by calculating
        all of their scores first."""
        candidates = list(candidates)
        vals = np.array(calc_candidate_scores(candidates))

        # If e1 is bigger than max_epsilon, assume it's infinite.
        if np.isinf(e1) or e1 > MAX_EPSILON:
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
    n, d = table.shape
    if random_init:
        x1 = np.random.randint(d)
    else:
        # Pick x1 based on entropy
        # consumes some privacy budget, but starting with a bad choice can lead
        # to a bad network.
        vals = [
            calc_entropy(
                cols,
                cols_noncommon,
                domains,
                {group_names[groups[x]]: AttrSelector(0, {col_names[x]: 0})},
            )
            for x in range(d)
        ]
        vals = np.array(vals)
        vals -= vals.max()

        delta = d * sens_entropy(n) / e1
        p = np.exp(vals / 2 / delta)
        p /= p.sum()

        if e1 > MAX_EPSILON:
            x1 = np.argmax(p)
        else:
            x1 = np.random.choice(range(d), size=1, p=p)[0]

    A = [a for a in range(d) if a != x1]
    t = (n * e2) / (2 * d * theta)

    # Allow for "unbounded" privacy budget without destroying the computer.
    if e1 > MAX_EPSILON:
        logger.warn("Baking without DP (e1=inf).")
    if t > n / 10 or e2 > MAX_EPSILON or t > MAX_T:
        t = min(n / 10, MAX_T)
        logger.warn(
            f"Considering e2={e2} unbounded, t will be bound to min(n/10, {MAX_T:.0e})={t:.2f} for computational reasons."
        )

    V = [x1]
    N = [(x1, empty_pset)]

    for _ in prange(1, d, desc="Finding Nodes: "):
        O = list()

        for x in piter(A, leave=False, desc="Finding Maximal Parent sets: "):
            psets = maximal_parents(V, t / domain[x][0])
            for pset in psets:
                O.append((x, pset))
            if not psets:
                O.append((x, empty_pset))

        node = pick_candidate(O)
        V.append(node[0])
        A.remove(node[0])
        N.append(node)

    nodes = []
    for x, pset in N:
        node = Node(
            attr=group_names[groups[x]],
            col=col_names[x],
            domain=domain[x][0],
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
    s += f"(PrivBayes e1={e1:.2f}, e2={e2:.2f}, theta={theta:.2f}, available t={t:.2f})"

    pset_len = 57

    s += f"\n┌{'─'*21}┬─────┬──────────┬{'─'*pset_len}┐"
    s += f"\n│{'Attribute':>20s} │ Dom │ Avail. t │ Parents{' '*(pset_len - 8)}│"
    s += f"\n├{'─'*21}┼─────┼──────────┼{'─'*pset_len}┤"
    for _, x, domain, p in nodes:
        s += f"\n│{x:>20s} │ {domain:>3d} │ {t/domain:>8.2f} │"

        line_str = ""
        for p_name, attr_sel in p.items():
            p_str = f" {p_name}."
            for col in attrs[p_name].cols:
                if col in attr_sel.cols:
                    p_str += str(attr_sel.cols[col])
                else:
                    p_str += "_"

            if len(p_str) + len(line_str) >= pset_len:
                s += f"{line_str:57s}│"
                s += f"\n│{' '*21}│     │          │"
                line_str = f" >{p_str}"
            else:
                line_str += p_str

        s += f"{line_str:57s}│"

    s += f"\n└{'─'*21}┴─────┴──────────┴{'─'*pset_len}┘"
    return s


def calc_noisy_marginals(
    attrs: Attributes, table: pd.DataFrame, nodes: Nodes, noise_scale: float
):
    """Calculates the marginals and adds laplacian noise with scale `noise_scale`."""
    cols, cols_noncommon, domains = expand_table(attrs, table)

    marginals = []
    for _, x, x_dom, p in nodes:
        xp = [x] + list(p.values())

        # Find integer dtype based on domain
        p_dom = 1
        for attr in p.values():
            for i, (n, h) in enumerate(attr.cols.items()):
                p_dom *= domains[n][h] - (attr.common if i > 0 else 0)

        dtype = "uint32"

        n, d = table.shape
        _sum_nd = np.zeros((n,), dtype=dtype)
        _tmp_nd = np.zeros((n,), dtype=dtype)

        mul = 1
        for attr in reversed(p.values()):
            common = attr.common
            l_mul = 1
            for i, (n, h) in enumerate(attr.cols.items()):
                if i == 0 or common == 0:
                    np.multiply(cols[n][h], mul * l_mul, out=_tmp_nd)
                else:
                    np.multiply(cols_noncommon[n][h], mul * l_mul, out=_tmp_nd)

                np.add(_sum_nd, _tmp_nd, out=_sum_nd)
                l_mul *= domains[n][h] - common
            mul *= l_mul + common

        np.multiply(cols[x][0], mul, out=_tmp_nd)
        np.add(_sum_nd, _tmp_nd, out=_sum_nd)

        counts = np.bincount(_sum_nd, minlength=p_dom * x_dom)
        margin = counts.reshape(x_dom, p_dom).astype("float32")
        noise = np.random.laplace(scale=noise_scale, size=margin.shape)

        marginal = (margin + noise).clip(0)
        marginal /= marginal.sum()
        marginals.append(marginal)

    return marginals


def sample_rows(nodes: list[Node], marginals: np.array, n: int) -> pd.DataFrame:
    dtype = "uint32"
    out = pd.DataFrame(dtype=dtype)

    for (x, p), marginal in zip(nodes, marginals):
        x_cols = x.cols[: len(x.cols) - x.h]
        p_cols = list(chain.from_iterable(a.cols[: len(a.cols) - a.h] for a in p))

        domain = {name: marginal.shape[i] for i, name in enumerate(p_cols + x_cols)}
        domain_p = np.product(marginal.shape[: len(p_cols)])
        domain_x = np.product(marginal.shape[-len(x_cols) :])

        # Create lookup tables where for each column there's
        # idx -> val, where idx in (0, domain_x)
        nums = np.arange(domain_x)
        luts = {}
        for name in reversed(x_cols):
            luts[name] = nums % domain[name]
            nums //= domain[name]

        if len(p) == 0:
            # No parents = use 1-way marginal
            # Concatenate m to avoid N dimensions and use lookup table to recover
            m = marginal.reshape(-1)
            idx = np.random.choice(domain_x, size=n, p=m)
        else:
            # Use conditional probability
            m = marginal.reshape((domain_p, domain_x))
            m = m / m.sum(axis=1, keepdims=True)

            # Some marginal groups were never sampled in the original data
            # However, noise from DP might lead to some of them being sampled
            # Store those groups and sample them uniformly when that happens.
            blacklist = np.any(np.isnan(m), axis=1)

            p_idx = np.zeros(n, dtype=dtype)
            for name in p_cols:
                p_idx *= np.uint16(domain[name])
                p_idx += out[name].to_numpy(dtype=dtype)

            # Apply conditional probability by group
            # groups are proportional to dom(P) = vectorized
            idx = np.empty(len(p_idx), dtype=dtype)
            for group in np.unique(p_idx):
                size = np.sum(p_idx == group)
                idx[p_idx == group] = np.random.choice(
                    domain_x, size=size, p=m[group, :] if not blacklist[group] else None
                )

        # Place columns in Dataframe using luts
        for name in x_cols:
            out[name] = luts[name][idx]

    return out


class PrivBayesSynth(Synth):
    name = "privbayes"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False

    def __init__(
        self,
        e1: float = 0.3,
        e2: float = 0.7,
        theta: float = 4,
        use_r: bool = True,
        seed: float | None = None,
        random_init: bool = False,
        **_,
    ) -> None:
        super().__init__()
        self.e1 = e1
        self.e2 = e2
        self.theta = theta
        self.use_r = use_r
        self.seed = seed
        self.random_init = random_init

    @make_deterministic
    def bake(
        self,
        attrs: dict[str, Attributes],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        assert len(data) == 1, "Only tabular data supported for now"

        table_name = next(iter(data.keys()))
        table = data[table_name]
        attrs = attrs[table_name]

        # Fit network
        nodes, t = greedy_bayes(
            table, attrs, self.e1, self.e2, self.theta, self.use_r, self.random_init
        )

        # Nodes are a tuple of a x attribute
        self.table_name = table_name
        self.d = len(table.keys())
        self.t = t
        self.attrs = attrs
        self.nodes = nodes
        logger.info(self)

    @make_deterministic
    def fit(
        self,
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        table = data[self.table_name]
        self.n = len(table)
        noise = 2 * self.d / (self.n * self.e2)
        if self.e2 > MAX_EPSILON:
            logger.warning(f"Considering e2={self.e2} unbounded, sampling without DP.")
            noise = 0
        self.marginals = calc_noisy_marginals(self.attrs, table, self.nodes, noise)

    @make_deterministic
    def sample(
        self, n: int = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        data = {
            self.table_name: sample_rows(
                self.nodes, self.marginals, self.n if n is None else n
            )
        }
        ids = {self.table_name: pd.DataFrame()}

        return data, ids

    def __str__(self) -> str:
        return print_tree(
            self.attrs,
            self.nodes,
            self.e1,
            self.e2,
            self.theta,
            self.t,
        )
