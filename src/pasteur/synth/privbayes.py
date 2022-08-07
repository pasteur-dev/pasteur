from itertools import chain
import logging
from functools import reduce
from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from pasteur.transform.table import Attribute

from .base import Synth, make_deterministic, process_in_parallel

logger = logging.getLogger(__name__)


class Node(NamedTuple):
    x: Attribute = None
    p: list[Attribute] = []


def calc_marginal(
    data: np.ndarray,
    domain: np.ndarray,
    x: list[int],
    p: list[int],
    rm_zeros: bool = False,
):
    """Calculates the 1 way and 2 way marginals between the set of columns in x
    and the set of columns in p."""

    sub_data = data[:, x + p]
    sub_domain = domain[x + p]
    margin, _ = np.histogramdd(sub_data, sub_domain)
    margin /= margin.sum()
    if rm_zeros:
        # Mutual info turns into NaN without this
        margin += 1e-24

    x_idx = tuple(range(len(x)))
    p_idx = tuple(range(-len(p), 0))

    x_mar = np.sum(margin, axis=p_idx).reshape(-1)
    p_mar = np.sum(margin, axis=x_idx).reshape(-1)
    j_mar = margin.reshape((len(x_mar), len(p_mar)))

    return j_mar, x_mar, p_mar


def sens_mutual_info(n: float):
    """Provides the the log2 sensitivity of the mutual information function for a given
    dataset size (n)."""
    return 2 / n * np.log2((n + 1) / 2) + (n - 1) / n * np.log2((n + 1) / (n - 1))


def calc_mutual_info(data: np.ndarray, domain: np.ndarray, x: list[int], p: list[int]):
    """Calculates mutual information I(X,P) for the provided data using log2."""

    j_mar, x_mar, p_mar = calc_marginal(data, domain, x, p, rm_zeros=True)
    return np.sum(j_mar * np.log2(j_mar / np.outer(x_mar, p_mar)))


def sens_r_function(n: float):
    """Provides the the R function sensitivity for a given dataset size (n)."""
    return 3 / n + 2 / (n**2)


def calc_r_function(data: np.ndarray, domain: np.ndarray, x: list[int], p: list[int]):
    """Calculates the R(X,P) function for the provided data."""

    j_mar, x_mar, p_mar = calc_marginal(data, domain, x, p)
    r = np.sum(np.abs(j_mar - np.outer(x_mar, p_mar))) / 2
    return r


def greedy_bayes(
    data: pd.DataFrame,
    attr_str: dict[str, Attribute],
    e1: float,
    e2: float,
    theta: float,
    use_r: bool,
):
    """Performs the greedy bayes algorithm for variable domain data.

    Supports variable e1, e2, where in the paper they are defined as
    `e1 = b * e` and `e2 = (1 - b) * e`, variable theta, and both
    mutual information and R functions.

    Binary domains are not supported due to computational intractability."""

    # Keep string names based on their index on a list.
    cols = list(data.columns)
    attr = []
    has_na = []
    for a in attr_str.values():
        has_na.append(a.has_na)
        attr.append([cols.index(col) for col in a.cols])

    # 30k is a sweet spot for categorical variables
    # Dropping pandas for a 5x in speed when calculating marginals
    data = data.to_numpy(dtype="uint16")
    domain = data.max(axis=0) + 1

    n = len(data)
    d = len(attr)

    calc_fun = calc_r_function if use_r else calc_mutual_info
    sens_fun = sens_r_function if use_r else sens_mutual_info

    #
    # Set up maximal parents algorithm as shown in paper
    # (recursive implementation is a bit slow)
    # (wrapped in a closure due to fancy syntactic sugar lambdas)
    #

    # Returns the height of a hierarchical attribute
    height = lambda a: len(attr[a])

    # Returns the domain of a hierarchical attribute at height h (h=0 is max)
    # TODO: check NA check is correct
    # Dom of a value is the product of its columns
    dom_no_na = lambda a, h: reduce(
        lambda k, l: k * l, [domain[c] for c in attr[a][: height(a) - h]], 1
    )
    # If theres an NA column it will be the first one, so it's skipped and 1 is added.
    dom_has_na = (
        lambda a, h: reduce(
            lambda k, l: k * l, [domain[c] for c in attr[a][1 : height(a) - h]], 1
        )
        + 1
    )
    # The only place to correct for an NA hierarchical value in the value function is here
    # Mutual information function should stay the same, the invalid values that have 0
    # don't affect entropy. Other than that, not adding noise to invalid marginals
    dom = lambda a, h: dom_has_na(a, h) if has_na[a] else dom_no_na(a, h)

    # Sets are tuples that contain the height of each attribute in them, or -1
    # if the attribute is not in them
    # create_pset = lambda a, h: tuple(h if i == a else -1 for i in range(len(attr)))
    add_to_pset = lambda z, a, h: tuple(h if i == a else c for i, c in enumerate(z))
    empty_pset = tuple(-1 for _ in range(len(attr)))

    def maximal_parent_sets(V: list[int], t: float) -> list[tuple[int, int]]:
        """Given a set V containing hierarchical attributes (by int) and a tau
        score that is divided by the size of the domain, return a set of all
        possible combinations of attributes, such that if t > 1 there isn't an
        attribute that can be indexed in a higher level"""

        if t < 1:
            return []
        if not V:
            return [empty_pset]

        S = []
        U = set()
        x = V[0]
        for h in range(height(x)):
            for z in maximal_parent_sets(V[1:], t / dom(x, h)):
                if z in U:
                    continue

                U.add(z)
                S.append(add_to_pset(z, x, h))

        for z in maximal_parent_sets(V[1:], t):
            if z not in U:
                S.append(z)

        return S

    #
    # Implement misc functions for summating the scores
    #
    score_cache = {}

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

            x_cols = attr[x]
            p_cols = []
            for p, h in enumerate(pset):
                if h == -1:
                    continue

                p_cols.extend(attr[p][: height(p) - h])

            to_be_processed.append({"x": x_cols, "p": p_cols})

        # Process new ones
        base_args = {"domain": domain, "data": data}
        new_mar = np.sum(~cached)
        all_mar = len(candidates)
        new_scores = process_in_parallel(
            calc_fun,
            to_be_processed,
            base_args,
            f"Calculating {new_mar}/{all_mar} ({all_mar/new_mar:.1f}x w/ cache) marginals",
        )

        # Update cache
        scores[~cached] = new_scores
        for i, candidate in enumerate(candidates):
            if not cached[i]:
                score_cache[candidate] = scores[i]

        return scores

    def pick_candidate(candidates: list[tuple[int, tuple[int]]]):
        """Selects a candidate based on the exponential mechanism by calculating
        all of their scores first."""
        candidates = list(candidates)
        vals = np.array(calc_candidate_scores(candidates))

        # If e1 is bigger than 1e3, assume it's infinite.
        if np.isinf(e1) or e1 > 1e3:
            return candidates[np.argmax(vals)]

        # np.exp is unstable for large vals
        # subtract max (taken from original source)
        # doesn't affect probabilities
        vals -= vals.max()

        delta = (d - 1) * sens_fun(n) / e1
        p = np.exp(vals / 2 / delta)
        p /= p.sum()

        choice = np.random.choice(len(candidates), size=1, p=p)[0]

        return candidates[choice]

    #
    # Implement greedy bayes (as shown in the paper)
    #
    x1 = np.random.randint(len(attr))
    A = [a for a in range(0, len(attr)) if a != x1]
    t = (n * e2) / (2 * d * theta)

    V = [x1]
    N = [(x1, empty_pset)]

    for _ in trange(1, d, desc="Finding Nodes: "):
        O = list()

        for x in tqdm(A, leave=False, desc="Finding Maximal Parent sets: "):
            psets = maximal_parent_sets(V, t / dom(x, 0))
            for pset in psets:
                O.append((x, pset))
            if not psets:
                O.append((x, empty_pset))

        node = pick_candidate(O)
        V.append(node[0])
        A.remove(node[0])
        N.append(node)

    domain = [dom(x, 0) for x in range(len(attr))]

    return N, domain, t


def print_tree(
    tree: list[tuple[int, tuple[int]]],
    domain: list[int],
    attr_names: list[str],
    e1: float,
    e2: float,
    theta: float,
    t: float,
):
    s = f"Bayesian Network Tree:\n"
    s += f"(PrivBayes e1={e1:.2f}, e2={e2:.2f}, theta={theta:.2f}, available t={t:.2f})"

    pset_len = 70

    s += f"\n┌{'─'*21}┬─────┬──────────┬{'─'*pset_len}┐"
    s += f"\n│{'Attribute':>20s} │ Dom │ Avail. t │{' '*pset_len}│"
    s += f"\n├{'─'*21}┼─────┼──────────┼{'─'*pset_len}┤"
    for a, pset in list(tree):
        a_name = attr_names[a]
        s += f"\n│{a_name:>20s} │ {domain[a]:>3d} │ {t/domain[a]:>8.2f} │"

        p_str = ""
        for p, h in enumerate(pset):
            if h == -1:
                continue

            p_name = attr_names[p]
            p_str += f"{p_name:>15s}.{h}"

        s += f"{p_str:70s}│"

    s += f"\n└{'─'*21}┴─────┴──────────┴{'─'*pset_len}┘"
    return s


def calc_noisy_marginals(data: pd.DataFrame, nodes: list[Node], noise_scale: float):
    """Calculates the marginals and adds laplacian noise with scale `noise_scale`."""

    marginals = []
    for x, p in nodes:
        x_cols = x.cols[: len(x.cols) - x.h]
        p_cols = list(chain.from_iterable(a.cols[: len(a.cols) - a.h] for a in p))
        cols = p_cols + x_cols

        sub_data = data[cols].to_numpy(dtype="int16")
        sub_domain = sub_data.max(axis=0) + 1

        margin, _ = np.histogramdd(sub_data, sub_domain)
        noise = np.random.laplace(scale=noise_scale, size=margin.shape)

        # Consider a0.col0 being a NA indicatior. When a0.col0=1 then a0 is NA.
        # This counts as 1 value in the domain, however it doubles the size of
        # the marginal. If we apply noise uniformely to the marginal then half of
        # it will go to the NA value, skewing its probability.

        # The following loop removes all noise from the marginals a0.col0=1,
        # except for the marginal a0.col0=1, a0.colN=0, such as that the NA is
        # also DP protected.
        for a in [x] + p:
            if not a.has_na:
                continue

            na_col = a.cols[0]

            # Find noise of NA
            na_idx = []
            for col in cols:
                if col == na_col:
                    na_idx.append(1)
                elif col in a.cols:
                    na_idx.append(0)
                else:
                    na_idx.append(slice(None))
            na_idx = tuple(na_idx)
            na_noise = noise[na_idx]

            # Zero out all other noise for na
            idx = []
            for col in cols:
                if col == na_col:
                    idx.append(1)
                else:
                    idx.append(slice(None))
            idx = tuple(idx)
            noise[idx] = 0

            # Replace noise again
            noise[na_idx] = na_noise

        marginal = (margin + noise).clip(0)
        marginal /= marginal.sum()
        marginals.append(marginal)

    return marginals


def sample_rows(nodes: list[Node], marginals: np.array, n: int) -> pd.DataFrame:
    out = pd.DataFrame(dtype="int16")

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

            p_idx = np.zeros(n, dtype="int16")
            for name in p_cols:
                p_idx *= domain[name]
                p_idx += out[name].to_numpy()

            # Apply conditional probability by group
            # groups are proportional to dom(P) = vectorized
            idx = np.empty(len(p_idx), dtype="int16")
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
    type = "bhr"
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
        **_,
    ) -> None:
        super().__init__()
        self.e1 = e1
        self.e2 = e2
        self.theta = theta
        self.use_r = use_r
        self.seed = seed

    @make_deterministic
    def bake(
        self,
        attrs: dict[str, dict[str, Attribute]],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        assert len(data) == 1, "Only tabular data supported for now"

        table_name = next(iter(data.keys()))
        table = data[table_name]
        attr = attrs[table_name]
        attr_names = list(attr.keys())

        # Fit network
        nodes_raw, domain_raw, t = greedy_bayes(
            table, attr, self.e1, self.e2, self.theta, self.use_r
        )

        # Create tuples based on attributes
        # A node is composed of an x atribute and a set of
        nodes: list[Node] = []
        for a, pset in nodes_raw:
            x_attr = attr[attr_names[a]]
            p_attrs = []

            for p, h in enumerate(pset):
                if h != -1:
                    p_attrs.append(attr[attr_names[p]]._replace(h=h))

            nodes.append(Node(x_attr, p_attrs))

        # Nodes are a tuple of a x attribute
        self.table_name = table_name
        self.d = len(table.keys())
        self.t = t
        self.attr = attr
        self.attr_names = attr_names
        self.nodes_raw = nodes_raw
        self.domain_raw = domain_raw
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
        self.marginals = calc_noisy_marginals(table, self.nodes, noise)

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
            self.nodes_raw,
            self.domain_raw,
            self.attr_names,
            self.e1,
            self.e2,
            self.theta,
            self.t,
        )
