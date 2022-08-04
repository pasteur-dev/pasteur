import logging
from functools import reduce

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import laplace

from ..metadata import Metadata
from ..transform import TableTransformer
from .base import Synth

logger = logging.getLogger(__name__)


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
    return np.sum(np.abs(j_mar - np.outer(x_mar, p_mar))) / 2


def greedy_bayes(
    data: pd.DataFrame,
    attr_str: list[list[str]],
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
    for a_cols in attr_str.values():
        attr.append([cols.index(col) for col in a_cols])

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
    dom = lambda a, h: reduce(
        lambda k, l: k * l, [domain[c] for c in attr[a][: height(a) - h]], 1
    )
    # Picks an item from a set, sets are ordered by default now so it's deterministic
    pick_from = lambda V: next(iter(V))

    # Sets are tuples that contain the height of each attribute in them, or -1
    # if the attribute is not in them
    # create_pset = lambda a, h: tuple(h if i == a else -1 for i in range(len(attr)))
    add_to_pset = lambda z, a, h: tuple(h if i == a else c for i, c in enumerate(z))
    empty_pset = tuple(-1 for _ in range(len(attr)))

    def maximal_parent_sets(V: set[int], t: float) -> list[tuple[int, int]]:
        """Given a set V containing hierarchical attributes (by int) and a tau
        score that is divided by the size of the domain, return a set of all
        possible combinations of attributes, such that if t > 1 there isn't an
        attribute that can be indexed in a higher level"""

        if t < 1:
            return set()
        if not V:
            return set([empty_pset])

        S = set()
        U = set()
        x = pick_from(V)
        for h in range(height(x)):
            for z in maximal_parent_sets(V - {x}, t / dom(x, h)):
                if z in U:
                    continue

                U.add(z)
                S.add(add_to_pset(z, x, h))

        for z in maximal_parent_sets(V - {x}, t):
            if z not in U:
                S.add(z)

        return S

    #
    # Implement misc functions for summating the scores
    #
    def calc_candidate_scores(candidates: list[tuple[int, tuple[int]]]):
        """Calculates the mutual information approximation score for each candidate
        marginal based on `calc_fun`"""
        candidate_scores = []
        for candidate in candidates:
            x, pset = candidate

            x_cols = attr[x]
            p_cols = []
            for p, h in enumerate(pset):
                if h == -1:
                    continue

                p_cols.extend(attr[p][: height(p) - h])

            score = calc_fun(data, domain, x_cols, p_cols)
            candidate_scores.append(score)

        return candidate_scores

    def pick_candidate(candidates: set[tuple[int, tuple[int]]]):
        """Selects a candidate based on the exponential mechanism by calculating
        all of their scores first."""
        candidates = list(candidates)
        vals = np.array(calc_candidate_scores(candidates))
        delta = (d - 1) * sens_fun(n) / e1

        p = np.exp(vals / delta)
        p /= p.sum()

        choice = np.random.choice(len(candidates), size=1, p=p)[0]
        return candidates[choice]

    #
    # Implement greedy bayes (as shown in the paper)
    #
    A = {a for a in range(len(attr))}
    x1 = pick_from(A)
    t = (n * e2) / (2 * d * theta)

    V = {x1}
    N = {(x1, empty_pset)}

    for _ in range(1, d):
        O = set()
        for x in A - V:
            psets = maximal_parent_sets(V, t / dom(x, 0))
            for pset in psets:
                O.add((x, pset))
            if not psets:
                O.add((x, empty_pset))

        node = pick_candidate(O)  # FIXME
        V.add(node[0])
        N.add(node)

    return N


def print_tree(tree: set[tuple[int, tuple[int]]], attr_names: list[str]):
    s = f"{'_Bayesian Network_':>20s}"

    for a, pset in reversed(list(tree)):
        a_name = attr_names[a]
        s += f"\n{a_name:>20s}: "

        for p, h in enumerate(pset):
            if h == -1:
                continue

            p_name = attr_names[p]
            s += f"{p_name:>15s}.{h}"

    return s


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

    def bake(
        self,
        transformers: dict[str, TableTransformer],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        assert len(data) == 1, "Only tabular data supported for now"

        self.table_name = next(iter(data.keys()))
        table = data[self.table_name]
        transformer = transformers[self.table_name]
        self.attr = transformer.get_attributes("bhr", table)

        self.nodes = greedy_bayes(
            table, self.attr, self.e1, self.e2, self.theta, self.use_r
        )

        logger.info(self)

    def fit(
        self,
        transformers: dict[str, pd.DataFrame],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        pass

    def __str__(self) -> str:
        return print_tree(self.nodes, list(self.attr.keys()))
