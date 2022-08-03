import logging
import random

import numpy as np
import pandas as pd
from scipy.stats import laplace

from ..metadata import Metadata
from .base import Synth

logger = logging.getLogger(__name__)


def calc_k(d: int, n: int, e2: float = 0.1, noise_ratio: float = 4):
    """Calculate the maximum degree k for a bayesian network with the given input
    that achieves the specified noise ratio (or usefulness).

    See PrivBayes Lemma 3. If there's no k that meets that requirement, returns k=0."""

    MAX_K = 40
    k_m = np.array(range(min(d, MAX_K)), dtype="float")
    usefulness_m = n * e2 / ((d - k_m) * (2 ** (k_m + 2)))

    candidates = usefulness_m > noise_ratio

    # (d - k_m) * (2 ** (k_m + 2)) is upwards monotonic, so we pick the last k
    # that's positive, or 0, if none of them have the required usefulness
    if not np.any(candidates):
        logger.warning(
            f"Target usefulness not reached, expected {noise_ratio: .2f} and got {usefulness_m[0]: .2f} for k=0"
        )
        return 0

    # Return the maximum k which has at least target_usefulness
    # k = idx
    k = np.argwhere(candidates)[-1, 0]
    logger.info(
        f"Selected k={k} with usefulness={usefulness_m[k]: .2f} > {noise_ratio: .2f}"
    )
    return k


def calc_mutual_information(data: pd.DataFrame, x: str, p: list[str]):
    """Calculates the mutual information score using log2.

    Therefore, the score returned is measured in bits.
    For a binary x or p, the maximum is 1."""

    joint_dist = data.groupby([x] + p).size()
    contigency_pd = pd.pivot(pd.DataFrame(joint_dist).reset_index(), p, x)

    contigency = contigency_pd.to_numpy(dtype="float")
    cg = contigency / contigency.sum()

    # Marginals
    x_mar = contigency.sum(axis=0)
    x_mar /= x_mar.sum()
    p_mar = contigency.sum(axis=1)
    p_mar /= p_mar.sum()

    return np.sum(cg * np.log2(cg / np.outer(p_mar, x_mar)))


def calc_sensitivity_bin(n: int | float):
    """The binary sensitivity according to Lemma 4.1 of PrivBayes.

    log2 is used to calculate it, so it is only valid when used with the mutual
    information function of this package."""
    n = float(n)
    return 1 / n * np.log2((n + 1) / 2) + (n - 1) / n * np.log2((n + 1) / (n - 1))


def calc_noisy_marginals(data: pd.DataFrame, cols: list[str], noise_scale: float):
    marginal = data[cols].groupby(cols).size()
    noise = laplace.rvs(loc=0, scale=noise_scale, size=marginal.shape)

    noisy_marginal = marginal - noise
    noisy_marginal = noisy_marginal.clip(0)


def greedy_bayes(data: pd.DataFrame, k: int):
    N = {}
    V = set()
    A = set(data.keys())


class BinPrivBayesSynth(Synth):
    name = "bin_privbayes"
    type = "bin"
    tabular = True
    multimodal = False
    timeseries = False

    DEFAULT_ARGS = {"e": 1, "beta": 0.1, "noise_ratio": 4}

    def fit(self, meta: dict, data: dict[str, pd.DataFrame]):
        assert len(data) == 1, "Only one table supported"
        name = list(data.keys())[0]
        table = data[name]
        self.meta = Metadata(meta, data).get_table(name)

        # FIXME: Add way of injecting args
        args = self.DEFAULT_ARGS
        e, beta, noise_ratio = args["e"], args["beta"], args["noise_ratio"]
        e1 = beta * e
        e2 = (1 - beta) * e

        n, d = table.shape
        k = calc_k(d, n, e2, noise_ratio)

        self.noise_scale = 2 * (d - k) / e2
        self.k = k

    def sample(self) -> dict[str, pd.DataFrame]:
        assert False, "Not implemented"
