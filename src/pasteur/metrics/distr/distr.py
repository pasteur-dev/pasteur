import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.special import rel_entr

from functools import reduce
from ...transform import get_dtype

KL_ZERO_FILL = 1e-24


def calc_marginal_1way(
    data: np.ndarray,
    domain: np.ndarray,
    x: list[int],
    zero_fill: float | None = None,
):
    """Calculates the 1 way marginal of x, returned as a 1D array."""

    x_dom = reduce(lambda a, b: a * b, domain[x], 1)
    dtype = get_dtype(x_dom)

    idx = np.zeros((len(data)), dtype=dtype)
    tmp = np.empty((len(data)), dtype=dtype)
    mul = 1
    for col in reversed(x):
        # idx += mul*data[:, col]
        np.add(idx, np.multiply(mul, data[:, col], out=tmp), out=idx)
        mul *= domain[col]

    counts = np.bincount(idx, minlength=x_dom)
    margin = counts.astype("float")
    margin /= margin.sum()
    if zero_fill is not None:
        # Mutual info turns into NaN without this
        margin += zero_fill

    return margin.reshape(-1)


def calc_chisquare(ref: pd.DataFrame, syn: pd.DataFrame):
    cols = list(ref.keys())
    a = ref[cols].to_numpy(dtype="uint16")
    b = syn[cols].to_numpy(dtype="uint16")

    domain = np.concatenate([a, b]).max(axis=0) + 1

    # Add at least one sample prob to distr chisquare valid
    zero_fill = 1 / len(a)

    res = []
    for i, col in enumerate(cols):
        k = calc_marginal_1way(a, domain, [i], zero_fill)
        j = calc_marginal_1way(b, domain, [i], zero_fill)
        chi, p = chisquare(k, j)
        res.append([col, chi, p])

    return pd.DataFrame(res, columns=["col", "X^2", "p"])


def calc_kl(ref: pd.DataFrame, syn: pd.DataFrame):
    cols = list(ref.keys())
    a = ref[cols].to_numpy(dtype="uint16")
    b = syn[cols].to_numpy(dtype="uint16")

    domain = np.concatenate([a, b]).max(axis=0) + 1

    res = []
    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            k = calc_marginal_1way(a, domain, [i, j], KL_ZERO_FILL)
            l = calc_marginal_1way(b, domain, [i, j], KL_ZERO_FILL)

            kl = rel_entr(k, l).sum()
            kl_norm = 1 / (1 + kl)
            res.append([col_i, col_j, kl, kl_norm, len(k)])

    return pd.DataFrame(
        res,
        columns=[
            "col_i",
            "col_j",
            "kl",
            "kl_norm",
            "mlen",
        ],
    )
