from functools import reduce

import numpy as np


def get_dtype(domain: int):
    # uint16 is 2x as fast as uint32 (5ms -> 3ms), use with marginals.
    # Marginal domain can not exceed max(uint16) size 65535 + 1
    if domain < 65535 + 1:
        return "uint16"
    return "uint32"


def calc_marginal(
    data: np.ndarray,
    domain: np.ndarray,
    x: list[int],
    p: list[int],
    zero_fill: float | None = None,
):
    """Calculates the 1 way and 2 way marginals between the set of columns in x
    and the set of columns in p."""

    xp = x + p
    x_dom = reduce(lambda a, b: a * b, domain[x], 1)
    p_dom = reduce(lambda a, b: a * b, domain[p], 1)
    dtype = get_dtype(x_dom * p_dom)

    idx = np.zeros((len(data)), dtype=dtype)
    tmp = np.empty((len(data)), dtype=dtype)
    mul = 1
    for col in reversed(xp):
        # idx += mul*data[:, col]
        np.add(idx, np.multiply(mul, data[:, col].astype(dtype), out=tmp), out=idx)
        mul *= domain[col]

    counts = np.bincount(idx, minlength=x_dom * p_dom)
    margin = counts.reshape(x_dom, p_dom).astype("float")

    margin /= margin.sum()
    if zero_fill is not None:
        # Mutual info turns into NaN without this
        margin += zero_fill

    j_mar = margin
    x_mar = np.sum(margin, axis=1)
    p_mar = np.sum(margin, axis=0)

    return j_mar, x_mar, p_mar


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
