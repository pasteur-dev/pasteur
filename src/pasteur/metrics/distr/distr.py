import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.special import rel_entr

from ...math import calc_marginal_1way

KL_ZERO_FILL = 1e-24


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
