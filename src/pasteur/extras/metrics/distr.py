from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import chisquare

from ...attribute import Attributes, IdxValue, get_dtype
from ...metric import TableMetric

if TYPE_CHECKING:
    from ...metadata import Metadata

KL_ZERO_FILL = 1e-24
FONT_SIZE = "13px"


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
        margin /= margin.sum()

    return margin.reshape(-1)


class ChiSquareMetric(TableMetric[dict[str, np.ndarray]]):
    name = "cs"
    encodings = ["idx"]

    def fit(
        self,
        table: str,
        meta: Metadata,
        attrs: dict[str, dict[str, Attributes]],
        tables: dict[str, dict[str, pd.DataFrame]],
        ids: pd.DataFrame | None = None,
    ):
        self.table = table
        table_attrs = attrs["idx"][table]

        self.domain = {}
        for attr in table_attrs.values():
            for name, val in attr.vals.items():
                assert isinstance(val, IdxValue)
                self.domain[name] = val.domain

    def process(
        self,
        tables: dict[str, dict[str, pd.DataFrame]],
        ids: pd.DataFrame | None = None,
    ):
        table = tables["idx"][self.table]
        domain = np.array(list(self.domain.values()))
        a = table[list(self.domain)].to_numpy(dtype="uint16")

        # Add at least one sample prob to distr chisquare valid
        zero_fill = 1 / len(a)

        res = {}
        for i, name in enumerate(self.domain):
            res[name] = calc_marginal_1way(a, domain, [i], zero_fill)

        return res

    def visualise(
        self,
        data: dict[str, np.ndarray],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):
        import mlflow

        from ...utils.mlflow import color_dataframe, gen_html_table

        splits = data.copy()
        wrk = splits.pop(wrk_set)

        results = {}
        for name, split in splits.items():
            res = []
            for col in self.domain:
                chi, p = chisquare(wrk[col], split[col])
                res.append([col, chi, p])

            results[name] = pd.DataFrame(res, columns=["col", "X^2", "p"])

        cs_formatters = {
            "X^2": {"precision": 3},
            "p": {"formatter": lambda x: f"{100*x:.1f}"},
        }
        style = color_dataframe(
            results,
            idx=["col"],
            cols=[],
            vals=["X^2", "p"],
            formatters=cs_formatters,
            split_ref=ref_set,
        )

        fn = (
            f"distr/cs.html" if self.table == "table" else f"distr/{self.table}_cs.html"
        )
        mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)


class KullbackLeiblerMetric(TableMetric[dict[str, np.ndarray]]):
    name = "kl"
    encodings = ["idx"]

    def fit(
        self,
        table: str,
        meta: Metadata,
        attrs: dict[str, dict[str, Attributes]],
        tables: dict[str, dict[str, pd.DataFrame]],
        ids: pd.DataFrame | None = None,
    ):
        self.table = table
        table_attrs = attrs["idx"][table]

        self.domain = {}
        for attr in table_attrs.values():
            for name, val in attr.vals.items():
                assert isinstance(val, IdxValue)
                self.domain[name] = val.domain

    def process(
        self,
        tables: dict[str, dict[str, pd.DataFrame]],
        ids: pd.DataFrame | None = None,
    ):
        table = tables["idx"][self.table]
        a = table[list(self.domain)].to_numpy(dtype="uint16")
        domain = np.array(list(self.domain.values()))

        res = {}
        for i, col_i in enumerate(self.domain):
            for j, col_j in enumerate(self.domain):
                res[(col_i, col_j)] = calc_marginal_1way(
                    a, domain, [i, j], KL_ZERO_FILL
                )

        return res

    def visualise(
        self,
        data: dict[str, np.ndarray],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):
        import mlflow

        from ...utils.mlflow import color_dataframe, gen_html_table

        splits = data.copy()
        wrk = splits.pop(wrk_set)

        results = {}
        for name, split in splits.items():
            res = []
            for key in split:
                col_i, col_j = key
                k = wrk[key]
                j = split[key]

                kl = rel_entr(k, j).sum()
                kl_norm = 1 / (1 + kl)
                res.append([col_i, col_j, kl, kl_norm, len(k)])

            results[name] = pd.DataFrame(
                res,
                columns=[
                    "col_i",
                    "col_j",
                    "kl",
                    "kl_norm",
                    "mlen",
                ],
            )

        kl_formatters = {"kl_norm": {"precision": 3}}
        style = color_dataframe(
            results,
            idx=["col_j"],
            cols=["col_i"],
            vals=["kl_norm"],
            formatters=kl_formatters,
            split_ref=ref_set,
        )

        fn = (
            f"distr/kl.html" if self.table == "table" else f"distr/{self.table}_kl.html"
        )
        mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)
