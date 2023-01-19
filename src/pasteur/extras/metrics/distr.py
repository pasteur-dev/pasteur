from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import chisquare

from ...attribute import Attributes, IdxValue, get_dtype
from ...metric import Summaries, TableData, TableMetric
from ...utils.progress import process_in_parallel

if TYPE_CHECKING:
    from ...metadata import Metadata

KL_ZERO_FILL = 1e-24
FONT_SIZE = "13px"

logger = logging.getLogger(__name__)

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


class ChiSquareMetric(
    TableMetric[Summaries[dict[str, np.ndarray]], Summaries[dict[str, np.ndarray]]]
):
    name = "cs"
    encodings = ["idx"]

    def fit(
        self,
        table: str,
        meta: Metadata,
        attrs: dict[str, dict[str, Attributes]],
        data: TableData,
    ):
        self.table = table
        table_attrs = attrs["idx"][table]

        self.domain = {}
        for attr in table_attrs.values():
            for name, val in attr.vals.items():
                assert isinstance(val, IdxValue)
                self.domain[name] = val.domain

    def process_chunk(
        self,
        table: pd.DataFrame,
    ):
        domain = np.array(list(self.domain.values()))
        a = table()[list(self.domain)].to_numpy(dtype="uint16")

        # Add at least one sample prob to distr chisquare valid
        zero_fill = 1 / len(a)

        res = {}
        for i, name in enumerate(self.domain):
            res[name] = calc_marginal_1way(a, domain, [i], zero_fill)

        return res

    def process_split(self, name: str, split: TableData):
        res = process_in_parallel(
            self.process_chunk,
            [{"table": t} for t in split["tables"]["idx"][self.table].values()],
            desc=f"Processing CS split {name}",
        )

        assert res, "Received empty data"

        cols = res[0].keys()
        a = {col: np.sum([r[col] for r in res], axis=0) for col in cols}
        # Normalize result
        return {n: r / np.sum(r) for n, r in a.items()}

    def preprocess(
        self, wrk: TableData, ref: TableData
    ) -> Summaries[dict[str, np.ndarray]]:
        return Summaries(self.process_split("wrk", wrk), self.process_split("ref", ref))

    def process(
        self,
        wrk: TableData,
        ref: TableData,
        syn: TableData,
        pre: Summaries[dict[str, np.ndarray]],
    ) -> Summaries[dict[str, np.ndarray]]:
        return Summaries(pre.wrk, pre.ref, self.process_split("syn", syn))

    def visualise(self, data: dict[str, Summaries[dict[str, np.ndarray]]]):
        import mlflow

        from ...utils.mlflow import color_dataframe, gen_html_table

        results = {}

        # Add ref split first
        name = "ref"
        res = []
        split = next(iter(data.values()))
        for col in self.domain:
            wrk, syn = split.wrk, split.ref
            assert syn is not None
            chi, p = chisquare(wrk[col], syn[col])
            res.append([col, chi, p])

        results[name] = pd.DataFrame(res, columns=["col", "X^2", "p"])

        for name, split in data.items():
            res = []
            for col in self.domain:
                wrk, syn = split.wrk, split.syn
                assert syn is not None
                chi, p = chisquare(wrk[col], syn[col])
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
            split_ref="ref",
        )

        fn = (
            f"distr/cs.html" if self.table == "table" else f"distr/{self.table}_cs.html"
        )
        mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)


class KullbackLeiblerMetric(
    TableMetric[
        Summaries[dict[tuple[str, str], np.ndarray]],
        Summaries[dict[tuple[str, str], np.ndarray]],
    ]
):
    name = "kl"
    encodings = ["idx"]

    def fit(
        self,
        table: str,
        meta: Metadata,
        attrs: dict[str, dict[str, Attributes]],
        data: TableData,
    ):
        self.table = table
        table_attrs = attrs["idx"][table]

        self.domain = {}
        for attr in table_attrs.values():
            for name, val in attr.vals.items():
                assert isinstance(val, IdxValue)
                self.domain[name] = val.domain

    def process_chunk(
        self,
        table: pd.DataFrame,
    ):
        a = table()[list(self.domain)].to_numpy(dtype="uint16")
        domain = np.array(list(self.domain.values()))

        res = {}
        for i, col_i in enumerate(self.domain):
            for j, col_j in enumerate(self.domain):
                res[(col_i, col_j)] = calc_marginal_1way(
                    a, domain, [i, j], KL_ZERO_FILL
                )

        return res

    def process_split(self, name: str, split: TableData):
        res = process_in_parallel(
            self.process_chunk,
            [{"table": t} for t in split["tables"]["idx"][self.table].values()],
            desc=f"Processing KL split {name}",
        )

        assert res, "Received empty data"

        pairs = res[0].keys()
        a = {pair: np.sum([r[pair] for r in res], axis=0) for pair in pairs}
        # Normalize result
        return {n: r / np.sum(r) for n, r in a.items()}

    def preprocess(
        self, wrk: TableData, ref: TableData
    ) -> Summaries[dict[str, np.ndarray]]:
        return Summaries(self.process_split("wrk", wrk), self.process_split("ref", ref))

    def process(
        self,
        wrk: TableData,
        ref: TableData,
        syn: TableData,
        pre: Summaries[dict[tuple[str, str], np.ndarray]],
    ) -> Summaries[dict[tuple[str, str], np.ndarray]]:
        return Summaries(pre.wrk, pre.ref, self.process_split("syn", syn))

    def visualise(self, data: dict[str, Summaries[dict[tuple[str, str], np.ndarray]]]):
        import mlflow

        from ...utils.mlflow import color_dataframe, gen_html_table

        results = {}
        ref_split = next(iter(data.values()))
        ref_split = Summaries(ref_split.wrk, ref_split.ref, ref_split.ref)
        for name, split in {
            "ref": ref_split,
            **data,
        }.items():
            wrk, syn = split.wrk, split.syn
            assert syn
            res = []
            for key in syn:
                col_i, col_j = key
                k = wrk[key]
                j = syn[key]

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
            logger.info(f"Split {name} mean norm KL={results[name]['kl_norm'].mean():.5f}.")
            mlflow.log_metric(f"kl_norm.{name}", results[name]['kl_norm'].mean())

        kl_formatters = {"kl_norm": {"precision": 3}}
        style = color_dataframe(
            results,
            idx=["col_j"],
            cols=["col_i"],
            vals=["kl_norm"],
            formatters=kl_formatters,
            split_ref="ref",
        )

        fn = (
            f"distr/kl.html" if self.table == "table" else f"distr/{self.table}_kl.html"
        )
        mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)
