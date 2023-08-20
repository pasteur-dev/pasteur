from __future__ import annotations

import logging
from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.special import rel_entr
from scipy.stats import chisquare

from pasteur.metric import Summaries
from pasteur.utils import LazyDataset

from ...attribute import Attributes, CatValue, SeqValue, get_dtype
from ...metric import Metric, Summaries
from ...utils import LazyChunk, LazyFrame, data_to_tables
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


def _visualise_cs(
    name: str, domain: dict[str, int], data: dict[str, Summaries[dict[str, np.ndarray]]]
):
    import mlflow

    from ...utils.mlflow import color_dataframe, gen_html_table

    results = {}

    # Add ref split first
    zfill = lambda x: (x + 1) / np.sum(x + 1)
    name = "ref"
    res = []
    split = next(iter(data.values()))
    for col in domain:
        wrk, syn = split.wrk, split.ref
        assert syn is not None

        chi, p = chisquare(zfill(wrk[col]), zfill(syn[col]))
        res.append([col, chi, p])

    results[name] = pd.DataFrame(res, columns=["col", "X^2", "p"])

    for name, split in data.items():
        res = []
        for col in domain:
            wrk, syn = split.wrk, split.syn
            assert syn is not None
            chi, p = chisquare(zfill(wrk[col]), zfill(syn[col]))
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

    fn = f"distr/cs.html" if name == "table" else f"distr/{name}_cs.html"
    mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)


def _visualise_kl(
    name: str, data: dict[str, Summaries[dict[tuple[str, str], np.ndarray]]]
):
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

            zfill = lambda x: (x + KL_ZERO_FILL) / np.sum(x + KL_ZERO_FILL)
            k = zfill(wrk[key])
            j = zfill(syn[key])

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
        mlflow.log_metric(f"kl_norm.{name}", results[name]["kl_norm"].mean())

    kl_formatters = {"kl_norm": {"precision": 3}}
    style = color_dataframe(
        results,
        idx=["col_j"],
        cols=["col_i"],
        vals=["kl_norm"],
        formatters=kl_formatters,
        split_ref="ref",
    )

    fn = f"distr/kl.html" if name == "table" else f"distr/{name}_kl.html"
    mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)


def _process_marginals_chunk(
    name: str,
    expand_parents: bool,
    domain: dict[str, dict[str, int]],
    ids: dict[str, LazyChunk],
    tables: dict[str, LazyChunk],
):
    assert not expand_parents, "Expanding parents not supported yet"

    table = tables[name]()[list(domain[name])].to_numpy(dtype="uint16")
    table_domain = domain[name]
    domain_arr = np.array(list(table_domain.values()))

    # One way for CS
    one_way: dict[str, ndarray] = {}
    for i, name in enumerate(table_domain):
        one_way[name] = calc_marginal_1way(table, domain_arr, [i], 0)

    # Two way for KL
    two_way: dict[tuple[str, str], ndarray] = {}
    for i, col_i in enumerate(table_domain):
        for j, col_j in enumerate(table_domain):
            two_way[(col_i, col_j)] = calc_marginal_1way(table, domain_arr, [i, j], 0)

    return one_way, two_way


class DistributionMetric(
    Metric[
        Summaries[dict[str, tuple[dict[str, ndarray], dict[tuple[str, str], ndarray]]]],
        Summaries[dict[str, tuple[dict[str, ndarray], dict[tuple[str, str], ndarray]]]],
    ]
):
    name = "dstr"
    encodings = "idx"

    def fit(
        self,
        meta: dict[str, Attributes],
        data: dict[str, LazyFrame],
    ):
        self.domain = defaultdict(dict)

        for table, attrs in meta.items():
            for attr in attrs.values():
                for name, val in attr.vals.items():
                    if isinstance(val, SeqValue):
                        continue
                    assert isinstance(val, CatValue)
                    self.domain[table][name] = val.domain

    def preprocess(
        self,
        wrk: dict[str, LazyDataset],
        ref: dict[str, LazyDataset],
    ) -> Summaries[
        dict[str, tuple[dict[str, ndarray], dict[tuple[str, str], ndarray]]]
    ]:
        per_call = []
        per_call_meta = []
        base_args = {"domain": self.domain}

        for cwrk, cref in LazyDataset.zip_values([wrk, ref]):
            for split, split_data in [("wrk", cwrk), ("ref", cref)]:
                ids, tables = data_to_tables(split_data)

                for table in self.domain:
                    per_call.append(
                        {
                            "name": table,
                            "expand_parents": False,
                            "ids": ids,
                            "tables": tables,
                        }
                    )
                    per_call_meta.append({"split": split, "table": table})

        # Process marginals
        out = process_in_parallel(
            _process_marginals_chunk,
            per_call,
            base_args=base_args,
            desc="Preprocessing distribution metrics",
        )

        # Intertwine results
        res = defaultdict(lambda: defaultdict(list))

        for meta, hist in zip(per_call_meta, out):
            res[meta["split"]][meta["table"]].append(hist)

        ret = defaultdict(dict)
        for split, split_hists in res.items():
            for table, table_hists in split_hists.items():
                one_way = {}
                for key in table_hists[0][0].keys():
                    one_way[key] = np.sum(
                        [table_hists[i][0][key] for i in range(len(table_hists))],
                        axis=0,
                    )

                two_way = {}
                for key in table_hists[0][1].keys():
                    two_way[key] = np.sum(
                        [table_hists[i][1][key] for i in range(len(table_hists))],
                        axis=0,
                    )

                ret[split][table] = one_way, two_way
        return Summaries(wrk=ret["wrk"], ref=ret["ref"])

    def process(
        self,
        wrk: dict[str, LazyDataset],
        ref: dict[str, LazyDataset],
        syn: dict[str, LazyDataset],
        pre: Summaries[
            dict[str, tuple[dict[str, ndarray], dict[tuple[str, str], ndarray]]]
        ],
    ) -> Summaries[
        dict[str, tuple[dict[str, ndarray], dict[tuple[str, str], ndarray]]]
    ]:
        per_call = []
        per_call_meta = []
        base_args = {"domain": self.domain}

        for csyn in LazyDataset.zip_values(syn):
            ids, tables = data_to_tables(csyn)

            for table in self.domain:
                per_call.append(
                    {
                        "name": table,
                        "expand_parents": False,
                        "ids": ids,
                        "tables": tables,
                    }
                )
                per_call_meta.append({"table": table})

        # Process marginals
        out = process_in_parallel(
            _process_marginals_chunk,
            per_call,
            base_args=base_args,
            desc="Processing distribution metrics",
        )

        # Intertwine results
        res = defaultdict(list)
        for meta, hist in zip(per_call_meta, out):
            res[meta["table"]].append(hist)

        ret = {}
        for table, table_hists in res.items():
            one_way = {}
            for key in table_hists[0][0].keys():
                one_way[key] = np.sum(
                    [table_hists[i][0][key] for i in range(len(table_hists))],
                    axis=0,
                )

            two_way = {}
            for key in table_hists[0][1].keys():
                two_way[key] = np.sum(
                    [table_hists[i][1][key] for i in range(len(table_hists))],
                    axis=0,
                )

            ret[table] = one_way, two_way
        return pre.replace(syn=ret)

    def visualise(
        self,
        data: dict[
            str,
            Summaries[
                dict[str, tuple[dict[str, ndarray], dict[tuple[str, str], ndarray]]]
            ],
        ],
    ):
        for name in self.domain:
            _visualise_cs(
                name,
                self.domain[name],
                {
                    k: Summaries(
                        wrk=v.wrk[name][0],
                        ref=v.ref[name][0],
                        syn=v.syn[name][0] if v.syn else None,
                    )
                    for k, v in data.items()
                },
            )
            _visualise_kl(
                name,
                {
                    k: Summaries(
                        wrk=v.wrk[name][1],
                        ref=v.ref[name][1],
                        syn=v.syn[name][1] if v.syn else None,
                    )
                    for k, v in data.items()
                },
            )
