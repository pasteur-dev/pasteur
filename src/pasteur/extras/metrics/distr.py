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
from scipy.stats.contingency import association

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

OneWaySummary = dict[str, ndarray]
TwoWaySummary = dict[tuple[str, str] | tuple[str, str | int, str], ndarray]
DistrSummary = Summaries[dict[str, tuple[OneWaySummary, TwoWaySummary]]]


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
        np.add(
            idx,
            np.multiply(mul, data[:, col], out=tmp, casting="unsafe"),
            out=idx,
        )
        mul *= domain[col]

    counts = np.bincount(idx, minlength=x_dom)
    assert (
        len(counts) == x_dom
    ), f"Overflow error, domain for columns `{x}` is wrong or there is a mistake in encoding."

    return counts


def _visualise_cs(
    table: str,
    domain: dict[str, int],
    data: dict[str, Summaries[dict[str, np.ndarray]]],
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

    fn = f"distr/cs.html" if table == "table" else f"distr/cs/{table}.html"
    mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)


def _get_histdata(val):
    if len(val) == 2 and val[0] == "None":
        v = _get_histdata(val[1])
        if v is None:
            return None
        return [float("NaN"), *v]

    out = []
    for v in val:
        if not isinstance(v, str):
            return None

        try:
            out.append(float(v))
            continue
        except ValueError:
            pass

        if len(v) < 4:
            return None

        if v[0] not in "([":
            return None

        if v[-1] not in ")]":
            return None

        try:
            l, r = v[1:-1].split(", ", maxsplit=1)
            out.append((float(l) + float(r)) / 2)
        except ValueError:
            return None

    return out


def _visualise_basetable(
    table: str,
    attrs: Attributes,
    data: dict[str, Summaries[dict[str, np.ndarray]]],
):
    import re
    from pasteur.hierarchy import RebalancedValue

    from ...utils.mlflow import gen_html_table, color_dataframe

    # Unroll splits
    ref_split = next(iter(data.values()))
    splits = {
        "wrk": ref_split.wrk,
        "ref": ref_split.ref,
    }

    for split, split_data in data.items():
        splits[split] = split_data.syn

    # Handle them individually
    out_num = []
    out_cat = []

    CAT_VALS = 5
    CAT_MIN_VAL = 0.001

    TRE = re.compile(r"\d{2}:\d{2}") # 12:34
    MRE = re.compile(r"\+?\d{2}:\d{2}") # +12:34

    hvals_prev = {}
    for attr in attrs.values():
        for name, col in attr.vals.items():
            if not hasattr(col, "head"):
                continue

            for sname, split in splits.items():
                bins = _get_histdata(getattr(col, "head"))

                if bins is None:
                    break

                bins = np.array(bins)

                mean = np.nansum(bins * split[name]) / np.nansum(split[name])
                std = np.sqrt(
                    np.nansum((bins - mean) ** 2 * split[name])
                    / (np.nansum(split[name]) - 1)
                )
                out_num.append(
                    {
                        "name": name,
                        "split": sname,
                        "mean": float(mean),
                        "std": float(std),
                    }
                )

            counts = splits["wrk"][name]
            try:
                hval = RebalancedValue(counts, col)  # type: ignore
            except Exception:
                logger.exception(f"Failed to get human values for {name}")
                hvals_prev[name] = (None, 0)
                continue

            height = 0
            for h in range(hval.height):
                height = h
                dom = hval.get_domain(h)

                tmp = [0 for _ in range(dom)]
                for i, j in enumerate(hval.get_mapping(h)):
                    tmp[j] += counts[i]

                # Missing values merge last, which can make
                # unecessary merges. Therefore, check the second largest
                # min is above min val
                vmins = sorted(tmp)[:2]

                if dom <= CAT_VALS and vmins[1] > CAT_MIN_VAL:
                    break

            hvals_prev[name] = (hval, height)

            vnames = [[] for _ in range(hval.get_domain(height))]
            hnames = getattr(hval.original, "head").get_human_values()

            for i, v in enumerate(hval.get_mapping(height)):
                vnames[v].append(hnames[i])

            def process_names(l):
                if not l:
                    return "[Empty]"
                if len(l) == 1:
                    return l[0]

                # Handle intervals
                if l[0] and l[0][0] in "[(" and l[-1] and l[-1][-1] in ")]":
                    return f"{l[0].split(',')[0]}, {l[-1].split(', ')[-1]}"

                # Handle times
                if re.match(TRE, l[0]) and re.match(TRE, l[-1]):
                    return f"{l[0]}-{l[-1]}"

                # Handle intervals (skip first +)
                if re.match(MRE, l[0]) and re.match(MRE, l[-1]):
                    return f"{l[0]}-{l[-1][1:]}"

                # Handle numbers
                if all(v.isnumeric() for v in l):
                    return f"[{min(l)}, {max(l)}]"

                return ", ".join([v for v in l if v])[:35]

            vnames = [process_names(v) for v in vnames]

            for i, vname in enumerate(vnames):
                for sname, split in splits.items():
                    nsum = np.sum(split[name])

                    mval = 0
                    for j, v in enumerate(hval.get_mapping(height)):
                        if v == i:
                            mval += split[name][j]

                    rate = mval / nsum

                    out_cat.append(
                        {
                            "name": name,
                            "split": sname,
                            "value": "[missing]" if vname == "None" else vname,
                            "rate": 100 * float(rate),
                        }
                    )

    import mlflow

    stylers = {}
    if out_num:
        stylers["Numerical"] = color_dataframe(
            out_num,
            idx=["name"],
            cols=[],
            vals=["mean", "std"],
            split_ref="wrk",
            split_col="split",
            formatters={"mean": {"precision": 3}, "std": {"precision": 3}},
        )

    if out_cat:
        stylers["Categorical"] = color_dataframe(
            out_cat,
            idx=["name", "value"],
            cols=[],
            vals=["rate"],
            split_ref="wrk",
            split_col="split",
            formatters={"rate": {"precision": 1}},
        )

    if stylers:
        fn = (
            f"distr/basetable.html"
            if table == "table"
            else f"distr/basetable/{table}.html"
        )
        mlflow.log_text(gen_html_table(stylers, FONT_SIZE), fn)


def _visualise_kl(
    table: str,
    data: dict[str, Summaries[TwoWaySummary]],
):
    return _visualise_2way(table, data, "kl")


ASSOC_METRICS = ["cramer", "tschuprow", "pearson"]
METRICS = ["kl", *ASSOC_METRICS]


def _visualise_2way(
    table: str, data: dict[str, Summaries[TwoWaySummary]], metr: str = "kl", domain=None
):
    import mlflow

    from ...utils.mlflow import color_dataframe, gen_html_table

    results = {}
    presults = {}

    ref_split = next(iter(data.values()))
    ref_split = Summaries(ref_split.wrk, ref_split.ref, ref_split.ref)
    for name, split in {
        "ref": ref_split,
        **data,
    }.items():
        wrk, syn = split.wrk, split.syn
        assert syn
        res = []
        pres = {}

        for key in syn:
            if len(key) == 3:
                col_i, p, col_j = key
            else:
                col_i, col_j = key
                p = None

            if metr == "kl":
                zfill = lambda x: (x + KL_ZERO_FILL) / np.sum(x + KL_ZERO_FILL)
                k = zfill(wrk[key])
                j = zfill(syn[key])

                kl = rel_entr(k / k.sum(), j).sum()
                kl_norm = 1 / (1 + kl)
                out = [col_i, col_j, kl, kl_norm, len(k)]
            elif metr in ASSOC_METRICS:
                assert domain

                if col_i == col_j and not p:
                    continue

                k = wrk[key] + 1
                j = syn[key] + 1

                dom_i = domain[table][col_i]
                m_wrk = association(k.reshape((dom_i, -1)), method=metr)
                m_syn = association(j.reshape((dom_i, -1)), method=metr)
                m_res = np.abs(m_wrk - m_syn)

                out = [col_i, col_j, m_res, m_syn, len(k)]
            else:
                assert False, f"Metric {metr} not supported."

            if p:
                if p not in pres:
                    pres[p] = []
                pres[p].append(out)
            else:
                res.append(out)

        results[name] = pd.DataFrame(
            res,
            columns=[
                "col_i",
                "col_j",
                "metr",
                "metr_norm",
                "mlen",
            ],
        )
        sname = name.replace(" ", "_").replace("=", "_")
        # mlflow.log_metric(f"{sname}.kl_norm.{table}", results[name]["kl_norm"].mean())

        if pres:
            presults[name] = {
                k: pd.DataFrame(
                    v,
                    columns=[
                        "col_i",
                        "col_j",
                        "metr",
                        "metr_norm",
                        "mlen",
                    ],
                )
                for k, v in pres.items()
            }
            for k, v in presults[name].items():
                corrected = k.replace("-", "o") if k.startswith("-") else k
                mlflow.log_metric(
                    f"{sname}.metr_norm.{table}.{corrected}",
                    v["metr_norm"].mean(),
                )

    kl_formatters = {"metr_norm": {"precision": 3}}
    kl_formatters_overall = {"mean_metr_norm": {"precision": 3}}

    res = {}
    for split in results:
        if split not in res:
            res[split] = []
        res[split].append(
            {
                "table": "!",
                "split": split,
                "mean_metr_norm": results[split]["metr_norm"].mean(),
            }
        )
        if presults:
            for p in presults[split]:
                res[split].append(
                    {
                        "table": p,
                        "split": split,
                        "mean_metr_norm": presults[split][p]["metr_norm"].mean(),
                    }
                )

    # Print results as a table
    outs = f"{metr.upper():>5s} Table '{table:15s}' results:\n"
    ores = []
    for v in res.values():
        ores.extend(v)
    outs += (
        pd.DataFrame(ores)
        .pivot(index=["table"], columns=["split"], values=["mean_metr_norm"])
        .xs("mean_metr_norm", axis=1)
        .sort_index()
        .to_markdown()
    )
    outs += "\n"
    logger.info(outs)

    for v in results.values():
        if v.empty:
            return res
    
    base = color_dataframe(
        results,
        idx=["col_j"],
        cols=["col_i"],
        vals=["metr_norm"],
        formatters=kl_formatters,
        split_ref="ref",
    )
    overall = color_dataframe(
        {k: pd.DataFrame(v) for k, v in res.items()},
        idx=["table"],
        cols=[],
        vals=["mean_metr_norm"],
        formatters=kl_formatters_overall,
        split_ref="ref",
    )
    dfs = {"overall": overall, "same table": base}

    if presults:
        for p in next(iter(presults.values())):
            dfs[p] = color_dataframe(
                {k: v[p] for k, v in presults.items()},
                idx=["col_i"],
                cols=["col_j"],
                vals=["metr_norm"],
                formatters=kl_formatters,
                split_ref="ref",
            )

    pref = ""
    if metr in ASSOC_METRICS:
        pref = "assoc/"
    fn = (
        f"distr/{pref}{metr}.html"
        if table == "table"
        else f"distr/{pref}{metr}/{table}.html"
    )
    mlflow.log_text(gen_html_table(dfs, FONT_SIZE), fn)
    return res


def _process_marginals_chunk(
    name: str,
    domain: dict[str, dict[str, int]],
    parents: dict[str, list[str]],
    seq: dict[str, SeqValue],
    ids: dict[str, LazyChunk],
    tables: dict[str, LazyChunk],
):
    tids = ids[name]()
    raw_table = tables[name]()
    table = raw_table[list(domain[name])].to_numpy(dtype="uint16")
    table_domain = domain[name]
    domain_arr = np.array(list(table_domain.values()))
    ofs = table.shape[1]

    # One way for CS
    one_way: dict[str, ndarray] = {}
    for i, cname in enumerate(table_domain):
        one_way[cname] = calc_marginal_1way(table, domain_arr, [i], 0)

    # Two way for KL
    two_way: dict[tuple[str, str] | tuple[str, str | int, str], ndarray] = {}
    for i, col_i in enumerate(table_domain):
        for j, col_j in enumerate(table_domain):
            two_way[(col_i, col_j)] = calc_marginal_1way(table, domain_arr, [i, j], 0)

    # Two way accross parents
    for p in parents[name]:
        p_table = (
            tids[[p]]
            .join(tables[p](), on=p)
            .drop(columns=[p])[list(domain[p])]
            .to_numpy(dtype="uint16")
        )
        p_domain = np.array(list(domain[p].values()))
        combined = np.concatenate((table, p_table), axis=1)
        combined_dom = np.concatenate((domain_arr, p_domain))

        for i, col_i in enumerate(table_domain):
            for j, col_j in enumerate(domain[p]):
                two_way[(col_i, p, col_j)] = calc_marginal_1way(
                    combined, combined_dom, [i, ofs + j], 0
                )

    _JOIN_NAME = "_id_zdjwk"
    _IDX_NAME = "_id_lkjijk"

    if name in seq:
        sval = seq[name]
        if sval.order:
            tseq = raw_table[sval.name]
            ids_seq = tids.join(tseq, how="right").reset_index(names=_IDX_NAME)
            for o in range(sval.order):
                ids_seq_prev = tids.join(tseq + o + 1, how="right").reset_index(
                    names=_JOIN_NAME
                )
                join_ids = ids_seq.merge(
                    ids_seq_prev, on=[*tids.columns, sval.name], how="inner"
                ).set_index(_IDX_NAME)[[_JOIN_NAME]]
                ref_df = join_ids.join(raw_table, on=_JOIN_NAME)[
                    list(domain[name])
                ].to_numpy(dtype="uint16")
                fkey = (
                    ~pd.isna(
                        ids_seq.set_index(_IDX_NAME)[[]].join(join_ids, how="left")
                    ).to_numpy()
                ).reshape(-1)
                combined = np.concatenate((table[fkey], ref_df), axis=1)
                combined_dom = np.concatenate([domain_arr, domain_arr])

                for i, col_i in enumerate(table_domain):
                    for j, col_j in enumerate(table_domain):
                        two_way[(col_i, f"{-o-1}", col_j)] = calc_marginal_1way(
                            combined, combined_dom, [i, ofs + j], 0
                        )
        pass

    return one_way, two_way


class DistributionMetric(Metric[DistrSummary, DistrSummary]):
    name = "distr"
    encodings = "idx"

    def fit(
        self,
        meta: dict[str, Attributes],
        data: dict[str, LazyFrame],
    ):
        self.domain = defaultdict(dict)
        self.attrs = meta

        self.parents = {
            k[:-4]: list(v.sample().columns)
            for k, v in data.items()
            if k.endswith("_ids")
        }
        self.seq = {}

        for table, attrs in meta.items():
            for attr in attrs.values():
                for name, val in attr.vals.items():
                    if isinstance(val, SeqValue):
                        self.seq[table] = val
                    else:
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
        base_args = {"domain": self.domain, "parents": self.parents, "seq": self.seq}

        for cwrk, cref in LazyDataset.zip_values([wrk, ref]):
            for split, split_data in [("wrk", cwrk), ("ref", cref)]:
                ids, tables = data_to_tables(split_data)

                for table in self.domain:
                    per_call.append(
                        {
                            "name": table,
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
        pre: DistrSummary,
    ) -> DistrSummary:
        per_call = []
        per_call_meta = []
        base_args = {"domain": self.domain, "parents": self.parents, "seq": self.seq}

        for csyn in LazyDataset.zip_values(syn):
            ids, tables = data_to_tables(csyn)

            for table in self.domain:
                per_call.append(
                    {
                        "name": table,
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
            DistrSummary,
        ],
    ):
        # import time 

        overall_metr = {}
        for name in self.domain:
            # start = time.perf_counter()
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
            # logger.info(f"cs {name} {time.perf_counter()-start:.2f}s")

            # start = time.perf_counter()
            _visualise_basetable(
                name,
                self.attrs[name],
                {
                    k: Summaries(
                        wrk=v.wrk[name][0],
                        ref=v.ref[name][0],
                        syn=v.syn[name][0] if v.syn else None,
                    )
                    for k, v in data.items()
                },
            )
            # logger.info(f"bs {name} {time.perf_counter()-start:.2f}s")

            for metric in METRICS:
                if metric not in overall_metr:
                    overall_metr[metric] = {}
                # start = time.perf_counter()
                overall_metr[metric][name] = _visualise_2way(
                    name,
                    {
                        k: Summaries(
                            wrk=v.wrk[name][1],
                            ref=v.ref[name][1],
                            syn=v.syn[name][1] if v.syn else None,
                        )
                        for k, v in data.items()
                    },
                    metric,
                    domain=self.domain,
                )
                # logger.info(f"2w {metric} {name} {time.perf_counter()-start:.2f}s")

        from pasteur.utils.styles import use_style
        import matplotlib.pyplot as plt
        import mlflow

        use_style("mlflow")

        for metr in METRICS:
            scores = {}
            scores_per_table = {}

            for table, table_res in overall_metr[metr].items():
                scores_per_table[table] = {}
                for split, split_res in table_res.items():
                    if split not in scores:
                        scores[split] = {
                            "intra": [],
                            "seq": [],
                            "hist": [],
                        }
                    if split not in scores_per_table[table]:
                        scores_per_table[table][split] = {
                            "intra": [],
                            "seq": [],
                            "hist": [],
                        }
                    for res in split_res:
                        if res["table"] == "!":
                            scores[split]["intra"].append(res["mean_metr_norm"])
                            scores_per_table[table][split]["intra"].append(
                                res["mean_metr_norm"]
                            )
                        elif res["table"].startswith("-"):
                            scores[split]["seq"].append(res["mean_metr_norm"])
                            scores_per_table[table][split]["seq"].append(
                                res["mean_metr_norm"]
                            )
                        else:
                            scores[split]["hist"].append(res["mean_metr_norm"])
                            scores_per_table[table][split]["hist"].append(
                                res["mean_metr_norm"]
                            )

            fancy_names = {
                "intra": "Intra-table",
                "seq": "Sequential",
                "hist": "Inter-table",
            }

            lines = {}
            mlflow.log_dict(scores, f"_raw/metrics/distr/{metr}_overall.json")
            mlflow.log_dict(
                scores_per_table, f"_raw/metrics/distr/{metr}_overall_per_table.json"
            )
            for table, split_scores_per_table in [
                ("_overall_single", scores),
                ("_overall", scores),
                *scores_per_table.items(),
            ]:
                combined = "_single" in table
                fig, ax = plt.subplots()
                bar_width = 0.3

                for split, split_scores in split_scores_per_table.items():
                    for stype, type_scores in split_scores.items():
                        if stype not in lines:
                            lines[stype] = {}
                        lines[stype][split] = np.mean(type_scores) if type_scores else 0

                l_res = 0
                split_scores = {}
                if combined:
                    l_res = len(split_scores_per_table)
                    split_scores = split_scores_per_table
                    for x, y in enumerate(split_scores_per_table.values()):
                        ax.bar(
                            x,
                            np.mean([np.mean(v) for v in y.values()]),
                        )
                else:
                    for i, (stype, split_scores) in enumerate(lines.items()):
                        l_res = len(split_scores)
                        x = np.arange(l_res)
                        ax.bar(
                            x + i * bar_width,
                            split_scores.values(),
                            bar_width,
                            label=fancy_names[stype],
                        )

                ax.set_xlabel("Experiment")
                ax.set_ylabel(f"Mean Norm {metr.upper()}")
                ax.set_title(f"Overall Mean Norm {metr.upper()}")

                max_len = 0
                labels = [k.split(" ") for k in split_scores.keys()]
                for params in labels:
                    for param in params:
                        max_len = max(max_len, len(param))

                ax.set_xticks(np.arange(l_res) + (0 if combined else 0.3))
                if max_len > 15 or l_res > 7:
                    tick_labels = [" ".join(l) for l in labels]
                    rot = min(3 * l_res, 90)
                    ax.set_xticklabels(tick_labels)
                    plt.setp(
                        ax.get_xticklabels(), rotation=rot, horizontalalignment="right"
                    )
                else:
                    tick_labels = ["\n".join(l) for l in labels]
                    ax.set_xticklabels(tick_labels)

                if combined:
                    # Dont use legend on combined graph
                    pass
                elif metr == "kl":
                    # ax.set_ylim([0.55, 1.03])
                    ax.legend(loc="lower right")
                elif metr in ASSOC_METRICS:
                    ax.legend(loc="upper right")
                else:
                    ax.legend(loc="lower right")

                # elif metr == "chi2":
                #     ax.set_ylim([0.5, 1.03])
                plt.tight_layout()
                pref = ""
                if metr in ASSOC_METRICS:
                    pref = "assoc/"
                mlflow.log_figure(fig, f"distr/{pref}{metr}_overall/{table}.png")
                plt.close()
