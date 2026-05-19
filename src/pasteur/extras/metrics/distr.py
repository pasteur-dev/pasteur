from __future__ import annotations

import logging
import re
from collections import OrderedDict, defaultdict
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable

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

    chi_fmt = lambda m, lo, hi: f"{m:.3f} ({lo:.3f}, {hi:.3f})"
    p_fmt = lambda m, lo, hi: f"{100*m:.1f} ({100*lo:.1f}, {100*hi:.1f})"

    agg_chi, display_chi = _aggregate_run_dfs(
        results, key_cols=["col"], val_col="X^2", fmt=chi_fmt
    )
    agg_p, display_p = _aggregate_run_dfs(
        results, key_cols=["col"], val_col="p", fmt=p_fmt
    )
    # Merge the two aggregated frames per base (one carries the mean X^2,
    # the other the mean p) so color_dataframe sees a single DataFrame per
    # base with both value columns.
    agg_results: dict[str, pd.DataFrame] = {}
    for base in agg_chi:
        chi_df = agg_chi[base][["col", "X^2"]]
        p_df = agg_p[base][["col", "p"]]
        agg_results[base] = chi_df.merge(p_df, on="col", how="outer")

    # Flatten the display lookup keyed by (val_name, base).
    display_strs = {("X^2", b): display_chi[b] for b in display_chi}
    display_strs.update({("p", b): display_p[b] for b in display_p})

    style = color_dataframe(
        agg_results,
        idx=["col"],
        cols=[],
        vals=["X^2", "p"],
        formatters=cs_formatters,
        split_ref="ref",
    )
    # cl = (val_name, base); rl = col_value
    style = _override_display(
        style,
        display_strs,
        lambda rl, cl: (
            ((cl[0], cl[-1]), (rl,)) if isinstance(cl, tuple) and len(cl) >= 2 else None
        ),
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

    TRE = re.compile(r"\d{2}:\d{2}")  # 12:34
    MRE = re.compile(r"\+?\d{2}:\d{2}")  # +12:34

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
METRICS = ["kl", "tvd", *ASSOC_METRICS]
PRINT_METRICS = ["kl", "tvd", "cramer"]
# Metrics where lower scores are better (so the worst-tail percentile
# flips: e.g. ``percentile_lower=5`` is plotted as the 95th percentile).
_METRIC_LOWER_IS_BETTER = {"tvd": True}


_RUN_SUFFIX_RE = re.compile(r"\s*r\d+$")


def _strip_run_suffix(name: str) -> str:
    return _RUN_SUFFIX_RE.sub("", name).rstrip()


def _darken_color(color, factor: float = 0.55):
    import matplotlib.colors as mcolors

    r, g, b, a = mcolors.to_rgba(color)
    return (r * factor, g * factor, b * factor, a)


def _draw_ci_caps(ax, xs, heights, p5, p95, bar_color, width):
    color = _darken_color(bar_color)
    half = width * 0.35
    for x, _h, lo, hi in zip(xs, heights, p5, p95):
        ax.hlines(lo, x - half, x + half, colors=[color], linewidth=1.8)
        ax.hlines(hi, x - half, x + half, colors=[color], linewidth=1.8)


def _aggregate_runs_scalar(
    values: dict[str, float],
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Group ``{split_name: scalar}`` by stripped base name. Returns mean
    per base plus (p5, p95) per base for groups with ≥ 2 runs."""
    groups: OrderedDict[str, list[float]] = OrderedDict()
    for name, val in values.items():
        groups.setdefault(_strip_run_suffix(name), []).append(float(val))
    means: OrderedDict[str, float] = OrderedDict()
    cis: dict[str, tuple[float, float]] = {}
    for base, vs in groups.items():
        means[base] = float(np.mean(vs)) if vs else 0.0
        if len(vs) >= 2:
            cis[base] = (
                float(np.percentile(vs, 5)),
                float(np.percentile(vs, 95)),
            )
    return means, cis


def _aggregate_run_dfs(
    results: dict[str, pd.DataFrame],
    key_cols: list[str],
    val_col: str,
    fmt: Callable[[float, float, float], str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[tuple, str]]]:
    """Group ``results`` by base name (stripping the ``rN`` run suffix).
    Within each multi-run group, aggregate per ``key_cols``: take the mean
    of ``val_col`` (the column kept in the returned DataFrame) and build a
    parallel ``"mean (p5, p95)"`` string lookup keyed by the key-col tuple.

    ``fmt`` overrides the default 3-decimal cell rendering; it receives
    ``(mean, p5, p95)`` and returns the displayed string.

    Single-run groups pass the original DataFrame through with an empty
    string lookup (so the caller falls back to its default formatter)."""
    if fmt is None:
        fmt = lambda m, lo, hi: f"{m:.3f} ({lo:.3f}, {hi:.3f})"
    groups: OrderedDict[str, list[pd.DataFrame]] = OrderedDict()
    for name, df in results.items():
        groups.setdefault(_strip_run_suffix(name), []).append(df)

    agg_results: OrderedDict[str, pd.DataFrame] = OrderedDict()
    display_strs: OrderedDict[str, dict[tuple, str]] = OrderedDict()

    def _p(q):
        def _f(s):
            v = s.dropna()
            return float(np.percentile(v, q)) if v.size else float("nan")

        return _f

    for base, dfs in groups.items():
        if len(dfs) == 1:
            agg_results[base] = dfs[0]
            display_strs[base] = {}
            continue

        merged = pd.concat(dfs, ignore_index=True)
        stats = merged.groupby(key_cols, sort=False)[val_col].agg(
            mean="mean", p5=_p(5), p95=_p(95)
        )

        agg = stats[["mean"]].rename(columns={"mean": val_col}).reset_index()
        other_cols = [c for c in dfs[0].columns if c not in key_cols and c != val_col]
        if other_cols:
            first_other = (
                merged.groupby(key_cols, sort=False)[other_cols].first().reset_index()
            )
            agg = agg.merge(first_other, on=key_cols, how="left")
        agg_results[base] = agg

        strs: dict[tuple, str] = {}
        for keys_vals, row in stats.iterrows():
            m = float(row["mean"])
            if np.isnan(m):
                continue
            tup = keys_vals if isinstance(keys_vals, tuple) else (keys_vals,)
            strs[tup] = fmt(m, float(row["p5"]), float(row["p95"]))
        display_strs[base] = strs

    return agg_results, display_strs


def _override_display(
    pts,
    display_strs: dict[str, dict[tuple, str]],
    key_extract: Callable,
):
    """Override cell display strings on a pandas Styler.

    ``key_extract`` receives ``(row_label, col_label)`` and returns
    ``(split_name, key_tuple)`` or ``None``. When the lookup hits, the
    cell's display function is replaced via ``Styler._display_funcs``."""
    if not any(display_strs.values()):
        return pts
    pt = pts.data
    for ridx, row_label in enumerate(pt.index):
        for cidx, col_label in enumerate(pt.columns):
            extracted = key_extract(row_label, col_label)
            if extracted is None:
                continue
            split, key_tuple = extracted
            text = display_strs.get(split, {}).get(key_tuple)
            if text is None:
                continue
            pts._display_funcs[(ridx, cidx)] = lambda _v, t=text: t
    return pts


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
            elif metr == "tvd":
                zfill = lambda x: (x + KL_ZERO_FILL) / np.sum(x + KL_ZERO_FILL)
                k = zfill(wrk[key])
                j = zfill(syn[key])

                tvd = float(0.5 * np.sum(np.abs(k - j)))
                out = [col_i, col_j, tvd, tvd, len(k)]
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
            # for k, v in presults[name].items():
            #     corrected = k.replace("-", "o") if k.startswith("-") else k
            #     mlflow.log_metric(
            #         f"{sname}.metr_norm.{table}.{corrected}",
            #         v["metr_norm"].mean(),
            #     )

    kl_formatters = {"metr_norm": {"precision": 3}}
    kl_formatters_overall = {"mean_metr_norm": {"precision": 3}}

    res = {}
    for split in results:
        if split not in res:
            res[split] = []
        intra_norms = results[split]["metr_norm"].dropna().tolist()
        res[split].append(
            {
                "table": "!",
                "split": split,
                "mean_metr_norm": results[split]["metr_norm"].mean(),
                "metr_norms": [float(x) for x in intra_norms],
            }
        )
        if presults:
            for p in presults[split]:
                p_norms = presults[split][p]["metr_norm"].dropna().tolist()
                res[split].append(
                    {
                        "table": p,
                        "split": split,
                        "mean_metr_norm": presults[split][p]["metr_norm"].mean(),
                        "metr_norms": [float(x) for x in p_norms],
                    }
                )

    # Print results as a table
    if metr in PRINT_METRICS:
        outs = f"{metr.upper():>5s} Table '{table:15s}' results:\n"

        print_agg, print_display = _aggregate_run_dfs(
            {k: pd.DataFrame(v) for k, v in res.items()},
            key_cols=["table"],
            val_col="mean_metr_norm",
        )
        ordered_bases = list(print_agg.keys())

        cells: dict[tuple[str, str], str] = {}
        for base in ordered_bases:
            df = print_agg[base]
            for _, row in df.iterrows():
                tbl = row["table"]
                mean = row["mean_metr_norm"]
                if pd.isna(mean):
                    cells[(tbl, base)] = ""
                    continue
                strs = print_display.get(base, {})
                fmt = strs.get((tbl,))
                cells[(tbl, base)] = fmt if fmt is not None else f"{float(mean):.3f}"

        tables = sorted({tbl for tbl, _ in cells.keys()})
        outs += pd.DataFrame(
            {
                base: [cells.get((tbl, base), "") for tbl in tables]
                for base in ordered_bases
            },
            index=tables,
        ).to_markdown()
        outs += "\n"
        logger.info(outs)

    for v in results.values():
        if v.empty:
            return res

    agg_results, base_display = _aggregate_run_dfs(
        results, key_cols=["col_i", "col_j"], val_col="metr_norm"
    )
    base = color_dataframe(
        agg_results,
        idx=["col_j"],
        cols=["col_i"],
        vals=["metr_norm"],
        formatters=kl_formatters,
        split_ref="ref",
    )
    # cl = ('metr_norm', col_i_value, split_name); rl = col_j_value
    base = _override_display(
        base,
        base_display,
        lambda rl, cl: (
            (cl[-1], (cl[1], rl)) if isinstance(cl, tuple) and len(cl) >= 3 else None
        ),
    )

    agg_res, overall_display = _aggregate_run_dfs(
        {k: pd.DataFrame(v) for k, v in res.items()},
        key_cols=["table"],
        val_col="mean_metr_norm",
    )
    overall = color_dataframe(
        agg_res,
        idx=["table"],
        cols=[],
        vals=["mean_metr_norm"],
        formatters=kl_formatters_overall,
        split_ref="ref",
    )
    # cl = ('mean_metr_norm', split_name); rl = table_value
    overall = _override_display(
        overall,
        overall_display,
        lambda rl, cl: (
            (cl[-1], (rl,)) if isinstance(cl, tuple) and len(cl) >= 2 else None
        ),
    )
    dfs = {"overall": overall, "same table": base}

    if presults:
        for p in next(iter(presults.values())):
            agg_p, p_display = _aggregate_run_dfs(
                {k: v[p] for k, v in presults.items() if p in v},
                key_cols=["col_i", "col_j"],
                val_col="metr_norm",
            )
            pair = color_dataframe(
                agg_p,
                idx=["col_i"],
                cols=["col_j"],
                vals=["metr_norm"],
                formatters=kl_formatters,
                split_ref="ref",
            )
            # cl = ('metr_norm', col_j_value, split_name); rl = col_i_value
            dfs[p] = _override_display(
                pair,
                p_display,
                lambda rl, cl: (
                    (cl[-1], (rl, cl[1]))
                    if isinstance(cl, tuple) and len(cl) >= 3
                    else None
                ),
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


def _parse_pretty_names(names: list[str]):
    """Parse prettified run names into (algorithm, step_label, run_idx) tuples.

    Pretty names follow the convention from `prettify_run_names`:
      - ``_alg`` produces the algorithm token (first, no ``=``)
      - ``_run`` produces ``rN`` (value-only param)
      - regular params produce ``param=value``
      - boolean params produce ``flag`` / ``no_flag``
      - other ``_``-prefixed value params produce bare values

    Returns a dict mapping each name to ``(algorithm, step, run_idx)``.
    ``step`` is ``None`` when there is no sweep hyperparameter and
    ``run_idx`` is ``None`` when ``-r`` was not used.
    """
    import re

    parsed = {}
    for name in names:
        tokens = name.split()
        if not tokens:
            parsed[name] = ("default", None, None)
            continue

        # First token is the algorithm name when it doesn't look like a
        # param=value or a run suffix (rN).
        if "=" not in tokens[0] and not re.match(r"^r\d+$", tokens[0]):
            alg = tokens[0]
            rest = tokens[1:]
        else:
            alg = "syn"
            rest = tokens

        run_idx = None
        step_parts = []

        for token in rest:
            if re.match(r"^r\d+$", token):
                run_idx = int(token[1:])
            else:
                step_parts.append(token)

        step = " ".join(step_parts) if step_parts else None
        parsed[name] = (alg, step, run_idx)

    return parsed


# Extra CSS layered on top of wrap_zoom_html's _ZOOM_CSS so the matplotlib
# SVGs sit on a padded white surface with a heading.  The zoom-CSS turns
# ``body`` into a grab-cursor surface; we re-introduce the section title
# styling without re-enabling default body margins (those would fight the
# scroll-zoom maths).
_MULTIPLOT_EXTRA_CSS = (
    "body{font-family:-apple-system,BlinkMacSystemFont,sans-serif}"
    "h2{color:#333;margin:18px 24px 8px}"
    "svg{margin:0 24px 24px}"
)


def _draw_multiplot_subplot(
    ax,
    label: str,
    scores: dict[str, float | list[float]],
    parsed: dict,
    algorithms: list[str],
    steps: list,
    has_runs: bool,
    percentile: float,
    cmap,
):
    """Draw a single multiplot subplot into *ax*.

    See :func:`_render_multiplot` for what ``scores`` looks like.
    """
    n_alg = len(algorithms)

    # Y-axis bounds are computed only from mean lines, run dots, and the
    # ref line — so the dashed worst-tail percentile is allowed to fall
    # off-screen for very bad runs (low privacy budgets).
    y_main: list[float] = []

    for alg_idx, alg in enumerate(algorithms):
        color = cmap[alg_idx % len(cmap)]

        pos_list: list[int] = []
        mean_list: list[float] = []
        vals_list: list[list[float]] = []
        combos_list: list[list[float]] = []

        for step_idx, step in enumerate(steps):
            run_values: list[float] = []
            combo_values: list[float] = []
            for name, (a, s, _r) in parsed.items():
                if a == alg and s == step:
                    score = scores.get(name)
                    if score is None:
                        continue
                    if isinstance(score, (list, tuple, np.ndarray)):
                        run_combos = [float(v) for v in score if not np.isnan(v)]
                        if not run_combos:
                            continue
                        run_values.append(float(np.mean(run_combos)))
                        combo_values.extend(run_combos)
                    else:
                        fval = float(score)
                        if np.isnan(fval):
                            continue
                        run_values.append(fval)
                        combo_values.append(fval)

            if not run_values:
                continue

            pos_list.append(step_idx)
            mean_list.append(float(np.mean(run_values)))
            vals_list.append(run_values)
            combos_list.append(combo_values)

        if not pos_list:
            continue

        # --- 10th percentile bound across per-combo values ---
        if any(len(c) > 1 for c in combos_list):
            pl = [float(np.percentile(c, percentile)) for c in combos_list]
            ax.plot(
                pos_list,
                pl,
                color=color,
                linestyle=(0, (4, 2)),
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )

        # --- Line through means ---
        ax.plot(
            pos_list,
            mean_list,
            color=color,
            label=alg,
            marker="o",
            markersize=5,
            linewidth=1.5,
            zorder=4,
        )
        y_main.extend(mean_list)
        for vals in vals_list:
            y_main.extend(vals)

        # --- Individual run dots + CI overlay ---
        if has_runs:
            offset = (alg_idx - (n_alg - 1) / 2) * 0.06
            for pos, vals in zip(pos_list, vals_list):
                if len(vals) > 1:
                    q25, q75 = np.percentile(vals, [25, 75])
                    vmin, vmax = float(np.min(vals)), float(np.max(vals))

                    # Thin whisker: full range
                    ax.plot(
                        [pos + offset, pos + offset],
                        [vmin, vmax],
                        color=color,
                        linewidth=1,
                        alpha=0.4,
                        zorder=2,
                    )
                    # Thick bar: IQR
                    ax.plot(
                        [pos + offset, pos + offset],
                        [q25, q75],
                        color=color,
                        linewidth=4,
                        alpha=0.3,
                        zorder=2,
                    )

                # Scatter individual dots with slight jitter
                jitter = np.linspace(-0.03, 0.03, len(vals)) + offset
                ax.scatter(
                    [pos + j for j in jitter],
                    vals,
                    color=color,
                    alpha=0.5,
                    s=15,
                    zorder=3,
                )

    # --- Horizontal reference line + 10th percentile bound ---
    ref_raw = scores.get("ref", float("nan"))
    if isinstance(ref_raw, (list, tuple, np.ndarray)):
        ref_combos = [float(v) for v in ref_raw if not np.isnan(v)]
        ref_score = float(np.mean(ref_combos)) if ref_combos else float("nan")
    else:
        ref_combos = []
        ref_score = float(ref_raw)
        if not np.isnan(ref_score):
            ref_combos = [ref_score]
    if not np.isnan(ref_score):
        ax.axhline(
            ref_score,
            color="grey",
            linestyle="-",
            linewidth=1,
            alpha=0.7,
            zorder=1,
            label="ref",
        )
        y_main.append(ref_score)
    if len(ref_combos) > 1:
        ax.axhline(
            float(np.percentile(ref_combos, percentile)),
            color="grey",
            linestyle=(0, (4, 2)),
            linewidth=1,
            alpha=0.7,
            zorder=1,
        )

    # Lock y-limits to the main artists; dashed worst-tail bounds may
    # extend below/above and get clipped.
    if y_main:
        ymin = float(min(y_main))
        ymax = float(max(y_main))
        if ymax > ymin:
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

    # --- Axes formatting ---
    ax.set_xticks(range(len(steps)))
    step_labels = [s if s is not None else "default" for s in steps]
    if any(len(str(lb)) > 10 for lb in step_labels) or len(step_labels) > 6:
        ax.set_xticklabels(step_labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticklabels(step_labels, fontsize=8)

    ax.set_title(label, fontweight="bold", fontsize=11)
    ax.set_ylabel("Score", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3, linewidth=0.5)


def _parse_sweep_axes(split_names: list[str]):
    """Resolve the algorithm / step / has-runs axes shared by every
    subplot in a multiplot.  Returns ``(parsed, algorithms, steps,
    has_runs)`` or ``None`` if the run set isn't worth plotting (single
    point with no variation)."""
    if not split_names:
        return None

    parsed = _parse_pretty_names(split_names)
    algorithms = list(dict.fromkeys(alg for alg, _, _ in parsed.values()))
    steps = list(dict.fromkeys(step for _, step, _ in parsed.values()))
    has_runs = any(r is not None for _, _, r in parsed.values())
    has_steps = len(steps) > 1 or (len(steps) == 1 and steps[0] is not None)

    if not has_steps and len(algorithms) <= 1 and not has_runs:
        return None

    if not has_steps:
        steps = [None]

    return parsed, algorithms, steps, has_runs


def _fig_to_html(fig, title: str, extra_css: str = _MULTIPLOT_EXTRA_CSS) -> str:
    """Render *fig* as an inlined SVG inside the shared zoom/pan HTML
    shell used by ``graph.html`` etc."""
    from io import BytesIO

    import matplotlib.pyplot as plt

    from ...utils.mlflow import strip_svg_preamble, wrap_zoom_html

    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    svg = buf.read().decode("utf-8")
    plt.close(fig)

    body = f"<h2>{title}</h2>\n{strip_svg_preamble(svg)}"
    return wrap_zoom_html(body, title, extra_css=extra_css)


def _render_multiplot(
    subplot_scores: dict[str, dict[str, float | list[float]]],
    title: str,
    artifact_path: str,
    percentile: float = 5,
):
    """Render a multiplot HTML page with one subplot per key in *subplot_scores*.

    Each entry in *subplot_scores* maps a subplot label to
    ``{split_name: score}``, where score is either a scalar or a list of
    per-column-combination (e.g. 2-way) values for that split.  When lists
    are provided, a translucent 10–90 percentile band is drawn around each
    algorithm's line, computed across all combos from every run that
    contributes to the line.  X-axis = sweep steps, lines = algorithms,
    dots + run-to-run whiskers when ``-r N`` runs exist.  A dashed grey
    *ref* baseline is drawn when available.  The result is logged to
    *artifact_path* in mlflow.
    """
    import matplotlib.pyplot as plt
    import mlflow

    from ...utils.styles import use_style

    use_style("mlflow")

    split_names = [k for k in next(iter(subplot_scores.values())).keys() if k != "ref"]
    axes_info = _parse_sweep_axes(split_names)
    if axes_info is None:
        return
    parsed, algorithms, steps, has_runs = axes_info

    subplot_labels = list(subplot_scores.keys())
    n_plots = len(subplot_labels)
    ncols = min(n_plots, 2)
    nrows = (n_plots + ncols - 1) // ncols

    fig_w = max(6, 3 + 1.2 * len(steps)) * ncols
    fig_h = 4.5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    cmap = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for idx, label in enumerate(subplot_labels):
        row, col = divmod(idx, ncols)
        _draw_multiplot_subplot(
            axes[row][col],
            label,
            subplot_scores[label],
            parsed,
            algorithms,
            steps,
            has_runs,
            percentile,
            cmap,
        )

    # Hide empty subplot slots
    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    mlflow.log_text(_fig_to_html(fig, title), artifact_path)


def _render_combined_multiplot(
    rows: list[tuple[str, dict[str, dict[str, float | list[float]]], float]],
    title: str,
    artifact_path: str,
):
    """Render one HTML page with a row per (metric, subplot_scores,
    percentile) in *rows*.  Subplot ordering and dedup are pre-applied
    by the caller (so ``subplot_scores`` is what should be drawn)."""
    import matplotlib.pyplot as plt
    import mlflow

    from ...utils.styles import use_style

    use_style("mlflow")

    if not rows:
        return

    first_scores = rows[0][1]
    split_names = [k for k in next(iter(first_scores.values())).keys() if k != "ref"]
    axes_info = _parse_sweep_axes(split_names)
    if axes_info is None:
        return
    parsed, algorithms, steps, has_runs = axes_info

    nrows = len(rows)
    ncols = max(len(scores) for _, scores, _ in rows)

    fig_w = max(6, 3 + 1.2 * len(steps)) * ncols
    fig_h = 4.5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    cmap = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for row_idx, (fancy, scores, percentile) in enumerate(rows):
        labels = list(scores.keys())
        for col_idx in range(ncols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(labels):
                ax.set_visible(False)
                continue
            label = labels[col_idx]
            # Prefix the first subplot in each row with the metric name
            # so the row is self-identifying even without a row title.
            sub_label = f"{fancy} — {label}" if col_idx == 0 else label
            _draw_multiplot_subplot(
                ax,
                sub_label,
                scores[label],
                parsed,
                algorithms,
                steps,
                has_runs,
                percentile,
                cmap,
            )

    plt.tight_layout()
    mlflow.log_text(_fig_to_html(fig, title), artifact_path)


_CORR_TYPES = {
    "intra": "Intra-table",
    "seq": "Sequential",
    "hist": "Inter-table",
}

_METRIC_FANCY = {
    "kl": "KL Divergence",
    "tvd": "Total Variation Distance",
    "cramer": "Cramér's V",
    "tschuprow": "Tschuprow's T",
    "pearson": "Pearson",
}


def _visualise_multiplot(overall_metr: dict, percentile_lower: float = 5):
    """Create one multiplot HTML per metric, each with subplots per
    correlation type (overall, intra-table, sequential, inter-table).

    Files are placed alongside the existing per-metric ``_overall`` PNGs:
    ``distr/kl_overall/multiplot.html``,
    ``distr/assoc/cramer_overall/multiplot.html``, etc.
    """
    from collections import defaultdict

    import numpy as np

    # ------------------------------------------------------------------
    # Collect per-(metric, split, corr_type) raw score lists
    # ------------------------------------------------------------------
    # raw[metr][corr_type][split] = list[float]
    raw: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    # Preserve the order splits first appear in the parent run, per metric.
    split_order: dict[str, list[str]] = defaultdict(list)

    for metr in METRICS:
        if metr not in overall_metr:
            continue
        for _table, table_res in overall_metr[metr].items():
            for split, split_res in table_res.items():
                if split not in split_order[metr]:
                    split_order[metr].append(split)
                for entry in split_res:
                    norms = entry.get("metr_norms")
                    if norms:
                        vals = [float(v) for v in norms if not np.isnan(v)]
                    else:
                        scalar = entry.get("mean_metr_norm")
                        vals = (
                            [float(scalar)]
                            if scalar is not None and not np.isnan(scalar)
                            else []
                        )
                    if not vals:
                        continue
                    if entry["table"] == "!":
                        raw[metr]["intra"][split].extend(vals)
                    elif entry["table"].startswith("-"):
                        raw[metr]["seq"][split].extend(vals)
                    else:
                        raw[metr]["hist"][split].extend(vals)

    if not raw:
        return

    # ------------------------------------------------------------------
    # One HTML per metric (+ a top-level overall.html across PRINT_METRICS)
    # ------------------------------------------------------------------
    combined_rows: list[
        tuple[str, dict[str, dict[str, float | list[float]]], float]
    ] = []

    for metr, corr_data in raw.items():
        subplot_scores: dict[str, dict[str, float | list[float]]] = {}
        ordered_splits = split_order[metr]

        # -- Overall subplot (per-combo values from every corr type) --
        overall: dict[str, float | list[float]] = {}
        for split in ordered_splits:
            combined: list[float] = []
            for ct_vals in corr_data.values():
                combined.extend(ct_vals.get(split, []))
            overall[split] = combined if combined else float("nan")
        subplot_scores["Overall"] = overall

        # -- Per corr-type subplots (in fixed order, only if present) --
        # When only one corr type is present (e.g. a single-table dataset
        # with no sequential/unrolled marginals) the lone per-type subplot
        # would just duplicate "Overall", so suppress it.
        if len(corr_data) > 1:
            for ct_key, ct_label in _CORR_TYPES.items():
                if ct_key not in corr_data:
                    continue
                ct_dict = corr_data[ct_key]
                subplot_scores[ct_label] = {
                    split: list(ct_dict[split]) if ct_dict.get(split) else float("nan")
                    for split in ordered_splits
                }

        # Artifact path mirrors existing _overall folders
        pref = "assoc/" if metr in ASSOC_METRICS else ""
        path = f"distr/{pref}{metr}_overall/multiplot.html"
        fancy = _METRIC_FANCY.get(metr, metr.upper())

        # For lower-is-better metrics the worst tail is the upper one, so
        # flip the percentile (e.g. 5 → 95).
        percentile = (
            100.0 - percentile_lower
            if _METRIC_LOWER_IS_BETTER.get(metr, False)
            else percentile_lower
        )

        _render_multiplot(
            subplot_scores,
            f"{fancy} - Sweep",
            path,
            percentile=percentile,
        )

        if metr in PRINT_METRICS:
            combined_rows.append((fancy, subplot_scores, percentile))

    if combined_rows:
        _render_combined_multiplot(
            combined_rows,
            "Distribution Metrics — Sweep",
            "overall.html",
        )


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

    # Two way across parents
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
            parent_id_cols = [c for c in tids.columns if c != tids.index.name]
            for o in range(sval.order):
                ids_seq_prev = tids.join(tseq + o + 1, how="right").reset_index(
                    names=_JOIN_NAME
                )
                join_ids = ids_seq.merge(
                    ids_seq_prev,
                    on=[*parent_id_cols, sval.name],
                    how="inner",
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

    percentile_lower: float = 5

    def __init__(
        self,
        percentile_lower: float = 5,
        *args,
        _from_factory: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, _from_factory=_from_factory, **kwargs)
        self.percentile_lower = percentile_lower

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

                nonempty_stypes = set()
                for split, split_scores in split_scores_per_table.items():
                    for stype, type_scores in split_scores.items():
                        if stype not in lines:
                            lines[stype] = {}
                        if type_scores:
                            nonempty_stypes.add(stype)
                        lines[stype][split] = np.mean(type_scores) if type_scores else 0

                # Drop stypes (seq / hist) that have no data in any split for
                # this plot, so we don't render bars of height 0.
                for stype in list(lines.keys()):
                    if stype not in nonempty_stypes:
                        del lines[stype]

                # Aggregate runs (strip `rN`) for each stype: bar height = mean
                # across runs; CI caps from `_aggregate_runs_scalar`.
                lines_agg: dict[str, dict[str, float]] = {}
                cis_agg: dict[str, dict[str, tuple[float, float]]] = {}
                for stype, split_means in lines.items():
                    lines_agg[stype], cis_agg[stype] = _aggregate_runs_scalar(
                        split_means
                    )

                # Scale per-bar width to the number of stypes so the group
                # fills ~0.9 of each x-slot regardless of how many stypes are
                # plotted (otherwise a single stype leaves a huge empty gap).
                bar_width = 0.9 / max(1, len(lines_agg))

                l_res = 0
                split_scores = {}
                if combined:
                    per_split_overall = {
                        split: float(
                            np.nanmean([np.nanmean(v) for v in y.values() if len(v)])
                        )
                        for split, y in split_scores_per_table.items()
                    }
                    combined_means, combined_cis = _aggregate_runs_scalar(
                        per_split_overall
                    )
                    l_res = len(combined_means)
                    split_scores = combined_means
                    xs = np.arange(l_res)
                    container = ax.bar(xs, list(combined_means.values()))
                    if combined_cis and container.patches:
                        bar_color = container.patches[0].get_facecolor()
                        cap_xs, cap_h, cap_p5, cap_p95 = [], [], [], []
                        for j, b in enumerate(combined_means.keys()):
                            if b in combined_cis:
                                cap_xs.append(j)
                                cap_h.append(combined_means[b])
                                cap_p5.append(combined_cis[b][0])
                                cap_p95.append(combined_cis[b][1])
                        if cap_xs:
                            _draw_ci_caps(
                                ax,
                                np.array(cap_xs),
                                np.array(cap_h),
                                np.array(cap_p5),
                                np.array(cap_p95),
                                bar_color,
                                0.8,
                            )
                else:
                    for i, (stype, split_scores) in enumerate(lines_agg.items()):
                        l_res = len(split_scores)
                        x = np.arange(l_res)
                        container = ax.bar(
                            x + i * bar_width,
                            list(split_scores.values()),
                            bar_width,
                            label=fancy_names[stype],
                        )
                        stype_cis = cis_agg.get(stype, {})
                        if stype_cis and container.patches:
                            bar_color = container.patches[0].get_facecolor()
                            cap_xs, cap_h, cap_p5, cap_p95 = [], [], [], []
                            for j, b in enumerate(split_scores.keys()):
                                if b in stype_cis:
                                    cap_xs.append(j + i * bar_width)
                                    cap_h.append(split_scores[b])
                                    cap_p5.append(stype_cis[b][0])
                                    cap_p95.append(stype_cis[b][1])
                            if cap_xs:
                                _draw_ci_caps(
                                    ax,
                                    np.array(cap_xs),
                                    np.array(cap_h),
                                    np.array(cap_p5),
                                    np.array(cap_p95),
                                    bar_color,
                                    bar_width,
                                )

                ax.set_xlabel("Experiment")
                ax.set_ylabel(f"Mean Norm {metr.upper()}")
                ax.set_title(f"Overall Mean Norm {metr.upper()}")

                max_len = 0
                labels = [k.split(" ") for k in split_scores.keys()]
                for params in labels:
                    for param in params:
                        max_len = max(max_len, len(param))

                xtick_offset = (
                    0 if combined else max(0, (len(lines_agg) - 1)) / 2 * bar_width
                )
                ax.set_xticks(np.arange(l_res) + xtick_offset)
                if max_len > 15 or l_res > 7:
                    tick_labels = [" ".join(l) for l in labels]
                    rot = min(3 * l_res, 90)
                    ax.set_xticklabels(tick_labels)
                    plt.setp(
                        ax.get_xticklabels(), rotation=rot, horizontalalignment="right"
                    )
                else:
                    tick_labels = ["\n".join(l) for l in labels]
                    try:
                        ax.set_xticklabels(tick_labels)
                    except Exception:
                        logger.warning(f"Could not set tick labels.", exc_info=True)

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
                plt.close("all")

        _visualise_multiplot(overall_metr, percentile_lower=self.percentile_lower)
