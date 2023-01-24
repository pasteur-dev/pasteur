from typing import NamedTuple, cast

import numpy as np
import pandas as pd

from ..attribute import Attributes, get_dtype, IdxValue

ZERO_FILL = 1e-24


class AttrSelector(NamedTuple):
    name: str
    common: int
    cols: dict[str, int]


AttrSelectors = dict[str, AttrSelector]

 
def expand_table(
    attrs: Attributes,
    table: pd.DataFrame,
    *,
    out_cols: dict[str, list[np.ndarray]] | None = None,
    out_noncommon: dict[str, list[np.ndarray]] | None = None
) -> tuple[
    dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, list[int]]
]:
    """Takes in the raw idx encoded table and precalculates all column-height combinations
    of hierarchical attributes, with special versions for marginal calculations with
    attributes that have an NA value.

    Returns:
    cols: A dictionary, list structure that can be accessed as cols[name][height]
    to get each row's group with column <name> and <height> height.

    cols_noncommon: A second version that's offset by 1 or 2 depending on whether the parent
    attribute has na values/unknown values (+1 for each).

    domain: The same structure containing the domain of each <name>,<height> combination.

    It is then possible to calculate the marginal of an attribute with cols a,b,c,
    heights d,e,f and na values by doing the following:

    ```
    groups = col[a][d] + domain[a][d]*(cols_noncommon[b][e] + (domain[b][e]-1)*cols_noncommon[c][f])
    np.bincount(groups, minlength=domain[a][d]*(domain[b][e] - 1)*(domain[c][f] - 1))
    ```

    The above expression only requires one vector multiplication and one vector
    addition per attribute added to the marginal, with `bincount()` scaling linearly
    with dataset size `n`.

    For a dataset with size n=500k and 6 columns used in the marginal, it has a
    wallsize of 1.3ms, to 30ms of np.histogramdd.
    """
    cols = {}
    cols_noncommon = {}

    domains = {}
    for attr in attrs.values():
        for name, col in attr.vals.items():
            if name not in table:
                continue

            col = cast(IdxValue, col)
            col_hier = []
            col_noncommon = []
            col_dom = []

            for height in range(col.height):
                domain = col.get_domain(height)
                col_dom.append(domain)

                col_lvl = col.get_mapping(height)[table[name]]
                col_lvl = col_lvl.astype(get_dtype(domain))
                if out_cols:
                    out_cols[name][height][:] = col_lvl
                else:
                    col_hier.append(col_lvl)

                if attr.common > 0:
                    nc = np.where(col_lvl > attr.common, col_lvl - attr.common, 0)

                    if out_noncommon:
                        out_noncommon[name][height][:] = col_lvl
                    else:
                        col_noncommon.append(nc)

            domains[name] = col_dom
            cols[name] = col_hier
            cols_noncommon[name] = col_noncommon

    return out_cols or cols, out_noncommon or cols_noncommon, domains


def get_domains(attrs: Attributes) -> dict[str, list[int]]:
    domains = {}
    for attr in attrs.values():
        for name, col in attr.vals.items():
            col = cast(IdxValue, col)
            col_dom = []

            for height in range(col.height):
                domain = col.get_domain(height)
                col_dom.append(domain)

            domains[name] = col_dom

    return domains


def calc_marginal(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelector,
    p: AttrSelectors,
    partial: bool = False,
    out: np.ndarray | None = None,
):
    """Calculates the 1 way and 2 way marginals between the subsection of the
    hierarchical attribute x and the attributes p(arents)."""

    # Find integer dtype based on domain
    p_dom = 1
    for attr in p.values():
        common = attr.common
        l_dom = 1
        for i, (n, h) in enumerate(attr.cols.items()):
            l_dom *= domains[n][h] - common
        p_dom *= l_dom + common
    x_dom = 1
    for i, (n, h) in enumerate(x.cols.items()):
        x_dom *= domains[n][h] - x.common
    x_dom += x.common

    dtype = get_dtype(p_dom * x_dom)

    n = len(next(iter(cols.values()))[0])
    _sum_nd = np.zeros((n,), dtype=dtype)
    _tmp_nd = np.zeros((n,), dtype=dtype)

    # Handle parents
    mul = 1
    for attr_name, attr in p.items():
        common = attr.common
        l_mul = 1
        p_partial = partial and attr_name == x.name
        for i, (n, h) in enumerate(attr.cols.items()):
            if common == 0 or i == 0:
                np.multiply(cols[n][h], mul * l_mul, out=_tmp_nd, dtype=dtype)
            else:
                np.multiply(cols_noncommon[n][h], mul * l_mul, out=_tmp_nd, dtype=dtype)

            np.add(_sum_nd, _tmp_nd, out=_sum_nd, dtype=dtype)
            l_mul *= domains[n][h] - common

        if p_partial:
            mul *= l_mul
        else:
            mul *= l_mul + common

    # Handle x
    common = x.common
    for i, (n, h) in enumerate(x.cols.items()):
        if common == 0 or (i == 0 and not partial):
            np.multiply(cols[n][h], mul, out=_tmp_nd, dtype=dtype)
        else:
            np.multiply(cols_noncommon[n][h], mul, out=_tmp_nd, dtype=dtype)

        np.add(_sum_nd, _tmp_nd, out=_sum_nd, dtype=dtype)
        mul *= domains[n][h] - common

    # Keep only non-common items if there is a parent to source the others
    if partial:
        n = next(iter(x.cols))
        _sum_nd = _sum_nd[cols[n][0] >= common]
        x_dom = x_dom - x.common

    counts = np.bincount(_sum_nd, minlength=p_dom * x_dom)
    if out is not None:
        out = out.reshape((-1,))
        out += counts
    else:
        out = counts

    return out.reshape((x_dom, p_dom))


def calc_marginal_1way(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelectors,
    out: np.ndarray | None = None,
):
    """Calculates the 1 way marginal of the subsections of attributes x"""

    # Find integer dtype based on domain
    dom = 1
    for attr in x.values():
        common = attr.common
        l_dom = 1
        for i, (n, h) in enumerate(attr.cols.items()):
            l_dom *= domains[n][h] - common
        dom *= l_dom + common
    dtype = get_dtype(dom)

    n = len(next(iter(cols.values()))[0])
    _sum_nd = np.zeros((n,), dtype=dtype)
    _tmp_nd = np.empty((n,), dtype=dtype)

    mul = 1
    for attr in reversed(x.values()):
        common = attr.common
        l_mul = 1
        for i, (n, h) in enumerate(attr.cols.items()):
            if common == 0 or i == 0:
                np.multiply(cols[n][h], mul * l_mul, out=_tmp_nd, dtype=dtype)
            else:
                np.multiply(cols_noncommon[n][h], mul * l_mul, out=_tmp_nd, dtype=dtype)

            np.add(_sum_nd, _tmp_nd, out=_sum_nd, dtype=dtype)
            l_mul *= domains[n][h] - common
        mul *= l_mul + common

    counts = np.bincount(_sum_nd, minlength=dom)
    if out is not None:
        out += counts
    else:
        out = counts

    return out


def normalize(counts: np.ndarray, zero_fill: float | None = ZERO_FILL):
    margin = counts.astype("float32")

    margin /= margin.sum()
    if zero_fill is not None:
        # Mutual info turns into NaN without this
        margin += zero_fill

    j_mar = margin
    x_mar = np.sum(margin, axis=1)
    p_mar = np.sum(margin, axis=0)

    return j_mar, x_mar, p_mar


def normalize_1way(counts: np.ndarray, zero_fill: float | None = ZERO_FILL):
    margin = counts.astype("float32")

    margin /= margin.sum()
    if zero_fill is not None:
        # Mutual info turns into NaN without this
        margin += zero_fill

    return margin
