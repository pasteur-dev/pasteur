from functools import reduce
from itertools import chain
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..transform import Attribute, Attributes, get_dtype


def expand_table(
    attrs: Attributes, table: pd.DataFrame
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
        for name, col in attr.cols.items():
            col_hier = []
            col_noncommon = []
            col_dom = []

            lvl = col.lvl
            for height in range(lvl.height):
                domain = lvl.get_domain(height)
                col_dom.append(domain)

                col_lvl = lvl.get_mapping(height)[table[name]]
                col_lvl = col_lvl.astype(get_dtype(domain))
                col_hier.append(col_lvl)

                if attr.common > 0:
                    nc = np.where(col_lvl > attr.common, col_lvl - attr.common, 0)
                    col_noncommon.append(nc)

            domains[name] = col_dom
            cols[name] = col_hier
            cols_noncommon[name] = col_noncommon

    return cols, cols_noncommon, domains


class AttrSelector(NamedTuple):
    common: int
    cols: dict[str, int]


AttrSelectors = list[AttrSelector]


def calc_marginal(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelector,
    p: AttrSelectors,
    zero_fill: float | None = None,
):
    """Calculates the 1 way and 2 way marginals between the subsection of the
    hierarchical attribute x and the attributes p(arents)."""
    xp = [x] + p

    # Find integer dtype based on domain
    p_dom = 1
    for attr in p:
        for i, (n, h) in enumerate(attr.cols.items()):
            p_dom *= domains[n][h] - (attr.common if i > 0 else 0)
    x_dom = 1
    for i, (n, h) in enumerate(x.cols.items()):
        x_dom *= domains[n][h] - (attr.common if i > 0 else 0)

    dtype = get_dtype(p_dom * x_dom)

    n = len(next(iter(cols.values()))[0])
    _sum_nd = np.zeros((n,), dtype=dtype)
    _tmp_nd = np.zeros((n,), dtype=dtype)

    mul = 1
    for attr in reversed(xp):
        for i, (n, h) in enumerate(attr.cols.items()):
            common = attr.common
            if i == 0 or common == 0:
                np.multiply(cols[n][h], mul, out=_tmp_nd)
            else:
                np.multiply(cols_noncommon[n][h], mul, out=_tmp_nd)

            np.add(_sum_nd, _tmp_nd, out=_sum_nd)
            mul *= domains[n][h] - (common if i > 0 else 0)

    counts = np.bincount(_sum_nd, minlength=p_dom * x_dom)
    margin = counts.reshape(x_dom, p_dom).astype("float32")

    margin /= margin.sum()
    if zero_fill is not None:
        # Mutual info turns into NaN without this
        margin += zero_fill

    j_mar = margin
    x_mar = np.sum(margin, axis=1)
    p_mar = np.sum(margin, axis=0)

    return j_mar, x_mar, p_mar


x = AttrSelector(1, {"dod_year": 0, "dod_day": 1})
p = [
    AttrSelector(1, {"deathtime_year": 0, "deathtime_week": 1}),
    AttrSelector(0, {"admission_type": 0}),
]


def calc_marginal_1way(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelectors,
    zero_fill: float | None = None,
):
    """Calculates the 1 way marginal of the subsections of attributes x"""

    # Find integer dtype based on domain
    dom = 1
    for attr in p:
        for i, (n, h) in enumerate(attr.cols.items()):
            dom *= domains[n][h] - (attr.common if i > 0 else 0)
    dtype = get_dtype(dom)

    n = len(next(iter(cols.values()))[0])
    _sum_nd = np.zeros((n,), dtype=dtype)
    _tmp_nd = np.empty((n,), dtype=dtype)

    mul = 1
    for attr in reversed(x):
        for i, (n, h) in enumerate(attr.cols.items()):
            common = attr.common
            if i == 0 or common == 0:
                np.multiply(cols[n][h], mul, out=_tmp_nd)
            else:
                np.multiply(cols_noncommon[n][h], mul, out=_tmp_nd)

            np.add(_sum_nd, _tmp_nd, out=_sum_nd)
            mul *= domains[n][h] - (common if i > 0 else 0)

    counts = np.bincount(_sum_nd, minlength=dom)
    margin = counts.astype("float32")
    margin /= margin.sum()
    if zero_fill is not None:
        # Mutual info turns into NaN without this
        margin += zero_fill

    return margin.reshape(-1)
