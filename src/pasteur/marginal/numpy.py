from collections import defaultdict
from typing import Mapping, NamedTuple, Sequence, TypeGuard, cast

import numpy as np
import pandas as pd

from ..attribute import (
    Attributes,
    CatValue,
    Grouping,
    SeqAttributes,
    StratifiedValue,
    get_dtype,
)

ChildSelector = dict[str, int]
CommonSelector = int
TableSelector = str | tuple[str, int] | None
AttrName = str | tuple[str, ...]


AttrSelector = tuple[TableSelector, AttrName, ChildSelector | CommonSelector]
AttrSelectors = Sequence[AttrSelector]

CalculationData = dict[tuple[TableSelector, str, bool], list[np.ndarray]]


class CalculationInfo(NamedTuple):
    domains: dict[tuple[TableSelector, str], list[int]]
    common: dict[tuple[TableSelector, AttrName | None], int]
    common_names: dict[tuple[TableSelector, AttrName], str]


def _calc_common_rec(common: Grouping, col: Grouping):
    assert len(col) == len(common)

    ofs = 0
    for k, l in zip(common, col):
        if isinstance(k, Grouping):
            assert isinstance(l, Grouping)

            ofs += _calc_common_rec(k, l)
        elif isinstance(l, Grouping):
            return ofs
        else:
            ofs += 1

    return ofs


def _calc_common_seq_rec(seq: Grouping, col: Grouping):
    assert len(col) == len(seq)

    ofs = 0
    for k, l in zip(seq, col):
        if isinstance(k, Grouping) and isinstance(l, Grouping):
            if len(k) != len(l):
                return ofs

            ofs += _calc_common_seq_rec(k, l)
        elif isinstance(l, Grouping) or isinstance(l, Grouping):
            return ofs
        else:
            ofs += 1

    return ofs


def _calc_common(col: CatValue, common: CatValue | None):
    if not common:
        return 0

    if not isinstance(col, StratifiedValue):
        return 0

    if not isinstance(common, StratifiedValue):
        return 0

    return _calc_common_rec(common.head, col.head)


def _calc_common_seq(col: CatValue, seq: CatValue | None):
    if not seq:
        return 0

    if not isinstance(col, StratifiedValue):
        return 0

    if not isinstance(seq, StratifiedValue):
        return 0

    return _calc_common_seq_rec(seq.head, col.head)


def _map_column(table: pd.DataFrame, col: CatValue, common: CatValue | None):
    cols = []
    cols_noncommon = []
    domains = []
    common_num = _calc_common(col, common)

    for height in range(col.height):
        domain = col.get_domain(height)
        domains.append(domain)

        col_lvl = col.get_mapping(height)[table[col.name]]
        col_lvl = col_lvl.astype(get_dtype(domain))
        cols.append(col_lvl)

        if common_num > 0:
            non_common = np.where(col_lvl > common_num, col_lvl - common_num, 0)
            cols_noncommon.append(non_common)

    return cols, cols_noncommon, domains, common_num


def expand_table(
    attrs: Mapping[str | None, Attributes | SeqAttributes] = {},
    tables: dict[TableSelector, pd.DataFrame] = {},
    *,
    prealloc: CalculationData | None = None,
) -> tuple[CalculationData, CalculationInfo]:
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
    out = {}
    domains = {}
    common = {}
    common_names = {}

    # For hierarchical data, attributes are provided per table
    # We separate them into historical data, which are optional,
    # and the main table, which will have the name 'None'
    # The reason the main table is set to None that it may be partially synthesized,
    # so its name will be in the historical data.
    for table_name, table_attrs in attrs.items():
        table_name: str | None
        table_attrs: Attributes | SeqAttributes

        # For each table, we have multiple versions when there are sequential data
        attr_sets: list[
            tuple[tuple[str, int] | str | None, StratifiedValue | None, Attributes]
        ]
        if isinstance(table_attrs, SeqAttributes):
            assert isinstance(table_name, str)
            attr_sets = [
                ((table_name, k), table_attrs.seq, v)
                for k, v in table_attrs.hist.items()
            ]
            if table_attrs.attrs is not None:
                attr_sets.append((table_name, None, table_attrs.attrs))
        else:
            attr_sets = [(table_name, None, table_attrs)]

        for table_sel, seq, attr_set in attr_sets:
            table_common = 1024
            for attr in attr_set.values():
                vals = list(attr.vals.items())
                if attr.common:
                    common_name = attr.common.name
                    vals.append((common_name, attr.common))
                    common_names[(table_sel, attr.name)] = common_name

                attr_common = 1024
                for name, col in vals:
                    if not isinstance(col, CatValue):
                        continue

                    if name not in tables[table_sel]:
                        continue

                    col_hier, col_noncommon, col_domain, col_common = _map_column(
                        tables[table_sel], col, attr.common
                    )

                    if prealloc:
                        for height, data in enumerate(col_hier):
                            prealloc[(table_sel, name, False)][height][:] = data
                        for height, data in enumerate(col_noncommon):
                            prealloc[(table_sel, name, True)][height][:] = data
                    else:
                        out[(table_sel, name, False)] = col_hier
                        if col_noncommon:
                            out[(table_sel, name, True)] = col_noncommon

                    domains[(table_sel, name)] = col_domain
                    attr_common = min(attr_common, col_common)
                    table_common = min(table_common, _calc_common_seq(col, seq))

                common[(table_sel, attr.name)] = (
                    attr_common if attr_common < 1000 else 0
                )

    return prealloc or out, CalculationInfo(domains, common, common_names)


def get_domains(attrs: Attributes) -> dict[str, list[int]]:
    domains = {}
    for attr in attrs.values():
        for name, col in attr.vals.items():
            col = cast(CatValue, col)
            col_dom = []

            for height in range(col.height):
                domain = col.get_domain(height)
                col_dom.append(domain)

            domains[name] = col_dom

    return domains


def calc_marginal(
    data: CalculationData,
    info: CalculationInfo,
    x: AttrSelectors,
    out: np.ndarray | None = None,
):
    """Calculates the 1 way marginal of the subsections of attributes x"""

    # Find integer dtype based on domain
    dom = 1
    for table, attr, sel in x:
        if isinstance(sel, dict):
            common = info.common[(table, attr)]
            l_dom = 1
            for n, h in sel.items():
                l_dom *= info.domains[(table, n)][h] - common
            dom *= l_dom + common
        else:
            dom *= info.domains[(table, info.common_names[(table, attr)])][sel]

    dtype = get_dtype(dom)

    ref_table, *_ = next(iter(x))
    n = len(next(iter(d for k, d in data.items() if k[0] == ref_table))[0])

    _sum_nd = np.zeros((n,), dtype=dtype)
    _tmp_nd = np.empty((n,), dtype=dtype)

    mul = 1
    for table, attr, sel in reversed(x):
        common = info.common[(table, attr)]
        if isinstance(sel, dict):
            l_mul = 1
            for i, (n, h) in enumerate(reversed(sel.items())):
                if common == 0 or i == 0:
                    np.multiply(
                        data[(table, n, False)][h].reshape(-1),
                        mul * l_mul,
                        out=_tmp_nd,
                        dtype=dtype,
                    )
                else:
                    np.multiply(
                        data[(table, n, True)][h].reshape(-1),
                        mul * l_mul,
                        out=_tmp_nd,
                        dtype=dtype,
                    )

                np.add(_sum_nd, _tmp_nd, out=_sum_nd, dtype=dtype)
                l_mul *= info.domains[(table, n)][h] - common
            mul *= l_mul + common
        else:
            np.multiply(
                data[(table, info.common_names[(table, attr)], False)][sel].reshape(-1),
                mul,
                out=_tmp_nd,
                dtype=dtype,
            )
            mul *= info.domains[(table, info.common_names[(table, attr)])][sel]

    counts = np.bincount(_sum_nd, minlength=dom)
    if out is not None:
        out += counts
    else:
        out = counts

    return out
