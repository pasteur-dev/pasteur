import numpy as np

from .native import marginal
from .numpy import AttrSelector, AttrSelectors, CalculationData, CalculationInfo

Op = tuple[int, np.ndarray]


def calc_marginal(
    data: CalculationData,
    info: CalculationInfo,
    x: AttrSelector,
    p: AttrSelectors,
    out: np.ndarray | None = None,
    simd: bool = True,
):
    """Calculates the 1 way and 2 way marginals between the subsection of the
    hierarchical attribute x and the attributes p(arents)."""

    # Keep only non-common items if there is a parent to source the others
    # if partial:
    #     n = next(iter(x.cols))
    #     mask = cols[n][0] >= x.common
    #     x_dom = x_dom - x.common

    # Find attribute domains
    p_dom = 1
    for (table, attr, sel) in p:
        if isinstance(sel, dict):
            common = info.common[(table, attr)]
            l_dom = 1
            for n, h in sel.items():
                l_dom *= info.domains[(table, n)][h] - common
            p_dom *= l_dom + common
        else:
            p_dom *= info.domains[(table, info.common_names[(table, attr)])][sel]

    table, attr, sel = x
    if isinstance(sel, dict):
        common = info.common[(table, attr)]
        x_dom = 1
        for n, h in sel.items():
            x_dom *= info.domains[(table, n)][h] - common
        x_dom *= common
    else:
        x_dom = info.domains[(table, info.common_names[(table, attr)])][sel]

    if out:
        out = out.reshape((-1,))

    out = calc_marginal_1way(data, info, [x, *p], out, simd)
    return out.reshape((x_dom, p_dom))


def calc_marginal_1way(
    data: CalculationData,
    info: CalculationInfo,
    x: AttrSelectors,
    out: np.ndarray | None = None,
    simd: bool = True,
):
    """Calculates the 1 way marginal of the subsections of attributes x"""

    ops: list[Op] = []
    mul = 1
    for (table, attr, sel) in x:
        common = info.common[(table, attr)]
        l_mul = 1
        if isinstance(sel, dict):
            for i, (n, h) in enumerate(sel.items()):
                if common == 0 or i == 0:
                    ops.append((l_mul * mul, data[(table, n, False)][h]))
                else:
                    ops.append((l_mul * mul, data[(table, n, True)][h]))
                l_mul *= info.domains[(table, n)][h] - common
            mul *= l_mul + common
        else:
            ops.append(
                (mul, data[(table, info.common_names[(table, attr)], False)][sel])
            )
            mul *= info.domains[(table, info.common_names[(table, attr)])][sel]

    if out is None:
        out = np.zeros((mul,), dtype=np.uint32)

    marginal(out, ops, simd)

    return out
