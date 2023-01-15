import numpy as np

from .native import marginal
from .numpy import AttrSelector, AttrSelectors

Op = tuple[int, np.ndarray]


def calc_marginal(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelector,
    p: AttrSelectors,
    partial: bool = False,
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

    # Handle parents
    ops: list[Op] = []
    mul = 1
    for attr_name, attr in p.items():
        common = attr.common
        l_mul = 1
        p_partial = partial and attr_name == x.name
        for i, (n, h) in enumerate(attr.cols.items()):
            if common == 0 or i == 0:
                ops.append((mul * l_mul, cols[n][h]))
            else:
                ops.append((mul * l_mul, cols_noncommon[n][h]))

            l_mul *= domains[n][h] - common

        if p_partial:
            mul *= l_mul
        else:
            mul *= l_mul + common
    p_dom = mul

    # Handle x
    common = x.common
    l_mul = 1
    for i, (n, h) in enumerate(x.cols.items()):
        if common == 0 or (i == 0 and not partial):
            ops.append((mul * l_mul, cols[n][h]))
        else:
            ops.append((mul * l_mul, cols_noncommon[n][h]))

        l_mul *= domains[n][h] - common

    if not partial:
        l_mul += common
    x_dom = l_mul
    dom = mul * l_mul

    if out is None:
        out = np.zeros((dom,), dtype=np.uint32)
    else:
        out = out.reshape((-1,))

    marginal(out, ops, simd)

    return out.reshape((x_dom, p_dom))


def calc_marginal_1way(
    cols: dict[str, list[np.ndarray]],
    cols_noncommon: dict[str, list[np.ndarray]],
    domains: dict[str, list[int]],
    x: AttrSelectors,
    out: np.ndarray | None = None,
    simd: bool = True,
):
    """Calculates the 1 way marginal of the subsections of attributes x"""

    ops: list[Op] = []
    mul = 1
    for attr in reversed(x.values()):
        common = attr.common
        l_mul = 1
        for i, (n, h) in enumerate(attr.cols.items()):
            if common == 0 or i == 0:
                ops.append((l_mul * mul, cols[n][h]))
            else:
                ops.append((l_mul * mul, cols_noncommon[n][h]))
            l_mul *= domains[n][h] - common
        mul *= l_mul + common

    if out is None:
        out = np.zeros((mul,), dtype=np.uint32)

    marginal(out, ops, simd)

    return out
