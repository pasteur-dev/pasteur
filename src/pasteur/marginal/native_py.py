import numpy as np

from .native import marginal
from .numpy import AttrSelectors, CalculationData, CalculationInfo

Op = tuple[int, np.ndarray]


def calc_marginal(
    data: CalculationData,
    info: CalculationInfo,
    x: AttrSelectors,
    out: np.ndarray | None = None,
    simd: bool = True,
):
    """Calculates the 1 way marginal of the subsections of attributes x"""
    ops: list[Op] = []
    mul = 1
    for (table, attr, sel) in reversed(x):
        common = info.common[(table, attr)]
        l_mul = 1
        if isinstance(sel, dict):
            for i, (n, h) in enumerate(reversed(sel.items())):
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
